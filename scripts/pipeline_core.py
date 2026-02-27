import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from experience_filter import ExperienceFilter
from generate_JsT_peptides import (  # noqa: F401
    ALPHA as JST_ALPHA,
    ALPHA_BAR as JST_ALPHA_BAR,
    BETA_SCHEDULE as JST_BETA_SCHEDULE,
    NUM_DIFFUSION_STEPS as JST_NUM_DIFFUSION_STEPS,
    sample_ddpm_x_prediction,
    trim_pad as jst_trim_pad,
    onehot_to_token_with_temperature as jst_onehot_to_token,
)
from generate_peptides import (  # noqa: F401
    ALPHA,
    ALPHA_BAR,
    BETA_SCHEDULE,
    NUM_DIFFUSION_STEPS,
    calculate_features_for_sequences,
    onehot_to_token_with_temperature,
    sample_ddpm,
    trim_pad,
)
from model import DiffusionModel
from model_JsT import DiffusionModel as JsTDiffusionModel
from utils import DEFAULT_MAX_LEN, vocab


@dataclass
class ConditioningSpec:
    min_length: int = 6
    max_length: int = DEFAULT_MAX_LEN
    min_charge: float = 3.5
    min_amphipathicity: float = 0.35
    ph: float = 7.4


@dataclass
class PipelineStats:
    batches: int = 0
    total_generated: int = 0
    heuristic_passed: int = 0
    classification_checked: int = 0
    heuristic_failures: Counter = field(default_factory=Counter)
    total_candidates: int = 0
    duration_sec: float = 0.0


class SequenceGenerator:
    def __init__(self, max_len: int = DEFAULT_MAX_LEN, device: torch.device = None):
        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_batch(self, batch_size: int, temperature: float) -> List[str]:
        raise NotImplementedError

    def generate_sequences(self, total: int, batch_size: int, temperature: float) -> List[str]:
        sequences: List[str] = []
        while len(sequences) < total:
            sequences.extend(self.generate_batch(batch_size, temperature))
        return sequences[:total]


class DiffusionSequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        model_path: Path,
        max_len: int = DEFAULT_MAX_LEN,
        device: torch.device = None,
    ):
        super().__init__(max_len, device)
        self.model = DiffusionModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=128,
            hidden_dim=512,
            max_len=max_len,
            pad_idx=vocab.pad_idx,
            dropout_rate=0.7,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def generate_batch(self, batch_size: int, temperature: float) -> List[str]:
        shape = (batch_size, self.max_len, vocab.vocab_size)
        gen_x = sample_ddpm(
            self.model,
            NUM_DIFFUSION_STEPS,
            shape,
            self.device,
            ALPHA,
            ALPHA_BAR,
            BETA_SCHEDULE,
        )
        token_tensor = onehot_to_token_with_temperature(gen_x, temperature=temperature)
        return self._tokens_to_sequences(token_tensor)

    @staticmethod
    def _tokens_to_sequences(token_tensor: torch.Tensor) -> List[str]:
        sequences = []
        for token_seq in token_tensor.cpu().tolist():
            seq = trim_pad(token_seq, vocab.pad_idx)
            aa_seq = vocab.decode(seq)
            if aa_seq:
                sequences.append(aa_seq)
        return sequences


class JsTSequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        model_path: Path,
        max_len: int = DEFAULT_MAX_LEN,
        device: torch.device = None,
    ):
        super().__init__(max_len, device)
        self.model = JsTDiffusionModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            max_len=max_len,
            pad_idx=vocab.pad_idx,
            dropout_rate=0.5,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def generate_batch(self, batch_size: int, temperature: float) -> List[str]:
        shape = (batch_size, self.max_len, vocab.vocab_size)
        gen_x = sample_ddpm_x_prediction(
            self.model,
            JST_NUM_DIFFUSION_STEPS,
            shape,
            self.device,
            JST_ALPHA,
            JST_ALPHA_BAR,
            JST_BETA_SCHEDULE,
        )
        token_tensor = jst_onehot_to_token(gen_x, temperature=temperature)
        return self._tokens_to_sequences(token_tensor)

    @staticmethod
    def _tokens_to_sequences(token_tensor: torch.Tensor) -> List[str]:
        sequences = []
        for token_seq in token_tensor.cpu().tolist():
            seq = jst_trim_pad(token_seq, vocab.pad_idx)
            aa_seq = vocab.decode(seq)
            if aa_seq:
                sequences.append(aa_seq)
        return sequences


def load_predictive_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_heuristics(
    sequence: str,
    filter_obj: ExperienceFilter,
    spec: ConditioningSpec,
) -> Tuple[List[str], float, float]:
    reasons: List[str] = []
    if not filter_obj.check_length(sequence, spec.min_length, spec.max_length):
        reasons.append("length")
    charge = filter_obj.calculate_net_charge_at_ph(sequence, spec.ph)
    if charge < spec.min_charge:
        reasons.append("charge")
    amphipathicity = filter_obj.calculate_amphipathicity(sequence)
    if amphipathicity < spec.min_amphipathicity:
        reasons.append("amphipathicity")
    return reasons, charge, amphipathicity


def classify_sequences(
    sequences: Iterable[str],
    classifier,
    amp_threshold: float,
    heuristics_map: defaultdict,
    stats: PipelineStats,
) -> List[dict]:
    sequence_list = list(sequences)
    if not sequence_list:
        return []
    features, sequence_series = calculate_features_for_sequences(sequence_list)
    if features.empty:
        return []
    probs = classifier.predict_proba(features)[:, 1]
    stats.classification_checked += len(sequence_series)
    results = []
    for seq, prob in zip(sequence_series.tolist(), probs.tolist()):
        heuristics = heuristics_map.get(seq)
        if not heuristics:
            continue
        heur_info = heuristics.pop(0)
        if prob < amp_threshold:
            continue
        results.append(
            {
                "sequence": seq,
                "probability": prob,
                "net_charge": heur_info["charge"],
                "amphipathicity": heur_info["amphipathicity"],
            }
        )
    stats.total_candidates += len(results)
    return results


def build_heuristics_map(info_list: List[dict]) -> defaultdict:
    heuristics_map = defaultdict(list)
    for info in info_list:
        heuristics_map[info["sequence"]].append(info)
    return heuristics_map


def parse_device(device_str: str) -> torch.device:
    if device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def instantiate_generator(
    generator_name: str, model_path: Path, max_len: int, device: torch.device
) -> SequenceGenerator:
    if generator_name == "diffusion":
        return DiffusionSequenceGenerator(model_path=model_path, max_len=max_len, device=device)
    if generator_name == "jst":
        return JsTSequenceGenerator(model_path=model_path, max_len=max_len, device=device)
    raise ValueError(f"Unknown generator: {generator_name}")
