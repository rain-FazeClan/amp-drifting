import torch
import os
import pickle
import numpy as np
import pandas as pd
import argparse
# 【修改点1】从 model_new 导入，确保使用的是针对 x-prediction 优化的架构
from model_JsT import DiffusionModel, EMBEDDING_DIM, HIDDEN_DIM, get_diffusion_beta_schedule, DROPOUT_RATE
from utils import vocab, DEFAULT_MAX_LEN, PAD_TOKEN
from featured_generated import calculate_all_descriptors

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Generate AMP peptides using JiT (x-prediction) Diffusion Model')

    # 【修改点2】默认模型路径改为新训练的模型
    parser.add_argument('--diffusion_model', type=str,
                        default='models/diffusion_model_x_pred.pth',
                        help='Path to trained x-prediction diffusion model')
    
    parser.add_argument('--classifier_model', type=str,
                        default='models/predictive_model.pkl',
                        help='Path to trained classifier model')

    # 生成参数
    parser.add_argument('--num_generate', type=int, default=4000,
                        help='Total number of peptides to generate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for generation')
    parser.add_argument('--max_len', type=int, default=DEFAULT_MAX_LEN,
                        help=f'Maximum sequence length')

    # 采样参数
    parser.add_argument('--temperature', type=float, default=1.0, # x-pred 通常不需要太高的温度
                        help='Sampling temperature for diversity (default: 1.0)')
    parser.add_argument('--min_length', type=int, default=6,
                        help='Minimum valid sequence length')

    # 过滤参数
    parser.add_argument('--amp_threshold', type=float, default=0.8,
                        help='AMP classification threshold')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results/generated_peptides_JsT',
                        help='Output directory for results')
    # 【修改点3】默认输出文件名区分
    parser.add_argument('--output_file', type=str, default='candidate_amps_JsT.csv',
                        help='Output filename')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Save checkpoint every N generated candidates')

    # 其他参数
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


# Diffusion参数 - 与训练保持一致
NUM_DIFFUSION_STEPS = 1000
BETA_SCHEDULE = get_diffusion_beta_schedule(NUM_DIFFUSION_STEPS, beta_start=1e-3, beta_end=0.05)
ALPHA = 1. - BETA_SCHEDULE
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)


# 【修改点4】核心创新：使用 x-prediction 的采样逻辑
def sample_ddpm_x_prediction(model, num_steps, shape, device, alpha, alpha_bar, beta_schedule):
    """
    基于 x0 预测的采样函数 (JiT/Back-to-Basics Approach)
    原理：利用模型预测的 x0 直接计算后验分布均值
    """
    model.eval()
    # 从纯高斯噪声开始
    x = torch.randn(shape, device=device)

    with torch.no_grad():
        for t_ in reversed(range(num_steps)):
            t = torch.full((shape[0],), t_, device=device, dtype=torch.long)
            
            # 1. 模型直接预测 x_start (Logits)
            pred_logits = model(x, t)
            
            # 2. 将 Logits 视为估计的 x0 
            # 在连续空间扩散中，我们直接使用输出作为 x0 的估计
            pred_x0 = pred_logits 

            # 3. 准备扩散参数
            beta_t = beta_schedule.to(device)[t_]
            alpha_t = alpha.to(device)[t_]
            alpha_bar_t = alpha_bar.to(device)[t_]
            
            if t_ > 0:
                alpha_bar_prev = alpha_bar.to(device)[t_ - 1]
            else:
                alpha_bar_prev = torch.tensor(1.0).to(device)

            # 4. 计算后验分布均值 (Posterior Mean)
            # 公式: mu_t = coef_x0 * x0_pred + coef_xt * x_t
            coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
            coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
            
            mu_t = coef_x0 * pred_x0 + coef_xt * x
            
            # 5. 计算后验分布方差并采样
            if t_ > 0:
                # 使用对数方差更稳定
                posterior_variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                noise = torch.randn_like(x)
                x = mu_t + torch.sqrt(posterior_variance) * noise
            else:
                x = mu_t # 最后一步直接取均值

    model.train()
    return x


def onehot_to_token_with_temperature(x, temperature=1.0):
    """温度采样函数"""
    # x 是 logits，除以温度控制分布平滑度
    probs = torch.softmax(x / temperature, dim=-1)
    batch, seq, vocab_size = probs.shape
    probs_2d = probs.view(-1, vocab_size)
    sampled = torch.multinomial(probs_2d, 1).view(batch, seq)
    return sampled


def trim_pad(seq, pad_idx):
    return seq[:seq.index(pad_idx)] if pad_idx in seq else seq


def calculate_features_for_sequences(sequences):
    """计算序列特征的函数 (保持不变)"""
    valid_sequences = [seq for seq in sequences if len(seq) > 0 and all(aa in vocab.word_to_idx for aa in seq)]
    if not valid_sequences:
        print("Warning: No valid sequences to calculate features for.")
        return pd.DataFrame(), []

    count = 1
    descriptors_list = []
    # 这里假设 calculate_all_descriptors 可以在外部导入或在此定义
    # 为保持代码简洁，假设它从 featured_generated 导入成功
    for seq in valid_sequences:
        try:
            descriptors = calculate_all_descriptors(seq, count)
            descriptors_list.append(descriptors)
            count += 1
        except Exception as e:
            continue

    if not descriptors_list:
        return pd.DataFrame(), []

    feature_df = pd.DataFrame(descriptors_list)
    feature_df['Sequence'] = valid_sequences
    
    # 处理可能的非数值列
    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if feature_df[col].isnull().any():
            feature_df[col] = feature_df[col].fillna(feature_df[col].mean())

    return feature_df.drop('Sequence', axis=1, errors='ignore'), feature_df['Sequence']


def generate_and_filter_peptides(diffusion_model_path, classifier_path, num_to_generate, batch_size_gen,
                               output_dir, max_len=DEFAULT_MAX_LEN, temperature=1.0, min_length=6,
                               amp_threshold=0.8, output_file='candidate_amps_JsT.csv', verbose=False,
                               checkpoint_interval=1000):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Device: {device}")
        print(f"Mode: JiT (x-prediction) Generation")
        print(f"Parameters:")
        print(f"  - Model: {diffusion_model_path}")
        print(f"  - Classifier: {classifier_path}")
        print(f"  - Generation: {num_to_generate} peptides")
        print(f"  - Sampling Temp: {temperature}")
        print("-" * 50)

    print(f"Loading x-prediction Diffusion model from {diffusion_model_path}")

    # 初始化模型 (使用 model_new 中的定义)
    model = DiffusionModel(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          max_len=max_len,
                          pad_idx=vocab.pad_idx,
                          dropout_rate=DROPOUT_RATE).to(device)

    try:
        model.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model state dict: {e}")
        print("Please ensure you are loading a model trained with train_new.py (x-prediction architecture)")
        return

    model.eval()

    print(f"Loading Classifier model from {classifier_path}")
    try:
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        print("Classifier loaded successfully.")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return

    print(f"Generating and filtering {num_to_generate} peptides...")
    generated_candidate_sequences = []
    total_generated = 0
    total_valid = 0
    
    # 断点恢复：尝试加载已有进度
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pkl')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        generated_candidate_sequences = checkpoint_data.get('sequences', [])
        total_generated = checkpoint_data.get('total_generated', 0)
        total_valid = checkpoint_data.get('total_valid', 0)
        print(f"Resumed from checkpoint: {len(generated_candidate_sequences)} candidates already generated")
    
    last_checkpoint_count = len(generated_candidate_sequences)

    with torch.no_grad():
        batch_count = 0
        for _ in range(0, num_to_generate, batch_size_gen):
            current_batch_size = min(batch_size_gen, num_to_generate - len(generated_candidate_sequences))
            if current_batch_size <= 0:
                break

            batch_count += 1
            if verbose:
                print(f"\nBatch {batch_count}: Generating {current_batch_size} sequences...")

            # 【修改点5】调用新的采样函数
            gen_x = sample_ddpm_x_prediction(model, NUM_DIFFUSION_STEPS,
                                           (current_batch_size, max_len, vocab.vocab_size),
                                           device, ALPHA, ALPHA_BAR, BETA_SCHEDULE)

            # 温度采样
            gen_tokens = onehot_to_token_with_temperature(gen_x, temperature=temperature)

            decoded_sequences = []
            for seq in gen_tokens.cpu().tolist():
                seq = trim_pad(seq, vocab.pad_idx)
                aa_seq = vocab.decode(seq)
                total_generated += 1
                if min_length <= len(aa_seq) <= max_len:
                    decoded_sequences.append(aa_seq)
                    total_valid += 1

            if verbose:
                print(f"  Valid sequences: {len(decoded_sequences)}/{current_batch_size}")

            if not decoded_sequences:
                continue

            # 特征计算与分类筛选
            try:
                generated_features, original_generated_sequences = calculate_features_for_sequences(decoded_sequences)
                if generated_features.empty:
                    continue
                
                # 确保特征列与分类器训练时一致
                # 这里假设 classifier 是 sklearn 模型，且输入特征顺序一致
                # 实际应用中可能需要对齐列名
                
                predicted_proba = classifier.predict_proba(generated_features)[:, 1]
                amp_candidates_indices = [i for i, proba in enumerate(predicted_proba) if proba >= amp_threshold]
                
                for idx in amp_candidates_indices:
                    generated_candidate_sequences.append(original_generated_sequences.iloc[idx])

                if verbose:
                    print(f"  AMP candidates: {len(amp_candidates_indices)}")

            except Exception as e:
                print(f"Error during classification filtering: {e}. Skipping batch.")
                continue

            if not verbose:
                print(f"Batch {batch_count}: Found {len(amp_candidates_indices) if 'amp_candidates_indices' in locals() else 0} candidates. Total: {len(generated_candidate_sequences)}")
            
            # 每checkpoint_interval个候选保存一次断点
            if len(generated_candidate_sequences) - last_checkpoint_count >= checkpoint_interval:
                checkpoint_data = {
                    'sequences': generated_candidate_sequences,
                    'total_generated': total_generated,
                    'total_valid': total_valid
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                last_checkpoint_count = len(generated_candidate_sequences)
                print(f"Checkpoint saved: {len(generated_candidate_sequences)} candidates")

    print(f"\nFinished generation. Found {len(generated_candidate_sequences)} potential AMP candidates.")
    print(f"Generation efficiency: {total_valid}/{total_generated} valid sequences ({100*total_valid/total_generated:.1f}%)")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    candidate_df = pd.DataFrame({'Sequence': generated_candidate_sequences})
    candidate_df.to_csv(output_path, index=False)
    print(f"Candidate AMPs saved to {output_path}")

    # 计算统计信息
    if len(generated_candidate_sequences) > 0:
        avg_length = sum(len(seq) for seq in generated_candidate_sequences) / len(generated_candidate_sequences)
        unique_sequences = len(set(generated_candidate_sequences))
        diversity_score = unique_sequences / len(generated_candidate_sequences)

        print(f"\nGeneration Statistics (JsT Model):")
        print(f"Average sequence length: {avg_length:.1f}")
        print(f"Unique sequences: {unique_sequences}/{len(generated_candidate_sequences)}")
        print(f"Diversity score: {diversity_score:.4f}")

        # 保存详细统计
        stats_path = os.path.join(output_dir, 'generation_stats_JsT.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Generation Statistics (JsT/x-prediction)\n")
            f.write(f"======================================\n")
            f.write(f"Total candidates: {len(generated_candidate_sequences)}\n")
            f.write(f"Diversity score: {diversity_score:.4f}\n")
            f.write(f"Sampling Temperature: {temperature}\n")
        print(f"Stats saved to {stats_path}")
    
    # 生成完成后删除checkpoint文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file removed (generation complete)")

if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    generate_and_filter_peptides(
        diffusion_model_path=args.diffusion_model,
        classifier_path=args.classifier_model,
        num_to_generate=args.num_generate,
        batch_size_gen=args.batch_size,
        output_dir=args.output_dir,
        max_len=args.max_len,
        temperature=args.temperature,
        min_length=args.min_length,
        amp_threshold=args.amp_threshold,
        output_file=args.output_file,
        verbose=args.verbose,
        checkpoint_interval=args.checkpoint_interval
    )