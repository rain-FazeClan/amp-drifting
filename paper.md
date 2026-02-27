|     |               | Generative |       | Modeling    | via      | Drifting   |     |     |
| --- | ------------- | ---------- | ----- | ----------- | -------- | ---------- | --- | --- |
|     | MingyangDeng1 |            | HeLi1 | TianhongLi1 | YilunDu2 | KaimingHe1 |     |     |
6202 beF 4  ]GL.sc[  1v07740.2062:viXra
Figure1.DriftingModel.Anetworkfperformsapushforwardoperation:q=f # p prior ,mappingapriordistributionp prior (e.g.,Gaussian,
notshownhere)toapushforwarddistributionq(orange). Thegoaloftrainingistoapproximatethedatadistributionp data (blue). As
trainingiterates,weobtainasequenceofmodels{f i },whichcorrespondstoasequenceofpushforwarddistributions{q i }.OurDrifting
Modelfocusesontheevolutionofthispushforwarddistributionattraining-time.Weintroduceadriftingfield(detailedinmaintext)that
approacheszerowhenqmatchesp
data .Thisdriftingfieldprovidesalossfunction(y-axis,inlog-scale)fortraining.
Abstract of a prior distribution p matches the data distribution,
prior
|                                           |                               |     |     |     | namely,      | f p ≈ p      | . Conceptually, | generative mod-    |
| ----------------------------------------- | ----------------------------- | --- | --- | --- | ------------ | ------------ | --------------- | ------------------ |
| Generativemodelingcanbeformulatedaslearn- |                               |     |     |     |              | # prior      | data            |                    |
|                                           |                               |     |     |     | eling learns | a functional | (here, f # )    | that maps from one |
| ingamappingf                              | suchthatitspushforwarddistri- |     |     |     |              |              |                 |                    |
function(here,adistribution)toanother.
| butionmatchesthedatadistribution. |     |     | Thepushfor- |     |     |     |     |     |
| --------------------------------- | --- | --- | ----------- | --- | --- | --- | --- | --- |
wardbehaviorcanbecarriedoutiterativelyatin- The“pushforward”behaviorcanberealizediterativelyat
ferencetime,e.g.,indiffusion/flow-basedmodels. inferencetime,e.g.,inprevailingparadigmssuchasDiffu-
Inthispaper,weproposeanewparadigmcalled sion(Sohl-Dicksteinetal.,2015)andFlowMatching(Lip-
DriftingModels,whichevolvethepushforward man et al., 2022). When generating, these models map
distribution during training and naturally admit noisiersamplestoslightlycleanerones,progressivelyevolv-
one-stepinference. Weintroduceadriftingfield ingthesampledistributiontowardthedatadistribution.This
thatgovernsthesamplemovementandachieves modelingphilosophycanbeviewedasdecomposingacom-
equilibriumwhenthedistributionsmatch. This plexpushforwardmap(i.e.,f )intoachainofmorefeasible
#
leadstoatrainingobjectivethatallowstheneu- transformations,appliedatinferencetime.
ralnetworkoptimizertoevolvethedistribution.
Inthispaper,weproposeDriftingModels,anewparadigm
Inexperiments,ourone-stepgeneratorachieves
|                  |            |          |          |     | forgenerativemodeling. |     | DriftingModelsarecharacterized |     |
| ---------------- | ---------- | -------- | -------- | --- | ---------------------- | --- | ------------------------------ | --- |
| state-of-the-art | results on | ImageNet | 256×256, |     |                        |     |                                |     |
bylearningapushforwardmapthatevolvesduringtraining
| with FID | 1.54 in latent space | and | 1.61 in | pixel |     |     |     |     |
| -------- | -------------------- | --- | ------- | ----- | --- | --- | --- | --- |
time,therebyremovingtheneedforaniterativeinference
| space. Wehopethatourworkopensupnewop- |     |     |     |     |            |             |                              |     |
| ------------------------------------- | --- | --- | --- | --- | ---------- | ----------- | ---------------------------- | --- |
|                                       |     |     |     |     | procedure. | Themappingf | isrepresentedbyasingle-pass, |     |
portunitiesforhigh-qualityone-stepgeneration.
|     |     |     |     |     | non-iterativenetwork. |     | Asthetrainingprocessisinherently |     |
| --- | --- | --- | --- | --- | --------------------- | --- | -------------------------------- | --- |
iterativeindeeplearningoptimization,itcanbenaturally
viewedasevolvingthepushforwarddistribution,f p ,
| 1.Introduction    |              |          |         |       |                      |     |           | # prior |
| ----------------- | ------------ | -------- | ------- | ----- | -------------------- | --- | --------- | ------- |
|                   |              |          |         |       | throughtheupdateoff. |     | SeeFig.1. |         |
| Generative models | are commonly | regarded | as more | chal- |                      |     |           |         |
Todrivetheevolutionofthetraining-timepushforward,we
| lengingthandiscriminativemodels. |     | Whilediscriminative |     |     |     |     |     |     |
| -------------------------------- | --- | ------------------- | --- | --- | --- | --- | --- | --- |
introduceadriftingfieldthatgovernsthesamplemovement.
modelingtypicallyfocusesonmappingindividualsamples
Thisfielddependsonthegenerateddistributionandthedata
totheircorrespondinglabels,generativemodelingconcerns
|                                      |     |     |              |     | distribution. | Bydefinition,thisfieldbecomeszerowhenthe |     |     |
| ------------------------------------ | --- | --- | ------------ | --- | ------------- | ---------------------------------------- | --- | --- |
| mappingfromonedistributiontoanother. |     |     | Thiscanbeex- |     |               |                                          |     |     |
twodistributionsmatch,therebyreachinganequilibriumin
| pressedaslearningamappingf |     | suchthatthepushforward |     |     |     |     |     |     |
| -------------------------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- |
whichthesamplesnolongerdrift.
1MIT2HarvardUniversity.
Buildingonthisformulation,weproposeasimpletraining
|     |     |     |     |     | objective | that minimizes | the drift of | the generated sam- |
| --- | --- | --- | --- | --- | --------- | -------------- | ------------ | ------------------ |
1

GenerativeModelingviaDrifting
ples. Thisobjectiveinducessamplemovementsandthereby term. ClassicalVAEsareone-stepgeneratorswhenusinga
evolves the underlying pushforward distribution through Gaussianprior. Today’sprevailingVAEapplicationsoften
iterativeoptimization(e.g.,SGD).Wefurtherintroducethe resorttopriorslearnedfromothermethods,e.g.,diffusion
designsofthedriftingfield,theneuralnetworkmodel,and (Rombachetal.,2022)orautoregressivemodels(Esseretal.,
thetrainingalgorithm. 2021),whereVAEseffectivelyactastokenizers.
Drifting Models naturally perform single-step (“1-NFE”) Normalizing Flows (NFs). NFs (Rezende & Mohamed,
generationandachievestrongempiricalperformance. On 2015;Dinhetal.,2016;Zhaietal.,2024)learnmappings
ImageNet 256×256, we obtain a 1-NFE FID of 1.54 un- fromdatatonoiseandoptimizethelog-likelihoodofsam-
derthestandardlatent-spacegenerationprotocol,achieving ples. These methods require invertible architectures and
a new state-of-the-art among single-step methods. This computableJacobians. Conceptually,NFsoperateasone-
resultremainscompetitiveevenwhencomparedwithmulti- stepgeneratorsatinference,withcomputationperformed
stepdiffusion-/flow-basedmodels. Further,underthemore bytheinverseofthenetwork.
| challengingpixel-spacegenerationprotocol(i.e., |     |     |     | without |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | ------- | --- | --- | --- | --- |
MomentMatching.Moment-matchingmethods(Dziugaite
latents),wereacha1-NFEFIDof1.61,substantiallyoutper-
etal.,2015;Lietal.,2015)seektominimizetheMaximum
| formingpreviouspixel-spacemethods. |     |     | Theseresultssug- |     |     |     |     |     |
| ---------------------------------- | --- | --- | ---------------- | --- | --- | --- | --- | --- |
MeanDiscrepancy(MMD)betweenthegeneratedanddata
gestthatDriftingModelsofferapromisingnewparadigm
|     |     |     |     |     | distributions. | MomentMatchinghasrecentlybeenextended |     |     |
| --- | --- | --- | --- | --- | -------------- | ------------------------------------- | --- | --- |
forhigh-quality,efficientgenerativemodeling.
|     |     |     |     |     | toone-/few-stepdiffusion(Zhouetal.,2025). |     | Relatedto |     |
| --- | --- | --- | --- | --- | ----------------------------------------- | --- | --------- | --- |
MMD,ourapproachalsoleveragestheconceptsofkernel
2.RelatedWork functionsandpositive/negativesamples. However,ourap-
proachfocusesonadriftingfieldthatexplicitlygovernsthe
| Diffusion-/Flow-based |        | Models.     | Diffusion models | (e.g., |                             |     |                           |     |
| --------------------- | ------ | ----------- | ---------------- | ------ | --------------------------- | --- | ------------------------- | --- |
|                       |        |             |                  |        | sampledriftsattrainingtime. |     | FurtherdiscussionisinC.2. |     |
| Sohl-Dickstein        | et al. | 2015; Ho et | al. 2020; Song   | et al. |                             |     |                           |     |
2020)andtheirflow-basedcounterparts(e.g.,Lipmanetal. ContrastiveLearning. Ourdriftingfieldisdrivenbyposi-
2022;Liuetal.2022;Albergoetal.2023)formulatenoise- tivesamplesfromthedatadistributionandnegativesamples
to-datamappingsthroughdifferentialequations(SDEsor fromthegenerateddistribution. Thisisconceptuallyrelated
ODEs). At the core of their inference-time computation to the positive and negative samples in contrastive repre-
isaniterativeupdate,e.g.,oftheformx = x +∆x , sentationlearning(Hadselletal.,2006;Oordetal.,2018).
|     |     |     | i+1 | i i |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
suchaswithanEulersolver. Theupdate∆x i dependson Theideaofcontrastivelearninghasalsobeenextendedto
theneuralnetworkf,andasaresult,generationinvolves generativemodels,e.g.,toGANs(Unterthineretal.,2017;
multiplestepsofnetworkevaluations. Kang&Park,2020)orFlowMatching(Stoicaetal.,2025).
Agrowingbodyofworkhasfocusedonreducingthesteps
ofdiffusion-/flow-basedmodels.Distillation-basedmethods 3.DriftingModelsforGeneration
(e.g.,Salimans&Ho2022;Luoetal.2023;Yinetal.2024;
WeproposeDriftingModels,whichformulategenerative
Zhouetal.2024)distillapretrainedmulti-stepmodelinto
modelingasatraining-timeevolutionofthepushforward
| a single-step | one. Another   | line of     | research aims  | to train |              |                       |                     |      |
| ------------- | -------------- | ----------- | -------------- | -------- | ------------ | --------------------- | ------------------- | ---- |
|               |                |             |                |          | distribution | via a drifting field. | Our model naturally | per- |
| one-step      | diffusion/flow | models from | scratch (e.g., | Song     |              |                       |                     |      |
formsone-stepgenerationatinferencetime.
etal.2023;Fransetal.2024;Boffietal.2025;Gengetal.
| 2025a). | To achieve | this goal, these | methods incorporate |     |     |     |     |     |
| ------- | ---------- | ---------------- | ------------------- | --- | --- | --- | --- | --- |
3.1.PushforwardatTrainingTime
theSDE/ODEdynamicsintotrainingbyapproximatingthe
inducedtrajectories. Incontrast,ourworkpresentsacon- Consideraneuralnetworkf :RC (cid:55)→RD. Theinputoff
ceptuallydifferentparadigmanddoesnotrelyonSDE/ODE
|     |     |     |     |     | isϵ∼p ϵ (e.g.,anynoiseofdimensionC),andtheoutput |     |     |     |
| --- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- |
formulationsasindiffusion/flowmodels.
|     |     |     |     |     | isdenotedbyx | = f(ϵ) ∈ RD. | Ingeneral, theinputand |     |
| --- | --- | --- | --- | --- | ------------ | ------------ | ---------------------- | --- |
outputdimensionsneednotbeequal.
| GenerativeAdversarialNetworks(GANs). |            |                 | GANs(Good-       |      |     |     |     |     |
| ------------------------------------ | ---------- | --------------- | ---------------- | ---- | --- | --- | --- | --- |
| fellow et                            | al., 2014) | are a classical | family of models | that |     |     |     |     |
Wedenotethedistributionofthenetworkoutputbyq,i.e.,
trainageneratorbydiscriminatinggeneratedsamplesfrom x=f(ϵ)∼q. Inprobabilitytheory,qisreferredtoasthe
| realdata. | LikeGANs,   | ourmethodinvolvesasingle-pass |                  |     |                            |     |                     |     |
| --------- | ----------- | ----------------------------- | ---------------- | --- | -------------------------- | --- | ------------------- | --- |
|           |             |                               |                  |     | pushforwarddistributionofp |     | ϵ underf,denotedby: |     |
| network   | f that maps | noise to data,                | whose “goodness” | is  |                            |     |                     |     |
evaluated by a loss function; however, unlike GANs, our q =f p . (1)
# ϵ
methoddoesnotrelyonadversarialoptimization.
|     |     |     |     |     | Here, “f ” | denotes the pushforward | induced by | f. Intu- |
| --- | --- | --- | --- | --- | ---------- | ----------------------- | ---------- | -------- |
#
| Variational | Autoencoders | (VAEs). | VAEs (Kingma | &   |                                |     |                         |     |
| ----------- | ------------ | ------- | ------------ | --- | ------------------------------ | --- | ----------------------- | --- |
|             |              |         |              |     | itively,thisnotationmeansthatf |     | transformsadistribution |     |
Welling,2013)optimizetheevidencelowerbound(ELBO),
|     |     |     |     |     | p intoanotherdistributionq. |     | Thegoalofgenerativemod- |     |
| --- | --- | --- | --- | --- | --------------------------- | --- | ----------------------- | --- |
ϵ
whichconsistsofareconstructionlossandaKLdivergence elingistofindf suchthatf p ≈p .
# ϵ data
2

GenerativeModelingviaDrifting
Sinceneuralnetworktrainingisinherentlyiterative(e.g., Thisequationmotivatesafixed-pointiterationduringtrain-
SGD), the training process produces a sequence of mod- ing. Atiterationi,weseektosatisfy:
els {f }, where i denotes the training iteration. This cor-
i (cid:0) (cid:1)
f (ϵ)←f (ϵ)+V f (ϵ) . (5)
respondstoasequenceofpushforwarddistributions{q
i
} θi+1 θi p,qθi θi
duringtraining,whereq i =[f i ] # p ϵ foreachi. Thetraining Weconvertthisupdateruleintoalossfunction:
processprogressivelyevolvesq tomatchp .
Whenthenetworkf isupdated,a i sampleattr d a at i a ningiteration L=E ϵ (cid:104)(cid:13) (cid:13) f θ (ϵ) −stopgrad (cid:0) f θ (ϵ)+V p,qθ (cid:0) f θ (ϵ) (cid:1)(cid:1)(cid:13) (cid:13) 2 (cid:105) .
(cid:124)(cid:123)(cid:122)(cid:125) (cid:124) (cid:123)(cid:122) (cid:125)
iisimplicitly“drifted”as:x i+1 =x i +∆x i ,where∆x i := prediction frozentarget
f (ϵ)−f (ϵ)arisesfromparameterupdatestof. This (6)
i+1 i
impliesthattheupdateoff determinesthe“residual”ofx, Here, the stop-gradient operation provides a frozen state
whichwerefertoasthe“drift”. fromthelastiteration,following(Chen&He,2021;Song
&Dhariwal,2023). Intuitively,wecomputeafrozentarget
andmovethenetworkpredictiontowardit.
3.2.DriftingFieldforTraining
We note that the value of our loss function L is equal to
Next,wedefineadriftingfieldtogovernthetraining-time
E (cid:2) ∥V(f(ϵ))∥2(cid:3) ,thatis,thesquarednormofthedrifting
evolution of the samples x and, consequently, the push- ϵ
fieldV. Withthestop-gradientformulation,oursolverdoes
forward distribution q. A drifting field is a function that
notdirectlyback-propagatethroughV,becauseVdepends
computes ∆x given x. Formally, denoting this field by
V (·): Rd →Rd,wehave: onq θ andback-propagatingthroughadistributionisnon-
p,q
trivial. Instead, our formulation minimizes this objective
x i+1 =x i +V p,qi (x i ), (2) indirectly: itmovesx = f θ (ϵ)towardsitsdriftedversion,
i.e.,towardsx+∆xthatisfrozenatthisiteration.
Here,x =f (ϵ)∼q andafterdriftingwedenotex ∼
i i i i+1
q i+1 . Thesubscriptsp,qdenotethatthisfielddependsonp 3.3.DesigningtheDriftingField
(e.g.,p=p )andthecurrentdistributionq.
data
The field V depends on two distributions p and q. To
p,q
Ideally, when p = q, we want all x to stop drifting i.e.,
obtainacomputableformulation,weconsidertheform:
V=0. Inthispaper,weconsiderthefollowingproposition:
V (x)=E E [K(x,y+,y−)], (7)
Proposition3.1. Considerananti-symmetricdriftingfield: p,q y+∼p y−∼q
whereK(·,·,·)isakernel-likefunctiondescribinginterac-
V (x)=−V (x), ∀x. (3)
p,q q,p tionsamongthreesamplepoints. Kcanoptionallydepend
onpandq. Ourframeworksupportsabroadclassoffunc-
Thenwehave: q =p ⇒ V (x)=0,∀x.
p,q
tionsK,aslongasV=0whenp=q.
The proof is straightforward1. Intuitively, anti-symmetry
Fortheinstantiationinthiswork,weintroduceaformofV
meansthatswappingpandqsimplyflipsthesignofthedrift.
drivenbyattractionandrepulsion. Wedefinethefollowing
Thispropositionimpliesthatifthepushforwarddistribution
fieldsinspiredbythemean-shiftmethod(Cheng,1995):
q matchesthedatadistributionp, thedriftiszeroforany
sampleandthemodelachievesanequilibrium. V+(x):= 1 E (cid:2) k(x,y+)(y+−x) (cid:3) ,
p Z p
p
We note that the converse implication, i.e., V = 0 ⇒ (8)
q =p,isfalseingeneralforarbitrarychoicesof p V ,q . Forour V−(x):= 1 E (cid:2) k(x,y−)(y−−x) (cid:3) .
q Z q
kernelizedformulation(Sec.3.3),wegivesufficientcondi- q
tionsunderwhichV p,q ≈0impliesq ≈p(AppendixC.1). Here,Z p andZ q arenormalizationfactors:
TrainingObjective. Thepropertyofequilibriummotivates Z (x):=E [k(x,y+)],
p p
a definition of a training objective. Let f be a network (9)
θ Z (x):=E [k(x,y−)].
parameterized by θ, and x = f (ϵ) for ϵ ∼ p . At the q q
θ ϵ
equilibrium where V = 0, we set up the following fixed- Intuitively,Eq.(8)computestheweightedmeanofthevec-
pointrelation: tor difference y−x. The weights are given by a kernel
(cid:0) (cid:1) k(·,·)normalizedby(9). WethendefineVas:
f (ϵ)=f (ϵ)+V f (ϵ) . (4)
θˆ θˆ p,q θˆ θˆ
V (x):=V+(x)−V−(x). (10)
p,q p q
Here,θˆdenotestheoptimalparametersthatcanachievethe
Intuitively,thisfieldcanbeviewedasattractingbythedata
equilibrium,andq denotesthepushforwardoff .
θˆ θˆ
distribution p and repulsing by the sample distribution q.
1q=p⇒V =V =−V ⇒V =0 ThisisillustratedinFig.2.
p,q q,p p,q p,q
3

GenerativeModelingviaDrifting
|     |     |     |     |     |     | Algorithm1TrainingLoss. |                                   | Note:forbrevity,heretheneg- |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------------------- | --------------------------------- | --------------------------- | --- | --- | --- |
|     |     |     |     |     |     | ativesamplesy           | negarefromthesamebatchofgenerated |                             |     |     |     |
data,thoughtheycanincludeothersourceofnegatives.
|     |     |     |     |     |     | # f: generator |                |              |           |                  |      |
| --- | --- | --- | --- | --- | --- | -------------- | -------------- | ------------ | --------- | ---------------- | ---- |
|     |     |     |     |     |     | # y_pos:       | [N_pos,        | D], data     | samples   |                  |      |
|     |     |     |     |     |     | e = randn([N,  | C])            | # noise      |           |                  |      |
|     |     |     |     |     |     | x = f(e)       | # [N, D],      | generated    |           | samples          |      |
|     |     |     |     |     |     | y_neg          | = x # reuse    | x as         | negatives |                  |      |
|     |     |     |     |     |     | V = compute    | V(x,           | y_pos,       | y_neg)    |                  |      |
|     |     |     |     |     |     | x_drifted      | = stopgrad(x   |              | + V)      |                  |      |
|     |     |     |     |     |     | loss =         | mse loss(x     | - x_drifted) |           |                  |      |
|     |     |     |     |     |     | that V         | ≈ 0 leads to q | ≈ p.         | While     | this implication | does |
notholdforarbitrarychoicesofV,weempiricallyobserve
Figure2.Illustrationofdriftingasample.Ageneratedsamplex
(black)driftsaccordingtoavector:V=V+−V−.Here,V+is thatdecreasingthevalueof∥V∥2correlateswithimproved
|     |     |     | p   | q   | p   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
themean-shiftvectorofthepositivesamples(blue)andV−isthe generationquality. InAppendixC.1,weprovideanidentifi-
q
mean-shiftvectorofthenegativesamples(orange):seeEq.(8).x abilityheuristic: forourkernelizedconstruction,thezero-
| isattractedbyV | +andrepulsedbyV |     | −.  |     |     |     |     |     |     |     |     |
| -------------- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
p q drift condition imposes a large set of bilinear constraints
on(p,q),andundermildnon-degeneracyassumptionsthis
forcespandqtomatch(approximately).
SubstitutingEq.(8)intoEq.(10),weobtain:
|     |     |     |     |     |     | StochasticTraining. | Instochastictraining(e.g.,mini-batch |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------- | ------------------------------------ | --- | --- | --- | --- |
1
V (x)= E (cid:2) k(x,y+)k(x,y−)(y+−y−) (cid:3) . optimization),weestimateVbyapproximatingtheexpec-
| p,q |     | p,q |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Z Z
|     | p q |     |     |     |     | tationsinEq.(11)withempiricalmeans. |     |     |     | Foreachtraining |     |
| --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --------------- | --- |
(11)
|                                                  |              |               |                           |                 |      | step,wedrawN                               | samplesofnoiseϵ                   |                              |              | ∼ p andcomputea |              |
| ------------------------------------------------ | ------------ | ------------- | ------------------------- | --------------- | ---- | ------------------------------------------ | --------------------------------- | ---------------------------- | ------------ | --------------- | ------------ |
| Here,thevectordifferencereducestoy+−y−;theweight |              |               |                           |                 |      |                                            |                                   |                              |              | ϵ               |              |
|                                                  |              |               |                           |                 |      | batchofx=f                                 | θ (ϵ)∼q.                          | Thegeneratedsamplesalsoserve |              |                 |              |
| iscomputedfromtwokernelsandnormalizedjointly.    |              |               |                           |                 | This |                                            |                                   |                              |              |                 |              |
|                                                  |              |               |                           |                 |      | asthenegativesamplesinthesamebatch,i.e.,y− |                                   |                              |              |                 | ∼q. On       |
| formisaninstantiationofEq.(7).                   |              |               | ItiseasytoseethatV        |                 |      |                                            |                                   |                              |              |                 |              |
|                                                  |              |               |                           |                 |      | theotherhand,wesampleN                     |                                   |                              | datapointsy+ |                 | ∼p . The     |
|                                                  |              |               |                           |                 |      |                                            |                                   | pos                          |              |                 | data         |
| isanti-symmetric:                                | V            | p,q =−V       | q,p . Ingeneral,ourmethod |                 |      |                                            |                                   |                              |              |                 |              |
|                                                  |              |               |                           |                 |      | drifting                                   | field V is computed               |                              | in this      | batch of        | positive and |
| does not                                         | require V to | be decomposed |                           | into attraction | and  |                                            |                                   |                              |              |                 |              |
|                                                  |              |               |                           |                 |      | negativesamples.                           | Alg.1providethepseudocodeforsucha |                              |              |                 |              |
repulsion;itonlyrequiresV=0whenp=q.
|     |     |     |     |     |     | trainingstep,wherecompute |     |     | VisgiveninSectionA.1. |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------- | --- | --- | --------------------- | --- | --- |
Kernel. Thekernelk(·,·)canbeafunctionthatmeasures
| thesimilarity. | Inthispaper,weadopt: |     |     |     |     |     |     |     |     |     |     |
| -------------- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
3.4.DriftinginFeatureSpace
(cid:18) 1 (cid:19) Thusfar,wehavedefinedtheobjective(6)directlyinthe
|     | k(x,y)=exp |     | − ∥x−y∥ | ,   | (12) |          |              |             |           |             |           |
| --- | ---------- | --- | ------- | --- | ---- | -------- | ------------ | ----------- | --------- | ----------- | --------- |
|     |            |     | τ       |     |      | raw data | space. Our   | formulation | can       | be extended | to any    |
|     |            |     |         |     |      | feature  | space. Let ϕ | denote      | a feature | extractor   | (e.g., an |
whereτ isatemperatureand∥·∥isℓ -distance. Weview imageencoder)operatingonrealorgeneratedsamples. We
2
| k˜(x,y)≜ | 1k(x,y)asanormalizedkernel,whichabsorbs |     |     |     |     |     |     |     |     |     |     |
| -------- | --------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
rewritetheloss(6)inthefeaturespaceas:
Z
thenormalizationinEq.(11).
|     |     | k˜  |     |     |     | (cid:20)(cid:13) |     |     |     |     | (cid:21) |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | -------- |
I n p r ac t i c e ,w e im p le m en t us i n g a s of t m a x o pe ra ti o n ,w it h (cid:16) (cid:1)(cid:17)(cid:13) 2
|     |     |     |     |     |     | E (cid:13) ϕ(x)−stopgrad |     |     |     | (cid:0) | (cid:13) |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | ------- | -------- |
lo g it s g i v e n by − 1 ∥ x − y ∥ ,w h e r e th e s o f tm a x is ta k e n ov e r (cid:13) ϕ(x)+V ϕ(x) (cid:13) . (13)
τ
| y. This  | softmax operation | is          | similar to | that of InfoNCE |        |     |     |     |     |     |     |
| -------- | ----------------- | ----------- | ---------- | --------------- | ------ | --- | --- | --- | --- | --- | --- |
| (Oord et | al., 2018) in     | contrastive | learning.  | In our          | imple- |     |     |     |     |     |     |
mentation,wefurtherapplyanextrasoftmaxnormalization Here,x=f (ϵ)istheoutput(e.g.,images)ofthegenerator.
θ
overthesetof{x}withinabatch,whichslightlyimproves Visdefinedinthefeaturespace:inpractice,thismeansthat
performanceinpractice. Thisadditionalnormalizationdoes ϕ(y+) and ϕ(y−) serve as the positive/negative samples.
notaltertheantisymmetricpropertyoftheresultingV. Itisworthnotingthatfeatureencodingisatraining-time
operationandisnotusedatinferencetime.
| EquilibriumandMatchedDistributions. |     |     |     | Sinceourtrain- |     |     |     |     |     |     |     |
| ----------------------------------- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
inglossinEq.(6)encouragesminimizing∥V∥2,wehope This can be further extended to multiple features, e.g., at
4

GenerativeModelingviaDrifting
| multiplescalesandlocations: |     |     |     |     |     |     |     | ingitinto(15),weobtain: |     |     |     |     |     |     |
| --------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- |
(cid:20)(cid:13) (cid:16) (cid:1)(cid:17)(cid:13) 2 (cid:21) (·|∅).
(cid:88) E (cid:13)ϕ (cid:0) (cid:13) q θ (·|c)=αp data (·|c)−(α−1)p data (16)
|     |            | (x)−stopgrad |     | ϕ (x)+V |     | ϕ (x) | .        |     |     |     |     |     |     |     |
| --- | ---------- | ------------ | --- | ------- | --- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- |
|     | (cid:13) j |              |     | j       |     | j     | (cid:13) |     |     |     |     |     |     |     |
1
| j   |     |     |     |     |     |     |      | whereα= | ≥1. | Thisimpliesthatq |     |     | θ (·|c)istoapproxi- |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------- | --- | ---------------- | --- | --- | ------------------- | --- |
|     |     |     |     |     |     |     | (14) |         | 1−γ |                  |     |     |                     |     |
matealinearcombinationofconditionalandunconditional
Here, ϕ j represents the feature vectors at the j-th scale datadistributions. ThisfollowsthespiritoforiginalCFG.
| and/or | location | from | an encoder | ϕ.  | With | a ResNet-style |     |     |     |     |     |     |     |     |
| ------ | -------- | ---- | ---------- | --- | ---- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
imageencoder(Heetal.,2016),wecomputedriftinglosses Inpractice,Eq.(15)meansthatwesampleextranegative
|                                                      |     |     |     |     |     |     |     | examplesfromthedatainp |     |     | (·|∅),inadditiontothegen- |     |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | ------------------------- | --- | --- | --- |
| acrossmultiplescalesandlocations,whichprovidesricher |     |     |     |     |     |     |     |                        |     |     | data                      |     |     |     |
gradientinformationfortraining. erateddata. Thedistributionq θ (·|c)correspondstoaclass-
|     |     |     |     |     |     |     |     | conditional | network | f   | (·|c), similar | to  | common | practice |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | --- | -------------- | --- | ------ | -------- |
θ
Thefeatureextractorplaysanimportantroleinthegenera- (Ho&Salimans,2022). Wenotethat,inourmethod,CFG
| tionofhigh-dimensionaldata. |     |     |     | Asourmethodisbasedon |     |     |     |                                   |     |     |     |                    |     |     |
| --------------------------- | --- | --- | --- | -------------------- | --- | --- | --- | --------------------------------- | --- | --- | --- | ------------------ | --- | --- |
|                             |     |     |     |                      |     |     |     | isatraining-timebehaviorbydesign: |     |     |     | theone-step(1-NFE) |     |     |
k(·,·)
the kernel for characterizing sample similarities, it propertyispreservedatinferencetime.
isdesiredforsemanticallysimilarsamplestostayclosein
| thefeaturespace. |     | Thisgoalisalignedwithself-supervised |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
4.ImplementationforImageGeneration
| learning(e.g.,Heetal.2020;Chenetal.2020a). |     |     |     |     |     |     | Weuse |     |     |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
pre-trainedself-supervisedmodelsasthefeatureextractor.
Wedescribeourimplementationforimagegenerationon
|          |     |            |       |     |               |     |      | ImageNet(Dengetal.,2009)atresolution256×256. |     |     |     |     |     | Full |
| -------- | --- | ---------- | ----- | --- | ------------- | --- | ---- | -------------------------------------------- | --- | --- | --- | --- | --- | ---- |
| Relation | to  | Perceptual | Loss. | Our | feature-space |     | loss |                                              |     |     |     |     |     |      |
is related to perceptual loss (Zhang et al., 2018) but is implementationdetailsareprovidedinAppendixA.
conceptually different. The perceptual loss minimizes: Tokenizer. By default, we perform generation in latent
| ∥ϕ(x)−ϕ(x |     | )∥2,thatis,theregressiontargetisϕ(x |     |     |     |     | )   |     |     |     |     |     |     |     |
| --------- | --- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
target 2 target space(Rombachetal.,2022). WeadoptthestandardSD-
| andrequirespairingxwithitstarget. |     |     |     |     | Incontrast,ourregres- |     |     |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
VAEtokenizer,whichproducesa32×32×4latentspacein
|                          |     |     |     | (cid:0) | (cid:1)           |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --- | ------- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| siontargetin(13)isϕ(x)+V |     |     |     | ϕ(x)    | ,wherethedrifting |     |     |     |     |     |     |     |     |     |
whichgenerationisperformed.
| is in                                             | the feature | space | and requires |     | no pairing. |     | In princi- |               |                |     |                        |     |     |     |
| ------------------------------------------------- | ----------- | ----- | ------------ | --- | ----------- | --- | ---------- | ------------- | -------------- | --- | ---------------------- | --- | --- | --- |
|                                                   |             |       |              |     |             |     |            | Architecture. | Ourgenerator(f |     | )hasaDiT-like(Peebles& |     |     |     |
| ple,ourfeature-spacelossaimstomatchthepushforward |             |       |              |     |             |     |            |               |                |     | θ                      |     |     |     |
distributionsϕ qandϕ p. Xie,2023)architecture.Itsinputis32×32×4-dimGaussian
|     |     | #   | #   |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
noiseϵ,anditsoutputisthegeneratedlatentxofthesame
RelationtoLatentGeneration. Ourfeature-spacelossis dimension. Weuseapatchsizeof2,i.e.,likeDiT/2. Our
orthogonaltotheconceptofgeneratorsinthelatentspace
modelusesadaLN-zero(Peebles&Xie,2023)forprocess-
| (e.g.,LatentDiffusion(Rombachetal.,2022)). |     |     |     |     |     | Inourcase, |     |     |     |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
ingclass-conditioningorotherextraconditioning.
| whenusingϕ,thegeneratorf |     |     |     | canstillproduceoutputsin |     |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the pixel space or the latent space of a tokenizer. If the CFG conditioning. We follow (Geng et al., 2025b) and
generatorf isinthelatentspaceandthefeatureextractorϕ adoptCFG-conditioning. Attrainingtime,aCFGscaleα
isinthepixelspace,thetokenizerdecoderisappliedbefore (Eq.(16))israndomlysampled. Negativesamplesarepre-
extractingfeaturesfromϕ. paredbasedonα(Eq.(15)),andthenetworkisconditioned
|     |     |     |     |     |     |     |     | onthisvalue. | Atinferencetime,αcanbefreelyspecified |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------------------------------------- | --- | --- | --- | --- | --- |
3.5.Classifier-FreeGuidance andvariedwithoutretraining. DetailsareinA.7.
|     |     |     |     |     |     |     |     | Batching. | Thepseudo-codeinAlg.1describesabatchof |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | -------------------------------------- | --- | --- | --- | --- | --- |
Classifier-freeguidance(CFG)(Ho&Salimans,2022)im-
provesgenerationqualitybyextrapolatingbetweenclass- N =N generatedsamples. Inpractice,whenclasslabels
neg
|             |     |                   |     |                |     |     |        | are involved, | we  | sample | a batch | of N | class labels. | For |
| ----------- | --- | ----------------- | --- | -------------- | --- | --- | ------ | ------------- | --- | ------ | ------- | ---- | ------------- | --- |
| conditional |     | and unconditional |     | distributions. |     | Our | method |               |     |        |         | c    |               |     |
naturallysupportsarelatedformofguidance. eachlabel,weperformAlg.1independently. Accordingly,
|     |     |     |     |     |     |     |     | theeffectivebatchsizeisB |     |     | = N | ×N, | whichconsistsof |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --------------- | --- |
c
Inourmodel,givenaclasslabelcasthecondition,theun- N ×N negativesandN ×N positives.
|                                        |     |     |     |     |     |     |            | c   |     |     | c pos |     |     |     |
| -------------------------------------- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | ----- | --- | --- | --- |
| derlyingtargetdistributionpnowbecomesp |     |     |     |     |     |     | (·|c),from |     |     |     |       |     |     |     |
data
|                                |           |     |      |          | y+      |          |           | Wedefinea“trainingepoch”basedonthenumberofgen- |     |                                      |     |     |     |     |
| ------------------------------ | --------- | --- | ---- | -------- | ------- | -------- | --------- | ---------------------------------------------- | --- | ------------------------------------ | --- | --- | --- | --- |
| whichwecandrawpositivesamples: |           |     |      |          |         | ∼ p data | (·|c). To |                                                |     |                                      |     |     |     |     |
|                                |           |     |      |          |         |          |           | eratedsamplesx.                                |     | Inparticular,eachiterationgeneratesB |     |     |     |     |
| achieve                        | guidance, | we  | draw | negative | samples | either   | from      |                                                |     |                                      |     |     |     |     |
generated samplesor real samplesfrom different classes. samples,andoneepochcorrespondstoN /Biterations
data
|                                              |     |     |     |     |     |     |     | foradatasetofsizeN |     |      | .   |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ---- | --- | --- | --- | --- |
| Formally,thenegativesampledistributionisnow: |     |     |     |     |     |     |     |                    |     | data |     |     |     |     |
q˜(·|c)≜(1−γ)q (·|∅). FeatureExtractor. Ourmodelistrainedwithdriftingloss
|        |                          |     | θ   | (·|c)+γp | data |                 | (15) |                 |       |                                  |           |         |           |      |
| ------ | ------------------------ | --- | --- | -------- | ---- | --------------- | ---- | --------------- | ----- | -------------------------------- | --------- | ------- | --------- | ---- |
|        |                          |     |     |          |      |                 |      | in a feature    | space | (Sec.                            | 3.4). The | feature | extractor | ϕ is |
| Here,γ | ∈[0,1)isamixingrate,andp |     |     |          |      | (·|∅)denotesthe |      |                 |       |                                  |           |         |           |      |
|        |                          |     |     |          | data |                 |      | animageencoder. |       | WemainlyconsideraResNet-style(He |           |         |           |      |
unconditionaldatadistribution2.
2Thisshouldbethedatadistributionexcludingtheclassc.For
| Thegoaloflearningistofindq˜(·|c)=p |     |     |     |     |      | (·|c). | Substitut- |                                                   |     |     |     |     |     |     |
| ---------------------------------- | --- | --- | --- | --- | ---- | ------ | ---------- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                                    |     |     |     |     | data |        |            | simplicity,weusetheunconditionaldatadistribution. |     |     |     |     |     |     |
5

GenerativeModelingviaDrifting
Figure4.Evolutionofsamples.Weshowgeneratedpointssam-
pledatdifferenttrainingiterations,alongwiththeirlossvalues.
Theloss(whosevalueequals∥V∥2)decreasesasthedistribution
convergestothetarget.(y-axisislog-scale.)
|     |     |     |     | Table 1. | Importance of anti-symmetry: |     | breaking the anti- |
| --- | --- | --- | --- | -------- | ---------------------------- | --- | ------------------ |
Figure3.Evolutionofthegenerateddistribution.Thedistribu-
symmetryleadstofailure.Here,theanti-symmetriccaseisdefined
tionq (orange)evolvestowardabimodaltargetp(blue)during
training.Weshowthreeinitializationsofq:(top):initializedbe- inEq.(10)andEq.(11); otherdestructivecasesaredefinedin
similarways.(Setting:B/2model,100epochs)
tweenthetwomodes;(middle):initializedfarfrombothmodes;
(bottom):initializedcollapsedontoonemode.Acrossallinitial-
|     |     |     |     | case |     | driftingfieldV | FID |
| --- | --- | --- | --- | ---- | --- | -------------- | --- |
izations,ourmethodapproximatesthetargetdistributionwithout
V+−V−
| modecollapse. |     |     |     | anti-symmetry(default) |     |     | 8.46 |
| ------------- | --- | --- | --- | ---------------------- | --- | --- | ---- |
1.5V+−V−
|     |     |     |     | 1.5×attraction |     |          | 41.05 |
| --- | --- | --- | --- | -------------- | --- | -------- | ----- |
|     |     |     |     | 1.5×repulsion  |     | V+−1.5V− | 46.28 |
etal.,2016)encoder,pre-trainedbyself-supervisedlearning,
|     |     |     |     | 2.0×attraction |     | 2V+−V− | 86.16 |
| --- | --- | --- | --- | -------------- | --- | ------ | ----- |
e.g., MoCo (He et al., 2020) and SimCLR (Chen et al., 2.0×repulsion V+−2V− 112.84
2020a). When these pre-trained models operate in pixel attraction-only V+ 177.14
space, weapplytheVAEdecodertomapourgenerator’s
latent-spaceoutputbacktopixelspaceforfeatureextraction.
Gradientsarebackpropagatedthroughthefeatureencoder othermodesofpwillattractthesamples,allowingthemto
continuemovingandpushingqtocontinueevolving.
| andVAEdecoder. | WealsostudyanMAE(Heetal.,2022) |     |     |     |     |     |     |
| -------------- | ------------------------------ | --- | --- | --- | --- | --- | --- |
pre-trainedinlatentspace(detailedinA.3).
|     |     |     |     | Evolutionofthesamples. |     | Figure4showsthetrainingpro- |     |
| --- | --- | --- | --- | ---------------------- | --- | --------------------------- | --- |
For all ResNet-style models, features are extracted from cessontwo2Dcases. AsmallMLPgeneratoristrained.
multiplestages(i.e.,multi-scalefeaturemaps). Thedrifting Theloss(whosevalueequals∥V∥2)decreasesasthegener-
lossin(13)iscomputedateachscaleandthencombined. ateddistributionconvergestothetarget. Thisisinlinewith
WeelaborateonthedetailsinA.6. ourmotivationthatreducingthedriftandpushingtowards
theequilibriumwillapproximatelyyieldp=q.
| Pixel-spaceGeneration. | Whileourexperimentsprimarily |     |     |     |     |     |     |
| ---------------------- | ---------------------------- | --- | --- | --- | --- | --- | --- |
focusonlatent-spacegeneration,ourmodelssupportpixel-
5.2.ImageNetExperiments
spacegeneration.Inthiscase,ϵandxareboth256×256×3.
Weuseapatchsizeof16(i.e.,DiT/16). Thefeatureextrac- WeevaluateourmodelsonImageNet256×256. Ablation
torϕisdirectlyonthepixelspace. studiesuseaB/2modelontheSD-VAElatentspace,trained
|     |     |     |     | for100epochs. | Thedriftinglossisinafeaturespacecom- |     |     |
| --- | --- | --- | --- | ------------- | ------------------------------------ | --- | --- |
5.Experiments puted by a latent-MAE encoder. We report FID (Heusel
|                    |     |     |     | et al., 2017)     | on 50K generated | images. | We analyze the |
| ------------------ | --- | --- | --- | ----------------- | ---------------- | ------- | -------------- |
| 5.1.ToyExperiments |     |     |     | resultsasfollows. |                  |         |                |
Evolution of the generated distribution. Figure 3 visu- Anti-symmetry. Ourderivationofequilibriumrequiresthe
alizes a 2D toy case, where q evolves toward a bimodal driftingfieldtobeanti-symmetric;seeEq.(3). InTable1,
distributionpattrainingtime,underthreeinitializations. we conduct a destructive study that intentionally breaks
|                      |            |              |            | thisanti-symmetry. | Theanti-symmetriccase(ourablation |     |     |
| -------------------- | ---------- | ------------ | ---------- | ------------------ | --------------------------------- | --- | --- |
| In this toy example, | our method | approximates | the target |                    |                                   |     |     |
default)workswell,whileothercasesfailcatastrophically.
| distributionwithoutexhibitingmodecollapse. |     |     | Thisholds |     |     |     |     |
| ------------------------------------------ | --- | --- | --------- | --- | --- | --- | --- |
evenwhenqisinitializedinacollapsedsingle-modestate Intuitively,forasamplex,wewantattractionfromptobe
(bottom). Thisprovidesintuitionintowhyourmethodis canceled by repulsion from q when p and q match. This
robust to mode collapse: if q collapses onto one mode, equilibriumisnotachievedinthedestructivecases.
6

GenerativeModelingviaDrifting
Table2.Allocationofpositiveandnegativesamples.Inbothsub- Table4.Fromablationtofinalsetting.Wetrainourmodelfor
tables,wecontrolthetotalcomputebyfixingtheepochs(100)and moreepochs,adjusthyper-parametersforthisregime,andusea
| thebatchsizeB                                     | =N           | ×N (4096).Here,N |          | isforclasslabels. |            | largermodelsize. |      |      |        |     |
| ------------------------------------------------- | ------------ | ---------------- | -------- | ----------------- | ---------- | ---------------- | ---- | ---- | ------ | --- |
|                                                   |              | c pos            |          | c                 |            |                  |      |      |        |     |
| Under the                                         | same budget, | increasing       | positive | samples           | (left) and |                  |      |      |        |     |
| negative samples(right)improvesgenerationquality. |              |                  |          |                   | (Setting:  |                  |      |      |        |     |
|                                                   |              |                  |          |                   |            |                  | case | arch | ep FID |     |
B/2model,100epochs)
|         |         |       |         |      |            |     | (a)baseline(fromTable3) | B/2      | 100 3.36 |     |
| ------- | ------- | ----- | ------- | ---- | ---------- | --- | ----------------------- | -------- | -------- | --- |
|         |         |       |         |      |            |     | (b)longer               | B/2      | 320 2.51 |     |
| Nc Npos | Nneg    | B FID | Nc Npos | Nneg | B FID      |     |                         |          |          |     |
|         |         |       |         |      |            |     | (c)longer+hyper-param.  | B/2 1280 | 1.75     |     |
| 64 1    | 64 4096 | 20.43 | 512     | 8 8  | 4096 11.82 |     |                         |          |          |     |
|         |         |       |         |      |            |     | (d)largermodel          | L/2 1280 | 1.54     |     |
| 64 16   | 64 4096 | 10.39 | 256 16  | 16   | 4096 10.16 |     |                         |          |          |     |
| 64 32   | 64 4096 | 8.97  | 128 32  | 32   | 4096 9.32  |     |                         |          |          |     |
64 64 64 4096 8.46 64 64 64 4096 8.46 Table5.System-levelcomparison:ImageNet256×256genera-
tioninlatentspace.FIDison50Kimages,allreportedwithCFG
ifapplicable.Theparameternumbersare“generator+decoder”.
Table3.Featurespacefordrifting.Wecompareself-supervised Allgeneratorsaretrainedfromscratch(i.e.,notdistilled).
learning(SSL)encoders.StandardSimCLRandMoCoencoders
achievecompetitiveresults,whereasourcustomizedlatent-MAE method space params NFE FID↓ IS↑
performsbestandbenefitsfromincreasedwidthandlongertrain-
Multi-stepDiffusion/Flows
ing.(Generatorsetting:B/2model,100epochs)
|           |     |        |                   |              |       | DiT-XL/2(Peebles&Xie,2023)       |     | SD-VAE 675M+49M | 250×2 2.27 | 278.2 |
| --------- | --- | ------ | ----------------- | ------------ | ----- | -------------------------------- | --- | --------------- | ---------- | ----- |
|           |     |        |                   |              |       | SiT-XL/2(Maetal.,2024)           |     |                 | 2.06       | 270.3 |
|           |     |        |                   |              |       |                                  |     | SD-VAE 675M+49M | 250×2      |       |
|           |     |        | featureencoder(ϕ) |              |       | SiT-XL/2+REPA(Yuetal.,2024)      |     |                 | 1.42       | 305.7 |
|           |     |        |                   |              |       |                                  |     | SD-VAE 675M+49M | 250×2      |       |
|           |     |        |                   |              |       | LightningDiT-XL/2(Yaoetal.,2025) |     | VA-VAE 675M+70M | 250×2 1.35 | 295.3 |
| SSLmethod |     | arch   | block             | width SSLep. | FID   |                                  |     |                 |            |       |
|           |     |        |                   |              |       | RAE+DiTDH-XL/2(Zhengetal.,2025)  |     | RAE 839M+415M   | 50×2 1.13  | 262.6 |
| SimCLR    |     | ResNet | bottleneck        | 256 800      | 11.05 |                                  |     |                 |            |       |
Single-stepDiffusion/Flows
| MoCo-v2 |     | ResNet | bottleneck | 256 800 | 8.41 |                              |     |             |         |     |
| ------- | --- | ------ | ---------- | ------- | ---- | ---------------------------- | --- | ----------- | ------- | --- |
|         |     |        |            |         |      | iCT-XL/2(Song&Dhariwal,2023) |     | SD-VAE 675M | 1 34.24 | –   |
latent-MAE(default) ResNet basic 256 192 8.46 Shortcut-XL/2(Fransetal.,2024) 10.60 –
|     |     |     |     |     |     |     |     | SD-VAE 675M | 1   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- |
latent-MAE ResNet basic 384 192 7.26 MeanFlow-XL/2(Gengetal.,2025a) 3.43 –
|     |     |     |     |     |     |     |     | SD-VAE 676M | 1   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- |
latent-MAE ResNet basic 512 192 6.49 AdvFlow-XL/2(Linetal.,2025) SD-VAE 673M 1 2.38 284.2
latent-MAE ResNet basic 640 192 6.30 iMeanFlow-XL/2(Gengetal.,2025b) SD-VAE 610M 1 1.72 282.0
| latent-MAE |     | ResNet | basic | 640 1280 | 4.28 | DriftingModels |     |     |     |     |
| ---------- | --- | ------ | ----- | -------- | ---- | -------------- | --- | --- | --- | --- |
latent-MAE+clsft ResNet basic 640 1280 3.36 DriftingModel,B/2 SD-VAE 133M 1 1.75 263.2
|     |     |     |     |     |     | DriftingModel,L/2 |     |             | 1.54 | 258.9 |
| --- | --- | --- | --- | --- | --- | ----------------- | --- | ----------- | ---- | ----- |
|     |     |     |     |     |     |                   |     | SD-VAE 463M | 1    |       |
AllocationofPositiveandNegativeSamples. Ourmethod The comparison in Table 3 shows that the quality of the
samplespositiveandnegativeexamplestoestimateV(see feature encoder plays an important role. We hypothesize
Alg.1). InTable2,westudytheeffectof N pos and N neg , thatthisisbecauseourmethoddependsonakernelk(·,·)
underfixedepochsandfixedbatchsizeB. (seeEq.(12))tomeasuresamplesimilarity. Samplesthat
|                             |     |     |      |                   |     | are                             | closer in feature space | generally yield       | stronger drift, |     |
| --------------------------- | --- | --- | ---- | ----------------- | --- | ------------------------------- | ----------------------- | --------------------- | --------------- | --- |
| Table2showsthatusinglargerN |     |     | andN |                   |     |                                 |                         |                       |                 |     |
|                             |     |     | pos  | neg isbeneficial. |     |                                 |                         |                       |                 |     |
|                             |     |     |      |                   |     | providingrichertrainingsignals. |                         | Thisgoalisalignedwith |                 |     |
Largersamplesizesareexpectedtoimprovetheaccuracy
|                                              |     |     |     |     |      | themotivationofself-supervisedlearning. |     |     | Astrongfeature |     |
| -------------------------------------------- | --- | --- | --- | --- | ---- | --------------------------------------- | --- | --- | -------------- | --- |
| oftheestimatedVandhencethegenerationquality. |     |     |     |     | This |                                         |     |     |                |     |
encoderreducestheoccurrenceofanearly“flat”kernel(i.e.,
observationalignswithresultsincontrastivelearning(Oord
k(·,·)vanishesbecauseallsamplesarefaraway).
etal.,2018;Heetal.,2020;Chenetal.,2020a),inwhich
largersamplesetsimproverepresentationlearning. Ontheotherhand,wereportthatwewereunabletomake
|                          |     |     |                           |     |     | ourmethodworkonImageNetwithoutafeatureencoder. |     |     |     | In  |
| ------------------------ | --- | --- | ------------------------- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- |
| FeatureSpaceforDrifting. |     |     | Ourmodelcomputesthedrift- |     |     |                                                |     |     |     |     |
thiscase,thekernelmayfailtoeffectivelydescribesimilar-
| ing loss | in a feature | space | (Sec. 3.4). | Table 3 | compares |     |     |     |     |     |
| -------- | ------------ | ----- | ----------- | ------- | -------- | --- | --- | --- | --- | --- |
ity,eveninthepresenceofalatentVAE.Weleavefurther
| thefeatureencoders. |     | Usingthepublicpre-trainedencoders |     |     |     |     |     |     |     |     |
| ------------------- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
studyofthislimitationforfuturework.
| from SimCLR | (Chen | et al., | 2020a) and | MoCo | v2 (Chen |     |     |     |     |     |
| ----------- | ----- | ------- | ---------- | ---- | -------- | --- | --- | --- | --- | --- |
etal.,2020b),ourmethodobtainsdecentresults. System-level Comparisons. In addition to the ablation
setting,wetrainstrongervariantsandsummarizethemin
Thesestandardencodersoperateinthepixeldomain,which
Table4. WecomparewithpreviousmethodsinTable5.
requiresrunningtheVAEdecoderattraining.Tocircumvent
this,wepre-trainaResNet-stylemodelwiththeMAEobjec- Ourmethodachieves1.54FIDwithnative1-NFEgenera-
tive(Heetal.,2022),directlyonthelatentspace. Thefea- tion. Itoutperformsallprevious1-NFEmethods,whichare
turespaceproducedbythis“latent-MAE”performsstrongly basedonapproximatingdiffusion-/flow-basedtrajectories.
(Table3). IncreasingtheMAEencoderwidthandthenum- Notably,ourBase-sizemodelcompeteswithpreviousXL-
berofpre-trainingepochsbothimprovegenerationquality; sizemodels. Ourbestmodel(FID1.54)usesaCFGscale
fine-tuning it with a classifier (‘cls ft’) boosts the results of1.0,whichcorrespondsto“noCFG”indiffusion-based
furtherto3.36FID. methods. OurCFGformulationexhibitsatradeoffbetween
7

GenerativeModelingviaDrifting
Table6.System-levelcomparison:ImageNet256×256genera- Table7.RoboticsControl:ComparisonwithDiffusionPolicy.
tioninpixelspace.FIDison50Kimages,allreportedwithCFG TheevaluationprotocolfollowsDiffusionPolicy(Chietal.,2023).
ifapplicable. Theparameternumbersareofthegenerator. All This table involves four single-stage tasks and two multi-stage
generatorsaretrainedfromscratch(i.e.,notdistilled). tasks. “DriftingPolicy”(ours)replacesthemulti-stepDiffusion
|     |     |     |     |     | Policygeneratorwithourone-stepgenerator. |     |     |     | Successratesare |     |
| --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --------------- | --- |
reportedastheaverageoverthelast10checkpoints.
| method |     |     | space params | NFE FID↓ | IS↑ |     |     |     |     |     |
| ------ | --- | --- | ------------ | -------- | --- | --- | --- | --- | --- | --- |
Multi-stepDiffusion/Flows
ADM-G(Dhariwal&Nichol,2021) pix 554M 250×2 4.59 186.7 DiffusionPolicy DriftingPolicy
SiD,UViT/2(Hoogeboometal.,2023) pix 2.5B 1000×2 2.44 256.3 Task Setting NFE:100 NFE:1
| VDM++,UViT/2(Kingma&Gao,2023) |     |     | pix 2.5B | 256×2 2.12 | 267.7 |     |     |     |     |     |
| ----------------------------- | --- | --- | -------- | ---------- | ----- | --- | --- | --- | --- | --- |
Single-StageTasks(State&VisualObservation)
| SiD2,UViT/2(Hoogeboometal.,2024) |     |     | pix –  | 512×2 1.73 | –     |      |        |      |      |     |
| -------------------------------- | --- | --- | ------ | ---------- | ----- | ---- | ------ | ---- | ---- | --- |
|                                  |     |     |        |            |       |      | State  | 0.98 | 1.00 |     |
| SiD2,UViT/1(Hoogeboometal.,2024) |     |     | pix –  | 512×2 1.38 | –     | Lift |        |      |      |     |
|                                  |     |     |        |            |       |      | Visual | 1.00 | 1.00 |     |
| JiT-G/16(Li&He,2025)             |     |     | pix 2B | 100×2 1.82 | 292.6 |      |        |      |      |     |
|                                  |     |     |        |            |       |      | State  | 0.96 | 0.98 |     |
PixelDiT/16(Yuetal.,2025) pix 797M 200×2 1.61 292.7 Can Visual 0.97 0.99
0.38
| Single-stepDiffusion/Flows |     |     |          |        |       | ToolHang | State  | 0.30 |      |     |
| -------------------------- | --- | --- | -------- | ------ | ----- | -------- | ------ | ---- | ---- | --- |
| EPG-L/16(Leietal.,2025)    |     |     | pix 540M | 1 8.82 | –     |          | Visual | 0.73 | 0.67 |     |
|                            |     |     |          |        |       |          | State  | 0.91 | 0.86 |     |
| GANs                       |     |     |          |        |       | PushT    |        |      |      |     |
|                            |     |     |          |        |       |          | Visual | 0.84 | 0.86 |     |
| BigGAN(Brocketal.,2018)    |     |     | pix 112M | 1 6.95 | 152.8 |          |        |      |      |     |
Multi-StageTasks(StateObservation)
| GigaGAN(Kangetal.,2023) |     |     | pix 569M | 1 3.45 | 225.5 |     |     |     |     |     |
| ----------------------- | --- | --- | -------- | ------ | ----- | --- | --- | --- | --- | --- |
StyleGAN-XL(Saueretal.,2022) pix 166M 1 2.30 265.1 Phase1 0.36 0.56
|                    |     |     |          |        |       | BlockPush | Phase2 | 0.11 | 0.16 |     |
| ------------------ | --- | --- | -------- | ------ | ----- | --------- | ------ | ---- | ---- | --- |
| DriftingModels     |     |     |          |        |       |           | Phase1 | 1.00 | 1.00 |     |
| DriftingModel,B/16 |     |     | pix 134M | 1 1.76 | 299.7 |           | Phase2 | 1.00 | 1.00 |     |
Kitchen
| DriftingModel,L/16 |     |     | pix 464M | 1 1.61 | 307.5 |     | Phase3 | 1.00 | 0.99 |     |
| ------------------ | --- | --- | -------- | ------ | ----- | --- | ------ | ---- | ---- | --- |
|                    |     |     |          |        |       |     | Phase4 | 0.99 | 0.96 |     |
FIDandIS(seeB.3),similartostandardCFG. acrossdifferentdomains.
WeprovideuncuratedqualitativeresultsinAppendixB.5,
6.DiscussionandConclusion
| Fig.7-10,withCFG1.0. |     | Moreover,Fig.11-15showaside- |     |     |     |     |     |     |     |     |
| -------------------- | --- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
by-sidecomparisonwithimprovedMeanFlow(iMF)(Geng
WepresentDriftingModels,anewparadigmforgenerative
etal.,2025b),arecentstate-of-the-artone-stepmethod.
|     |     |     |     |     | modeling. | Atthecoreofourmodelistheideaofmodeling |     |     |     |     |
| --- | --- | --- | --- | --- | --------- | -------------------------------------- | --- | --- | --- | --- |
Pixel-spaceGeneration. Ourmethodcannaturallywork theevolutionofpushforwarddistributionsduringtraining.
without the latent VAE, i.e., the generator f directly pro- This allows us to focus on the update rule, i.e., x =
i+1
duces256×256×3images. Thefeatureencoderisapplied x +∆x ,duringtheiterativetrainingprocess. Thisisin
|                 |        |     |                   |       | i   | i   |     |     |     |     |
| --------------- | ------ | --- | ----------------- | ----- | --- | --- | --- | --- | --- | --- |
| on thegenerated | images | for | computingdrifting | loss. | We  |     |     |     |     |     |
contrastwithdiffusion-/flow-basedmodels,whichperform
adopt a configuration similar to that of the latent variant; theiterativeupdateatinferencetime. Ourmethodnaturally
implementationdetailsareinAppendixA. performsone-stepinference.
Table 6 compares different pixel-space generators. Our Giventhatourmethodologyissubstantiallydifferent,many
one-step, pixel-space method achieves 1.61 FID, which open questions remain. For example, although we show
outperforms or competes with previous multi-step meth- that q = p ⇒ V = 0, the converse implication does not
ods. Comparingwithotherone-step,pixel-spacemethods generallyholdintheory. WhileourdesignedVperforms
| (GANs), | our method | achieves | 1.61 FID | using only | 87G |     |     |     |     |     |
| ------- | ---------- | -------- | -------- | ---------- | --- | --- | --- | --- | --- | --- |
wellempirically,itremainsunclearunderwhatconditions
FLOPs;bycomparison,StyleGAN-XLproduces2.30FID V→0leadstoq →p.
| using1574GFLOPs. |     | MoreablationsareinB.1. |     |     |      |             |             |          |           |          |
| ---------------- | --- | ---------------------- | --- | --- | ---- | ----------- | ----------- | -------- | --------- | -------- |
|                  |     |                        |     |     | From | a practical | standpoint, | although | our paper | presents |
aneffectiveinstantiationofdriftingmodeling,manyofour
5.3.ExperimentsonRoboticControl
|     |     |     |     |     | design | decisions | may remain | sub-optimal. | For | example, |
| --- | --- | --- | --- | --- | ------ | --------- | ---------- | ------------ | --- | -------- |
Beyondimagegeneration,wefurtherevaluateourmethod thedesignofthedriftingfieldanditskernels,thefeature
onroboticscontrol. Ourexperimentdesignsandprotocols encoder, and the generator architecture remain open for
futureexploration.
| follow Diffusion | Policy | (Chi | et al., 2023). | At the core | of  |     |     |     |     |     |
| ---------------- | ------ | ---- | -------------- | ----------- | --- | --- | --- | --- | --- | --- |
DiffusionPolicyisamulti-step,diffusion-basedgenerator;
|                                          |     |     |     |            | From | a broader | perspective, | our work | reframes | iterative |
| ---------------------------------------- | --- | --- | --- | ---------- | ---- | --------- | ------------ | -------- | -------- | --------- |
| wereplaceitwithourone-stepDriftingModel. |     |     |     | Wedirectly |      |           |              |          |          |           |
neuralnetworktrainingasamechanismfordistributionevo-
computedriftinglossontherawrepresentationsforcontrol,
lution,incontrasttothedifferentialequationsunderlying
| usingnofeaturespace. |     | ResultsareinTable7. |                      | Our1-NFE  |                              |     |     |                        |     |     |
| -------------------- | --- | ------------------- | -------------------- | --------- | ---------------------------- | --- | --- | ---------------------- | --- | --- |
|                      |     |                     |                      |           | diffusion-/flow-basedmodels. |     |     | Wehopethatthisperspec- |     |     |
| model matches        | or  | exceeds             | the state-of-the-art | Diffusion |                              |     |     |                        |     |     |
tivewillinspiretheexplorationofotherrealizationsofthis
Policythatuses100NFE.Thiscomparisonsuggeststhat
mechanisminfuturework.
DriftingModelscanserveasapromisinggenerativemodel
8

GenerativeModelingviaDrifting
Acknowledgements Dziugaite,G.K.,Roy,D.M.,andGhahramani,Z. Training
generativeneuralnetworksviamaximummeandiscrep-
WegreatlythankGoogleTPUResearchCloud(TRC)for
|          |           |          |     |       |         |          | ancy optimization. |     |     | arXiv preprint |     | arXiv:1505.03906, |     |     |
| -------- | --------- | -------- | --- | ----- | ------- | -------- | ------------------ | --- | --- | -------------- | --- | ----------------- | --- | --- |
| granting | us access | to TPUs. | We  | thank | Michael | Albergo, |                    |     |     |                |     |                   |     |     |
2015.
| ZiqianZhong, | ZhengyangGeng, |     |     | HanhongZhao, |     | Jiangqi |     |     |     |     |     |     |     |     |
| ------------ | -------------- | --- | --- | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
Dai,AlexFan,andShauryaAgrawalforhelpfuldiscussions. Esser,P.,Rombach,R.,andOmmer,B.Tamingtransformers
MingyangDengispartiallysupportedbyfundingfromMIT- forhigh-resolutionimagesynthesis.InCVPR,pp.12873–
| IBMWatsonAILab. |     |     |     |     |     |     | 12883,2021. |         |     |         |         |         |     |     |
| --------------- | --- | --- | --- | --- | --- | --- | ----------- | ------- | --- | ------- | ------- | ------- | --- | --- |
|                 |     |     |     |     |     |     | Frans, K.,  | Hafner, | D., | Levine, | S., and | Abbeel, | P.  | One |
References
|     |     |     |     |     |     |     | step diffusion |     | via | shortcut | models. | arXiv | preprint |     |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | -------- | ------- | ----- | -------- | --- |
Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E. arXiv:2410.12557,2024.
| Stochasticinterpolants: |     |                                     | Aunifyingframeworkforflows |     |     |     |                                             |     |     |     |     |               |     |      |
| ----------------------- | --- | ----------------------------------- | -------------------------- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | ------------- | --- | ---- |
|                         |     |                                     |                            |     |     |     | Geng,Z.,Deng,M.,Bai,X.,Kolter,J.Z.,andHe,K. |     |     |     |     |               |     | Mean |
| anddiffusions.          |     | arXivpreprintarXiv:2303.08797,2023. |                            |     |     |     |                                             |     |     |     |     |               |     |      |
|                         |     |                                     |                            |     |     |     | flowsforone-stepgenerativemodeling.         |     |     |     |     | arXivpreprint |     |      |
arXiv:2505.13447,2025a.
| Boffi,N.M.,Albergo,M.S.,andVanden-Eijnden,E. |     |     |     |     |            | Flow        |                                      |     |          |                |        |               |     |        |
| -------------------------------------------- | --- | --- | --- | --- | ---------- | ----------- | ------------------------------------ | --- | -------- | -------------- | ------ | ------------- | --- | ------ |
| mapmatchingwithstochasticinterpolants:       |     |     |     |     |            | Amathemati- |                                      |     |          |                |        |               |     |        |
|                                              |     |     |     |     |            |             | Geng, Z.,                            | Lu, | Y., Wu,  | Z., Shechtman, |        | E., Kolter,   |     | J. Z., |
| calframeworkforconsistencymodels.            |     |     |     |     | TMLR,2025. |             |                                      |     |          |                |        |               |     |        |
|                                              |     |     |     |     |            |             | and He,                              | K.  | Improved | mean           | flows: | On            | the | chal-  |
|                                              |     |     |     |     |            |             | lengesoffastforwardgenerativemodels. |     |          |                |        | arXivpreprint |     |        |
Brock,A.,Donahue,J.,andSimonyan,K.LargescaleGAN
arXiv:2512.02012,2025b.
| trainingforhighfidelitynaturalimagesynthesis. |     |     |     |     |     | arXiv |     |     |     |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
preprintarXiv:1809.11096,2018. Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B.,
Warde-Farley,D.,Ozair,S.,Courville,A.,andBengio,Y.
| Chen, T., | Kornblith, | S., | Norouzi, | M., | and Hinton, | G. A |                            |     |     |               |     |     |     |     |
| --------- | ---------- | --- | -------- | --- | ----------- | ---- | -------------------------- | --- | --- | ------------- | --- | --- | --- | --- |
|           |            |     |          |     |             |      | Generativeadversarialnets. |     |     | NeurIPS,2014. |     |     |     |     |
simpleframeworkforcontrastivelearningofvisualrep-
resentations. InICML,2020a. Hadsell, R., Chopra, S., and LeCun, Y. Dimensionality
|                 |     |                                   |     |     |     |     | reductionbylearninganinvariantmapping. |     |     |     |     |     | InCVPR,pp. |     |
| --------------- | --- | --------------------------------- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- | --- | ---------- | --- |
| Chen,X.andHe,K. |     | Exploringsimplesiameserepresenta- |     |     |     |     |                                        |     |     |     |     |     |            |     |
1735–1742,2006.
| tionlearning. | InCVPR,pp.15750–15758,2021. |     |     |     |     |     |                |     |          |         |      |         |          |     |
| ------------- | --------------------------- | --- | --- | --- | --- | --- | -------------- | --- | -------- | ------- | ---- | ------- | -------- | --- |
|               |                             |     |     |     |     |     | He, K., Zhang, |     | X., Ren, | S., and | Sun, | J. Deep | residual |     |
Chen, X., Fan, H., Girshick, R., and He, K. Improved learningforimagerecognition. InCVPR,pp.770–778,
| baselines | with | momentum |     | contrastive | learning. | arXiv |     |     |     |     |     |     |     |     |
| --------- | ---- | -------- | --- | ----------- | --------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
2016.
preprintarXiv:2003.04297,2020b.
|     |     |     |     |     |     |     | He, K., Fan, | H., | Wu, | Y., Xie, | S., and | Girshick, | R.  | Mo- |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | -------- | ------- | --------- | --- | --- |
Cheng,Y.Meanshift,modeseeking,andclustering.TPAMI, mentumcontrastforunsupervisedvisualrepresentation
| 1995. |     |     |     |     |     |     | learning. | InCVPR,pp.9729–9738,2020. |     |     |     |     |     |     |
| ----- | --- | --- | --- | --- | --- | --- | --------- | ------------------------- | --- | --- | --- | --- | --- | --- |
Chi,C.,Feng,S.,Du,Y.,Xu,Z.,Cousineau,E.,Burchfiel, He,K.,Chen,X.,Xie,S.,Li,Y.,Dolla´r,P.,andGirshick,
B., and Song, S. Diffusion policy: Visuomotor policy R. Maskedautoencodersarescalablevisionlearners. In
CVPR,2022.
| learningviaactiondiffusion. |     |     |     | InRSS,2023. |     |     |            |             |     |               |     |         |       |     |
| --------------------------- | --- | --- | --- | ----------- | --- | --- | ---------- | ----------- | --- | ------------- | --- | ------- | ----- | --- |
|                             |     |     |     |             |     |     | Henry, A., | Dachapally, |     | P. R., Pawar, | S.  | S., and | Chen, | Y.  |
Deng,J.,Dong,W.,Socher,R.,Li,L.-J.,Li,K.,andFei-Fei,
L. ImageNet: Alarge-scalehierarchicalimagedatabase. Query-keynormalizationfortransformers. InEMNLP,
pp.4246–4253,2020.
InCVPR,pp.248–255.Ieee,2009.
Heusel,M.,Ramsauer,H.,Unterthiner,T.,Nessler,B.,and
| Dhariwal,P.andNichol,A. |     |     | DiffusionmodelsbeatGANs |     |     |     |     |     |     |     |     |     |     |     |
| ----------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
onimagesynthesis. NeurIPS,34:8780–8794,2021. Hochreiter,S. GANstrainedbyatwotime-scaleupdate
|                                        |     |     |                                |     |                |     | ruleconvergetoalocalnashequilibrium. |     |     |                                   |     | NeurIPS,2017. |     |     |
| -------------------------------------- | --- | --- | ------------------------------ | --- | -------------- | --- | ------------------------------------ | --- | --- | --------------------------------- | --- | ------------- | --- | --- |
| Dinh,L.,Sohl-Dickstein,J.,andBengio,S. |     |     |                                |     | Densityestima- |     |                                      |     |     |                                   |     |               |     |     |
|                                        |     |     |                                |     |                |     | Ho,J.andSalimans,T.                  |     |     | Classifier-freediffusionguidance. |     |               |     |     |
| tionusingrealNVP.                      |     |     | arXivpreprintarXiv:1605.08803, |     |                |     |                                      |     |     |                                   |     |               |     |     |
arXivpreprintarXiv:2207.12598,2022.
2016.
|              |                  |        |                 |           |     |              | Ho,J.,Jain,A.,andAbbeel,P. |     |                            |     | Denoisingdiffusionproba- |     |     |     |
| ------------ | ---------------- | ------ | --------------- | --------- | --- | ------------ | -------------------------- | --- | -------------------------- | --- | ------------------------ | --- | --- | --- |
| Dosovitskiy, | A.,              | Beyer, | L., Kolesnikov, |           | A., | Weissenborn, |                            |     |                            |     |                          |     |     |     |
|              |                  |        |                 |           |     |              | bilisticmodels.            |     | NeurIPS,33:6840–6851,2020. |     |                          |     |     |     |
| D., Zhai,    | X., Unterthiner, |        | T.,             | Dehghani, | M., | Minderer,    |                            |     |                            |     |                          |     |     |     |
M.,Heigold,G.,Gelly,S.,Uszkoreit,J.,andHoulsby,N. Hoogeboom,E.,Heek,J.,andSalimans,T.Simplediffusion:
Animageisworth16x16words: Transformersforimage End-to-enddiffusionforhighresolutionimages.InICML,
| recognitionatscale. |     | InICLR,2021. |     |     |     |     | pp.13213–13232.PMLR,2023. |     |     |     |     |     |     |     |
| ------------------- | --- | ------------ | --- | --- | --- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- |
9

GenerativeModelingviaDrifting
Hoogeboom,E.,Mensink,T.,Heek,J.,Lamerigts,K.,Gao, Oord,A.v.d.,Li,Y.,andVinyals,O. Representationlearn-
R.,andSalimans,T. Simplerdiffusion(SiD2): 1.5fidon ing with contrastive predictive coding. arXiv preprint
ImageNet512withpixel-spacediffusion. arXivpreprint arXiv:1807.03748,2018.
arXiv:2410.19324,2024.
|     |     |     |     |     |     | Peebles, | W. and Xie, | S. Scalable | diffusion |     | models with |
| --- | --- | --- | --- | --- | --- | -------- | ----------- | ----------- | --------- | --- | ----------- |
Ioffe,S.andSzegedy,C. Batchnormalization:Accelerating transformers. InCVPR,pp.4195–4205,2023.
deepnetworktrainingbyreducinginternalcovariateshift.
Radford,A.,Kim,J.W.,Hallacy,C.,Ramesh,A.,Goh,G.,
InICML,pp.448–456.pmlr,2015.
|                   |     |                                  |     |     |     | Agarwal,                      | S., Sastry, | G., Askell, | A.,                  | Mishkin, | P., Clark, |
| ----------------- | --- | -------------------------------- | --- | --- | --- | ----------------------------- | ----------- | ----------- | -------------------- | -------- | ---------- |
|                   |     |                                  |     |     |     | J.,Krueger,G.,andSutskever,I. |             |             | Learningtransferable |          |            |
| Kang,M.andPark,J. |     | ContraGAN:Contrastivelearningfor |     |     |     |                               |             |             |                      |          |            |
conditionalimagegeneration.NeurIPS,33:21357–21369, visual models from natural language supervision. In
ICML,pp.8748–8763.PmLR,2021.
2020.
|                     |                             |                               |     |                      |     | Rezende,D.andMohamed,S. |        |          | Variationalinferencewith |     |       |
| ------------------- | --------------------------- | ----------------------------- | --- | -------------------- | --- | ----------------------- | ------ | -------- | ------------------------ | --- | ----- |
| Kang, M.,           | Zhu, J.-Y.,                 | Zhang,                        | R., | Park, J., Shechtman, | E., |                         |        |          |                          |     |       |
|                     |                             |                               |     |                      |     | normalizing             | flows. | In ICML, | pp. 1530–1538.           |     | PMLR, |
| Paris,S.,andPark,T. |                             | ScalingupGANsfortext-to-image |     |                      |     |                         |        |          |                          |     |       |
| synthesis.          | InCVPR,pp.10124–10134,2023. |                               |     |                      |     | 2015.                   |        |          |                          |     |       |
Kingma,D.andGao,R. Understandingdiffusionobjectives Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and
|                                      |     |     |     |     |          | Ommer,B.         | High-resolutionimagesynthesiswithlatent |     |     |     |     |
| ------------------------------------ | --- | --- | --- | --- | -------- | ---------------- | --------------------------------------- | --- | --- | --- | --- |
| astheELBOwithsimpledataaugmentation. |     |     |     |     | NeurIPS, |                  |                                         |     |     |     |     |
|                                      |     |     |     |     |          | diffusionmodels. | InCVPR,pp.10684–10695,2022.             |     |     |     |     |
36:65484–65516,2023.
|                          |     |     |                          |     |     | Ronneberger,O.,Fischer,P.,andBrox,T. |     |     |     | U-Net: | Convolu- |
| ------------------------ | --- | --- | ------------------------ | --- | --- | ------------------------------------ | --- | --- | --- | ------ | -------- |
| Kingma,D.P.andWelling,M. |     |     | Auto-encodingvariational |     |     |                                      |     |     |     |        |          |
bayes. arXivpreprintarXiv:1312.6114,2013. tionalnetworksforbiomedicalimagesegmentation. In
MICCAI,2015.
Lei,J.,Liu,K.,Berner,J.,Yu,H.,Zheng,H.,Wu,J.,and
|                                                |                                        |     |     |     |       | Salimans,     | T. and Ho, | J.        | Progressive | distillation | for      |
| ---------------------------------------------- | -------------------------------------- | --- | --- | --- | ----- | ------------- | ---------- | --------- | ----------- | ------------ | -------- |
| Chu,X.                                         | ThereisnoVAE:End-to-endpixel-spacegen- |     |     |     |       |               |            |           |             |              |          |
|                                                |                                        |     |     |     |       | fast sampling | of         | diffusion | models.     | arXiv        | preprint |
| erativemodelingviaself-supervisedpre-training. |                                        |     |     |     | arXiv |               |            |           |             |              |          |
arXiv:2202.00512,2022.
preprintarXiv:2510.12586,2025.
|                |               |                                     |     |                        |     | Sauer,A.,Schwarz,K.,andGeiger,A.   |     |     | StyleGAN-XL:Scal- |             |     |
| -------------- | ------------- | ----------------------------------- | --- | ---------------------- | --- | ---------------------------------- | --- | --- | ----------------- | ----------- | --- |
| Li,T.andHe,K.  | Backtobasics: |                                     |     | Letdenoisinggenerative |     |                                    |     |     |                   |             |     |
|                |               |                                     |     |                        |     | ingStyleGANtolargediversedatasets. |     |     |                   | InSIGGRAPH, |     |
| modelsdenoise. |               | arXivpreprintarXiv:2511.13720,2025. |     |                        |     |                                    |     |     |                   |             |     |
pp.1–10,2022.
| Li, Y., Swersky, | K.,       | and | Zemel, | R. Generative  | moment |          |                 |         |     |              |       |
| ---------------- | --------- | --- | ------ | -------------- | ------ | -------- | --------------- | ------- | --- | ------------ | ----- |
|                  |           |     |        |                |        | Shazeer, | N. GLU variants | improve |     | transformer. | arXiv |
| matching         | networks. | In  | ICML,  | pp. 1718–1727. | PMLR,  |          |                 |         |     |              |       |
preprintarXiv:2002.05202,2020.
2015.
|                                         |                                     |     |     |     |             | Sohl-Dickstein,        | J., Weiss,                            | E.,                       | Maheswaranathan, |     | N., and |
| --------------------------------------- | ----------------------------------- | --- | --- | --- | ----------- | ---------------------- | ------------------------------------- | ------------------------- | ---------------- | --- | ------- |
| Lin,S.,Yang,C.,Lin,Z.,Chen,H.,andFan,H. |                                     |     |     |     | Adversarial |                        |                                       |                           |                  |     |         |
|                                         |                                     |     |     |     |             | Ganguli,S.             | Deepunsupervisedlearningusingnonequi- |                           |                  |     |         |
| flowmodels.                             | arXivpreprintarXiv:2511.22475,2025. |     |     |     |             |                        |                                       |                           |                  |     |         |
|                                         |                                     |     |     |     |             | libriumthermodynamics. |                                       | InICML,pp.2256–2265.pmlr, |                  |     |         |
2015.
| Lipman, | Y., Chen, | R.T., | Ben-Hamu, | H., Nickel, | M., and |     |     |     |     |     |     |
| ------- | --------- | ----- | --------- | ----------- | ------- | --- | --- | --- | --- | --- | --- |
Le,M. Flowmatchingforgenerativemodeling. arXiv Song,Y.andDhariwal,P. Improvedtechniquesfortraining
preprintarXiv:2210.02747,2022.
|     |     |     |     |     |     | consistencymodels. |     | arXivpreprintarXiv:2310.14189, |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | ------------------------------ | --- | --- | --- |
2023.
| Liu, X., | Gong, C., | and Liu, | Q.  | Flow straight | and fast: |     |     |     |     |     |     |
| -------- | --------- | -------- | --- | ------------- | --------- | --- | --- | --- | --- | --- | --- |
Learningtogenerateandtransferdatawithrectifiedflow.
Song,Y.,Sohl-Dickstein,J.,Kingma,D.P.,Kumar,A.,Er-
arXivpreprintarXiv:2209.03003,2022. mon,S.,andPoole,B. Score-basedgenerativemodeling
|                           |     |     |                           |     |     | throughstochasticdifferentialequations. |     |     |     | arXivpreprint |     |
| ------------------------- | --- | --- | ------------------------- | --- | --- | --------------------------------------- | --- | --- | --- | ------------- | --- |
| Loshchilov,I.andHutter,F. |     |     | Decoupledweightdecayregu- |     |     |                                         |     |     |     |               |     |
arXiv:2011.13456,2020.
| larization.       | InICLR,2019.   |                                   |          |             |            |                                             |       |     |     |     |         |
| ----------------- | -------------- | --------------------------------- | -------- | ----------- | ---------- | ------------------------------------------- | ----- | --- | --- | --- | ------- |
|                   |                |                                   |          |             |            | Song,Y.,Dhariwal,P.,Chen,M.,andSutskever,I. |       |     |     |     | Consis- |
| Luo, W.,          | Hu, T., Zhang, |                                   | S., Sun, | J., Li, Z., | and Zhang, |                                             |       |     |     |     |         |
|                   |                |                                   |          |             |            | tencymodels.                                | 2023. |     |     |     |         |
| Z. Diff-Instruct: |                | Auniversalapproachfortransferring |          |             |            |                                             |       |     |     |     |         |
knowledgefrompre-traineddiffusionmodels. NeurIPS, Stoica, G., Ramanujan, V., Fan, X., Farhadi, A., Krishna,
36:76525–76546,2023. R., andHoffman, J. Contrastiveflowmatching. arXiv
preprintarXiv:2506.05350,2025.
Ma,N.,Goldstein,M.,Albergo,M.S.,Boffi,N.M.,Vanden-
Eijnden,E.,andXie,S.SiT:Exploringflowanddiffusion- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y.
basedgenerativemodelswithscalableinterpolanttrans- Roformer: Enhanced transformer with totary position
formers. InECCV,pp.23–40.Springer,2024. embedding. IJON,568:127063,2024.
10

GenerativeModelingviaDrifting
Unterthiner, T., Nessler, B., Seward, C., Klambauer, G.,
Heusel,M.,Ramsauer,H.,andHochreiter,S. Coulomb
GANs: Provably optimal nash qquilibria via potential
fields. arXivpreprintarXiv:1708.08819,2017.
Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon,
I. S., and Xie, S. ConvNeXt V2: Co-designing and
scalingConvNetswithmaskedautoencoders. InCVPR,
pp.16133–16142,2023.
Wu, Y. and He, K. Group normalization. In ECCV, pp.
3–19,2018.
Yao,J.,Yang,B.,andWang,X. Reconstructionvs.gener-
ation: Tamingoptimizationdilemmainlatentdiffusion
models. InCVPR,pp.15703–15712,2025.
Yin,T.,Gharbi,M.,Zhang,R.,Shechtman,E.,Durand,F.,
Freeman, W. T., and Park, T. One-step diffusion with
distribution matching distillation. In CVPR, pp. 6613–
6623,2024.
Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J.,
and Xie, S. Representation alignment for generation:
Trainingdiffusiontransformersiseasierthanyouthink.
arXivpreprintarXiv:2410.06940,2024.
Yu,Y.,Xiong,W.,Nie,W.,Sheng,Y.,Liu,S.,andLuo,J.
PixelDiT:Pixeldiffusiontransformersforimagegenera-
tion. arXivpreprintarXiv:2511.20645,2025.
Zhai, S., Zhang, R., Nakkiran, P., Berthelot, D., Gu, J.,
Zheng, H., Chen, T., Bautista, M. A., Jaitly, N., and
Susskind, J. Normalizingflowsarecapablegenerative
models. arXivpreprintarXiv:2412.06329,2024.
Zhang,B.andSennrich,R. Rootmeansquarelayernormal-
ization. NeurIPS,32,2019.
Zhang,R.,Isola,P.,Efros,A.A.,Shechtman,E.,andWang,
O. Theunreasonableeffectivenessofdeepfeaturesasa
perceptualmetric. InCVPR,2018.
Zheng,B.,Ma,N.,Tong,S.,andXie,S. Diffusiontrans-
formerswithrepresentationautoencoders. arXivpreprint
arXiv:2510.11690,2025.
Zhou,L.,Ermon,S.,andSong,J. Inductivemomentmatch-
ing. arXivpreprintarXiv:2503.07565,2025.
Zhou, M., Zheng, H., Wang, Z., Yin, M., and Huang, H.
Scoreidentitydistillation: Exponentiallyfastdistillation
ofpretraineddiffusionmodelsforone-stepgeneration. In
ICML,2024.
11

GenerativeModelingviaDrifting
A.AdditionalImplementationDetails Algorithm2ComputingthedriftingfieldV.
Table8summarizestheconfigurationsandhyper-parameters def compute V(x, y_pos, y_neg, T):
| forablationstudiesandsystem-levelcomparisons. |     |     |     |     |     | Wepro- | # x: [N, | D]  |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | ------ | -------- | --- | --- | --- | --- | --- |
videdetailedexperimentalconfigurationsforreproducibil- # y_pos: [N_pos, D]
|          |          |         |       |          |         |        | # y_neg: | [N_neg, |     | D]  |     |     |
| -------- | -------- | ------- | ----- | -------- | ------- | ------ | -------- | ------- | --- | --- | --- | --- |
| ity. All | ablation | studies | share | a common | default | setup, |          |         |     |     |     |     |
# T: temperature
| while | system-level | comparisons |     | use scaled-up |     | configura- |     |     |     |     |     |     |
| ----- | ------------ | ----------- | --- | ------------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
tions. Moreimplementationdetailsaredescribedasfollows. # compute pairwise distance
|     |     |     |     |     |     |     | dist_pos | = cdist(x, |     | y_pos) | # [N, | N_pos] |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | ------ | ----- | ------ |
|     |     |     |     |     |     |     | dist_neg | = cdist(x, |     | y_neg) | # [N, | N_neg] |
A.1.Pseudo-codeforComputingDriftingFieldV
|        |          |                 |     |               |     |        | # ignore | self | (if    | y_neg | is x) |     |
| ------ | -------- | --------------- | --- | ------------- | --- | ------ | -------- | ---- | ------ | ----- | ----- | --- |
| Alg. 2 | provides | the pseudo-code |     | for computing |     | V. The |          |      |        |       |       |     |
|        |          |                 |     |               |     |        | dist_neg | +=   | eye(N) | 1e6   |       |     |
*
computationisbasedontakingempiricalmeansinEq.(11)
and(12),whichareimplementedassoftmaxovery-sample # compute logits
|     |     |     |     |     |     |     | logit_pos | =   | -dist_pos | /   | T   |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | --------- | --- | --- | --- |
axis. Inpractice,wefurthernormalizeoverthex-sample
|     |     |     |     |     |     |     | logit_neg | =   | -dist_neg | /   | T   |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | --------- | --- | --- | --- |
axis,alsoimplementedassoftmaxonthesamelogitmatrix.
WeablateitsinfluenceinB.2.
|                     |        |      |                                   |     |           |     | # concat    | for             | normalization |                 |             |        |
| ------------------- | ------ | ---- | --------------------------------- | --- | --------- | --- | ----------- | --------------- | ------------- | --------------- | ----------- | ------ |
|                     |        |      |                                   |     |           |     | logit =     | cat([logit_pos, |               |                 | logit_neg], | dim=1) |
| It is worth         | noting | that | this implementation               |     | preserves | the |             |                 |               |                 |             |        |
| desiredpropertyofV. |        |      | Inprinciple,thisimplementationcan |     |           |     |             |                 |               |                 |             |        |
|                     |        |      |                                   |     |           |     | # normalize |                 | along         | both dimensions |             |        |
beviewedasaMonteCarloestimationofadriftingfield: A_row = logit.softmax(dim=-1)
|     |       |       |          |                 |     |     | A_col =        | logit.softmax(dim=-2) |     |        |     |     |
| --- | ----- | ----- | -------- | --------------- | --- | --- | -------------- | --------------------- | --- | ------ | --- | --- |
| V   | (x)=E | [K˜   | (x,y+)K˜ | (x,y−)(y+−y−)], |     |     |                |                       |     |        |     |     |
| p,q |       | B,p,q | B        | B               |     |     | A = sqrt(A_row |                       | *   | A_col) |     |     |
(17)
where B consists of other samples in the batch and K˜ # back to [N, N_pos] and [N, N_neg]
B
|     |     |     |     |     |     |     | A_pos, | A_neg | = split(A, |     | [N_pos,], | dim=1) |
| --- | --- | --- | --- | --- | --- | --- | ------ | ----- | ---------- | --- | --------- | ------ |
denotenormalizingthedistancebasedonstatisticswithin
| B. ThisV  | alsosatisfiesV |     |                             | (x) = 0,sincewhenp |     | = q, |           |                               |         |        |     |     |
| --------- | -------------- | --- | --------------------------- | ------------------ | --- | ---- | --------- | ----------------------------- | ------- | ------ | --- | --- |
|           |                |     | p,p                         |                    |     |      | # compute | the                           | weights |        |     |     |
| thetermK˜ | (y+,x)K˜       |     | (y−,x)(y+−y−)cancelsoutwith |                    |     |      |           |                               |         |        |     |     |
|           | B              | B   |                             |                    |     |      | W_pos =   | A_pos                         | # [N,   | N_pos] |     |     |
| thetermK˜ | (y−,x)K˜       |     | (y+,x)(y−−y+).              |                    |     |      | W_neg =   | A_neg                         | # [N,   | N_neg] |     |     |
|           | B              |     | B                           |                    |     |      |           |                               |         |        |     |     |
|           |                |     |                             |                    |     |      | W_pos *=  | A_neg.sum(dim=1,keepdim=True) |         |        |     |     |
|           |                |     |                             |                    |     |      | W_neg *=  | A_pos.sum(dim=1,keepdim=True) |         |        |     |     |
A.2.GeneratorArchitecture
|                 |     |                                  |     |     |     |     | drift_pos | =   | W_pos | @ y_pos | # [N_x, | D]  |
| --------------- | --- | -------------------------------- | --- | --- | --- | --- | --------- | --- | ----- | ------- | ------- | --- |
| Inputandoutput. |     | Theinputtothegeneratorconsistsof |     |     |     |     |           |     |       |         |         |     |
|                 |     |                                  |     |     |     |     | drift_neg | =   | W_neg | @ y_neg | # [N_x, | D]  |
randomnoisealongwithconditioning:
|     |     |     |                    |     |     |     | V = drift_pos |     | - drift_neg |     |     |     |
| --- | --- | --- | ------------------ | --- | --- | --- | ------------- | --- | ----------- | --- | --- | --- |
|     |     | f   | :(ϵ,c,α)(cid:55)→x |     |     |     |               |     |             |     |     |     |
|     |     | θ   |                    |     |     |     | return        | V   |             |     |     |     |
whereϵdenotesrandomvariables,cisaclasslabel,andαis
| theCFGstrength. |     | ϵmayconsistofbothcontinuousrandom |     |     |     |     |     |     |     |     |     |     |
| --------------- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
variables(e.g.,Gaussiannoise)anddiscreteones(e.g.,uni- summingtheprojectedconditioningvectorwithpositional
formlydistributedintegers;seerandomstyleembeddings). embeddings. Randomstyleembeddings. Ourframework
Forlatent-spacemodels,theoutputx∈R32×32×4isinthe allowsarbitrarynoisedistributionsbeyondGaussians. In-
SD-VAElatentspace. Forpixel-spacemodels,theoutput spiredbyStyleGAN(Saueretal.,2022),weintroducean
x∈R256×256×3isdirectlyanimage. additional32“styletokens”: eachofwhichisarandomin-
|              |     |          |     |                       |     |       | dexintoacodebookof64learnableembeddings. |     |     |     |     | Theseare |
| ------------ | --- | -------- | --- | --------------------- | --- | ----- | ---------------------------------------- | --- | --- | --- | --- | -------- |
| Transformer. |     | We adopt | a   | DiT-style Transformer |     | (Pee- |                                          |     |     |     |     |          |
summedandaddedtotheconditioningvector.Thisdoesnot
| bles & | Xie, 2023). | Following |     | (Yao et | al., 2025), | we use |     |     |     |     |     |     |
| ------ | ----------- | --------- | --- | ------- | ----------- | ------ | --- | --- | --- | --- | --- | --- |
changethesequencelengthandintroducesnegligibleover-
| SwiGLU | (Shazeer, | 2020), | RoPE | (Su et | al., | 2024), RM- |                                  |     |     |     |                  |     |
| ------ | --------- | ------ | ---- | ------ | ---- | ---------- | -------------------------------- | --- | --- | --- | ---------------- | --- |
|        |           |        |      |        |      |            | headintermsofparametersandFLOPs. |     |     |     | Thistablereports |     |
SNorm(Zhang&Sennrich,2019),andQK-Norm(Henry
theeffectofstyleembeddingsonourablationdefault:
| et al., | 2020). | The input | Gaussian | noise | is patchified | into |     |     |     |     |     |     |
| ------- | ------ | --------- | -------- | ----- | ------------- | ---- | --- | --- | --- | --- | --- | --- |
256=16×16tokens(patchsize2×2forlatent,16×16for
|     |     |     |     |     |     |     |     |     |     | w/ostyle | w/style |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------- | --- |
pixel). Conditioning(c,α)isprocessedbyadaLN,aswell
|                                   |     |     |     |                    |     |     |     |     | FID | 8.86 | 8.46 |     |
| --------------------------------- | --- | --- | --- | ------------------ | --- | --- | --- | --- | --- | ---- | ---- | --- |
| asbyin-contextconditioningtokens. |     |     |     | Theoutputtokensare |     |     |     |     |     |      |      |     |
unpatchifiedbacktothetargetshape.
Incontrasttodiffusion-/flow-basedmethods,ourmethod
In-contexttokens. Following(Li&He,2025),weprepend can naturally handle different types of noise or random
16 learnable tokens to the sequence for in-context condi- variables. With random style embeddings, the input ran-
tioning(Peebles&Xie,2023). Thesetokensareformedby domvariablesconsistoftwoparts: (1)Gaussiannoise,and
12

GenerativeModelingviaDrifting
Table8.ConfigurationsforImageNet256×256.
ablationdefault B/2,latent(Table5) L/2,latent(Table5) B/16,pixel(Table6) L/16,pixel(Table6)
GeneratorArchitecture
| arch                 |     | DiT-B/2 | DiT-B/2 | DiT-L/2 | DiT-B/16 | DiT-L/16 |
| -------------------- | --- | ------- | ------- | ------- | -------- | -------- |
| inputsize            |     | 32×32×4 | 32×32×4 | 32×32×4 | 32×32×4  | 32×32×4  |
| patchsize            |     | 2×2     | 2×2     | 2×2     | 16×16    | 16×16    |
| hiddendim            |     | 768     | 768     | 1024    | 768      | 1024     |
| depth                |     | 12      | 12      | 24      | 12       | 24       |
| registertokens       |     | 16      | 16      | 16      | 16       | 16       |
| styleembeddingtokens |     | 32      | 32      | 32      | 32       | 32       |
FeatureEncoderforDriftingLoss
arch ResNet ResNet ResNet ResNet+ConvNeXt-V2 ResNet+ConvNeXt-V2
SSLpre-trainmethod latent-MAE latent-MAE latent-MAE pixel-MAE pixel-MAE
| ResNet:inputsize    |     | 32×32×4 | 32×32×4 | 32×32×4    | 256×256×3 | 256×256×3 |
| ------------------- | --- | ------- | ------- | ---------- | --------- | --------- |
| ResNet:conv1stride  |     | 1       | 1       | 1          | 8         | 8         |
| ResNet:basewidth    |     | 256     | 640     | 640        | 640       | 640       |
| ResNet:blocktype    |     |         |         | bottleneck |           |           |
| ResNet:blocks/stage |     |         |         | [3,4,6,3]  |           |           |
[322,162,82,42]
ResNet:size/stage
| MAE:maskingratio       |     |     |         | 50%     |         |         |
| ---------------------- | --- | --- | ------- | ------- | ------- | ------- |
| MAE:pre-trainepochs    |     | 192 | 1280    | 1280    | 1280    | 1280    |
| classificationfinetune |     | No  | 3ksteps | 3ksteps | 3ksteps | 3ksteps |
GeneratorOptimizer
| optimizer      |     |       | AdamW(β1=0.9,β2=0.95) |                              |      |      |
| -------------- | --- | ----- | --------------------- | ---------------------------- | ---- | ---- |
| learningrate   |     | 2e-4  | 4e-4                  | 4e-4                         | 2e-4 | 4e-4 |
| weightdecay    |     | 0.01  | 0.0                   | 0.01                         | 0.01 | 0.01 |
| warmupsteps    |     | 5k    | 10k                   | 10k                          | 10k  | 10k  |
| gradientclip   |     | 2.0   | 2.0                   | 2.0                          | 2.0  | 2.0  |
| trainingsteps  |     | 30k   | 200k                  | 200k                         | 100k | 100k |
| trainingepochs |     | 100   | 1280                  | 1280                         | 640  | 640  |
| EMAdecay       |     | 0.999 |                       | {0.999,0.9995,0.9998,0.9999} |      |      |
DriftingLossComputation
| classlabelsNc            |     | 64   | 128  | 128  | 128  | 128  |
| ------------------------ | --- | ---- | ---- | ---- | ---- | ---- |
| positivesamplesNpos      |     | 64   | 128  | 64   | 128  | 128  |
| generatedsamplesNneg     |     | 64   | 64   | 64   | 64   | 64   |
| effectivebatchB(Nc×Nneg) |     | 4096 | 8192 | 8192 | 8192 | 8192 |
temperaturesτ {0.02,0.05,0.2}:onelossperτ,sumalllossterms
CFGConfiguration
| train:CFGαrange |     | [1,4] | [1,4] | [1,4] | [1,4] | [1,4] |
| --------------- | --- | ----- | ----- | ----- | ----- | ----- |
train:CFGαsampling p(α)∝α−3 p(α)∝α−5 50%:α=1,50%:p(α)∝α−3 p(α)∝α−5 p(α)∝α−5
| train:uncondsamplesNuncond |     | 16  | 32  | 32        | 32  | 32  |
| -------------------------- | --- | --- | --- | --------- | --- | --- |
| inference:CFGαsearch       |     |     |     | [1.0,3.5] |     |     |
(2)discreteindicesforstyleembeddings. Ourmodelf pro- MAEEncoder. TheencoderfollowsaclassicalResNet(He
ducesthepushforwarddistributionoftheirjointdistribution. etal.,2016)design. Itmapsaninputtomulti-scalefeature
maps(4scalesinResNet):
A.3.ImplementationofResNet-styleMAE
|             |             |                 |                 | Encoder:x(cid:55)→{f | ,f ,f ,f | }   |
| ----------- | ----------- | --------------- | --------------- | -------------------- | -------- | --- |
|             |             |                 |                 |                      | 1 2 3    | 4   |
| In addition | to standard | self-supervised | learning models |                      |          |     |
(MoCo (He et al., 2020), SimCLR(Chen et al., 2020a)), Here, a feature map f has dimension H ×W ×C , with
|     |     |     |     |     | i   | i i i |
| --- | --- | --- | --- | --- | --- | ----- |
wedevelopacustomizedResNet-styleMAEmodelasthe H ×W ∈ {322,162,82,42} and C ∈ {C,2C,4C,8C}
|     |     |     |     | i i | i   |     |
| --- | --- | --- | --- | --- | --- | --- |
featureencoderfordriftingloss.
forabasewidthC.
Overview. UnlikestandardMAE(Heetal.,2022),whichis ThearchitecturefollowsstandardResNet(Heetal.,2016)
basedonViT(Dosovitskiyetal.,2021),ourMAEtrainsa design,withGroupNorm(GN)(Wu&He,2018)usedin
convolutionalResNetthatprovidesmulti-scalefeatures.For place of BatchNorm (BN) (Ioffe & Szegedy, 2015). All
latent-spacemodels,theinputandoutputhavedimension residualblocksare“basic”blocks(i.e.,eachconsistingof
32×32×4; for pixel-space models, the input and output two 3×3 convolutions). Following the standard ResNet-
havedimension256×256×3. 34 (He et al., 2016): the encoder has a 3×3 convolution
(withoutdownsampling)and4stageswith[3,4,6,3]blocks;
OurMAEconsistsofaResNet-styleencoderpairedwith
downsampling(stride2)happensatthefirstblockofstages
adeconvolutionaldecoderinaU-Net-style(Ronneberger
2to4.
| etal.,2015)encoder-decoderarchitecture. |     |     | Weonlyusethe |     |     |     |
| --------------------------------------- | --- | --- | ------------ | --- | --- | --- |
ResNet-styleencoderforfeatureextractionwhencomputing Forlatent-space(i.e.,latent-MAE),theinputofthisResNet
thedriftingloss. is32×32×4;forpixel-space,the256×256×3inputisfirst
13

GenerativeModelingviaDrifting
patchified(bya8×8patch)into32×32×192. TheResNet self-supervised pre-trained model using the MAE objec-
operatesontheinputwithH×W =32×32. tive, followed by classification fine-tuning. Like ResNet,
ConvNeXt-V2isamulti-stagearchitecture.
| MAEDecoder. | Thedecoderreturnstotheinputshapevia |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
deconvolutionsandskipconnections:
A.5.Multi-scaleFeaturesforDriftingLoss
|     | Decoder:{f |     | ,f ,f | ,f }(cid:55)→xˆ. |     |     |     |                                                   |     |     |     |     |     |     |
| --- | ---------- | --- | ----- | ---------------- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|     |            |     | 4 3   | 2 1              |     |     |     | Givenanimage,thefeatureencoderproducesfeaturemaps |     |     |     |     |     |     |
atmultiplescales,withmultiplespatiallocationsperscale.
| It starts            | with a 3×3 | convolutional |                              | block | on  | f 4 , followed |     |            |              |      |             |     |            |       |
| -------------------- | ---------- | ------------- | ---------------------------- | ----- | --- | -------------- | --- | ---------- | ------------ | ---- | ----------- | --- | ---------- | ----- |
|                      |            |               |                              |       |     |                |     | We compute | one drifting | loss | per feature |     | (e.g., per | scale |
| by4upsamplingblocks. |            |               | Eachupsamplingblockperforms: |       |     |                |     |            |              |      |             |     |            |       |
bilinear2×2upsampling→concatenatingwithencoder’s and/orperlocation). Specifically,wecomputethekernel,
thedrift,andtheresultinglossindependentlyforeachfea-
skipconnection→GN→two3×3convolutionswithGN
|           |         |     |             |     |          |     |        | ture. Theresultinglossesaresummed. |     |     |     |     |     |     |
| --------- | ------- | --- | ----------- | --- | -------- | --- | ------ | ---------------------------------- | --- | --- | --- | --- | --- | --- |
| and ReLU. | A final | 1×1 | convolution |     | produces | the | output |                                    |     |     |     |     |     |     |
channels. Forthepixel-space,thedecoderunpatchifiesback For each stage in a ResNet, we extract features from the
totheoriginalresolutionafterthelastlayer. output of every 2 residual blocks, together with the final
|                                                   |         |     |         |                |     |          |     | output. | This yields a                  | set of | feature | maps, | each of | shape |
| ------------------------------------------------- | ------- | --- | ------- | -------------- | --- | -------- | --- | ------- | ------------------------------ | ------ | ------- | ----- | ------- | ----- |
| Masking.                                          | The MAE | is  | trained | to reconstruct |     | randomly |     |         |                                |        |         |       |         |       |
|                                                   |         |     |         |                |     |          |     | H ×W ×C | . Foreachfeaturemap,weproduce: |        |         |       |         |       |
| maskedinputs.UnliketheViT-basedMAE(Heetal.,2022), |         |     |         |                |     |          |     | i i     | i                              |        |         |       |         |       |
whichremovesthemaskedtokensfromthesequence,we
|                             |     |     |     |                     |     |     |     | (a) H ×W | vectors,oneperlocation(eachC |     |     |     | -dim); |     |
| --------------------------- | --- | --- | --- | ------------------- | --- | --- | --- | -------- | ---------------------------- | --- | --- | --- | ------ | --- |
|                             |     |     |     |                     |     |     |     | i        | i                            |     |     |     | i      |     |
| simplyzerooutmaskedpatches. |     |     |     | Fortheinputofashape |     |     |     |          |                              |     |     |     |        |     |
H×W =32×32(ineitherthelatent-orpixel-basedcase), (b) 1globalmeanand1globalstd(eachC i -dim);
wemask2×2patchesbyzeroing. Eachpatchisindepen- (c) Hi×Wi vectorsofmeansand Hi×Wi vectorsofstds
|     |     |     |     |     |     |     |     | 2   | 2   |     |     | 2 2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
dentlymaskedwith50%probability.
|     |     |     |     |     |     |     |     | (eachC | i -dim),computedover2×2patches; |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------------------------------- | --- | --- | --- | --- | --- |
MAE training. We minimize the ℓ 2 reconstruction loss (d) Hi×Wi vectorsofmeansand Hi×Wi vectorsofstds
|               |          |     |        |       |             |     |     | 4      | 4                               |     |     | 4 4 |     |     |
| ------------- | -------- | --- | ------ | ----- | ----------- | --- | --- | ------ | ------------------------------- | --- | --- | --- | --- | --- |
| on the masked | regions. |     | We use | AdamW | (Loshchilov |     | &   |        |                                 |     |     |     |     |     |
|               |          |     |        |       |             |     |     | (eachC | i -dim),computedover4×4patches. |     |     |     |     |     |
Hutter,2019)withlearningrate4×10−3andabatchsizeof
8192. EMAwithdecay0.9995isused. Following(Heetal., Inaddition,fortheencoder’sinput(H 0 ×W 0 ×C 0 ),wecom-
2022),weapplyrandomresizedcropaugmentationtothe putethemeanofsquaredvalues(x2)perchannelandobtain
|                                                    |     |     |     |     |     |     |     | aC -dimvector. |     |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
| input(forthelatentsetting,imagesareaugmentedbefore |     |     |     |     |     |     |     | 0              |     |     |     |     |     |     |
beingpassedthroughtheVAEencoder).
|     |     |     |     |     |     |     |     | All resulting | vectors | here are | C -dim. | We  | compute | one |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ------- | -------- | ------- | --- | ------- | --- |
i
|                |              |     |     |     |              |         |     | drifting | loss for each | of these | C -dim | vectors. | All | these |
| -------------- | ------------ | --- | --- | --- | ------------ | ------- | --- | -------- | ------------- | -------- | ------ | -------- | --- | ----- |
| Classification | fine-tuning. |     | For | our | best feature | encoder |     |          |               |          | i      |          |     |       |
(last row of Table 3), we fine-tune the MAE model with losses,inadditiontothevanilladriftinglosswithoutϕ,are
alinearclassifierhead. ThelossisλL +(1−λ)L . summed. Thistablecomparestheeffectofthesedesignson
|                                                   |     |     |     |     | cls |     | recon |                     |     |     |     |     |     |     |
| ------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----- | ------------------- | --- | --- | --- | --- | --- | --- |
| Wefine-tuneallparametersinthisMAEfor3kiterations, |     |     |     |     |     |     |       | ourablationdefault: |     |     |     |     |     |     |
whereλfollowsalinearwarmupschedule,increasingfrom
|     |     |     |     |     |     |     |     |     |     | (a,b) | (a-c) | (a-d) |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ----- | ----- | --- | --- |
0to0.1overthefirst1kiterationsandremainingconstant
| at0.1fortherestofthetraining. |     |     |     |     |     |     |     |     | FID | 9.58 | 9.10 | 8.46 |     |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- | --- | --- |
Thisshowsthatourmethodbenefitsfromricherfeaturesets.
A.4.OtherPretrainedFeatureEncoders
|     |     |     |     |     |     |     |     | We note  | that once the        | feature | encoder             | is  | run, the  | compu- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------------------- | ------- | ------------------- | --- | --------- | ------ |
|     |     |     |     |     |     |     |     | tational | cost of our drifting |         | loss is negligible: |     | computing |        |
InadditiontoourcustomizedMAE,wealsoevaluateother
featureencodersforcomputingthedriftingloss. multi-scale,multi-locationlossesincurslittleoverheadcom-
paredtocomputingasingleloss.
MoCoandSimCLR.Weevaluatepubliclyavailableself-
| supervised | encoders | trained | on  | ImageNet | in  | pixel | space: |     |     |     |     |     |     |     |
| ---------- | -------- | ------- | --- | -------- | --- | ----- | ------ | --- | --- | --- | --- | --- | --- | --- |
A.6.FeatureandDriftNormalization
MoCo(Heetal.,2020;Chenetal.,2020b)SimCLR(Chen
et al., 2020a). We use the ResNet-50 variant. For latent- Tobalancethemultiplelosstermsfrommultiplefeatures,
spacegeneration,weapplytheVAEdecodertomapgen- we perform normalization for each feature ϕ , where, ϕ
|     |     |     |     |     |     |     |     |     |     |     |     |     | j   | j   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
eratoroutputsfromlatentspace(32×32×4)topixelspace denotesafeatureataspecificspatiallocationwithinagiven
(256×256×3)beforefeatureextraction. Gradientsareback- scale(seeA.5). Intuitively,wewanttoperformnormaliza-
propagatedthroughboththefeatureextractorandtheVAE tionsuchthatthekernelk(·,·)andthedriftVareinsensitive
| decoder. |     |     |     |     |     |     |     | totheabsolutemagnitudeoffeatures.Thisallowsourmodel |     |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
torobustlysupportdifferentfeatureencoders(seeTable3)
| MAE with | ConvNeXt-V2. |     | In  | our | pixel-space | genera- |     |     |     |     |     |     |     |     |
| -------- | ------------ | --- | --- | --- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
aswellasarichsetoffeaturesfromoneencoder.
| tor, we | also investigate |     | ConvNeXt-V2 |     | (Woo | et al., | 2023) |     |     |     |     |     |     |     |
| ------- | ---------------- | --- | ----------- | --- | ---- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- |
as the feature encoder. We note that ConvNeXt-V2 is a FeatureNormalization. Considerafeatureϕ ∈RCj. We
j
14

GenerativeModelingviaDrifting
defineanormalizationscaleS ∈ R1 andthenormalized Withthenormalizedfeatureandnormalizeddrift,thedrift-
j
featureisdenotedby: inglossofthefeatureϕ is:
j
ϕ˜ j :=ϕ j /S j . (18) L j =MSE(ϕ˜ j (x)−sg(ϕ˜ j (x)+V˜ j )), (26)
Whenusingϕ˜ ,theℓ distancecomputedinEq.(12)is: whereMSEdenotesmeansquarederror. Theoveralllossis
j 2 (cid:80)
thesumacrossallfeatures: L= L .
j j
dist (x,y)=∥ϕ˜ (x)−ϕ˜ (y)∥, (19)
j j j Multiple temperatures. Using normalized feature dis-
tances,thevalueoftemperatureτ determineswhatiscon-
wherexdenotesageneratedsampleandydenotesaposi-
sidered“nearby”. Toimproverobustnessacrossdifferent
tive/negativesample,andϕ˜ (·)meansextractingtheirfea-
j (cid:112) features and across different pretrained models we study,
tureatj. Wewanttheaveragedistancetobe C :
j weadoptmultipletemperatures.
(cid:112)
E E [dist (x,y)]≈ C . (20) Formally,foreachτ value,wecomputethenormalizeddrift
x y j j
asdescribedabove,denotedbyV˜ . Thenwecomputean
j,τ
Toachievethis,wesetthenormalizationscaleS j as: aggregatedfield: V˜ j ← (cid:80) τ V˜ j,τ ,anduseitforthelossin
Equation(26).
1
S = E E [∥ϕ (x)−ϕ (y)∥] (21)
j (cid:112) C x y j j Thistableshowstheeffectofmultipletemperaturesonour
j
ablationdefault:
Inpractice,weuseallxandysamplesinabatchtocompute
theempiricalmeaninplaceoftheexpectation. Wereusethe τ 0.02 0.05 0.2 {0.02,0.05,0.2}
cdistcomputationinAlg.2forcomputingthepairwise FID 10.62 8.67 8.96 8.46
distances. Weapplystop-gradienttoS ,becausethisscalar
j
isconceptuallycomputedfromsamplesfromtheprevious Usingmultipletemperaturescanachieveslightlybetterre-
batch. sultsthanusingasingleoptimaltemperature. Wefixτ ∈
{0.02,0.05,0.2}anddonotrequiretuningthishyperparam-
Withthenormalizedfeature,thekernelinEq.(12)issetas:
eteracrossdifferentconfigurations.
(cid:18) (cid:19)
1 Normalizationacrossspatiallocations. Forafeaturemap
k(x,y)=exp − ∥ϕ˜ (x)−ϕ˜ (y)∥ , (22)
τ˜ j j j of resolution H i ×W i , there are H i ×W i per-location fea-
tures. Separately computing the normalization for each
(cid:112)
where τ˜ := τ· C . By doing so, the value of tempera- locationwouldbeslowandunnecessary. Weassumethat
j j
tureτ doesnotdependonthefeaturemagnitudeorfeature featuresatdifferentlocationswithinthesamefeaturemap
dimensionality. We set τ ∈ {0.02, 0.05, 0.2} (discussed sharethesamenormalizationscale. Accordingly,wecon-
next). catenateallH ×W locationsandcomputethenormaliza-
i i
tionscaleoverallofthem. Thefeaturenormalizationand
DriftNormalization. Whenusingthefeatureϕ ,theresult-
j driftnormalizationarebothperformedinthisway.
ingdriftisinthesamefeaturespaceasϕ ,denotedasV .
j j
WeperformadriftnormalizationonV ,foreachfeature
j A.7.Classifier-FreeGuidance(CFG)
ϕ . Formally,wedefineanormalizationscaleλ ∈R1and
j j
denote: TosupportCFG,attrainingtime,weincludeN additional
unc
V˜ j :=V j /λ j . (23) unconditionalsamples(realimagesfromrandomclasses)
asextranegatives. Thesesamplesareweightedbyafactor
Again,wewantthenormalizeddrifttobeinsensitivetothe
w whencomputingthekernel. Forageneratedsamplex,
featuremagnitude:
theeffectivenegativedistributionitcompareswithis:
(cid:20) (cid:21)
E C 1 ∥V˜ j ∥2 ≈1. (24) q˜(·|c)≜ (N neg −1)·q θ (·|c)+N unc w·p data (·|∅) .
j (N −1)+N w
neg unc
Toachievethis,wesetλ as:
j ComparingthisequationwithEq.(15)(16),wehave:
λ j = (cid:115) E (cid:20) C 1 ∥V j ∥2 (cid:21) . (25) γ = (N neg − N 1 u ) nc + w N unc w
j
and
Inpractice,theexpectationisreplacedwiththeempirical 1 (N −1)+N w
α= = neg unc .
meancomputedovertheentirebatch. 1−γ N −1
neg
15

GenerativeModelingviaDrifting
Table9.Ablationsonpixel-spacegeneration.Westudygener-
GivenaCFGstrengthα,wecomputewaccordingly,which
ationdirectlyinpixelspace(withoutVAE).Applyingthesame
| isusedtoweightthekernel. |     |     | Thesameweightingwisalso |     |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
MAErecipeasinlatentspaceyieldshigherFID,indicatingthat
appliedwhencomputingtheglobaldistancenormalization. pixel-spacegenerationismorechallenging.CombiningMAEwith
|          |                |                  |     |       |         | ConvNeXt-V2helpsclosethisgap. |     |     |     | Latent-spaceresultsshown |     |     |     |
| -------- | -------------- | ---------------- | --- | ----- | ------- | ----------------------------- | --- | --- | --- | ------------------------ | --- | --- | --- |
| We train | our model with | CFG-conditioning |     | (Geng | et al., |                               |     |     |     |                          |     |     |     |
forreference.Theresultsbelowfollowtheablationsetting(B/16
2025b). Ateachiteration,werandomlysampleαfollowing modelforpixel-space,100epochs).
| a pre-defined                                  | distribution | (see | Table | 8) and compute | the |     |     |     |     |     |     |     |     |
| ---------------------------------------------- | ------------ | ---- | ----- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resultingwforweightingtheunconditionalsamples. |              |      |       |                | The |     |     |     |     |     |     |     |     |
FID(100-epoch)
value of α is a condition input to the network f (ϵ,c,α), featureencoderϕ latent(B/2) pixel(B/16)
θ
alongsidetheclasslabelc.
|                                     |     |     |     |               |     | MAE(width256,epoch192)         |     |     |     |     | 8.46 |     | 32.11 |
| ----------------------------------- | --- | --- | --- | ------------- | --- | ------------------------------ | --- | --- | --- | --- | ---- | --- | ----- |
|                                     |     |     |     |               |     | MAE(width640,epoch1280)+clsft. |     |     |     |     | 3.36 |     | 9.35  |
| Atinferencetime,wespecifyavalueofα. |     |     |     | Theinference- |     |                                |     |     |     |     |      |     |       |
|                                     |     |     |     |               |     | +MAEw/ConvNeXt-V2              |     |     |     |     |      | -   | 3.70  |
timecomputationremainstobeone-step(1-NFE).
Table10.Pixel-spacegeneration:fromablationtofinalsetting.
A.8.SampleQueue
Beyondtheablationsetting,wecomparethesettingsthatleadto
theresultsinTable6.
Ourmethodrequiresaccesstorandomlysampledreal(pos-
| itive/unconditional)data. |     | Thiscanbeimplementedusinga  |     |     |     |     |                         |     |     |      |     |      |     |
| ------------------------- | --- | --------------------------- | --- | --- | --- | --- | ----------------------- | --- | --- | ---- | --- | ---- | --- |
|                           |     |                             |     |     |     |     | case                    |     |     | arch | ep  | FID  |     |
| specializeddataloader.    |     | Instead,weadoptasamplequeue |     |     |     |     |                         |     |     |      |     |      |     |
|                           |     |                             |     |     |     |     | (a)baseline(fromTable9) |     |     | B/16 | 100 | 3.70 |     |
ofcacheddata,similartothequeueusedinMoCo(Heetal.,
2020). Thisimplementationsamplesdatainastatistically (b)longer+hyper-param. B/16 320 2.19
|     |     |     |     |     |     |     | (c)longer |     |     | B/16 | 640 | 1.76 |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | ---- | --- | ---- | --- |
similarwaytoaspecializeddataloader. Forcompleteness, (d)largermodel L/16 640 1.61
| we describe | our implementation |     | as  | follows, while | noting |     |     |     |     |     |     |     |     |
| ----------- | ------------------ | --- | --- | -------------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
thatadataloaderwouldbeamoreprincipledsolution.
B.AdditionalExperimentalResults
Foreachclasslabel,wekeepaqueueofsize128;forun-
conditionalsamples(usedinCFG),wemaintainaseparate B.1.AblationsonPixel-SpaceGeneration
globalqueueofsize1000.Ateachtrainingstep,wepushthe
|     |     |     |     |     |     | We  | provide | more ablations |     | on pixel-space |     | generation | in  |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------------- | --- | -------------- | --- | ---------- | --- |
latest64newreal(positive/unconditional)samples,along-
|     |     |     |     |     |     | Table9and10. |     | Table9comparestheeffectofthefeature |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------ | --- | ----------------------------------- | --- | --- | --- | --- | --- |
sidetheirlabels,intothecorrespondingqueues;theearliest
|                  |                                 |     |     |     |     | encoder | on  | the pixel-space |     | generator. | It shows |     | that the |
| ---------------- | ------------------------------- | --- | --- | --- | --- | ------- | --- | --------------- | --- | ---------- | -------- | --- | -------- |
| onesaredequeued. | Whensampling,positivesamplesare |     |     |     |     |         |     |                 |     |            |          |     |          |
choiceoffeatureencoderplaysamoresignificantrolein
drawnfromthequeueofthecorrespondingclass,andun-
|                                               |     |     |     |     |     | pixel-space |     | generation | quality. | A weaker |     | MAE | encoder |
| --------------------------------------------- | --- | --- | --- | --- | --- | ----------- | --- | ---------- | -------- | -------- | --- | --- | ------- |
| conditionalsamplesaredrawnfromtheglobalqueue. |     |     |     |     | We  |             |     |            |          |          |     |     |         |
yieldsanFIDof32.11,whereasastrongerMAEencoder
samplewithoutreplacement.
|     |     |     |     |     |     | improvesperformancetoanFIDof9.35. |     |     |     |     | Wefurtheradd |     |     |
| --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | ------------ | --- | --- |
anotherfeatureencoder,ConvNeXt-V2(Wooetal.,2023),
A.9.TrainingLoop
|     |     |     |     |     |     | which | is also | pre-trained | with | the | MAE objective. |     | This |
| --- | --- | --- | --- | --- | --- | ----- | ------- | ----------- | ---- | --- | -------------- | --- | ---- |
Insummary,inthetrainingloop,eachstepproceedsas: furtherimprovestheresulttoanFIDof3.70.
1. Sampleabatch(N )ofclasslabels. Table 10 reports the results of training longer and using
c
|     |     |     |     |     |     | alargermodel. |     | Duetolimitedtime,wetrainpixel-space |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | ----------------------------------- | --- | --- | --- | --- | --- |
2. Foreachlabelc,sampleaCFGscaleα.
modelsfor640epochs(vs.thelatentcounterpart’s1280);
| 3. Sampleabatch(N |     | )ofnoiseϵ. |     | Feed(ϵ,c,α)tothe |     |                                                     |     |     |     |     |     |     |     |
| ----------------- | --- | ---------- | --- | ---------------- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|                   |     | neg        |     |                  |     | weexpectthatlongertrainingwouldyieldfurtherimprove- |     |     |     |     |     |     |     |
generatorf toproducegeneratedsamples; ments.WeachieveanFIDof1.61forpixel-spacegeneration.
Thisisourresultinthemainpaper(Table6).
| 4. Samplepositivesamples(sameclass,N |     |     |     | )anduncon- |     |     |     |     |     |     |     |     |     |
| ------------------------------------ | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
pos
| ditionalsamples(forCFG,N |     |     |     | );  |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
unc
B.2.AblationonKernelNormalization
5. Extractfeaturesonallgenerated,positive,anduncon-
ditionalsamples In Eq. (11), our drifting field is weighted by normalized
6. Computethedriftinglossusingthefeatures. kernels,whichcanbewrittenas:
7. Runbackpropagationandparameterupdate. V(x)=E [k˜(x,y+)k˜(x,y−)(y+−y−)],
|     |     |     |     |     |     |     |     | p,q |     |     |     |     | (27) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
wherek˜(·,·)=
|     |     |     |     |     |     |     |     | 1k(·,·)denotesthenormalizedkernel. |     |     |     |     | In  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | --- |
Z
principle,thisnormalizationisapproximatedbyasoftmax
|     |     |     |     |     |     | operationovertheaxisofysamples. |            |         |         |      | Ourimplementation |     |        |
| --- | --- | --- | --- | --- | --- | ------------------------------- | ---------- | ------- | ------- | ---- | ----------------- | --- | ------ |
|     |     |     |     |     |     | (Alg.                           | 2) further | applies | softmax | over | the axis          | of  | x sam- |
ples. Wecomparethesedesigns,alongwithanothervariant
16

GenerativeModelingviaDrifting
Table11. Ablationonkernelnormalization. Softmaxnormal- Generated Retrieved Generated Retrieved
izationoverboththexandyaxesperformsbetter.Ontheother
hand,evenusingnonormalizationperformsdecently,showingthe
robustnessofourmethod.(Setting:B/2model,100epochs)
| kernelnormalization       |     | FID   |     |                                                     |                          |               |       |                     |          |          |
| ------------------------- | --- | ----- | --- | --------------------------------------------------- | ------------------------ | ------------- | ----- | ------------------- | -------- | -------- |
| softmaxoverxandy(default) |     | 8.46  |     |                                                     |                          |               |       |                     |          |          |
| softmaxovery              |     | 8.92  |     |                                                     |                          |               |       |                     |          |          |
| nonormalization           |     | 10.54 |     |                                                     |                          |               |       |                     |          |          |
|                           |     |       |     | Figure6.                                            | Nearestneighboranalysis. |               |       | Eachpanelshowsagen- |          |          |
|                           |     |       |     | eratedsampletogetherwithitstop-10nearestrealimages. |                          |               |       |                     |          | The      |
|                           |     |       |     | nearest neighbors                                   |                          | are retrieved | from  | the ImageNet        | training | set      |
|                           |     |       |     | based on                                            | the cosine               | similarity    | using | a CLIP              | encoder  | (Radford |
etal.,2021).Ourmethodgeneratesnovelimagesthatarevisually
distinctfromtheirnearestneighbors.
B.3.AblationonCFG
InFigure5,weinvestigatetheCFGscaleαusedatinference
|     |     |     |     | time. It               | shows    | that the | CFG formulation          |     | developed     | for   |
| --- | --- | --- | --- | ---------------------- | -------- | -------- | ------------------------ | --- | ------------- | ----- |
|     |     |     |     | our models             | exhibits | behavior | similar                  | to  | that observed | in    |
|     |     |     |     | diffusion-/flow-based  |          | models.  | Increasing               |     | the CFG       | scale |
|     |     |     |     | leadstohigherISvalues, |          |          | whereasbeyondtheFIDsweet |     |               |       |
spot,furtherincreasesinIScomeatthecostofworseFID.
|     |     |     |     | Notably, | with our | best | model (L/2), | the | optimal | FID is |
| --- | --- | --- | --- | -------- | -------- | ---- | ------------ | --- | ------- | ------ |
achievedatα=1.0,whichisoftenregardedas“w/oCFG”
|     |     |     |     | in diffusion-/flow-based |     |     | models (even | though | their | “w/o |
| --- | --- | --- | --- | ------------------------ | --- | --- | ------------ | ------ | ----- | ---- |
Figure5. EffectofCFGscaleα. (a): FIDvs.α. (b): ISvs.α. CFG”settingcanreduceNFEbyhalf). Whileourmethod
(c):ISvs.FID.WeshowtheL/2(solid)andB/2(dashed)models. neednotrunanunconditionalmodelatinferencetime(in
Consistent with common observations in diffusion-/flow-based contrasttostandardCFG),trainingisinfluencedbytheuse
models,theCFGscaleeffectivelytradesoffdistributionalcoverage
ofunconditionalrealsamplesasnegatives.
(asreflectedbyFID)againstper-imagequality(measuredbyIS).
Notably,withtheL/2model,theoptimalFIDisachievedatα=1.0,
whichisoftenregardedas“w/oCFG”indiffusion-/flow-based B.4.NearestNeighborAnalysis
models.ForB/2,theoptimalFIDisachievedatα=1.1.
InFigure6,weshowgeneratedimagestogetherwiththeir
|                        |      |     |     | nearest                                      | real images. | The | nearest | neighbors | are | retrieved |
| ---------------------- | ---- | --- | --- | -------------------------------------------- | ------------ | --- | ------- | --------- | --- | --------- |
| withoutnormalization(Z | =1). |     |     |                                              |              |     |         |           |     |           |
|                        |      |     |     | fromtheImageNettrainingsetusingCLIPfeatures. |              |     |         |           |     | These     |
Table 11 compares the three designs. Using the y-only visualizationssuggestthatourmethodgeneratesnovelim-
|                  |                  |         | x          | agesthatarevisuallydistinctfromtheirnearestneighbors, |     |     |     |     |     |     |
| ---------------- | ---------------- | ------- | ---------- | ----------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| softmax performs | well (8.92 FID), | whereas | using both |                                                       |     |     |     |     |     |     |
andysoftmaximprovestheresult(8.46FID).Ontheother ratherthanmerelymemorizingtrainingsamples.
| hand, even without | normalization, | performance | remains |     |     |     |     |     |     |     |
| ------------------ | -------------- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- |
decent,demonstratingtherobustnessofourmethod. B.5.QualitativeResults
Wenotethatallthreevariantssatisfytheequilibriumcon- Fig.7-10showuncuratedsamplesfromourmodel. Fig.11-
dition V p,q (x) = 0 when p = q. This explains why all 15provideside-by-sidecomparisonwithimprovedMean-
variantsperformreasonablywellandwhyeventhedestruc- Flow(iMF)(Gengetal.,2025b),thecurrentstate-of-the-art
| tivesetting(nonormalization)avoidscatastrophicfailure. |     |     |     | one-stepmethod. |     |     |     |     |     |     |
| ------------------------------------------------------ | --- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- |
17

GenerativeModelingviaDrifting
C.AdditionalDerivations withdN ≫m2,suchlinearindependenceisanaturalnon-
degeneracycondition.
C.1.OnIdentifiabilityoftheZero-DriftEquilibrium
Uniquenessoftheequilibrium. Thezero-driftcondition
InSec.3,weshowedthatanti-symmetryimpliesp=q ⇒
V(x) ≡ 0impliesV = 0. Groupingtermsbytheinde-
X
V(x) ≡ 0. Hereweinvestigatetheconverse: underwhat
pendentbasisvectors{U } ,wehave:
ij i<j
conditionsdoesV(x) ≈ 0implyp ≈ q? Generally, this
(cid:88)
isnotguaranteedforarbitraryvectorfields. However,we (a b −a b )U =0. (32)
i j j i ij
arguethatforourspecificconstruction,thezero-driftcondi-
1≤i<j≤m
tionimposesstrongconstraintsonthedistributions.
By the linear independence assumption, the coefficients
Toavoidboundaryissues,weassumethatpandqhavefull mustvanish: a b −a b =0foralli,j. Thisimpliesthat
i j j i
supportonRd(e.g.,viainfinitesimalGaussiansmoothing).
thevectoraisparalleltob(i.e.,a∝b). Sincepandqare
Consequently,ensuringtheequilibriumconditionV(x)≈ probabilitydensities(implying (cid:82) p = (cid:82) q = 1), wemust
0forgeneratedsamplesx∼qeffectivelyenforcesV(x)≈ havea=b,andthusp=q.
0forallx∈Rd.
Connectiontothemeanshiftfield. Themean-shiftfield
Setup. Consider a general interaction kernel fitsthisframework. Theupdatevector(beforenormaliza-
K(x,y+,y−)∈Rdandthedriftingfield tion)isE [k(x,y+)k(x,y−)(y+−y−)]. Assumingthe
p,q
V (x):=E (cid:2) K(x,y+,y−) (cid:3) . (28) normalization factors Z p and Z q are finite, the condition
p,q y+∼p,y−∼q V(x)=0impliesthenumeratorintegralvanishes,which
correspondstoaninteractionkerneloftheform:
Weassumethatpandqbelongtoafinite-dimensionalmodel
classspannedbyalinearlyindependentbasis{φ i }m i=1 : K(x,y+,y−)=k(x,y+)k(x,y−)(y+−y−). (33)
m m
(cid:88) (cid:88) Thiskernelgeneratesthebilinearstructureanalyzedabove.
p(y)= a φ (y), q(y)= b φ (y), (29)
i i i i SincewecanchooseN suchthatdN ≫m2,thedimension
i=1 i=1
of the test space is much larger than the number of basis
wherea,b∈Rmarecoefficientvectors. pairs.Thus,thelinearindependenceof{U }isexpectedto
ij
holdforgenericconfigurations. Finally,forgeneraldistribu-
Bilinearexpansionovertestlocations. Considerasetof
tionspandq,wecanapproximatethemusingasufficiently
testlocations(probes)X ={x }N withsufficientlylarge
k k=1 largebasisexpansion,turningintop˜andq˜. Whenthebasis
N (e.g., N ≫ m2). For each pair of basis indices (i,j),
approximationissufficientlyaccurate,p˜≈pandq˜≈q,and
wedefinetheinducedinteractionvectorU ∈ Rd×N by
ij thedriftfieldV ≈ V ≈ 0. Bytheargumentabove,
computingitscolumn: p˜,q˜ p,q
p˜≈q˜,andthusp≈q.
(cid:90)(cid:90)
U [:,x]≜ K(x,y+,y−)φ (y+)φ (y−)dy+dy− Theargumentaboveworksforgeneralformofdriftingfield,
ij i j
undermildanti-degeneracyassumptions.
(30)
evaluated at all x ∈ X. Substituting the basis expansion
C.2.TheDriftingFieldofMMD
intoEq.(28),thedriftingfieldevaluatedonX (storedasa
matrixV X )isabilinearcombination: Inprinciple,ifamethodminimizesadiscrepancybetween
twodistributionspandq andreachesminimumatp = q,
m m
V ≜ (cid:88)(cid:88) a b U . (31) then from the perspective of our framework, a drifting
X i j ij
fieldV existsthatgovernssamplemovement: wecanlet
i=1j=1
V∝−∂L, whichiszerowhenp = q. Wediscussthefor-
∂x
Here,V ∈Rd×N. Attheequilibrium,wehaveV =0, mulation of this V for a loss based on Maximum Mean
X X
whichyieldsdN linearequations. Discrepancy(MMD)(Lietal.,2015;Dziugaiteetal.,2015).
Linear independence assumption. Our anti-symmetry GradientsofDriftingLoss. Withx=f (ϵ),ourdrifting
θ
conditionimpliesthatswitchingpandq negatesthefield. lossinEq.(6)canbewrittenas:
In terms of basis interactions, this means U = −U
(and consequently U ii = 0). We make the g i e j neric no j n i - L=E x∼q [L(x)]=E x∼q (cid:104)(cid:13) (cid:13)x−sg (cid:0) x+V(x) (cid:1)(cid:13) (cid:13) 2 (cid:105) , (34)
degeneracyassumption: Thesetofvectors{U }
ij 1≤i<j≤m
islinearlyindependentinRdN. Thisassumptionrequires where “sg” is short for stop-gradient. The gradient w.r.t.
theprobesX andkernelK tobenon-degenerate; ifallx
theparametersθiscomputedby:
yield identical constraints, independence would fail. For ∂L (cid:104)∂L(x)∂x(cid:105)
generic choices of K and sufficiently diverse probes X ∂θ =E x∼q ∂x ∂θ . (35)
18

GenerativeModelingviaDrifting
where ∂L(x)=2(x−sg(x+V(x)))=−2V(x). Thisgives: In (Li et al., 2015), the Gaussian kernel is used:
∂x
ξ(x,y) = exp(− 1 ∥x−y∥2),leadingtoξ′(∥x−y∥2) =
1∂L(x) − 1 exp(− 1 ∥x 2σ − 2 y∥2).
V(x)=− (36) 2σ2 2σ2
2 ∂x
RelationsandDifferences. Whenusingourdefinitionof
We note that this formulation is general and imposes no V=V+−V−(i.e.,Eq.(10)),wehave:
constraintsonV,exceptthatV=0whenp=q.
(cid:104) (cid:105)
Our method does not require L to define a discrepancy
V(x)=E
y+∼p
k˜(x,y+)(y+−x)
betweenpandq. However,forothermethodsthatdepend (cid:104) (cid:105) (44)
−E k˜(x,y−)(y−−x)
onminimizingadiscrepancyL, wecaninduceadrifting y−∼q
fieldvia(36). ThisisvalidifLisminimizedwhenp=q.
Comparing (43) with (44), we show that the underlying
GradientsofMMDLoss. InMMD-basedmethods(e.g.,
kernelusedtobuildthedriftingfieldofMMDis:
Lietal.2015),thedifferencebetweentwodistributionsp
andqismeasuredbysquaredMMD:
k˜ (x,y)=−2ξ′(∥x−y∥2). (45)
MMD
L (p,q)=E [ξ(x,x′)]−2E [ξ(y,x)]
MMD2 x,x′∼q y∼p,x∼q When ξ is a Gaussian function, we have: k˜(x,y) =
+const.
1 exp(− 1 ∥x−y∥2). Withoutnormalization,theresult-
(37) σ2 2σ2
ing drift no longer satisfies the assumptions underlying
Here,theconstanttermisE [ξ(y,y′)],whichdepends
y,y′∼p Alg.2,andthemean-shiftinterpretationbreaksdown.
onlyonthetargetdistributionpandremainsunchanged. ξ
isakernelfunction. As a comparison, our general formulation enables to use
normalizedkernels:
Considerx=f (ϵ)withϵ∼p . Thegradientestimation
θ ϵ
performedin(Lietal.,2015)correspondsto: 1 1
k˜(x,y)= k(x,y)= k(x,y), (46)
Z(x) E [k(x,y)]
∂L (cid:104)∂L (x)∂x(cid:105) y
MMD2 =E MMD2 (38)
∂θ x∼q ∂x ∂θ where the expectation is over p or q. Only when we use
wherethegradientw.r.txiscomputedby: normalizedkernels,wehave(seeEq.(11)):
(cid:104) (cid:105)
∂L MMD2 (x) =2E (cid:104)∂ξ(x,x′)(cid:105) −2E (cid:104)∂ξ(x,y)(cid:105) . V(x)=E p,q k˜(x,y+)k˜(x,y−)(y+−y−) , (47)
∂x x′∼q ∂x y∼p ∂x
(39)
onwhichourAlg.2isbased.
Usingournotationofpositivesandnegatives,werename
thevariablesandrewriteas: Giventhisrelation,wesummarizethekeydifferencesbe-
tweenourmodelandtheMMD-basedmethodsasfollows:
∂L (x) (cid:104)∂ξ(x,y−)(cid:105) (cid:104)∂ξ(x,y+)(cid:105)
MMD2 =2E −2E .
∂x y−∼q ∂x y+∼p ∂x (i) OurmethodisformulatedaroundthedriftingfieldV,
(40)
whichismoreflexibleandgeneral.
ComparingwithEq.(36),weobtain:
(ii) Ourmethodsupportsandleveragesnormalizedkernels
(cid:104)∂ξ(x,y+)(cid:105) (cid:104)∂ξ(x,y−)(cid:105) 1k(x,y) that cannot be naturally derived from the
V (x)≜E −E Z
MMD y+∼p ∂x y−∼q ∂x MMDperspective.
(41) (iii) OurV-centricformulationenablesaflexiblestepsize
Thisistheunderlyingdriftingfieldthatcorrespondstothe fordrifting(i.e.,x←x+ηV)andthereforenaturally
MMDlossL MMD2 . supportsV-normalization(seeA.6).
Foraradialkernelξ(x,y) = ξ(R)whereR = ∥x−y∥2, (iv) OurV-centricformulationallowstheequilibriumcon-
thegradientofkernelis: cepttobenaturallyextendedtosupportCFG,whereas
aCFGvariantforMMDremainsunexplored.
∂ξ(x,y)
=2ξ′(∥x−y∥2)(x−y) (42)
Insummary,althoughaspecialcaseofourmethodreduces
∂x
to MMD, our V-centric framework is more general and
whereξ′isthederivativeofthefunctionξ(R). Accordingly,
enablesuniquepossibilitiesthatareimportantinpractice.
Eq.(41)becomes:
Inourexperiments,wewerenotabletoobtainreasonable
(cid:104) (cid:105) resultsusingtheMMDframework.
V (x)=E 2ξ′(∥x−y+∥2)(x−y+) .
MMD y+∼p
(43)
(cid:104) (cid:105)
−E 2ξ′(∥x−y−∥2)(x−y−)
y−∼q
19

GenerativeModelingviaDrifting
Class012:housefinch,linnet,Carpodacusmexicanus Class017:jay
Class021:kite Class022:baldeagle,Americaneagle,Haliaeetusleucocephalus
Class024:greatgreyowl,greatgrayowl,Strixnebulosa Class031:treefrog,tree-frog
Class088:macaw Class090:lorikeet
Class092:beeeater Class095:jacamar
Figure7.Uncuratedsamplesfromourlatent-L/2modelwithCFG=1.0(page1/4).FID=1.54,IS=258.9.
20

GenerativeModelingviaDrifting
Class108:seaanemone,anemone Class145:kingpenguin,Aptenodytespatagonica
Class270:whitewolf,Arcticwolf,Canislupustundrarum Class279:Arcticfox,whitefox,Alopexlagopus
Class288:leopard,Pantherapardus Class291:lion,kingofbeasts,Pantheraleo
Class296:icebear Class323:monarch,Danausplexippus
Class349:bighorn,bighornsheep,Oviscanadensis Class386:Africanelephant,Loxodontaafricana
Figure8.Uncuratedsamplesfromourlatent-L/2modelwithCFG=1.0(page2/4).FID=1.54,IS=258.9.
21

GenerativeModelingviaDrifting
Class388:giantpanda,Ailuropodamelanoleuca Class425:barn
Class448:birdhouse Class483:castle
Class580:greenhouse,nursery,glasshouse Class649:megalith,megalithicstructure
Class698:palace Class718:pier
Class755:radiotelescope,radioreflector Class780:schooner
Figure9.Uncuratedsamplesfromourlatent-L/2modelwithCFG=1.0(page3/4).FID=1.54,IS=258.9.
22

GenerativeModelingviaDrifting
Class829:streetcar,tram,tramcar,trolley,trolleycar Class927:trifle
Class958:hay Class970:alp
Class973:coralreef Class975:lakeside,lakeshore
Class979:valley,vale Class980:volcano
Class985:daisy Class992:agaric
Figure10.Uncuratedsamplesfromourlatent-L/2modelwithCFG=1.0(page4/4).FID=1.54,IS=258.9.
23

GenerativeModelingviaDrifting
ours improvedMeanFlow(iMF)
Class012:Housefinch
Class014:Indigobunting
Class022:Baldeagle
Class042:Agama
Class081:Ptarmigan
Figure11.Side-by-sidecomparisonwithimprovedMeanFlow(iMF)(Gengetal.,2025b)(page1/5).Uncuratedsamplesfromour
method(left)andiMF(right)onallImageNetclassesvisualizedintheiMFpaper.Bothmethodsgenerateimageswithasingleneural
functionevaluation(1-NFE).TheiMFvisualizationsuseCFGω=6.0andinterval[t ,t ]=[0.2,0.8],achievingFID3.92andIS348.2
min max
(DiT-XL/2).Forfaircomparison,wesettheCFGscaletomatchtheISofiMFvisualizations,whichleadstoFID3.01andIS354.4(at
CFG=1.5)forourmethod(DiT-L/2).
24

GenerativeModelingviaDrifting
ours improvedMeanFlow(iMF)
Class108:Seaanemone
Class140:Red-backedsandpiper
Class289:Snowleopard
Class291:Lion
Class387:Lesserpanda
Figure12.Side-by-sidecomparisonwithimprovedMeanFlow(iMF)(Gengetal.,2025b)(page2/5).Uncuratedsamplesfromour
method(left)andiMF(right)onallImageNetclassesvisualizedintheiMFpaper.Bothmethodsgenerateimageswithasingleneural
functionevaluation(1-NFE).TheiMFvisualizationsuseCFGω=6.0andinterval[t ,t ]=[0.2,0.8],achievingFID3.92andIS348.2
min max
(DiT-XL/2).Forfaircomparison,wesettheCFGscaletomatchtheISofiMFvisualizations,whichleadstoFID3.01andIS354.4(at
CFG=1.5)forourmethod(DiT-L/2).
25

GenerativeModelingviaDrifting
ours improvedMeanFlow(iMF)
Class437:Beacon
Class483:Castle
Class540:Drillingplatform
Class562:Fountain
Class649:Megalith
Figure13.Side-by-sidecomparisonwithimprovedMeanFlow(iMF)(Gengetal.,2025b)(page3/5).Uncuratedsamplesfromour
method(left)andiMF(right)onallImageNetclassesvisualizedintheiMFpaper.Bothmethodsgenerateimageswithasingleneural
functionevaluation(1-NFE).TheiMFvisualizationsuseCFGω=6.0andinterval[t ,t ]=[0.2,0.8],achievingFID3.92andIS348.2
min max
(DiT-XL/2).Forfaircomparison,wesettheCFGscaletomatchtheISofiMFvisualizations,whichleadstoFID3.01andIS354.4(at
CFG=1.5)forourmethod(DiT-L/2).
26

GenerativeModelingviaDrifting
ours improvedMeanFlow(iMF)
Class698:Palace
Class738:Pot
Class963:Pizza
Class970:Alp
Class973:Coralreef
Figure14.Side-by-sidecomparisonwithimprovedMeanFlow(iMF)(Gengetal.,2025b)(page4/5).Uncuratedsamplesfromour
method(left)andiMF(right)onallImageNetclassesvisualizedintheiMFpaper.Bothmethodsgenerateimageswithasingleneural
functionevaluation(1-NFE).TheiMFvisualizationsuseCFGω=6.0andinterval[t ,t ]=[0.2,0.8],achievingFID3.92andIS348.2
min max
(DiT-XL/2).Forfaircomparison,wesettheCFGscaletomatchtheISofiMFvisualizations,whichleadstoFID3.01andIS354.4(at
CFG=1.5)forourmethod(DiT-L/2).
27

GenerativeModelingviaDrifting
ours improvedMeanFlow(iMF)
Class975:Lakeside
Class976:Promontory
Class985:Daisy
Figure15.Side-by-sidecomparisonwithimprovedMeanFlow(iMF)(Gengetal.,2025b)(page5/5).Uncuratedsamplesfromour
method(left)andiMF(right)onallImageNetclassesvisualizedintheiMFpaper.Bothmethodsgenerateimageswithasingleneural
functionevaluation(1-NFE).TheiMFvisualizationsuseCFGω=6.0andinterval[t ,t ]=[0.2,0.8],achievingFID3.92andIS348.2
min max
(DiT-XL/2).Forfaircomparison,wesettheCFGscaletomatchtheISofiMFvisualizations,whichleadstoFID3.01andIS354.4(at
CFG=1.5)forourmethod(DiT-L/2).
28