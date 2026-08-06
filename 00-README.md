# The Machinery of Learning

### A complete reference textbook on Machine Learning and Deep Learning, with the mathematics kept in

---

## What this book is

A single, self-contained curriculum covering the mathematics, algorithms, architectures, and empirical phenomena of modern machine learning — from the bias–variance decomposition to superposition in transformer residual streams, from CART splitting criteria to GRPO.

It is written to be **re-read**. Chapters are self-contained, cross-referenced, and organized so that any concept can be looked up in isolation. Derivations are complete: there is no "it can be shown that."

### Assumed background

- Multivariable calculus and linear algebra.
- Comfort reading $\sum$, $\mathbb{E}$, $\nabla$, and matrix notation.
- Some prior exposure to ML (an introductory course, or having built models).

Everything else is built from the ground up, including the parts that introductory courses skip.

---

## Structure of every section

1. **The one-line idea** — plain English.
2. **The analogy** — anchored in something physical.
3. **The mathematics** — derived in full.
4. **The numbers** — a worked example with real magnitudes, to build calibration rather than symbol-recognition.
5. **What goes wrong** — the failure modes, which are a separate kind of knowledge from the formula.

Equations that must be memorized are marked **▸**. If you are skimming, read only those.

---

## The complete table of contents

### Part I — Foundations

| # | Chapter | Contents |
|---|---|---|
| 01 | **[Mathematical Foundations](01-mathematical-foundations.md)** | Linear algebra for DL; SVD and low-rank structure; norms and Lipschitz constants; Johnson–Lindenstrauss; matrix calculus; forward vs reverse-mode AD; Hessians, Gauss–Newton, Fisher; probability; softmax Jacobian; entropy, cross-entropy, KL, mutual information; Jensen and the ELBO. |
| 02 | **[Learning Theory & Generalization](02-learning-theory-and-generalization.md)** | ERM; bias–variance derived; Hoeffding and concentration; union bound; VC dimension and Sauer–Shelah; Rademacher complexity; PAC-Bayes and non-vacuous bounds; why uniform convergence fails for deep nets. |
| 03 | **[Resampling & Noisy Evaluation](03-resampling-and-noisy-evaluation.md)** | Bootstrap (percentile, BCa, .632, block, wild); jackknife and influence functions; cross-validation and the correlated-average variance formula; permutation tests; Monte Carlo standard error; **the record-statistics analysis of validation curves**. |

### Part II — Optimization

| # | Chapter | Contents |
|---|---|---|
| 04 | **[Optimization I: Gradient Descent](04-optimization-i-gradient-descent.md)** | Smoothness, strong convexity, the descent lemma; convergence rates derived; condition number; the quadratic model; heavy-ball and Nesterov momentum; SGD, its noise floor, and the SDE view; Newton, natural gradient, K-FAC, L-BFGS; gradient clipping; weight averaging. |
| 05 | **[Optimization II: Adaptive Methods & Schedules](05-optimization-ii-adaptive-methods-and-schedules.md)** | AdaGrad → RMSProp → Adam → **AdamW**; bias correction derived; the L2-vs-decoupled-decay distinction; every LR schedule with its formula; warmup and RAdam's rectification; **Edge of Stability**; Adam-vs-SGD generalization; Lion, Shampoo, Muon, SAM. |

### Part III — Neural Networks

| # | Chapter | Contents |
|---|---|---|
| 06 | **[Neural Networks & Backpropagation](06-neural-networks-and-backpropagation.md)** | Universal approximation and depth separation; backprop derived in index and matrix form; VJP rules; vanishing/exploding gradients quantified; Xavier/He initialization derived; zero-init residual branches; the activation zoo (GELU, SwiGLU); debugging protocol. |
| 07 | **[Normalization & Regularization](07-normalization-and-regularization.md)** | BatchNorm and its train/eval gap; LayerNorm, RMSNorm, GroupNorm, WeightNorm; scale invariance and the effective-LR/weight-decay coupling; dropout as ensembling and as Bayesian approximation; label smoothing; mixup/CutMix; early stopping as implicit $\ell_2$; stochastic depth. |
| 08 | **[Convolutions, ResNets & Vision Architectures](08-convolutions-resnets-vision.md)** | Convolution as constrained matmul; receptive field arithmetic; dilated, separable, transposed convolutions; the three competing explanations of why ResNets work; BatchNorm–ResNet interaction; Inception, DenseNet, EfficientNet, ConvNeXt; U-Net; detection and segmentation heads. |
| 09 | **[Sequence Models: RNNs, LSTMs, Seq2Seq](09-sequence-models-rnn-lstm.md)** | BPTT derived; the eigenvalue condition for exploding/vanishing recurrence; LSTM gates and the constant error carousel; GRU; bidirectional and deep RNNs; encoder–decoder; Bahdanau and Luong attention — the direct ancestor of the transformer; CTC; teacher forcing and exposure bias. |

### Part IV — Transformers and Large Language Models

| # | Chapter | Contents |
|---|---|---|
| 10 | **[Tokenization, Embeddings & Vectorization](10-tokenization-embeddings-vectorization.md)** | One-hot → count vectors → TF-IDF → word2vec (skip-gram with negative sampling, derived) → GloVe → contextual embeddings; BPE, WordPiece, Unigram/SentencePiece with the algorithms; vocabulary size trade-offs; embedding matrices, tying, and factorization; sentence embeddings and vector databases; ANN search (HNSW, IVF-PQ). |
| 11 | **[Attention & the Transformer](11-attention-and-transformers.md)** | Attention from first principles; the $\sqrt{d_k}$ derivation; self- vs cross-attention; multi-head and what heads specialize in; causal masking; the full encoder and decoder block; pre-LN vs post-LN; the FFN as key–value memory; residual stream as a communication channel; complete FLOP and parameter accounting; Mixture-of-Experts. |
| 12 | **[Positional Information & Long Context](12-positional-encoding-long-context.md)** | Why permutation equivariance forces the issue; sinusoidal, learned, relative (T5, Shaw); **RoPE derived in full**; ALiBi; NoPE; context extension (PI, NTK-aware, YaRN); efficient attention: sparse, Longformer, Linformer, Performer; **FlashAttention's tiling and online softmax**; MQA/GQA/MLA; state-space models (S4, Mamba) and linear attention. |
| 13 | **[GPT: Autoregressive Language Modelling](13-gpt-autoregressive-language-models.md)** | The next-token objective and why it is enough; the GPT-1→2→3→4 architectural lineage; in-context learning and induction heads; decoding strategies (greedy, beam, top-$k$, nucleus, min-$p$, temperature, contrastive, typical); repetition and degeneration; perplexity and its pitfalls; BERT/T5 contrast and the encoder–decoder-vs-decoder-only question. |
| 14 | **[Training LLMs at Scale](14-training-llms-at-scale.md)** | Data: sourcing, dedup (MinHash/SimHash), filtering, curriculum, contamination; the training stack; fp16/bf16/fp8 and loss scaling; data/tensor/pipeline/sequence parallelism; ZeRO stages; gradient checkpointing arithmetic; the memory budget formula; loss spikes and instabilities (z-loss, QK-norm); throughput accounting (MFU). |
| 15 | **[Scaling Laws & Emergence](15-scaling-laws-and-emergence.md)** | Kaplan power laws; **Chinchilla derived**, the compute-optimal $N \propto C^{0.5}$, $D\propto C^{0.5}$ result; inference-aware scaling; data-constrained scaling; $\mu$P and hyperparameter transfer derived; emergence and the metric-discontinuity critique; test-time compute scaling. |
| 16 | **[Post-Training: SFT, RLHF, DPO, Reasoning](16-post-training-rlhf-dpo-reasoning.md)** | Instruction tuning; the Bradley–Terry reward model derived; PPO for language with the KL penalty; **DPO derived from the RLHF optimum in full**; IPO/KTO/ORPO/SimPO; GRPO and RLVR; process vs outcome supervision; constitutional AI and RLAIF; reasoning models, chain-of-thought, self-consistency; reward hacking and length bias. |
| 17 | **[Efficient Inference & Compression](17-efficient-inference-and-compression.md)** | The KV-cache memory formula; prefill vs decode and arithmetic intensity; continuous batching, PagedAttention; speculative and Medusa decoding with the acceptance-rate math; quantization (PTQ, GPTQ, AWQ, SmoothQuant, QAT, straight-through estimator); pruning (magnitude, structured, SparseGPT); knowledge distillation derived; LoRA/QLoRA/DoRA and the rank-choice argument; model merging (task arithmetic, TIES, SLERP). |
| 18 | **[Retrieval, RAG, Tools & Agents](18-retrieval-rag-agents.md)** | Sparse vs dense retrieval; BM25 derived; bi-encoders vs cross-encoders; contrastive retriever training; chunking; hybrid search and reciprocal rank fusion; rerankers; RAG failure modes and evaluation; long-context vs retrieval; tool use and function calling; ReAct, planning, memory; multi-agent patterns; agent evaluation. |

### Part V — Generative Modelling

| # | Chapter | Contents |
|---|---|---|
| 19 | **[Generative Models: The Taxonomy](19-generative-models-taxonomy.md)** | The trilemma (quality / diversity / speed); autoregressive models; **VAEs with the full ELBO, reparameterization, and posterior collapse**; VQ-VAE and straight-through; **GANs with the optimal discriminator and JS-divergence result derived**; mode collapse; WGAN and the Kantorovich–Rubinstein duality; normalizing flows and the change-of-variables formula; energy-based models and contrastive divergence; evaluation (FID, IS, precision/recall, and their failure modes). |
| 20 | **[Diffusion Models](20-diffusion-models.md)** | The forward process and its closed form; the exact posterior derived; the variational bound term by term; why $\epsilon$-prediction; $v$-prediction and SNR weighting; **the score-matching connection**; the SDE/probability-flow-ODE view; DDIM and deterministic sampling; noise schedules (linear, cosine, shifted); classifier and classifier-free guidance derived; latent diffusion; consistency and distillation models; flow matching and rectified flow. |
| 21 | **[Discrete Diffusion & Conditional Generation](21-discrete-diffusion-and-conditioning.md)** | Categorical forward processes; **D3PM transition matrices $Q_t$, cumulative $\bar Q_t$, exact posterior, and the full loss including the auxiliary $x_0$ term**; uniform, absorbing, and structured kernels; SEDD and the concrete score; MDLM; conditioning mechanisms compared (concatenation, cross-attention, FiLM, **AdaLN-Zero**); the DiT architecture block by block; discrete guidance; conditioning collapse diagnostics. |

### Part VI — Classical Machine Learning

| # | Chapter | Contents |
|---|---|---|
| 22 | **[Classical Supervised Learning](22-classical-supervised-learning.md)** | OLS with the normal equations and the geometry of projection; ridge (with the SVD shrinkage formula), LASSO, elastic net, and the soft-threshold derivation; logistic regression and IRLS; generalized linear models; **SVMs: the margin, the dual, KKT conditions, kernels, and the representer theorem**; naive Bayes; kNN and the curse of dimensionality; the imbalanced-data toolkit. |
| 23 | **[Trees & Gradient Boosting](23-trees-and-gradient-boosting.md)** | CART; Gini vs entropy vs MSE; pruning; bagging and the correlated-ensemble variance formula; random forests and OOB; extra trees; **AdaBoost as coordinate descent on exponential loss, derived**; gradient boosting as functional gradient descent; **XGBoost's second-order objective with the leaf-weight and gain formulas derived**; LightGBM (histogram, GOSS, EFB, leaf-wise growth); CatBoost (ordered boosting, ordered target statistics); tuning order; SHAP and permutation importance. |
| 24 | **[Unsupervised Learning & Dimensionality Reduction](24-unsupervised-learning.md)** | PCA derived three ways (variance, reconstruction, SVD); probabilistic PCA; kernel PCA; ICA and the non-Gaussianity objective; NMF; random projection; **t-SNE and UMAP objectives with their failure modes**; k-means, k-means++, and its EM interpretation; **GMM with the complete EM derivation**; DBSCAN/HDBSCAN; spectral clustering and the graph-Laplacian argument; hierarchical clustering; cluster validity metrics; anomaly detection. |
| 25 | **[Self-Supervised & Representation Learning](25-self-supervised-representation-learning.md)** | Pretext tasks; **InfoNCE as a mutual-information bound, derived**; SimCLR, MoCo, and the queue; the role of temperature, hard negatives, and batch size; **BYOL/SimSiam and why they don't collapse**; Barlow Twins and VICReg; DINO and self-distillation; masked autoencoders; alignment and uniformity; evaluation protocols (linear probe, k-NN, fine-tune) and their disagreements. |

### Part VII — Reinforcement Learning

| # | Chapter | Contents |
|---|---|---|
| 26 | **[Reinforcement Learning Foundations](26-rl-foundations.md)** | MDPs and the Markov property; return, discounting, and why $\gamma$ exists; **the Bellman expectation and optimality equations derived**; policy evaluation, policy iteration, value iteration; **the contraction-mapping convergence proof**; Monte Carlo vs TD; TD($\lambda$) and eligibility traces; SARSA vs Q-learning (on- vs off-policy); importance sampling; the deadly triad; exploration (ε-greedy, UCB, Thompson, and the bandit regret bounds). |
| 27 | **[Deep Reinforcement Learning](27-deep-reinforcement-learning.md)** | DQN and every stabilizing trick; Double, Dueling, PER, n-step, distributional (C51/QR-DQN), NoisyNet, Rainbow; **the policy gradient theorem derived**; REINFORCE, baselines, and the variance argument; actor–critic; **GAE derived**; TRPO's natural gradient and KL trust region; **PPO's clipped surrogate**; DDPG, TD3, and **SAC with the max-entropy objective**; model-based RL (Dyna, MuZero, Dreamer); offline RL (BCQ, CQL, IQL, Decision Transformer); multi-agent RL; the sim-to-real and reward-specification problems. |

### Part VIII — Modalities & Structure

| # | Chapter | Contents |
|---|---|---|
| 28 | **[Vision Transformers & Multimodal Models](28-vision-transformers-and-multimodal.md)** | ViT patchification and the inductive-bias trade; DeiT, Swin, hierarchical ViTs; **CLIP and the symmetric InfoNCE objective**; SigLIP; zero-shot classification as retrieval; captioning; VLM architectures (frozen encoder + projector, cross-attention, early fusion); Flamingo/LLaVA/Qwen-VL patterns; visual tokenizers; audio and speech (spectrograms, wav2vec 2.0, Whisper, neural codecs); video; any-to-any models. |
| 29 | **[Graph Neural Networks & Geometric Deep Learning](29-graph-neural-networks.md)** | Graphs, adjacency, and the Laplacian; spectral convolution and its localization; GCN, GraphSAGE, GAT, GIN; **the message-passing framework and the Weisfeiler–Lehman expressivity bound**; over-smoothing and over-squashing with the mathematics; pooling; equivariance and invariance; E(3)-equivariant networks and tensor-field networks; graph transformers; applications to molecules and physics. |

### Part IX — The Science of Deep Learning

| # | Chapter | Contents |
|---|---|---|
| 30 | **[Double Descent, Grokking & the NTK](30-double-descent-grokking-ntk.md)** | Model-wise, epoch-wise, and sample-wise double descent; **the minimum-norm interpolation analysis and the pole at the interpolation threshold**; effective vs raw parameter count; benign overfitting; the neural tangent kernel derived and its lazy-training regime; feature learning vs kernel regime; **grokking: the delayed-generalization phenomenon, the weight-norm mechanism, and progress measures**; the role of weight decay and representation formation. |
| 31 | **[Neural Collapse, Implicit Bias & Lottery Tickets](31-neural-collapse-implicit-bias-lottery-tickets.md)** | **Neural collapse NC1–NC4 and the simplex ETF geometry derived**; the unconstrained-features model; **the implicit bias of gradient descent toward the max-margin solution**, with the $O(1/\log t)$ rate; Adam's different bias; flat minima, sharpness measures, and their reparameterization problem; the SDE temperature argument; **the Lottery Ticket Hypothesis, iterative magnitude pruning, rewinding, transfer, and the strong LTH**; linear mode connectivity and permutation symmetry. |
| 32 | **[Mechanistic Interpretability & Superposition](32-mechanistic-interpretability-superposition.md)** | Features, circuits, and the linear representation hypothesis; **superposition, the toy model, and the sparsity/importance phase diagram**; polysemanticity; **sparse autoencoders, the $\ell_1$ objective, dead features, and JumpReLU/TopK variants**; the residual stream as a bandwidth-limited channel; QK and OV circuits; induction heads and in-context learning; activation patching, path patching, and causal scrubbing; probing and its confounds; steering vectors; what interpretability has and has not established. |
| 33 | **[Calibration, Uncertainty & Robustness](33-calibration-uncertainty-robustness.md)** | Proper scoring rules; ECE and its estimator bias; temperature scaling, Platt, isotonic; aleatoric vs epistemic uncertainty; deep ensembles, MC dropout, SWAG, Laplace approximation; conformal prediction with the coverage guarantee derived; OOD detection (MSP, energy, Mahalanobis, ODIN); distribution shift taxonomy; adversarial examples, PGD, adversarial training, certified defences; spurious correlations and group robustness (IRM, GroupDRO). |

### Part X — Reference

| # | Chapter | Contents |
|---|---|---|
| 34 | **[The Interview Bank](34-interview-bank.md)** | ~200 questions with worked answers, organized by topic and difficulty, including every question in this book's scope: the classics, the derivations, the systems questions, the "why does this work" questions, and the research-level ones. Plus the ten answers that most reliably separate strong candidates. |
| — | **[Verification Notes](VERIFICATION.md)** | The result of a full page-by-page audit of the mathematics, examples, and analogies in this book: what was checked, what was corrected, and the known limitations. |

---

## Reading paths

**Path A — Full curriculum (12–16 weeks).** 01 → 34 in order. This is the correct path.

**Path B — Interview preparation (3–4 weeks).** 02, 03, 05, 06, 07, 11, 12, 13, 15, 16, 19, 20, 23, 26, 27, 30, 31, 32, 34.

**Path C — LLM specialist.** 10 → 18, then 15, 16, 32.

**Path D — Generative modelling specialist.** 19 → 21, then 20's flow-matching section, then 28.

**Path E — Classical ML / applied DS.** 02, 03, 22, 23, 24, 33.

**Path F — Frontier phenomena first.** 30 → 31 → 32, back-filling from the prerequisite lines at the top of each.

**Daily maintenance.** Each chapter ends with a **Check for Understanding** — a single compressed sentence. Reading those 34 sentences takes ten minutes and is a genuine refresher. The **▸** equations are the second layer of that same idea.

---

## Notation

| Symbol | Meaning |
|---|---|
| $x, y$ | input, label |
| $\theta, w$ | parameters |
| $f_\theta$ | the model |
| $\mathcal{L}, \hat{\mathcal{L}}$ | population loss, empirical loss |
| $\eta, \alpha$ | learning rate |
| $g_t$ | gradient at step $t$ |
| $n$ | training set size |
| $d, d_{\text{model}}$ | input / hidden dimension |
| $N$ or $p$ | parameter count |
| $D$ | number of training tokens (Ch. 15) |
| $B$ | batch size |
| $L$ | number of layers |
| $T$ | sequence length (Part IV) or horizon (Part VII) or diffusion steps (Part V) — disambiguated per chapter |
| $H(\cdot)$, $\mathrm{KL}(\cdot\|\cdot)$ | entropy, KL divergence |
| $\odot$ | elementwise product |
| $\lambda_{\max}(H)$ | top Hessian eigenvalue ("sharpness") |
| $\gamma$ | discount factor (Part VII) |
| $\beta_t, \alpha_t, \bar\alpha_t$ | diffusion schedule quantities |

Vectors are columns. $\nabla_\theta\mathcal{L}$ has the same shape as $\theta$.

---

## Case Study A — the book's running numerical example

A single concrete training run is used across Chapters 1–5, 20, and 21 so that abstract quantities acquire real magnitudes. It is a discrete-diffusion transformer:

```
Model:        DiT backbone, D3PM-style discrete diffusion
Data:         145,515 training sequences, 15,864 validation
Batch size:   64          →  2,274 optimizer steps per epoch
Optimizer:    AdamW, lr = 3e-4, weight_decay = 0.01, no scheduler
Validation:   16 batches (~1,024 items), fresh random timestep t per batch
Metric:       token-level cross-entropy ("realCE")
Log:          best 1.556 @ epoch 22 → 1.547 @ 36 → 1.524 @ 37 → next @ 43
```

Derived facts referenced later:

▸ A 13-epoch gap with no new best equals **29,562 parameter updates** — the model changed enormously; only the record stood still.

▸ AdamW with $\beta_2=0.999$ has a memory horizon of $1/(1-\beta_2) = 1{,}000$ steps $= 0.44$ epochs. **The optimizer's entire memory is shorter than half an epoch**, so it cannot encode a multi-epoch plateau even in principle.

▸ Under a flat-performance null, the probability of no new record across epochs 23–35 is exactly $\prod_{k=23}^{35}(1-1/k) = 22/35 = 0.629$. **The dry spell is the single most likely outcome and carries no information.** Back-to-back records at 36 and 37 have probability $\frac{1}{36}\cdot\frac{1}{37} = 7.5\times10^{-4}$ under the same null — *that* is the evidence of genuine improvement.

The full analysis is Chapter 3 §3.6. The pattern generalizes to any noisy metric tracked by its running minimum.

---

## How to use this

Read a section, close it, and re-derive the boxed equation on paper. If you can't, you recognized it rather than learned it — and recognition is the failure mode that feels exactly like understanding.

Start at [Chapter 01](01-mathematical-foundations.md).
