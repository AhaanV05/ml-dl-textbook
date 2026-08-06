# Chapter 29 — Graph Neural Networks & Geometric Deep Learning

> **Prerequisites:** Ch. 8, Ch. 11, Ch. 24 (§24.4 spectral clustering).

---

## 29.1 Why graphs need their own machinery

### The one-line idea

Graph data has no canonical ordering of its nodes, so any architecture must produce the same answer under any relabelling — and that permutation-invariance requirement determines almost everything about the design.

### The analogy

Describing a molecule. If you hand two chemists the same molecule with the atoms numbered differently, they must agree it is the same molecule. A model that reads an adjacency matrix row by row like an image would give two different answers, because it would be reading the *numbering* as if it were content. GNNs are built so the numbering cannot be seen.

### The formal requirement

▸ For a permutation matrix $P$: **node-level outputs must be equivariant**, $f(PAP^\top, PX)=Pf(A,X)$, and **graph-level outputs must be invariant**, $g(PAP^\top,PX)=g(A,X)$.

This is the same design principle as convolution's translation equivariance (Ch. 8 §8.1) applied to a different symmetry group. **Geometric deep learning is the program of deriving architectures from the symmetry group of the data**: translation → CNN, permutation of a set → DeepSets, permutation of a graph → GNN, rotation/translation in 3-D → E(3)-equivariant networks, sequence order → RNN/transformer with positional encoding.

### Notation

$G=(V,E)$, adjacency $A$, degree $D=\mathrm{diag}(d_i)$, node features $X\in\mathbb{R}^{n\times d}$, neighbourhood $\mathcal{N}(i)$.
**Laplacian:** $L = D-A$; normalized $L_{\text{sym}}=I-D^{-1/2}AD^{-1/2}$, with eigenvalues in $[0,2]$.

---

## 29.2 From spectral to spatial

### Spectral convolution

The graph Fourier transform uses the Laplacian's eigenvectors: $L=U\Lambda U^\top$, $\hat x = U^\top x$. Convolution becomes multiplication in the spectral domain:
$$g_\theta \star x = U\,g_\theta(\Lambda)\,U^\top x$$

**Problems:** $O(n^3)$ eigendecomposition, $O(n^2)$ multiplication, filters are not localized, and the eigenbasis is graph-specific so nothing transfers between graphs.

### ChebNet

Approximate $g_\theta(\Lambda)$ by a degree-$K$ Chebyshev polynomial. Because $L^k$ has support exactly on the $k$-hop neighbourhood, ▸ **a degree-$K$ polynomial filter is automatically $K$-hop localized**, and it needs no eigendecomposition — just $K$ sparse matrix–vector products.

### GCN — the first-order simplification

Take $K=1$, set $\lambda_{\max}\approx2$, tie the two coefficients, and add a **renormalization trick** ($\tilde A = A+I$, $\tilde D$ its degree matrix) to keep the spectrum stable under stacking:

▸ $$\boxed{\ H^{(l+1)} = \sigma\!\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\right)\ }$$

▸ **Read it as: each node averages its neighbours' features (plus its own, via the self-loop), with degree normalization, then applies a shared linear map and a nonlinearity.** The spectral derivation motivates the formula, but the formula itself is purely local — which is why the whole field moved to the spatial view.

---

## 29.3 Message passing

▸ The general framework subsuming nearly every GNN:

$$m_{ij}^{(l)} = \phi\big(h_i^{(l)},\,h_j^{(l)},\,e_{ij}\big)$$
$$a_i^{(l)} = \bigoplus_{j\in\mathcal{N}(i)} m_{ij}^{(l)} \qquad\text{(a permutation-invariant aggregator)}$$
$$h_i^{(l+1)} = \psi\big(h_i^{(l)},\,a_i^{(l)}\big)$$

**Aggregate → Update.** $\bigoplus$ must be permutation-invariant: sum, mean, max, or attention-weighted sum. Everything else is a design choice.

▸ **$L$ layers means each node sees its $L$-hop neighbourhood** — the exact analogue of a CNN's receptive field (Ch. 8 §8.1).

### The main architectures

| Model | Aggregation | Note |
|---|---|---|
| **GCN** | degree-normalized mean | simple, strong baseline, transductive |
| **GraphSAGE** | mean / LSTM / max-pool over a **sampled** fixed-size neighbourhood, then concatenate with self | **inductive** — generalizes to unseen nodes; neighbour sampling makes it scale to billions of edges |
| **GAT** | attention-weighted sum | $\alpha_{ij}=\mathrm{softmax}_j\big(\mathrm{LeakyReLU}(a^\top[Wh_i\|Wh_j])\big)$; learned, anisotropic neighbour weighting; multi-head |
| **GIN** | **sum**, then an MLP | maximally expressive under the WL bound (§29.4) |
| **MPNN / GINE** | edge features in the message | essential for molecules (bond types) |
| **PNA** | multiple aggregators concatenated with degree scalers | one aggregator cannot capture everything |

▸ **A transformer is a GNN on a complete graph** with attention aggregation and positional encodings standing in for structure. Saying this correctly connects Chapters 11 and 29 and is a strong interview answer.

---

## 29.4 Expressivity and the Weisfeiler–Lehman bound

### The 1-WL test

An algorithm for graph isomorphism testing: initialize each node's colour by its degree; repeatedly set each node's new colour to a hash of (its colour, the multiset of its neighbours' colours); if two graphs' colour histograms ever differ, they are not isomorphic.

### The theorem

▸ **Message-passing GNNs are at most as powerful as the 1-WL test at distinguishing non-isomorphic graphs** (Xu et al., 2019; Morris et al., 2019).

**Why:** a message-passing layer computes exactly a function of (node feature, multiset of neighbour features) — structurally identical to a WL refinement round. It can never distinguish two graphs that WL cannot.

▸ **The concrete consequence to remember: a message-passing GNN cannot count triangles, cannot detect cycles of a given length, and cannot distinguish two 3-regular graphs on the same number of nodes.** In a 3-regular graph every node looks identical to WL forever, so a GNN assigns them all the same representation regardless of global structure. For chemistry — where rings determine everything — this is not a theoretical curiosity.

### GIN — matching the bound

To reach the WL bound, the aggregator must be **injective on multisets**. Mean and max are not: $\{a,a\}$ and $\{a\}$ have the same mean and max. **Sum is** (for countable features).

▸ $$h_i^{(l+1)} = \mathrm{MLP}^{(l)}\left((1+\epsilon^{(l)})\,h_i^{(l)} + \sum_{j\in\mathcal{N}(i)}h_j^{(l)}\right)$$

The MLP provides the universal function approximation, the sum provides injectivity, and $\epsilon$ distinguishes the node from its neighbours. **GIN is as expressive as any message-passing GNN can be.**

*(Note the practical caveat: mean aggregation is better when the neighbourhood *distribution* matters more than its size, and max is better for detecting a distinctive neighbour. Maximal expressivity is not always maximal accuracy.)*

### Going beyond WL

- **Higher-order GNNs (k-WL):** message-pass over $k$-tuples of nodes. More expressive, $O(n^k)$ cost.
- **Positional/structural encodings:** add Laplacian eigenvectors, random-walk return probabilities, or random node identifiers as features. Cheap and effective; breaks strict permutation invariance in the random-ID case (recovered in expectation).
- **Subgraph GNNs:** represent a graph by a bag of its subgraphs.
- **Substructure counting:** just add triangle/ring counts as features. Inelegant, works well, standard in molecular ML.

---

## 29.5 The two structural pathologies

### Over-smoothing

▸ Repeated neighbourhood averaging is a diffusion process. As $L\to\infty$, node representations converge to a limit determined only by degree — **all nodes become identical** and the model loses discriminative power.

**The mathematics:** the GCN propagation matrix $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ has $\lambda_{\max}=1$ with all other $|\lambda_i|<1$. So $\hat A^L$ converges to a rank-one projection onto the dominant eigenvector, and everything else decays as $|\lambda_2|^L$. **Exponential collapse in depth.**

**Consequence:** most GNNs are 2–4 layers deep. **This is the single biggest structural difference from CNNs and transformers**, which get better with depth.

**Mitigations:** residual/initial connections (GCNII adds $\alpha H^{(0)}$ at every layer, provably slowing the collapse), jumping-knowledge (concatenate all layers' outputs), PairNorm/DropEdge, and simply using fewer layers with wider neighbourhood sampling.

### Over-squashing

▸ A node's $L$-hop neighbourhood can grow exponentially, but its representation is a **fixed-size vector**. Information from an exponentially large receptive field is squashed through $O(d)$ bits.

**The mathematics:** the sensitivity $\left\|\frac{\partial h_i^{(L)}}{\partial x_j}\right\|$ for a node $j$ at distance $L$ decays with the **normalized adjacency powers**, and it is controlled by the graph's **Cheeger constant / spectral gap** — a bottleneck edge in the graph is a bottleneck in information flow. This is why long-range tasks on tree-like or bottlenecked graphs fail even when the model is deep enough to reach.

**Mitigations:** graph rewiring (add edges to improve the spectral gap — e.g. stochastic discrete Ricci flow, expander-graph augmentation), a **fully-connected "virtual node"** connected to everything (a cheap global shortcut), and graph transformers.

▸ **Note the tension: over-smoothing pushes you to fewer layers; over-squashing pushes you to more.** Long-range dependencies on graphs remain genuinely hard, and this pair of constraints is why.

---

## 29.6 Pooling and readout

**Graph-level readout** must be permutation-invariant: sum (preserves size information), mean (size-invariant), max, or attention-weighted (Set2Set, GlobalAttention).

▸ **Sum vs mean is a real modelling decision.** For molecules, many properties are extensive (scale with size — e.g. molecular weight, total energy) so sum is correct; others are intensive (solubility, band gap) so mean is correct. Getting this wrong is a common and silent source of poor performance.

**Hierarchical pooling:** DiffPool (learn a soft cluster assignment matrix), TopK/SAGPool (score and keep the top nodes), EdgePool. Useful but often not better than a good flat readout.

---

## 29.7 Graph transformers

Run full attention over all nodes, injecting structure through the encodings rather than through the message passing.

**Structural signals:**
- **Laplacian eigenvectors** as positional encodings (the graph analogue of sinusoids — note the sign ambiguity of eigenvectors must be handled, usually by random sign flipping).
- **Random-walk structural encodings**: the diagonal of $(D^{-1}A)^k$ for several $k$.
- **Shortest-path distance as an attention bias** (Graphormer) — the graph analogue of T5's relative position bias (Ch. 12 §12.3).
- **Degree centrality** as a node feature.

▸ **The advantage: no over-squashing (every node is one hop from every other) and no over-smoothing.** The costs: $O(n^2)$ attention, loss of the sparsity prior, and greater data hunger — exactly the ViT-versus-CNN trade of Chapter 28 §28.1, appearing again on a different symmetry group.

**GraphGPS** and similar hybrids interleave message-passing layers (local, efficient) with attention layers (global). This is currently the strongest general recipe, and it mirrors the local/global interleaving that won in long-context language models (Ch. 12 §12.5).

---

## 29.8 Equivariance in 3-D

For molecules, proteins, and physical systems, node positions $\vec r_i\in\mathbb{R}^3$ matter, and predictions must respect **E(3)** (rotation, translation, reflection) or **SE(3)** (no reflection — important for chirality).

▸ **Energy must be invariant; forces must be equivariant** ($\vec F\to R\vec F$ under a rotation $R$).

**Three strategies:**

1. **Invariant features only.** Use distances $\|\vec r_i-\vec r_j\|$, angles, and dihedrals — all invariant by construction. SchNet (distances via radial basis functions), DimeNet (adds angles), GemNet (adds dihedrals). ▸ Simple and effective; the limitation is that distance-only descriptors are provably incomplete — some distinct geometries share all pairwise distances.

2. **Equivariant vector features.** Keep vector-valued node states alongside scalars and combine them only in ways that preserve equivariance: scalars scale vectors, dot products of vectors give scalars, cross products give vectors. **EGNN** is the minimal version:
▸ $$m_{ij}=\phi_e\big(h_i,h_j,\|\vec r_i-\vec r_j\|^2\big),\qquad \vec r_i \leftarrow \vec r_i + \sum_{j\ne i}(\vec r_i-\vec r_j)\,\phi_r(m_{ij})$$
Positions are updated only along **differences of positions** scaled by invariant quantities — so the update rotates with the input automatically. Simple, cheap, and a strong baseline.

3. **Spherical harmonics / tensor fields.** Represent features as irreducible representations of SO(3) and combine them with Clebsch–Gordan tensor products (Tensor Field Networks, NequIP, MACE, e3nn). Maximum expressivity and the best accuracy on force fields; substantial complexity and cost.

▸ **Why equivariance is worth the trouble:** it is a hard constraint, so the model never wastes capacity learning that physics is rotation-invariant, and it generalizes perfectly to unseen orientations. Empirically it improves data efficiency by roughly an order of magnitude on molecular property prediction. **This is the strongest available demonstration that the right symmetry prior beats more data.**

**The counter-trend worth noting:** at very large data scale, non-equivariant models with rotation augmentation (e.g. large transformer force fields) have become competitive — the same "prior versus scale" story as Chapter 28 §28.1. The prior is still winning in the low-data regime that most chemistry lives in.

---

## 29.9 Applications and practicalities

**Molecules:** property prediction (QM9, MoleculeNet), force fields replacing DFT at $10^5$–$10^6\times$ the speed, generation (Ch. 21 — a molecular graph is a discrete object, which is exactly what discrete diffusion handles), retrosynthesis, and docking.

**Proteins:** AlphaFold2's Evoformer is a graph-attention system over residue pairs; AlphaFold3 and successors use diffusion over atomic coordinates. Structure prediction is now largely a solved problem for single chains.

**Other:** recommendation (PinSage — GraphSAGE at billion-node scale), traffic forecasting, physics simulation (GNS), combinatorial optimization, fraud detection, and knowledge graphs.

**The practical warnings:**
- ▸ **Scaffold splitting, not random splitting**, for molecular data. Random splits leak analogues between train and test and inflate scores by 10–20 points. This is Chapter 3's grouped-CV point, and it is violated constantly in published chemistry ML.
- Featurization often matters more than architecture. Try a fingerprint (ECFP) + gradient boosting baseline first (Ch. 23); on many molecular benchmarks it wins.
- Watch for extensive/intensive readout mismatch (§29.6).
- Large graphs need sampling (GraphSAGE, Cluster-GCN) or historical-embedding methods; full-batch training on a 100M-node graph is not possible.

---

## Check for Understanding

**GNNs are the architecture you get by demanding permutation equivariance, which forces every layer into the aggregate-then-update message-passing form — and that form caps their expressivity at the 1-WL test, so they cannot count triangles; depth is limited from one side by over-smoothing (repeated averaging collapses all nodes to the dominant eigenvector) and from the other by over-squashing (an exponentially large receptive field compressed into a fixed vector), which is why graph transformers and hybrid local/global models exist.**

---

**Next:** [Chapter 30 — Double Descent, Grokking & the NTK](30-double-descent-grokking-ntk.md)
