# Chapter 29 — Graph Neural Networks & Geometric Deep Learning

> **Prerequisites:** Ch. 8, Ch. 11, Ch. 24 (§24.4 spectral clustering).

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; every entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $G=(V,E)$ | "G equals V, E" | A **graph**: a bag of nodes $V$ and a bag of edges $E$ joining them |
| $A$ | "the adjacency matrix" | An $n\times n$ grid of 0s and 1s; $A_{ij}=1$ means "node $i$ is wired to node $j$" |
| $d_i$ | "the degree of i" | How many neighbours node $i$ has — the row sum $\sum_j A_{ij}$ |
| $D=\mathrm{diag}(d_i)$ | "the degree matrix" | The degrees laid down the diagonal, zeros everywhere else |
| $L = D-A$ | "the graph Laplacian" | "Your own degree, minus your neighbours" — a difference operator on the graph |
| $L_{\text{sym}}$ | "the symmetric normalized Laplacian" | The same operator rescaled so popular nodes don't shout down quiet ones |
| $\mathcal{N}(i)$ | "the neighbourhood of i" | The set of nodes directly wired to $i$ |
| $P$ | "a permutation matrix" | A relabelling: exactly one 1 per row and per column, zeros elsewhere |
| $PAP^\top$ | "P A P transpose" | The very same graph, with its nodes renumbered |
| $X \in \mathbb{R}^{n\times d}$ | "X in R n-by-d" | Node features: one row of $d$ numbers per node |
| $h_i^{(l)}$ | "h-i at layer l" | Node $i$'s hidden vector after $l$ rounds of message passing |
| $m_{ij}$ | "the message from j to i" | What neighbour $j$ posts to node $i$ this round |
| $\bigoplus$ | "the aggregator" | Any operation that ignores the order of its inputs — sum, mean, max |
| $\tilde A = A+I$ | "A-tilde" | The adjacency matrix with a **self-loop** bolted onto every node |
| $\alpha_{ij}$ | "alpha-i-j" | An attention weight — how much node $i$ chooses to listen to node $j$ |
| $\vec r_i \in \mathbb{R}^3$ | "r-vector-i" | Node $i$'s position in 3-D space (an atom's coordinates) |
| $\lambda_2$ | "lambda-two" | The second-largest eigenvalue — it governs how fast information spreads |

### Full forms of every abbreviation in this chapter

Read each full form aloud once. Most of these stop being intimidating the moment you hear what they stand for.

| Short | Full form |
|---|---|
| GNN | Graph Neural Network |
| GCN | Graph Convolutional Network |
| GAT | Graph ATtention network |
| GIN | Graph Isomorphism Network |
| GraphSAGE | Graph **SA**mple and aggre**G**at**E** |
| MPNN | Message Passing Neural Network |
| GINE | GIN with **E**dge features |
| PNA | Principal Neighbourhood Aggregation |
| WL | Weisfeiler–Lehman (graph isomorphism test) |
| GCNII | GCN with **I**nitial residual and **I**dentity mapping |
| DiffPool | **Diff**erentiable **Pool**ing |
| SAGPool | **S**elf-**A**ttention **G**raph **Pool**ing |
| GraphGPS | Graph transformer that is **G**eneral, **P**owerful and **S**calable |
| EGNN | E($n$)-Equivariant Graph Neural Network |
| E(3) | the **E**uclidean group in 3-D: rotations, translations **and** reflections |
| SE(3) | the **S**pecial **E**uclidean group: rotations and translations, **no** reflections |
| SO(3) | the **S**pecial **O**rthogonal group in 3-D: rotations only |
| DFT | **D**ensity **F**unctional **T**heory (quantum chemistry — *not* discrete Fourier transform here) |
| ECFP | Extended-Connectivity FingerPrint |
| QM9 | a **Q**uantum-**M**echanical dataset of molecules with up to **9** heavy atoms |
| GNS | Graph Network–based Simulator |
| CNN | Convolutional Neural Network |
| ViT | Vision Transformer |
| MLP | Multi-Layer Perceptron |

---

## 29.1 Why graphs need their own machinery

### The one-line idea

Graph data has no canonical ordering of its nodes, so any architecture must produce the same answer under any relabelling — and that permutation-invariance requirement determines almost everything about the design.

### The analogy

Describing a molecule. If you hand two chemists the same molecule with the atoms numbered differently, they must agree it is the same molecule. A model that reads an adjacency matrix row by row like an image would give two different answers, because it would be reading the *numbering* as if it were content. GNNs are built so the numbering cannot be seen.

#### Examples and non-examples: data that  needs a GNN

Before any of the machinery, the prior question: when is this the right tool at all?

**✅  graph-structured**

| Example | The nodes | The edges | Why nothing simpler works |
|---|---|---|---|
| A caffeine molecule | 24 atoms | 25 bonds | There is no canonical atom order; the same molecule admits $24!$ valid numberings |
| The Cora citation network (2,708 papers) | Papers | "cites" | A paper's topic depends on what it cites, which depends on what *those* cite |
| A road network for travel-time prediction | Junctions | Road segments | Congestion propagates along roads, not across a grid |
| A protein's residue contact map | Amino acids | Pairs within about 8 Å | Residues far apart along the chain can be adjacent in space |
| A Pinterest pin–board graph (~3 billion nodes) | Pins and boards | "pinned to" | Relationships are the entire signal; there is no ordering to exploit |

**❌ Near-misses — often called graphs, but a GNN is the wrong tool**

| Looks like it | Why a GNN is not the answer | What to use instead |
|---|---|---|
| A grid of pixels | It *is* a graph — but one with a **canonical ordering**. Pixel $(3,7)$ is always in the same place | A CNN, which exploits exactly the ordering a GNN throws away |
| A sentence, "a chain graph of tokens" | Also a graph with a canonical ordering, and here the order carries the meaning | A transformer with positional encodings (Ch. 11–12) |
| A table of customer rows | Rows are independent; there are no edges to pass messages along | Gradient boosting (Ch. 23) |
| A DOM tree whose children have a fixed order | Sibling order is real information that a permutation-invariant layer would delete | A tree-structured model, or serialize and use a sequence model |
| A molecule written as a SMILES string | The string is a *serialization* — a canonical ordering imposed after the fact | Either a GNN on the graph or a language model on the string; the point is that you have now chosen one |

▸ **The boundary:** reach for a GNN when **the numbering of the entities is arbitrary and the relationships are the content.** If a natural order exists — left to right, first to last, row by row — you have a sequence or a grid, and discarding that order costs you information rather than protecting you from it.

> **Common misconception.** *"An image is a grid, a grid is a graph, so a CNN is a special case of a GNN and I could always just use the GNN."* The inclusion is true and the conclusion is backwards. A CNN's power comes precisely from the thing a GNN is built to ignore: a fixed spatial order lets it learn a *different weight per relative position* — one weight for "the pixel above," another for "the pixel to the left." A GNN on a pixel grid is forced to use the same weight for every neighbour, because it is forbidden from knowing which neighbour is which. **Every symmetry you build in is a piece of structure you have promised not to use.** You impose a symmetry because it is true of your data, not because it is a free upgrade.

### The formal requirement

▸ For a permutation matrix $P$: **node-level outputs must be equivariant**, $f(PAP^\top, PX)=Pf(A,X)$, and **graph-level outputs must be invariant**, $g(PAP^\top,PX)=g(A,X)$.

This is the same design principle as convolution's translation equivariance (Ch. 8 §8.1) applied to a different symmetry group. **Geometric deep learning is the program of deriving architectures from the symmetry group of the data**: translation → CNN, permutation of a set → DeepSets, permutation of a graph → GNN, rotation/translation in 3-D → E(3)-equivariant networks, sequence order → RNN/transformer with positional encoding.

#### Equivariance and invariance, decoded

Two words that sound like synonyms and mean  different things. Both are worth being able to say out loud.

- **Invariant** = *"the answer doesn't move."* Shuffle the input, get the identical output.
- **Equivariant** = *"the answer moves along with it."* Shuffle the input, and the output gets shuffled in exactly the same way.

> **Analogy.** A class photograph. **Invariant**: "how many people are in this photo?" — reshuffle everyone and the answer is still 31. **Equivariant**: "who is wearing a red shirt?" — reshuffle everyone and the answer reshuffles too; the *list* changes, the *content* doesn't. Predicting a property of the whole molecule is the first kind. Predicting a property of each atom is the second.

**Now the symbols.** A **permutation matrix** $P$ is a relabelling machine: it has exactly one $1$ in every row and every column and zeros everywhere else. Multiplying by it shuffles rows; multiplying by $P^\top$ on the right shuffles columns.

- $PX$ — "shuffle the rows of the feature table," i.e. renumber the nodes.
- $PAP^\top$ — "shuffle **both** the rows and the columns of the adjacency matrix." Both, because $A_{ij}$ refers to node $i$ *and* node $j$; renumber the nodes and both indices have to move together.

**A concrete two-node example.** Take a graph with one edge between node 1 and node 2, and swap the labels:

$$A = \begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad P = \begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad PAP^\top = \begin{pmatrix}0&1\\1&0\end{pmatrix}$$

The adjacency matrix came back unchanged, because "node 1 is joined to node 2" says the same thing however you number them. **That is the whole point: $PAP^\top$ and $A$ are the same graph wearing different name tags.**

**Reading the two conditions aloud:**

| Condition | Read aloud | What it forbids |
|---|---|---|
| $f(PAP^\top, PX)=Pf(A,X)$ | "renumber the input, and the per-node answers come back renumbered the same way" | Any layer whose output depends on *which* node you called node 1 |
| $g(PAP^\top,PX)=g(A,X)$ | "renumber the input, and the whole-graph answer is bit-for-bit identical" | Any readout that reads the node ordering as if it were data |

▸ **Why this single requirement determines the architecture.** An image has a canonical order — pixel $(3,7)$ is always in the same place, so a CNN can safely learn a weight *per position*. A graph has no such order: your adjacency matrix and mine describe the same molecule with the rows in a different sequence. Feeding an adjacency matrix into a fully-connected layer would train the model on the arbitrary output of whatever numbering scheme your data loader happened to use. **Permutation equivariance isn't a nicety added for elegance; it is the only way to stop the model learning the filing system instead of the file.** Everything else in this chapter — message passing, sum aggregation, the WL bound — falls out of taking that constraint seriously.

**How many orderings are we ruling out?** A graph on $n$ nodes has $n!$ possible numberings. For a modest 50-atom molecule that is $50! \approx 3\times10^{64}$ — more than the number of atoms in the Earth. A model that isn't permutation-equivariant would have to see a meaningful fraction of those relabellings to learn that they all mean the same thing. It never will. Building the symmetry in is not an optimization; it is the difference between possible and impossible.

#### Examples and non-examples: permutation-invariant functions

Every aggregation step in the rest of this chapter must be one of these, so it is worth being able to sort candidates by eye.

**✅  permutation-invariant** functions of a bag of vectors $\{v_1,\dots,v_k\}$

| Example | Why it qualifies |
|---|---|
| $\sum_j v_j$ | Addition is commutative — reorder the summands, same total |
| $\frac{1}{k}\sum_j v_j$ | The same, divided by a count that also does not depend on order |
| $\max_j v_j$, elementwise | The largest entry is the largest entry however you list them |
| $\sum_j \alpha_j v_j$ with $\alpha_j=\mathrm{softmax}_j(f(v_j))$ | Each weight is computed **from its own vector**, then summed — this is GAT's aggregation |
| $\log\sum_j e^{v_j}$ (log-sum-exp) | A smooth maximum; still a sum underneath |
| $\big(\sum_j v_j,\ \max_j v_j,\ \min_j v_j\big)$ concatenated | Concatenating *several invariant summaries* is fine — nothing is indexed by $j$ |

**❌ Near-misses — look order-agnostic, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| $[\,v_1 \Vert v_2 \Vert v_3\,]$ | $[v_1\Vert v_2]\neq[v_2\Vert v_1]$ — position inside the vector *is* the ordering | A set-to-sequence operation; it needs a canonical order you do not have |
| "Take the first neighbour" | Depends entirely on which one your data loader happened to list first | Reading the file system as though it were chemistry |
| Sort the neighbours by norm, then run an LSTM | Sorting is invariant only if the sort key breaks all ties — equal vectors leave the order undetermined | Invariant *in expectation* at best. GraphSAGE's LSTM aggregator handles this by shuffling every pass |
| Weights $\alpha_j$ read from a lookup table indexed by $j$ | The weight depends on the **label** $j$, not on $v_j$ | A per-position weight — exactly what a CNN may do and a GNN may not |
| Subtract $v_1$ from every other neighbour | Singles out one element by index | A choice of origin, which the numbering was not entitled to make |

▸ **The boundary:** a function is permutation-invariant when **every input reaches the output through the same channel** — no input can be reached *by its position*. The reliable test is a code test: write the operation as a loop accumulating into one running total, touching each element with identical code. If your loop needs `j` for anything other than fetching `v[j]`, it is not invariant.

**What would break, concretely.** Take a nitrogen atom with three hydrogen neighbours whose feature vectors are $h_a,h_b,h_c$. Sum gives $h_a+h_b+h_c$ no matter how the loader ordered them. Concatenation gives one of $3!=6$ different vectors depending on the ordering, so **the same ammonia molecule produces six different predictions** — and during training the model would spend its capacity trying to make those six agree, learning by brute force a fact you could have made true by construction.

> **Common misconception.** *"A permutation-invariant model can't do node-level prediction — if the output doesn't change when you shuffle the nodes, how can it say anything about a particular node?"* The invariance is demanded at the **graph** level; at the **node** level the requirement is equivariance, which is a different condition. A node classifier returns $n$ predictions; shuffle the input and you get the same $n$ predictions, shuffled the same way. Nothing about individual nodes is lost. The belief is tempting because "invariant" sounds like "blind to individuals" — but the only thing discarded is the arbitrary numbering: the labels on the filing cabinet, not the files.

> **Common misconception.** *"Graph neural networks learn the graph structure."* They are **given** the structure and learn a function on top of it. The adjacency matrix is an input, fixed before the forward pass, exactly as an image's pixel grid is fixed before a CNN runs. Inferring *which* edges ought to exist is a separate and much harder problem (latent graph inference, graph structure learning). The confusion is tempting because a trained GNN's predictions depend on structure in intricate, clearly learned ways — but the *dependence* is learned; the structure is handed over.

> **Where geometric deep learning came from.** The organizing idea — *"choose your architecture by first naming the symmetry group of your data"* — was named and systematized by **Michael Bronstein, Joan Bruna, Taco Cohen and Petar Veličković** in their 2021 monograph *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics and Gauges*, building on a 2017 survey by Bronstein, Bruna, Yann LeCun, Arthur Szlam and Pierre Vandergheynst. Its intellectual ancestor is **Felix Klein's Erlangen Program** of 1872, which reorganized all of geometry around the question "what transformations leave this structure alone?" — Euclidean geometry is the study of what survives rigid motions, projective geometry of what survives projections, and so on. Klein was 23 when he wrote it. The claim of geometric deep learning is that neural architectures admit exactly the same taxonomy: a CNN is what you get from the translation group, a GNN from the permutation group, an E(3)-network from the group of rigid motions in space. It is one of the rare cases in this book where a 150-year-old idea transferred with almost no modification.

### Notation

$G=(V,E)$, adjacency $A$, degree $D=\mathrm{diag}(d_i)$, node features $X\in\mathbb{R}^{n\times d}$, neighbourhood $\mathcal{N}(i)$.
**Laplacian:** $L = D-A$; normalized $L_{\text{sym}}=I-D^{-1/2}AD^{-1/2}$, with eigenvalues in $[0,2]$.

#### The graph vocabulary, with a worked example

Every symbol above becomes obvious on one tiny graph. Take a **path of three nodes**, $1 - 2 - 3$ (think: three atoms in a row).

**The adjacency matrix $A$** — a table with a $1$ wherever two nodes are joined:

$$A = \begin{pmatrix} 0&1&0 \\ 1&0&1 \\ 0&1&0\end{pmatrix}$$

Read row 2: "node 2 is joined to node 1 and node 3." The diagonal is zero because no node is joined to itself. The matrix is symmetric because the graph is undirected — friendship goes both ways.

**The degree $d_i$** — how many neighbours node $i$ has, which is simply that node's row sum. Here $d_1=1$, $d_2=2$, $d_3=1$. The two end nodes have one neighbour each; the middle node has two.

**The degree matrix $D$** — those numbers on the diagonal and nothing else:

$$D = \begin{pmatrix} 1&0&0 \\ 0&2&0 \\ 0&0&1\end{pmatrix}$$

**The Laplacian $L = D - A$** — subtract one from the other:

$$L = \begin{pmatrix} 1&-1&0 \\ -1&2&-1 \\ 0&-1&1\end{pmatrix}$$

▸ **What $L$ does is the whole point.** Multiply it by a vector $x$ of one number per node and look at entry $i$:

$$(Lx)_i = d_i x_i - \sum_{j\in\mathcal{N}(i)} x_j = \sum_{j \in \mathcal{N}(i)} (x_i - x_j)$$

**In English: "how much do I differ from my neighbours, summed up."** For our path with $x = (5, 5, 5)$ — everyone agrees — $Lx = (0,0,0)$. For $x = (10, 0, 10)$ — the ends disagree violently with the middle — $Lx = (10, -20, 10)$.

> **Analogy.** Heat in a metal rod. The Laplacian is the operator that says "heat flows from hot spots to cold spots at a rate proportional to the temperature difference." Every node compares itself with its neighbours and the imbalance is the driving force. A uniform temperature gives zero flow, which is exactly $Lx=0$ for constant $x$. **The graph Laplacian is the discrete version of the second-derivative operator $\nabla^2$ from physics**, which is where the name comes from — Pierre-Simon Laplace's operator, transplanted from continuous space onto a network.

**Why the "normalized" version exists.** In the raw Laplacian, a celebrity node with 10,000 followers produces enormous entries while a node with 2 neighbours produces tiny ones — the scale of the operator depends on how popular you are. $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$ divides out the degrees, so that entry $(i,j)$ carries a factor $1/\sqrt{d_i d_j}$ instead of a raw count.

▸ **The payoff is the guarantee "eigenvalues in $[0,2]$."** The eigenvalues of the raw $L$ can be as large as the maximum degree, which for a social network might be $10^5$. Normalizing pins the whole spectrum into a fixed window regardless of the graph. That matters because you are about to *stack* these operators, and §1.1.2's $\lambda^k$ argument says anything with an eigenvalue above 1 explodes with depth. **Bounded spectrum is the licence to build a deep model.**

- $\lambda = 0$ always occurs, with the all-ones vector as eigenvector — "everybody agrees" is always a zero-disagreement state. Its multiplicity counts the number of **connected components** of the graph, which is a  useful diagnostic.
- $\lambda = 2$ is only reached by bipartite graphs — a perfectly alternating pattern is the most "disagreeing" signal a graph can carry.

#### Examples and non-examples: what a Laplacian actually tells you

**✅  facts you can read out of $L$**

| Fact | How you get it | On the path $1-2-3$ |
|---|---|---|
| Node $i$'s degree | The diagonal entry $L_{ii}$ | $L_{22}=2$ — the middle node has two neighbours |
| Whether $i$ and $j$ are adjacent | $L_{ij}=-1$ if joined, $0$ if not | $L_{13}=0$: the two ends are not joined |
| Total disagreement of a signal | $x^\top L x=\sum_{(i,j)\in E}(x_i-x_j)^2$ | $x=(10,0,10)$ gives $100+100=200$ |
| The number of connected components | The multiplicity of eigenvalue $0$ | Multiplicity 1 — the path is connected |
| How bottlenecked the graph is | $\lambda_2$, the spectral gap (§29.5) | Small for a long chain, large for a clique |
| The number of spanning trees | Delete any row and column, take the determinant | $1$ — a path is its own only spanning tree |

**❌ Near-misses — things people expect $L$ to know, and it doesn't**

| Assumed | Why it fails | What it actually is |
|---|---|---|
| "$L$ encodes distances between nodes" | $L$ has one entry per **edge** — it is a one-hop object | Distances live in $L^k$ and in the *spectrum*, not in a single entry |
| "$L_{ij}=-1$ means $i$ and $j$ are far apart" | The minus sign is subtraction, not distance | $-1$ means **adjacent**; $0$ means not adjacent |
| "The biggest eigenvalue counts the nodes" | $\lambda_{\max}\le 2\max_i d_i$, which has nothing to do with $n$ | A bound on the sharpest oscillation the graph supports |
| "$L$ is invertible, so smoothing can be undone" | $L\mathbf{1}=\mathbf{0}$ always — $L$ is **singular** on every graph | The constant signal is in the null space, which is the mathematical form of over-smoothing (§29.5) |
| "$L_{\text{sym}}$ is $L$ with the numbers made prettier" | The rescaling changes which operator you iterate, and so changes the fixed point | A different operator with a bounded spectrum — the thing that makes depth legal |

▸ **The boundary:** $L$ is a strictly **local** operator — one number per edge — whose *powers and eigenvalues* encode global structure. Anything you can extract from a single multiplication by $L$ is a one-hop fact. Everything global costs you either a power of $L$ or an eigendecomposition.

**What would break, in numbers.** Take a social graph whose most-followed account has $d_{\max}=10^5$. The raw Laplacian then has eigenvalues up to about $2\times10^5$. Stack three layers of it and §1.1.2's $\lambda^k$ argument predicts amplification of roughly $(2\times10^5)^3 = 8\times10^{15}$ before a single weight has been trained — every activation overflows fp16 on layer one. Normalizing pins the whole spectrum into $[0,2]$ regardless of how popular anyone is. **The normalization is not cosmetic; it is the difference between a network that runs and one that returns `NaN`.**

**And a second thing that breaks.** $L_{\text{sym}}=I-D^{-1/2}AD^{-1/2}$ needs $d_i>0$ for every node, because $D^{-1/2}$ takes $1/\sqrt{d_i}$. An **isolated node** — a molecule fragment with no bonds, a user with no friends — has $d_i=0$ and the expression divides by zero. This is not hypothetical: it is one of the two or three most common runtime errors in graph libraries, and it is the second reason (after spectral stability) that the GCN's self-loop trick $\tilde A=A+I$ exists. With self-loops, $\tilde d_i \ge 1$ always.

> **Where the graph Laplacian came from.** **Gustav Kirchhoff** wrote this matrix down in 1847, while analysing electrical circuits — the same Kirchhoff of Kirchhoff's current and voltage laws. He proved what is now called the **matrix-tree theorem**: delete any one row and column from $L$, take the determinant, and you get the exact number of spanning trees of the network. He was counting current paths, not doing data analysis, and the object is still sometimes called the Kirchhoff matrix. Graph theory itself begins a century earlier with **Leonhard Euler's** 1736 paper on the **Seven Bridges of Königsberg**, in which he proved you cannot walk a route crossing each of the city's seven bridges exactly once. Euler's insight was that the shapes and distances were irrelevant — only *what connects to what* mattered. That deliberate throwing-away of geometry is the founding act of the whole field, and it is precisely why a GNN is allowed to ignore how you drew the picture.

---

## 29.2 From spectral to spatial

### Spectral convolution

The graph Fourier transform uses the Laplacian's eigenvectors: $L=U\Lambda U^\top$, $\hat x = U^\top x$. Convolution becomes multiplication in the spectral domain:
$$g_\theta \star x = U\,g_\theta(\Lambda)\,U^\top x$$

**Problems:** $O(n^3)$ eigendecomposition, $O(n^2)$ multiplication, filters are not localized, and the eigenbasis is graph-specific so nothing transfers between graphs.

#### What spectral convolution actually says

This is the one  abstract passage in the chapter, and it rewards being slowed down.

**Start with the fact it is imitating.** For ordinary signals, the **convolution theorem** says: convolving two signals in the time domain is the same as *multiplying* them in the frequency domain. That is why audio engineers reach for the Fourier transform — a messy smearing operation becomes plain multiplication once you change coordinates.

The recipe is always three steps: **transform → multiply → transform back.**

**Now the graph version, term by term:**

| Piece | Read aloud | What it does |
|---|---|---|
| $L = U\Lambda U^\top$ | "eigendecompose the Laplacian" | Find the graph's natural vibration patterns (§1.1.2) |
| $U^\top x$ | "U transpose x" | **Forward transform.** Rewrite the node signal in terms of those patterns |
| $g_\theta(\Lambda)$ | "g-theta of Lambda" | **The filter.** A learned number multiplying each pattern — turn this frequency up, that one down |
| $U$ | "U" | **Inverse transform.** Convert back to per-node values |
| $\star$ | "convolved with" | The operation being defined |

**What are the "frequencies" of a graph?** The Laplacian's eigenvectors, sorted by eigenvalue. Recall from the worked example above that $x^\top L x = \sum_{(i,j)\in E}(x_i-x_j)^2$ measures total disagreement across edges. So:

- **Small $\lambda$** → eigenvectors that vary *slowly* across the graph. Neighbours hold similar values. **Low frequency = smooth.**
- **Large $\lambda$** → eigenvectors that flip sign between adjacent nodes. **High frequency = jagged.**

> **Analogy.** A drum skin. Strike it and it vibrates in a superposition of characteristic modes — a slow whole-membrane wobble, then progressively more intricate patterns with more nodal lines. Those modes are the eigenvectors of the continuous Laplacian; the graph Laplacian's eigenvectors are the same idea for a network instead of a membrane. A graph "Fourier transform" asks: **how much of each vibration mode is present in this signal?**

**Now read the four problems, and feel each one.**

- **$O(n^3)$ eigendecomposition.** For a social graph with $n = 10^6$ nodes, that is $10^{18}$ operations — days on a fast machine. Worse, $U$ itself is $n\times n$: $10^{12}$ numbers, **4 terabytes in fp32.** You cannot even store the answer.
- **$O(n^2)$ multiplication** per filter application, per layer. Also hopeless at scale.
- **Filters are not localized.** $g_\theta(\Lambda)$ is free to do anything to any frequency, and a generic frequency-domain filter is *global* in node space — one node's value can influence a node on the far side of the graph. That throws away the locality prior that makes convolution work on images in the first place.
- **The eigenbasis is graph-specific.** $U$ is computed from *this* graph. Train a filter on caffeine and it is expressed in caffeine's vibration modes, which are meaningless for aspirin. **Nothing transfers.** For molecular machine learning, where every training example is a different graph, this is fatal on its own.

▸ **The whole rest of the chapter is the retreat from this formula.** ChebNet fixes localization and cost; GCN fixes everything but at the price of expressivity; message passing abandons the spectral picture entirely. The spectral derivation is worth knowing because it explains *why* the surviving formula has the shape it does — but nobody computes a graph Fourier transform in production.

### ChebNet

Approximate $g_\theta(\Lambda)$ by a degree-$K$ Chebyshev polynomial. Because $L^k$ has support exactly on the $k$-hop neighbourhood, ▸ **a degree-$K$ polynomial filter is automatically $K$-hop localized**, and it needs no eigendecomposition — just $K$ sparse matrix–vector products.

#### Unpacking the polynomial trick

Two separate ideas are doing the work here. Take them one at a time.

**Idea 1 — a polynomial in $L$ is automatically local.**

The key observation is about what powers of a matrix mean on a graph:

▸ $(A^k)_{ij}$ **counts the number of walks of length exactly $k$ from node $i$ to node $j$.**

Check it on the path $1-2-3$ from above. $A^2$ has $(A^2)_{13} = 1$: there is exactly one 2-step walk from node 1 to node 3, via node 2. And $(A^2)_{11}=1$: you can walk to node 2 and back. But $(A^3)_{14}$ doesn't exist because there's no node 4 — and in general, **if $j$ is more than $k$ hops from $i$, then $(A^k)_{ij}=0$, because no walk that short can get there.**

The same is true of $L$, which is built from $A$. So:

| Filter | Reaches | Cost |
|---|---|---|
| $c_0 I$ | the node itself | free |
| $c_0 I + c_1 L$ | 1 hop | one sparse multiply |
| $c_0I + c_1L + \dots + c_KL^K$ | exactly $K$ hops, no further | $K$ sparse multiplies |

▸ **So "restrict the filter to a polynomial of degree $K$" and "make the filter $K$-hop local" are the same instruction.** You do not have to add locality as a constraint — you get it for free by choosing the function class. This is the single most elegant step in the chapter.

**Idea 2 — you never need $U$ again.** A filter $g_\theta(\Lambda)$ that is a polynomial can be moved outside the transform entirely:

$$U\,g_\theta(\Lambda)\,U^\top = U\left(\sum_k c_k \Lambda^k\right)U^\top = \sum_k c_k\, U\Lambda^k U^\top = \sum_k c_k L^k$$

The middle step uses the same cancellation as §1.1.2: $U\Lambda^kU^\top = (U\Lambda U^\top)^k = L^k$, because the interior $U^\top U = I$ pairs annihilate. **The eigendecomposition vanishes from the formula.** You went in needing a $4\text{-terabyte}$ matrix and came out needing $K$ multiplications by a sparse matrix that has one entry per edge.

**Concrete cost.** A graph with $n=10^6$ nodes and $10^7$ edges, filter width $K=3$: three sparse multiplies at $10^7$ operations each is $3\times10^7$ operations — versus $10^{18}$ for the eigendecomposition. That is a factor of **thirty billion**.

**Why *Chebyshev* polynomials specifically, rather than just $1, L, L^2, \dots$?** The naive power basis is numerically appalling: $L^k$ for growing $k$ points more and more in the same direction (that is exactly what power iteration exploits, §1.1.2), so the coefficients $c_k$ become wildly ill-conditioned. Chebyshev polynomials are the family that stays maximally spread out over an interval — they minimize the worst-case deviation from zero — and they satisfy a cheap recurrence $T_k(x) = 2xT_{k-1}(x)-T_{k-2}(x)$, so you compute each term from the previous two with one multiply. Same function class, far better arithmetic.

> **Where Chebyshev polynomials came from.** **Pafnuty Chebyshev** introduced them in the 1850s while working on a problem in mechanical engineering: he was studying **linkages** — the jointed rod mechanisms that convert a steam engine's rotary motion into straight-line motion — and wanted to know how close to a perfect straight line such a mechanism could get. That question is "what polynomial of given degree deviates least from a target, in the worst case?", and its answer is the polynomial family that bears his name. Chebyshev's name is transliterated from Cyrillic in a famously large number of ways — Tchebychev, Tschebyscheff, Chebyshov, and many more — which is why the polynomials are universally written $T_k$ rather than $C_k$: the $T$ comes from the French spelling *Tchebychef*. A notation quirk in modern graph learning is a fossil of 19th-century transliteration practice.

### GCN — the first-order simplification

Take $K=1$, set $\lambda_{\max}\approx2$, tie the two coefficients, and add a **renormalization trick** ($\tilde A = A+I$, $\tilde D$ its degree matrix) to keep the spectrum stable under stacking:

▸ $$\boxed{\ H^{(l+1)} = \sigma\!\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\right)\ }$$

▸ **Read it as: each node averages its neighbours' features (plus its own, via the self-loop), with degree normalization, then applies a shared linear map and a nonlinearity.** The spectral derivation motivates the formula, but the formula itself is purely local — which is why the whole field moved to the spatial view.

#### The GCN layer, with real numbers in it

This one boxed formula is the most-implemented line in all of graph learning, so let us build it from parts and then run it.

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $H^{(l)}$ | "H at layer l" | The whole feature table at layer $l$: $n$ rows (one per node), $d_l$ columns |
| $W^{(l)}$ | "W at layer l" | A learned $d_l \times d_{l+1}$ matrix, **shared by every node** |
| $\tilde A = A+I$ | "A-tilde" | Adjacency plus self-loops — "count yourself as your own neighbour" |
| $\tilde D$ | "D-tilde" | Degrees computed from $\tilde A$, so $\tilde d_i = d_i + 1$ |
| $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ | "the propagation matrix" | The mixing recipe: how much of each neighbour each node absorbs |
| $\sigma$ | "sigma" | The activation function — here it is *not* a standard deviation or a singular value (§0.4) |

▸ **The reason $W$ is shared across all nodes is the permutation requirement from §29.1.** A per-node weight matrix would be a weight indexed by a node's *label*, which is exactly the arbitrary thing we are forbidden from reading. Weight sharing across nodes is the graph analogue of a CNN sharing one kernel across all pixel positions, and it is forced by the symmetry for exactly the same reason.

**Now run it on the path $1-2-3$.** Add self-loops: $\tilde A$ has ones on the diagonal too, so $\tilde d_1 = 2$, $\tilde d_2 = 3$, $\tilde d_3 = 2$. The propagation matrix $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ has entries $\hat A_{ij} = 1/\sqrt{\tilde d_i \tilde d_j}$ wherever $\tilde A_{ij}=1$:

$$\hat A = \begin{pmatrix} 1/2 & 1/\sqrt6 & 0 \\ 1/\sqrt6 & 1/3 & 1/\sqrt6 \\ 0 & 1/\sqrt6 & 1/2 \end{pmatrix} \approx \begin{pmatrix} 0.500 & 0.408 & 0 \\ 0.408 & 0.333 & 0.408 \\ 0 & 0.408 & 0.500\end{pmatrix}$$

So node 2's new feature, before the weights and nonlinearity, is

$$0.408\,h_1 \;+\; 0.333\,h_2 \;+\; 0.408\,h_3$$

**Read that as a sentence: "take 41% of each neighbour and 33% of yourself, and add them up."** No learned parameters are involved in the mixing at all — the mixing weights come entirely from the graph's shape. All the learning lives in $W^{(l)}$, which is applied identically to every node afterwards.

**Why $1/\sqrt{d_i d_j}$ rather than $1/d_i$?** The symmetric form treats the edge, not the node, as the unit. A message crossing an edge is damped by *both* endpoints' popularity — half a discount from the sender, half from the receiver. The consequence is that $\hat A$ stays **symmetric**, which in turn guarantees real eigenvalues and an orthogonal eigenbasis (§1.1.2), which is what makes the depth analysis in §29.5 possible at all. Note it is *not* row-stochastic: row 2 above sums to $1.149$, not $1$. The design chose symmetry over exact averaging.

▸ **What the renormalization trick actually buys.** Without self-loops, the eigenvalues of the normalized operator live in $[-1, 1]$, and the $-1$ end is a disaster: an eigenvalue near $-1$ raised to the power of the depth *oscillates* rather than decaying, so features flip sign every layer and training destabilizes. Adding $I$ pulls the spectrum away from that boundary — the negative end contracts toward zero — so stacking layers damps signals smoothly instead of ringing. **Kipf and Welling reported this as an empirical trick, and it is the difference between a GCN that trains and one that does not.**

**A tiny numerical sanity check on the self-loop.** Suppose node 2 is a carbon atom whose own feature says "I am carbon" and whose neighbours are both hydrogens. Without the self-loop, node 2's update would be $0.5h_1 + 0.5h_3$ — **pure hydrogen, with the carbon-ness erased.** A node that only listens to its neighbours forgets itself entirely after one layer. The self-loop is not bookkeeping; it is the mechanism by which a node's own identity survives the round.

> **Common misconception.** *"The GCN learns how much attention to pay to each neighbour."* It learns nothing of the sort. Every mixing coefficient in $\hat A$ is $1/\sqrt{\tilde d_i\tilde d_j}$ — determined entirely by the graph's shape and computed once, before training starts. The only learned object in the layer is $W^{(l)}$, which is applied **identically to every node after the mixing**. The misconception is tempting because the layer is written as one matrix product and everything in a neural network is usually learned; here, half the product is a fixed constant. The architecture that *does* learn neighbour weights is GAT, and the difference between the two is precisely this point.

> **Common misconception.** *"The GCN's propagation matrix is a weighted average, so each row sums to 1."* It does not. Row 2 of the worked example sums to $0.408+0.333+0.408=1.149$. The symmetric normalization $1/\sqrt{\tilde d_i \tilde d_j}$ was chosen so that $\hat A$ stays **symmetric** — which buys real eigenvalues and an orthogonal eigenbasis, and is what makes the over-smoothing analysis in §29.5 possible at all. The row-stochastic alternative $\tilde D^{-1}\tilde A$ is a  average and is what GraphSAGE's mean aggregator uses; it is a different design with a different spectrum. **"Averaging" is the right intuition and the wrong arithmetic.**

#### Examples and non-examples: what one GCN layer can and cannot represent

**✅ Functions a single GCN layer  computes**

| Example | How |
|---|---|
| "Predict this atom's type from its neighbours' types" | One round of neighbour mixing, then $W$ |
| "Smooth a noisy per-node signal" | The mixing step *is* Laplacian smoothing — this is its original use |
| "Detect a node whose neighbourhood differs from itself" | The self-loop keeps $h_i$ available alongside the neighbour average |
| "Project 1,433 bag-of-words features down to 16" | $W^{(l)}$ is an ordinary dense layer applied per node |

**❌ Near-misses — things a single GCN layer is often assumed to do**

| Assumed | Why it can't | What you would need |
|---|---|---|
| Count how many neighbours a node has | Degree normalization divides the count back out | Sum aggregation (GIN), or degree as an explicit input feature |
| Treat an oxygen neighbour differently from a hydrogen neighbour | The mixing weight depends only on degrees, never on features | Attention (GAT), or edge features in the message (MPNN/GINE) |
| Use the bond type | $A$ is binary; bond order never enters | An edge-featured message function |
| See a node two hops away | One layer is one hop, by construction | A second layer |
| Distinguish node 1 from node 2 in a symmetric graph | Both are structurally identical and start with the same features | Positional or structural encodings (§29.4) |

▸ **The boundary:** a GCN layer computes exactly **one degree-normalized weighted sum over a one-hop neighbourhood, followed by a shared linear map.** Anything requiring a comparison *between* neighbours, a count, or a reach beyond one hop is outside a single layer — and knowing which of those you need is how you choose between GCN, GAT, GIN and MPNN.

> **Where the GCN came from.** **Thomas Kipf and Max Welling** at the University of Amsterdam posted *Semi-Supervised Classification with Graph Convolutional Networks* in late 2016 (ICLR 2017). Its lineage runs backwards through **Michaël Defferrard, Xavier Bresson and Pierre Vandergheynst's** ChebNet (2016) to **Joan Bruna, Wojciech Zaremba, Arthur Szlam and Yann LeCun's** *Spectral Networks* (2014), the first attempt to define convolution on a graph via the Laplacian's eigenbasis. What made the GCN paper land was not the derivation but the **simplification**: three layers of approximation reduce a formidable spectral construction to one line that fits in a tweet and beats everything on the standard citation-network benchmarks. Kipf's accompanying blog post and the accompanying figures did as much for adoption as the paper. An older lineage exists too — **Marco Gori, Gabriele Monfardini and Franco Scarselli** introduced the name "graph neural network" in 2005 and formalized it in 2009, using a contraction-mapping fixed point rather than a fixed stack of layers: their model ran message passing to convergence via the Banach fixed-point theorem. It was ahead of its hardware, and largely uncited until the field caught up a decade later.

---

## 29.3 Message passing

▸ The general framework subsuming nearly every GNN:

$$m_{ij}^{(l)} = \phi\big(h_i^{(l)},\,h_j^{(l)},\,e_{ij}\big)$$
$$a_i^{(l)} = \bigoplus_{j\in\mathcal{N}(i)} m_{ij}^{(l)} \qquad\text{(a permutation-invariant aggregator)}$$
$$h_i^{(l+1)} = \psi\big(h_i^{(l)},\,a_i^{(l)}\big)$$

**Aggregate → Update.** $\bigoplus$ must be permutation-invariant: sum, mean, max, or attention-weighted sum. Everything else is a design choice.

▸ **$L$ layers means each node sees its $L$-hop neighbourhood** — the exact analogue of a CNN's receptive field (Ch. 8 §8.1).

#### Reading the three message-passing equations

These three lines subsume essentially every GNN ever published, so it is worth being able to recite them as a story rather than as algebra.

**The story: a village of gossips.** Every node is a person holding an opinion (its feature vector $h_i$). Each round has three beats:

1. **Compose.** Every person writes a note to each neighbour. The note depends on what they think, what the neighbour thinks, and the nature of their relationship.
2. **Collect.** Every person gathers the pile of notes they received — *as an unordered pile*, because the order the postman delivered them is not information.
3. **Update.** Every person revises their opinion based on the pile plus what they already believed.

Repeat $L$ times. That is a graph neural network, completely.

**Now the symbols, line by line.**

**Line 1 — the message.** $m_{ij}^{(l)} = \phi(h_i^{(l)}, h_j^{(l)}, e_{ij})$

- $\phi$ (phi) is a small learned network — usually an MLP (multi-layer perceptron). *"How do I turn a pair of opinions into a note?"*
- $e_{ij}$ is the **edge feature**: what kind of relationship this is. For a molecule it is the bond type (single, double, aromatic). For a citation graph there may be none, and the term is dropped.
- Note the subscript order: $m_{ij}$ is the message **from $j$ to $i$**, which is the opposite of what most people guess on first reading. The convention matches $A_{ij}$ — first index is the receiver.

**Line 2 — the aggregation.** $a_i^{(l)} = \bigoplus_{j\in\mathcal{N}(i)} m_{ij}^{(l)}$

- $\bigoplus$ is a *placeholder* for whichever pooling operation you chose. It is written as a big generic operator precisely because the framework doesn't care which one you pick — only that it satisfies the constraint.
- $\mathcal{N}(i)$ is the neighbour set, so this is a loop over "everyone wired to me."

▸ **This single line is where permutation invariance is enforced, and it is the only place it needs to be.** $\phi$ and $\psi$ can be arbitrarily complicated networks and the whole model stays equivariant, provided $\bigoplus$ ignores order. Sum, mean, and max all do. **Concatenation does not** — `[m1, m2]` and `[m2, m1]` are different vectors, so a concatenating aggregator would read the node numbering as data. That is the one design rule you cannot break.

**Line 3 — the update.** $h_i^{(l+1)} = \psi(h_i^{(l)}, a_i^{(l)})$

- $\psi$ (psi) is another small learned network. *"How do I fold the incoming pile into what I already believed?"*
- $h_i^{(l)}$ appears on both sides, which is what keeps a node's own identity alive across rounds — the same job the self-loop did in the GCN.

**Check the GCN against this template.** $\phi(h_i,h_j) = \frac{1}{\sqrt{\tilde d_i\tilde d_j}}h_j$ (the message is just a scaled copy of the neighbour), $\bigoplus$ is sum, and $\psi(h_i,a_i) = \sigma((a_i + \text{self term})W)$. **The GCN is the simplest possible instance of the framework** — which is exactly why the field reorganized around message passing rather than around the spectral derivation.

**Now the receptive-field claim, with numbers.** After one layer, node $i$'s vector depends on $i$ and its immediate neighbours. After two, on everything within two hops, because its neighbours' vectors already absorbed *their* neighbours. In general:

| Layers $L$ | Node $i$ sees | If each hop reaches ~4 new nodes |
|---|---|---|
| 1 | 1-hop neighbours | ~5 nodes |
| 2 | 2-hop | ~21 nodes |
| 3 | 3-hop | ~85 nodes |
| 6 | 6-hop | ~5,461 nodes |

▸ **The counts grow like $4^L$ — exponentially — whereas a CNN's receptive field grows like $L$ times the kernel width, which is *linear*.** Hold on to this asymmetry: it is the root cause of both pathologies in §29.5. A CNN can be 100 layers deep because its receptive field creeps outward politely. A GNN's receptive field swallows the entire graph in a handful of layers, and everything it swallows must fit into one fixed-size vector.

#### Examples and non-examples: is this thing a message-passing GNN?

The three equations are broad enough that the interesting question is what falls *outside* them.

**✅  instances of the framework**

| Example | $\phi$ (message) | $\bigoplus$ (aggregate) | $\psi$ (update) |
|---|---|---|---|
| GCN | scale the neighbour by $1/\sqrt{\tilde d_i\tilde d_j}$ | sum | linear map, then $\sigma$ |
| GAT | scale the neighbour by a learned $\alpha_{ij}$ | sum | linear map, then $\sigma$ |
| GIN | copy the neighbour unchanged | sum | MLP of $(1+\epsilon)h_i + a_i$ |
| A transformer layer | value vector times attention weight | sum over **all** tokens | the feed-forward block |
| A CNN on an image, viewed abstractly | multiply the neighbour pixel by its kernel weight | sum over the $3\times3$ window | activation |
| Belief propagation on a factor graph | the classical message update | product (or sum in log-space) | normalize |

Note the CNN row carefully: a CNN **is** message passing on a grid, with the crucial addition that the message function is allowed to depend on *which* neighbour it is. That extra dependence is exactly the permutation symmetry a GNN gives up.

**❌ Near-misses — graph methods that are not message passing**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| node2vec / DeepWalk embeddings | There is no message function and no update — you look up a stored vector per node | A **shallow embedding table** trained by a random-walk objective; nothing generalizes to a new node |
| Full spectral convolution, $U g_\theta(\Lambda)U^\top x$ | A generic frequency filter is global; there is no local neighbourhood step | A graph Fourier filter, which is why §29.2 is a story of retreat |
| PageRank | Fixed update rule, no learned $\phi$ or $\psi$ | A **fixed-point iteration**; message passing with the learning removed |
| Feeding $A$ into an MLP row by row | Reads node indices as features; not permutation-equivariant at all | An architecture that will learn your data loader |
| ECFP fingerprint plus gradient boosting | The fingerprint step is WL colour refinement, but nothing is learned inside it | A **fixed featurizer** plus a tabular model — and often a very strong baseline (§29.9) |
| DeepSets over a bag of atoms | Correct framework, but the edge set is empty | The $\mathcal{N}(i)=\emptyset$ special case: a per-element MLP followed by a pool |

▸ **The boundary:** it is message passing when there is a **learned function of (my state, the unordered bag of my neighbours' states)** applied repeatedly with shared weights. Drop "learned" and you have PageRank. Drop "unordered" and you have a CNN. Drop the neighbours and you have DeepSets. Drop the repetition and you have a featurizer.

> **Common misconception.** *"More layers means a bigger receptive field, so deeper GNNs see more and should do better."* Both halves of the premise are true and the conclusion still fails. The receptive field does grow — exponentially — but the vector holding it does not grow at all, and the repeated averaging that does the growing is simultaneously erasing the distinctions you needed. **Depth in a GNN buys reach and spends resolution**, which is why the two pathologies in §29.5 pull in opposite directions and why most deployed GNNs are two or three layers deep. The intuition is imported wholesale from CNNs and transformers, where depth  is close to free.

> **Where message passing came from.** The unification was published by **Justin Gilmer, Samuel Schoenholz, Patrick Riley, Oriol Vinyals and George Dahl** at Google Brain in 2017, in *Neural Message Passing for Quantum Chemistry*. Their contribution was largely taxonomic: they showed that eight or so architectures which had been published as separate inventions — spectral networks, ChebNet, GCN, the 2015 molecular fingerprint network of **David Duvenaud** and colleagues, and others — were all the same three equations with different choices of $\phi$, $\bigoplus$ and $\psi$. A paper whose main result is "these are all the same thing" rarely gets attention; this one reorganized a field. It also let the authors search the design space systematically rather than by invention, which is how they got state-of-the-art results on QM9 in the same paper.

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

#### The architecture table, decoded

Six rows, six  different bets about what matters. Here is each one in plain language, with its name spelled out.

**GCN — Graph Convolutional Network.** Degree-normalized mean, as worked through above. *Transductive* means it can only score nodes that were present in the graph at training time: the model learned an embedding for *this* graph, and a new node arriving tomorrow has no representation. Fine for a fixed citation network, useless for a social network that grows hourly.

**GraphSAGE — Graph SAmple and aggreGatE.** Two changes, both decisive.

- **Sampling.** Instead of aggregating over *all* neighbours, draw a fixed number — say 25 at layer 1 and 10 at layer 2. This caps the work per node at $25\times10 = 250$ regardless of whether a node has three neighbours or three million. Without it, a single celebrity node in a social graph makes one training step unbounded.
- **Concatenate rather than merge.** $h_i^{(l+1)} = \sigma\big(W\cdot[\,h_i^{(l)} \,\Vert\, a_i^{(l)}\,]\big)$, where $\Vert$ means "stick the two vectors end to end." Keeping self and neighbourhood in separate slots means the model can weigh them differently, rather than having them pre-blended.

▸ **This is what makes it *inductive*: it learns a function of a neighbourhood, not a table of node embeddings.** Show it a node it has never seen and it computes an embedding on the spot. That is the difference between a research artifact and something you can run at Pinterest.

**GAT — Graph ATtention network.** Decode the formula:

$$\alpha_{ij}=\mathrm{softmax}_j\big(\mathrm{LeakyReLU}(a^\top[Wh_i \,\Vert\, Wh_j])\big)$$

Read right to left: project both nodes with the shared $W$; glue the two projections together; squash the pair down to a *single number* with the learned vector $a$; pass it through LeakyReLU; then softmax across all of node $i$'s neighbours so the weights sum to one. The result $\alpha_{ij}$ is *"the fraction of node $i$'s attention that goes to node $j$."*

**"Anisotropic" is the word to remember.** It means "not the same in all directions." A GCN is *isotropic*: every neighbour gets a weight fixed by the graph's shape, and the model has no say. GAT lets the model decide, per pair, per head. For a molecule that means a carbon can learn to attend hard to an attached oxygen and barely at all to a hydrogen — a chemically sensible asymmetry that a GCN cannot express.

**GIN — Graph Isomorphism Network.** Sum, then an MLP. The next section is entirely about why those two choices are exactly the right ones.

**MPNN / GINE — Message Passing Neural Network / GIN with Edge features.** Puts $e_{ij}$ into the message, which for chemistry is not optional. Benzene and cyclohexane are the *same graph* — six carbons in a ring — and differ only in their bond types. A model that ignores edge features cannot tell them apart, and they behave completely differently.

**PNA — Principal Neighbourhood Aggregation.** Runs several aggregators at once (mean, max, min, standard deviation) and concatenates the results, then rescales by functions of the degree. The motivating theorem is that **no single aggregator can distinguish all multisets of neighbour features** when features are continuous — you need multiple views. The "degree scalers" exist because sum-like aggregators grow with degree and mean-like ones do not, so the network otherwise has no clean way to reason about how many neighbours it had.

> **Common misconception.** *"GAT's attention weights tell you which neighbours matter."* They tell you which neighbours the model **routed information through**, which is a weaker and different claim. A neighbour can receive a large $\alpha_{ij}$ and carry a value vector that barely moves the output; a neighbour with a small weight can be the one whose absence flips the prediction. This is the same trap as reading transformer attention maps as explanations (Ch. 32), and it has the same source: attention weights are visible, legible, and sum to one, which makes them *feel* like an explanation. **The honest test of whether a neighbour matters is to remove it and re-run the model**, not to read a coefficient.

> **Common misconception.** *"GraphSAGE is 'GCN with sampling.'"* Sampling is the famous part, but the load-bearing change is **inductivity** — learning a function of a neighbourhood rather than a table of per-node embeddings. A transductive model must be retrained when a node arrives; an inductive one embeds the newcomer on the spot. That distinction is what let graph learning move from fixed academic citation networks to social platforms where a hundred thousand nodes appear per hour.

> **Where GraphSAGE and GAT came from.** **William L. Hamilton, Rex Ying and Jure Leskovec** at Stanford introduced GraphSAGE — *Inductive Representation Learning on Large Graphs* — at NeurIPS 2017; the acronym is *SAmple and aggreGatE*, and the paper's target was explicitly the setting the academic benchmarks were hiding, where the graph changes faster than you can retrain. The same group deployed it at Pinterest as **PinSage** (KDD 2018, with Ruining He, Kaifeng Chen, Pong Eksombatchai and Chantat Eksombatchai) on a graph of roughly three billion nodes — at the time, by a wide margin, the largest deep architecture ever run on relational data. **GAT** — *Graph Attention Networks* — came from **Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò and Yoshua Bengio** and appeared at ICLR 2018, with Veličković then a PhD student at Cambridge. Its arrival sits almost exactly between the transformer paper (mid-2017) and the field's realization that the two mechanisms were the same one; Veličković later co-authored the geometric deep learning monograph that made the connection explicit.

#### Why "a transformer is a GNN on a complete graph" is exactly true

Line up the definitions and they coincide:

| Transformer | Graph attention network |
|---|---|
| Tokens | Nodes |
| Every token attends to every other | The graph is **complete** — every node adjacent to every node |
| $\mathrm{softmax}(q_i^\top k_j/\sqrt{d})$ | $\alpha_{ij}$, an attention weight over neighbours |
| Weighted sum of values | Attention-weighted aggregation, $\bigoplus$ |
| The feed-forward block | The update function $\psi$ |
| Positional encodings | Structural encodings (§29.7) |

▸ **Self-attention *is* the aggregation step of a message-passing layer, run on the graph where everything is connected to everything.** And the reason a transformer needs positional encodings is now visible from the graph side: **attention is permutation-invariant by construction, so a bare transformer cannot tell "the cat sat" from "sat cat the."** Positional encodings are how you smuggle the ordering back in after the architecture has thrown it away. Chapter 12's whole subject is that smuggling operation.

The reverse framing is just as useful: a GNN is a transformer whose attention pattern has been **masked by the adjacency matrix.** Sparse attention and message passing are the same idea, discovered from opposite directions.

---

## 29.4 Expressivity and the Weisfeiler–Lehman bound

### The 1-WL test

An algorithm for graph isomorphism testing: initialize each node's colour by its degree; repeatedly set each node's new colour to a hash of (its colour, the multiset of its neighbours' colours); if two graphs' colour histograms ever differ, they are not isomorphic.

#### The 1-WL test, run by hand

**First, the vocabulary.**

- **Isomorphic** — "the same graph, relabelled." Two graphs are isomorphic if you can renumber one to get the other exactly. This is precisely the $PAP^\top$ relation from §29.1.
- **Multiset** — a bag that allows duplicates but has no order. $\{a, a, b\}$ is a multiset; it is different from $\{a,b\}$ (duplicates count) but the same as $\{a,b,a\}$ (order doesn't).
- **Hash** — any function that maps distinct inputs to distinct outputs. Here it just means "give this combination a fresh, unique colour name."
- **1-WL** — the "1" means colours are attached to *single nodes*. The $k$-WL variants attach colours to $k$-tuples of nodes and are strictly more powerful.

**Now run it.** Take a triangle with a tail: nodes $\{1,2,3\}$ form a triangle, and node 4 hangs off node 3.

**Round 0 — colour by degree.** $d_1=2, d_2=2, d_3=3, d_4=1$.

| Node | Colour |
|---|---|
| 1 | **A** (degree 2) |
| 2 | **A** (degree 2) |
| 3 | **B** (degree 3) |
| 4 | **C** (degree 1) |

**Round 1 — each node's new colour is (my colour, the bag of my neighbours' colours).**

| Node | (own colour, bag of neighbour colours) | New colour |
|---|---|---|
| 1 | $(\text{A},\ \{\text{A},\text{B}\})$ | **D** |
| 2 | $(\text{A},\ \{\text{A},\text{B}\})$ | **D** |
| 3 | $(\text{B},\ \{\text{A},\text{A},\text{C}\})$ | **E** |
| 4 | $(\text{C},\ \{\text{B}\})$ | **F** |

Nodes 1 and 2 stayed identical — and they  *are* interchangeable in this graph, so that is correct behaviour, not a failure. **The refinement has stabilized**: no further round will split anything, so the algorithm halts.

**The final histogram** is $\{\text{D}{:}2,\ \text{E}{:}1,\ \text{F}{:}1\}$. Compare two graphs by comparing histograms.

▸ **The critical asymmetry: WL can prove two graphs are *different*, but never that they are the *same*.** Different histograms $\Rightarrow$ definitely not isomorphic. Identical histograms $\Rightarrow$ *maybe* isomorphic, and maybe not. It is a one-sided test, like a smoke alarm: silence is not proof of no fire.

> **Analogy.** Two identical-looking suitcases. You weigh them — same. Shake them — same rattle. X-ray them — same silhouette. Every test you can think of agrees, and you still cannot be certain they hold the same objects. WL is that battery of tests: cheap, powerful, and provably incomplete.

**Why the graph community cares about a 1968 isomorphism heuristic at all** is that it turns out to be *exactly* as powerful as message passing — no more, no less. That coincidence is the theorem below, and it is the reason a paper about colouring graphs is the tightest known ceiling on a whole class of neural architectures.

> **Where the WL test came from.** **Boris Weisfeiler and Andrei Leman** published the algorithm in 1968 in a short Russian paper, *A reduction of a graph to a canonical form and an algebra arising during this reduction*. They were not testing isomorphism for its own sake; they were building an algebraic object (now called a coherent configuration, or Weisfeiler–Leman algebra) and the colour refinement was the construction step. Leman has publicly noted that the standard English rendering "Lehman" is a mistranslation and that his name is properly transliterated **Leman** — you will see both spellings in the literature, and they refer to the same person. Boris Weisfeiler emigrated to the United States and became a professor at Penn State; in January 1985 he **disappeared while hiking alone in southern Chile**, and later Chilean judicial investigations connected the case to the German expatriate settlement Colonia Dignidad and to the Pinochet-era security services. The case has never been fully resolved. The 1-WL test is also known to be **incomplete**: in 1992, **Jin-Yi Cai, Martin Fürer and Neil Immerman** constructed pairs of graphs that $k$-WL cannot distinguish for any fixed $k$, closing off the hope that simply raising $k$ would solve graph isomorphism. That result is exactly why "go to higher-order GNNs" is a treadmill rather than a road.

### The theorem

▸ **Message-passing GNNs are at most as powerful as the 1-WL test at distinguishing non-isomorphic graphs** (Xu et al., 2019; Morris et al., 2019).

**Why:** a message-passing layer computes exactly a function of (node feature, multiset of neighbour features) — structurally identical to a WL refinement round. It can never distinguish two graphs that WL cannot.

▸ **The concrete consequence to remember: a message-passing GNN cannot count triangles, cannot detect cycles of a given length, and cannot distinguish two 3-regular graphs on the same number of nodes.** In a 3-regular graph every node looks identical to WL forever, so a GNN assigns them all the same representation regardless of global structure. For chemistry — where rings determine everything — this is not a theoretical curiosity.

#### What the WL bound actually forbids — the canonical counterexample

The proof sketch in one sentence: **a message-passing layer takes in (my current vector, the unordered bag of my neighbours' vectors) and produces a new vector — which is character-for-character the same input–output signature as a WL refinement round.** Two things computing from identical information cannot reach different conclusions. Anything WL cannot see, message passing cannot see. That is the entire argument, and its simplicity is why the bound is airtight.

**Now the counterexample you should be able to draw on a napkin.** Take two graphs, each with **6 nodes, 9 edges, every node of degree exactly 3**:

- **The triangular prism** — two triangles joined by three rungs, like a Toblerone box.
- **$K_{3,3}$** — three nodes on the left, three on the right, every left node joined to every right node (the "three utilities" graph).

They are unmistakably different objects. The prism is stuffed with triangles; $K_{3,3}$ is **bipartite** and therefore contains no odd cycle at all — **not a single triangle.** A chemist would call these completely different molecules.

**But watch WL fail.** Round 0: every node has degree 3, so everyone is colour **A** — in *both* graphs. Round 1: every node's signature is $(\text{A}, \{\text{A},\text{A},\text{A}\})$ — again identical everywhere, in both graphs. Round 2: the same. **The refinement never splits anything, in either graph, forever.** Both histograms read "6 nodes of one colour," and the test reports no difference.

▸ **So a message-passing GNN produces *literally identical* embeddings for every node of both graphs, and identical graph-level readouts.** Not approximately similar — bit-for-bit the same. No amount of width, depth, training data, or clever loss function can change this; it is a statement about what information the architecture receives.

**Why it hurts most in chemistry.** Aromaticity, ring strain, and reactivity are all consequences of *cycle structure*. Benzene is a 6-ring; hexane is a 6-chain; their properties are utterly different. A GNN can distinguish those two (the degree sequences differ), but as soon as molecules become regular enough that local degree patterns match, the model is blind to the thing chemists care about most. **This is why "add triangle and ring counts as input features" — the least elegant fix in §29.4 — is standard practice in molecular machine learning.** Handing the model the answer to the question it cannot ask is not cheating; it is engineering around a proved limitation.

> **Analogy.** Describing a building by walking around it and only ever noting "the room I'm in has three doors, and each of those rooms also has three doors." You could wander a prison and a hotel for a thousand years and produce the identical report. The information you are permitted to collect simply does not contain the floor plan. Adding more layers is walking for longer; it does not change what you are allowed to write down.

#### Examples and non-examples: pairs of graphs a message-passing GNN can and cannot tell apart

**✅ Pairs it distinguishes easily**

| Pair | What separates them | Which round of WL notices |
|---|---|---|
| Benzene (6-ring) vs hexane (6-chain) | Degree sequence: all 2s versus $1,2,2,2,2,1$ | Round 0 |
| A triangle vs a 3-node path | Degrees $3\times2$ versus $1,2,1$ | Round 0 |
| A star $K_{1,5}$ vs a 6-cycle | One node of degree 5 versus all degree 2 | Round 0 |
| A triangle-with-a-tail vs a 4-path | Degrees $2,2,3,1$ versus $1,2,2,1$ | Round 0 |
| Two trees of the same size but different shape | Colours refine differently after one or two rounds | Round 1–2 |

**❌ Pairs it provably cannot** — every one of these gets **bit-for-bit identical** node embeddings and graph readouts

| Pair | Why WL is blind | What it would take |
|---|---|---|
| One 6-cycle vs **two disjoint triangles** | Every node has degree 2 in both, forever. Chemically: cyclohexane versus two molecules of cyclopropane | A ring-size feature, or a cycle counter |
| Triangular prism vs $K_{3,3}$ | Both 3-regular on 6 nodes; the prism is full of triangles, $K_{3,3}$ has none | 3-WL, a subgraph GNN, or triangle counts as input |
| Any two non-isomorphic $d$-regular graphs on $n$ nodes | Regularity means every node's signature is identical at every round | Structural encodings, or random node identifiers |
| Two molecules differing only in bond order (benzene vs cyclohexane as bare graphs) | $A$ is binary; bond type never enters the message | Edge features (MPNN / GINE) |
| Two 3-D conformations with the same bond graph | Connectivity is identical; only coordinates differ | 3-D geometry and the machinery of §29.8 |

▸ **The boundary:** a message-passing GNN sees exactly what **iterated local degree-and-colour refinement** can see. If two graphs agree on every node's "my colour plus the bag of my neighbours' colours," at every round, they are the same graph as far as the architecture is concerned — regardless of how obviously different they look to you.

**Work the cyclohexane example by hand; it takes twenty seconds and it is the fastest way to believe the theorem.** Graph A: six carbons in one ring. Graph B: two separate triangles of carbons. Round 0 — every node has degree 2, so every node is colour **A**, in both graphs, 6 of them each. Round 1 — every node's signature is $(\text{A},\{\text{A},\text{A}\})$, so every node becomes colour **B**, again 6 of them in each graph. Round 2, and every round after: identical. **The histograms never diverge.** A GNN with sum readout therefore emits the same graph vector for one hexagon and for two triangles, which are not merely different molecules but different *numbers* of molecules.

> **Common misconception.** *"A deeper GNN, or a wider one, or one trained on more data, will eventually learn to tell those apart."* It cannot, and the reason is not about capacity. Depth, width, training data and loss function all affect what the network *does with* the information it receives. The WL bound is a statement about what information the architecture **receives at all** — and if two inputs produce identical intermediate states at every layer, no downstream machinery can separate them. This is the most useful thing to be clear about in an interview: the bound is an information argument, not an optimization argument.

> **Common misconception.** *"So GNNs can't do chemistry."* They do chemistry very well, because real molecular datasets are full of graphs whose degree patterns differ, and because practitioners routinely staple ring and substructure counts onto the input features (§29.4). The bound describes a *specific* blind spot — regular and near-regular structures, and cycle counting — not a general incapacity. The honest reading is: **the failure cases are rare in general graphs and unusually common in exactly the domain the field cares most about**, which is why the fix became standard practice rather than a footnote.

### GIN — matching the bound

To reach the WL bound, the aggregator must be **injective on multisets**. Mean and max are not: $\{a,a\}$ and $\{a\}$ have the same mean and max. **Sum is** (for countable features).

▸ $$h_i^{(l+1)} = \mathrm{MLP}^{(l)}\left((1+\epsilon^{(l)})\,h_i^{(l)} + \sum_{j\in\mathcal{N}(i)}h_j^{(l)}\right)$$

The MLP provides the universal function approximation, the sum provides injectivity, and $\epsilon$ distinguishes the node from its neighbours. **GIN is as expressive as any message-passing GNN can be.**

*(Note the practical caveat: mean aggregation is better when the neighbourhood *distribution* matters more than its size, and max is better for detecting a distinctive neighbour. Maximal expressivity is not always maximal accuracy.)*

#### Why sum, and not mean or max

**"Injective on multisets" is the whole requirement, and it is worth saying in English: different bags of neighbours must produce different totals.** If two  different neighbourhoods collapse to the same aggregate, the information is destroyed at that instant and no later layer can recover it.

**Put numbers on the failure.** Let a neighbour's feature be the single number $a = 3$.

| Neighbourhood | Sum | Mean | Max |
|---|---|---|---|
| $\{3\}$ | **3** | 3 | 3 |
| $\{3,3\}$ | **6** | 3 | 3 |
| $\{3,3,3\}$ | **9** | 3 | 3 |

▸ **Mean and max cannot count.** All three neighbourhoods look identical to them. Sum separates all three cleanly. For chemistry this is not subtle: a carbon with one hydrogen and a carbon with three hydrogens are different atoms in different molecules, and a mean-aggregating network cannot tell them apart from that signal alone.

There is a second, complementary failure worth seeing:

| Neighbourhood | Sum | Mean | Max |
|---|---|---|---|
| $\{1,5\}$ | 6 | 3 | **5** |
| $\{2,4\}$ | 6 | 3 | **4** |
| $\{3,3\}$ | 6 | 3 | **3** |

Here **sum and mean both fail** and max succeeds. No single aggregator is injective on all multisets of real-valued features — which is exactly the theorem behind PNA's "use several at once" design.

**Reading the GIN formula, term by term.**

$$h_i^{(l+1)} = \mathrm{MLP}^{(l)}\left((1+\epsilon^{(l)})\,h_i^{(l)} + \sum_{j\in\mathcal{N}(i)}h_j^{(l)}\right)$$

| Piece | Job |
|---|---|
| $\sum_{j\in\mathcal{N}(i)}h_j^{(l)}$ | **Injectivity.** Plain sum, so the bag of neighbours survives intact |
| $(1+\epsilon^{(l)})h_i^{(l)}$ | **Self-distinction.** Scale your own vector differently from your neighbours' so "me" and "one of my neighbours" don't blur |
| $\mathrm{MLP}^{(l)}$ | **Universality.** By the universal approximation theorem an MLP can represent any function of the aggregate, so nothing is lost downstream |
| $\epsilon^{(l)}$ | A scalar, either learned or fixed at 0 |

▸ **Why $\epsilon$ matters, in one example.** Set $\epsilon = 0$ and the formula becomes $\mathrm{MLP}(h_i + \sum_j h_j)$ — you have added yourself into the same pot as your neighbours, and a node with feature $a$ and one neighbour $a$ becomes indistinguishable from a node with feature $2a$ and no neighbours. **Any $\epsilon \ne 0$ (even an irrational constant) restores the distinction**, which is why the paper notes that a fixed nonzero value works nearly as well as a learned one. The parameter is doing a job that requires only that it not be zero.

**The design in one sentence:** *sum for injectivity, MLP for expressiveness, $\epsilon$ for self-awareness.* Three components, three separate jobs, no redundancy.

**And the caveat deserves emphasis, because it is where theory and practice part company.** Maximal expressivity means "can distinguish the most graphs." It does **not** mean "generalizes best." Sum aggregation makes a node's representation scale with its degree, which is a large, high-variance input to the next layer; on citation networks where degrees span four orders of magnitude, that instability often costs more than the extra expressivity gains. **GIN wins on synthetic graph-distinguishing benchmarks and frequently loses to a plain GCN on real node-classification tasks.** Both facts are true and neither is a scandal — they are answers to different questions.

#### Examples and non-examples: injective aggregation

**✅ Aggregators that are injective on the multisets they are used for**

| Example | Injective on | Why |
|---|---|---|
| Sum, over one-hot node types | Any finite multiset of one-hot vectors | The total is literally a count vector: $\{$C,C,H$\}\mapsto(2,1,0,\dots)$ |
| Sum, over countable features | Multisets drawn from a countable set | The GIN paper's stated condition — the proof needs countability |
| Sum + max + min + std, concatenated (PNA) | Far more multisets than any one of them | Different aggregators fail on different pairs; concatenating covers more |
| Sum of $f(v_j)$ for a learned injective $f$ | The general construction | This is exactly GIN: MLP outside, sum inside |

**❌ Near-misses — aggregators that quietly destroy information**

| Looks fine | The collision | Consequence |
|---|---|---|
| Mean | $\{3\},\{3,3\},\{3,3,3\}$ all give $3$ | Cannot count. A carbon with one hydrogen and a carbon with three become identical |
| Max | $\{1,5\}$ and $\{5,5\}$ both give $5$ | Sees only the extreme; the rest of the neighbourhood is invisible |
| Sum, over **real**-valued features | $\{1,5\}$ and $\{2,4\}$ both give $6$ | Injectivity holds for countable features, not continuous ones — this is PNA's motivating theorem |
| Sum, in fp16, over a node with 20,000 neighbours | Values above 65,504 overflow to `inf` | Injective in theory, saturated in practice. The reason large-degree graphs need normalization or sampling |
| "Sum, then LayerNorm" | Normalization divides the magnitude back out | You have re-introduced the mean's blindness one line later — a  common bug |

▸ **The boundary:** an aggregator is injective when **no two different bags of neighbours produce the same output.** Everything else in GIN — the MLP, the $\epsilon$ — is machinery for preserving that property through the rest of the layer.

> **Common misconception.** *"GIN is the best GNN, because it is provably the most expressive."* Maximal expressivity means "can distinguish the most graphs." It says nothing about which function the model will actually *find*, or how it will generalize. Sum aggregation makes a node's representation scale with its degree; on a citation network where degrees span four orders of magnitude, that is a high-variance input to every subsequent layer, and it routinely costs more than the extra expressivity gains. **GIN dominates synthetic graph-distinguishing benchmarks and often loses to a plain GCN on real node classification.** The misconception is tempting because "provably maximal" sounds like a total ordering — it is a ceiling on one specific capability, and capability is not the binding constraint in most real tasks.

> **Where the WL bound came from.** Two groups proved essentially the same theorem independently and published at the same conference in 2019. **Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka** (MIT and Stanford) framed it as *How Powerful are Graph Neural Networks?* and introduced GIN as the construction that attains the bound. **Christopher Morris, Martin Ritzert, Matthias Fey, William Hamilton, Jan Eric Lenssen, Gaurav Rattan and Martin Grohe** proved the same ceiling and developed the $k$-WL generalization in *Weisfeiler and Leman Go Neural*. Grohe's group came at it from finite model theory and descriptive complexity — a branch of mathematical logic — rather than from machine learning, which is why the two papers read so differently while proving the same thing. **The result changed the field's self-image overnight:** before it, GNN expressivity was discussed in vague terms; afterwards, every new architecture was expected to state where it sat relative to 1-WL.

### Going beyond WL

- **Higher-order GNNs (k-WL):** message-pass over $k$-tuples of nodes. More expressive, $O(n^k)$ cost.
- **Positional/structural encodings:** add Laplacian eigenvectors, random-walk return probabilities, or random node identifiers as features. Cheap and effective; breaks strict permutation invariance in the random-ID case (recovered in expectation).
- **Subgraph GNNs:** represent a graph by a bag of its subgraphs.
- **Substructure counting:** just add triangle/ring counts as features. Inelegant, works well, standard in molecular ML.

#### The four escape routes, decoded

All four are attempts to break the WL ceiling, and they trade elegance against cost in different ratios.

**1. Higher-order GNNs ($k$-WL).** Instead of giving each *node* a colour, give each **$k$-tuple of nodes** a colour, and pass messages between tuples. Since a triangle is a 3-tuple, a 3-WL scheme can see triangles by construction.

The cost is brutal. On a graph with $n=1000$ nodes:

| Order | Objects tracked | Count |
|---|---|---|
| 1-WL | nodes | $10^3$ |
| 2-WL | pairs | $10^6$ |
| 3-WL | triples | $10^9$ |

▸ **Memory grows as $n^k$, so $k=3$ is already the practical ceiling and $k=4$ is out of reach for any real graph.** And by the Cai–Fürer–Immerman result, *no fixed $k$ suffices in general* — there are always graph pairs that defeat you. You pay exponentially for a bounded amount of extra sight.

**2. Positional and structural encodings.** Give each node extra input features that encode where it sits, so that WL has something to distinguish nodes *with*. Three common choices:

- **Laplacian eigenvectors.** The first few eigenvectors of $L$, evaluated at each node. This is the direct graph analogue of a transformer's sinusoidal positional encoding — and for a path graph, the Laplacian eigenvectors literally *are* sinusoids, which is a satisfying consistency check.
- **Random-walk return probabilities.** "If I take $k$ random steps from this node, what is the chance I come back?" Formally the diagonal entry of $(D^{-1}A)^k$. A node inside a triangle has a high 3-step return probability; a node on a chain has essentially none. **This directly injects the cycle information the WL bound forbids.**
- **Random node identifiers.** Give every node a random vector. Now every node is distinguishable and the WL bound simply does not apply.

▸ **The random-identifier trick breaks permutation invariance, and the fix is to accept it in expectation.** Sample fresh identifiers every forward pass and average: any single run is not invariant, but the *distribution* of outputs is, so the expected prediction is. It is the same bargain as dropout — deliberately injected randomness whose bias cancels in aggregate.

**3. Subgraph GNNs.** Represent the graph as a *bag of subgraphs* — for example, the $n$ graphs you get by marking one node as special — run a GNN on each, and pool. Marking a node breaks the symmetry that was blinding you, so the prism and $K_{3,3}$ become distinguishable. Cost: $n$ forward passes per graph.

**4. Substructure counting.** Count the triangles, the 5-rings, the 6-rings, and staple those numbers onto each node's feature vector before the first layer.

▸ **This is the one people actually use, and the honest reason is worth stating: it is $O(\text{cheap})$, it is computed once during preprocessing, and it works.** Chemistry already has excellent hand-built substructure counters going back decades. The theoretician's objection — "you're not *learning* the structure, you're being told it" — is correct and irrelevant to anyone shipping a model. **When a limitation is proved, the right response is often to route around it rather than to defeat it.**

---

## 29.5 The two structural pathologies

### Over-smoothing

▸ Repeated neighbourhood averaging is a diffusion process. As $L\to\infty$, node representations converge to a limit determined only by degree — **all nodes become identical** and the model loses discriminative power.

**The mathematics:** the GCN propagation matrix $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ has $\lambda_{\max}=1$ with all other $|\lambda_i|<1$. So $\hat A^L$ converges to a rank-one projection onto the dominant eigenvector, and everything else decays as $|\lambda_2|^L$. **Exponential collapse in depth.**

**Consequence:** most GNNs are 2–4 layers deep. **This is the single biggest structural difference from CNNs and transformers**, which get better with depth.

**Mitigations:** residual/initial connections (GCNII adds $\alpha H^{(0)}$ at every layer, provably slowing the collapse), jumping-knowledge (concatenate all layers' outputs), PairNorm/DropEdge, and simply using fewer layers with wider neighbourhood sampling.

#### Over-smoothing, decoded

**The phenomenon in one sentence: averaging, repeated, destroys differences — and a GCN layer is an average.**

> **Analogy.** A room of people, each holding a paint colour. Every minute, everyone mixes their paint with their neighbours'. After one round, local patches blur. After twenty, the entire room is the same shade of brown. Nothing was lost by accident and no step was wrong — **averaging is a lossy operation, and iterating a lossy operation converges to the loss.** The information wasn't destroyed by a bug; it was destroyed by doing exactly what you asked.

**The mathematics, unpacked.** $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ is symmetric, so §1.1.2 applies: it has an eigendecomposition $\hat A = Q\Lambda Q^\top$, and $\hat A^L = Q\Lambda^L Q^\top$. Its eigenvalues satisfy $\lambda_1 = 1$ (with eigenvector proportional to $\sqrt{\tilde d}$, the degree profile) and $\lvert\lambda_i\rvert < 1$ for every other $i$ on a connected graph.

Now raise them to the power $L$:

| $\lambda_i$ | $\lambda_i^{4}$ | $\lambda_i^{10}$ | $\lambda_i^{30}$ |
|---|---|---|---|
| $1.0$ | $1.0$ | $1.0$ | $1.0$ |
| $0.9$ | $0.656$ | $0.349$ | $0.042$ |
| $0.7$ | $0.240$ | $0.028$ | $2\times10^{-5}$ |
| $0.5$ | $0.063$ | $0.001$ | $9\times10^{-10}$ |

▸ **One eigenvalue survives and every other decays geometrically.** After enough layers, $\hat A^L \approx q_1q_1^\top$ — a **rank-one** matrix, meaning every node's representation is a scalar multiple of the *same* vector. The only thing distinguishing node $i$ from node $j$ is the scalar, and that scalar is fixed by degree. **Two atoms with the same number of bonds become literally indistinguishable, whatever the rest of the molecule looks like.**

**This is the same $\lambda^k$ argument as vanishing gradients in Chapter 6 and adversarial amplification in §1.1.4** — the third appearance in this book of "repeatedly applying a matrix amplifies its top eigenvector and annihilates everything else." Once you have the pattern, all three are one fact.

**The size of $1-\lambda_2$ is called the spectral gap, and it sets the timescale.** A well-connected graph has a large gap and smooths fast; a stringy, bottlenecked graph has a tiny gap and smooths slowly. Counterintuitively, **the better-connected your graph, the fewer layers you can afford.**

▸ **Why this is the single biggest structural difference from CNNs and transformers.** Both of those get monotonically better with depth: ResNet-152 beats ResNet-50, and a 96-layer transformer beats a 12-layer one. A GCN typically peaks at **2 layers** and degrades past 4. That is not a tuning failure; it is the operator converging. Every mitigation below is an attempt to stop a fixed point from being reached.

**The mitigations, each in one line:**

| Fix | What it does | Why it works |
|---|---|---|
| **GCNII** initial residual | Adds $\alpha H^{(0)}$ back at every layer | Keeps re-injecting the un-smoothed signal, so the fixed point is never reached |
| **Jumping knowledge** | Concatenates every layer's output before the readout | Layer 2's un-smoothed features are still available even if layer 8's are mush |
| **PairNorm** | Rescales so the average pairwise distance between node features stays constant | Directly forbids the collapse it is trying to prevent |
| **DropEdge** | Randomly deletes edges each step | Fewer edges means slower mixing — and it is a regularizer besides |
| **Fewer layers, wider sampling** | 2 layers over 25 sampled neighbours each | Sidesteps the problem instead of fighting it. Usually the right answer |

> **Where over-smoothing was identified.** **Qimai Li, Zhichao Han and Xiao-Ming Wu** named and analysed it in 2018, in *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning*. Their framing was the crucial move: they showed a GCN layer **is** Laplacian smoothing, a technique that had existed in graph signal processing and mesh processing for years as a *deliberate denoising operation*. Computer graphics people had been using exactly this operator to smooth out bumpy 3-D meshes since the 1990s. **The GCN's core operation was already known, under a different name, as a tool whose entire purpose is to erase detail** — which makes it slightly less surprising that stacking it erases detail.

### Over-squashing

▸ A node's $L$-hop neighbourhood can grow exponentially, but its representation is a **fixed-size vector**. Information from an exponentially large receptive field is squashed through $O(d)$ bits.

**The mathematics:** the sensitivity $\left\|\frac{\partial h_i^{(L)}}{\partial x_j}\right\|$ for a node $j$ at distance $L$ decays with the **normalized adjacency powers**, and it is controlled by the graph's **Cheeger constant / spectral gap** — a bottleneck edge in the graph is a bottleneck in information flow. This is why long-range tasks on tree-like or bottlenecked graphs fail even when the model is deep enough to reach.

**Mitigations:** graph rewiring (add edges to improve the spectral gap — e.g. stochastic discrete Ricci flow, expander-graph augmentation), a **fully-connected "virtual node"** connected to everything (a cheap global shortcut), and graph transformers.

▸ **Note the tension: over-smoothing pushes you to fewer layers; over-squashing pushes you to more.** Long-range dependencies on graphs remain  hard, and this pair of constraints is why.

#### Over-squashing, decoded

Over-smoothing and over-squashing sound similar and are opposite problems. Keep them apart with one line each:

| | Over-smoothing | Over-squashing |
|---|---|---|
| The failure | Everything becomes the **same** | Distant things never **arrive** |
| Cause | Averaging repeated too often | Too much information forced through too narrow a pipe |
| Symptom | Deep model, all nodes identical | Deep model, long-range task still fails |
| Pushes you to | **Fewer** layers | **More** layers |

**The arithmetic that makes it inevitable.** Suppose each hop reaches about 4 new nodes and your hidden dimension is $d=128$.

| Layers | Nodes in the receptive field | Numbers available to hold them |
|---|---|---|
| 2 | ~21 | 128 |
| 4 | ~341 | 128 |
| 6 | ~5,461 | 128 |
| 8 | ~87,381 | 128 |

▸ **The left column grows exponentially; the right column is a constant.** By layer 6 you are asking 128 numbers to summarize five thousand nodes. Something must be discarded, and gradient descent decides what — usually whatever is furthest away, because it arrived most attenuated.

> **Analogy.** A funnel. Pour in a bucket and it passes fine. Pour in a swimming pool through the same neck and most of it goes over the sides. **The neck did not get narrower; the volume got larger.** A GNN's hidden dimension is the neck, and every layer you add pours in exponentially more.

**Reading the sensitivity formula.** $\left\lVert\dfrac{\partial h_i^{(L)}}{\partial x_j}\right\rVert$ asks: *"if I nudge the input feature of node $j$, how much does node $i$'s final representation move?"* It is exactly the influence of $j$ on $i$ (§0.7 — a partial derivative is a sensitivity). If it is near zero, node $j$ is invisible to node $i$ no matter what the loss wants, because **no gradient can flow along a path whose derivative has already vanished.**

That quantity is bounded by entries of the powered normalized adjacency, and those entries decay with distance. So the model can *reach* node $j$ in $L$ hops while being unable to *feel* it.

**The spectral gap and the Cheeger constant, in plain terms.** The **Cheeger constant** $h(G)$ measures the worst bottleneck in a graph: consider every way of cutting the graph into two pieces, and report the smallest ratio of (edges cut) to (nodes on the smaller side). A dumbbell — two dense blobs joined by a single edge — has a tiny Cheeger constant. A well-mixed random graph has a large one.

Cheeger's inequality ties this shape fact to the spectrum: **a small Cheeger constant forces a small spectral gap $\lambda_2$, and a small spectral gap means slow mixing.** So:

▸ **A structural bottleneck in the graph is, provably, a bottleneck in information flow.** This is why long-range tasks fail on trees and molecules-with-long-chains even when the model is deep enough to reach across. The obstruction is in the graph, not in the network.

**The mitigations, decoded:**

- **Graph rewiring** — add or reweight edges to raise the spectral gap. **Expander graphs** are the target: graphs that are sparse (few edges per node) yet have no bottlenecks at all. Adding a small number of well-chosen shortcuts can transform mixing time without meaningfully increasing cost.
- **A virtual node** — one extra node joined to every real node. Now every pair of nodes is **2 hops apart** via the hub, regardless of the original diameter. Cost: $n$ extra edges, which is nothing. **It is the cheapest fix in the chapter and often the most effective**, and its downside is that it is a shared channel: everything routed through one vector will interfere.
- **Graph transformers** — abolish the distance concept entirely (§29.7).

> **Analogy for the virtual node.** An office where people can only talk to whoever sits adjacent. Adding a company-wide notice board puts everyone one step from everyone. It is enormously effective and it also becomes a place where messages get lost in the noise. That is the exact trade.

#### Examples and non-examples: diagnosing which pathology you have

These two get confused constantly, and the fixes are opposites, so a wrong diagnosis makes things worse. Here is how each one actually presents.

**✅ Symptoms that  indicate over-smoothing**

| Symptom | Why it points there |
|---|---|
| Node embeddings' pairwise cosine similarity climbs toward 1 as depth increases | The rank-one collapse, measured directly |
| Accuracy peaks at 2 layers and falls monotonically from 4 onward | The operator is converging to its fixed point |
| Predictions become nearly constant across the graph | Every node is a multiple of the same vector |
| Two nodes with the same degree get identical outputs | The surviving eigenvector is the degree profile |
| Adding a residual/initial connection (GCNII) recovers the lost accuracy | The fix targets exactly this failure |

**✅ Symptoms that  indicate over-squashing**

| Symptom | Why it points there |
|---|---|
| The task needs information from 6+ hops away and accuracy is at chance for those cases | Distant gradients have already decayed |
| Short-range versions of the same task work fine | The model is capable; the signal is not arriving |
| The graph is tree-like, chain-like, or has an obvious bottleneck | Small Cheeger constant, small spectral gap, slow mixing |
| Adding a **virtual node** helps substantially | The fix targets exactly this failure |
| Widening $d$ from 128 to 512 helps more than adding layers | You were bandwidth-limited, not reach-limited |

**❌ Near-misses — failures that look like a graph pathology and aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Training loss also rises with depth | Over-smoothing degrades *test* performance by collapsing representations; it does not usually make the model unable to fit | An optimization problem — vanishing gradients, a bad learning rate, missing normalization |
| Accuracy drops when you add layers **and** parameters together | Two changes, one measurement | Possibly plain overfitting; hold parameter count fixed and re-test |
| Long-range task fails and the model is 2 layers deep | The information physically cannot arrive in 2 hops | Insufficient depth — a reach problem, not a squashing problem |
| Node features become identical because they *started* identical | Nothing collapsed; there was never anything to collapse | A featurization problem. Add degree, or structural encodings |
| Deep GNN diverges to `NaN` | Collapse is a smooth convergence, not an explosion | Unnormalized propagation, or an isolated node dividing by zero |

▸ **The boundary:** over-smoothing is a **convergence** failure — signals become identical; over-squashing is a **bandwidth** failure — signals never arrive. The one-question diagnostic: *are my node representations too similar to each other, or are they fine but ignoring something far away?* The first says use fewer layers; the second says use more, or a shortcut.

> **Common misconception.** *"My GNN gets worse with depth, so it's overfitting."* Over-smoothing degrades **training** performance as readily as test performance, which overfitting by definition does not. A 20-layer GCN that has collapsed cannot fit its training set either — it has thrown away the distinctions it would need. The misconception is tempting because "more layers, worse accuracy" is the standard signature of overfitting everywhere else in deep learning. **Plot the training curve; that single check separates the two diagnoses.**

> **Common misconception.** *"Over-smoothing is just vanishing gradients under a different name."* They share the $\lambda^k$ mathematics and they are different failures. Vanishing gradients are about the **backward** pass: the update signal cannot reach early layers. Over-smoothing is about the **forward** pass: the representations themselves become identical, and would do so even if you never computed a gradient at all. You can watch it happen with a randomly initialized, untrained network — which is the cleanest way to convince yourself the two are distinct.

> **Where over-squashing came from.** **Uri Alon and Eran Yahav** named the phenomenon in 2021, in *On the Bottleneck of Graph Neural Networks and its Practical Implications*. Their proposed remedy was almost comically simple — make the *last* layer fully adjacent, connecting every node to every other for one round — and it improved long-range benchmarks substantially, which was itself the argument that the diagnosis was right. In 2022, **Jake Topping, Francesco Di Giovanni, Benjamin Chamberlain, Xiaowen Dong and Michael Bronstein** connected over-squashing to **discrete Ricci curvature**: negatively-curved edges are exactly the bottlenecks, and you can rewire by running a discrete **Ricci flow** to smooth them out. Ricci flow is the tool **Richard Hamilton** introduced in 1982 and **Grigori Perelman** used to prove the **Poincaré conjecture** in 2002–2003 — for which Perelman was awarded the Fields Medal in 2006 and the million-dollar Millennium Prize in 2010, and declined both. **The Cheeger constant** is named for **Jeff Cheeger**, who proved the continuous version in 1970 in Riemannian geometry; the graph analogue was worked out in the mid-1980s by Dodziuk and by Alon and Milman. As with so much of this chapter, the machinery predates the application by decades and came from a field with no interest in it.

---

## 29.6 Pooling and readout

**Graph-level readout** must be permutation-invariant: sum (preserves size information), mean (size-invariant), max, or attention-weighted (Set2Set, GlobalAttention).

▸ **Sum vs mean is a real modelling decision.** For molecules, many properties are extensive (scale with size — e.g. molecular weight, total energy) so sum is correct; others are intensive (solubility, band gap) so mean is correct. Getting this wrong is a common and silent source of poor performance.

**Hierarchical pooling:** DiffPool (learn a soft cluster assignment matrix), TopK/SAGPool (score and keep the top nodes), EdgePool. Useful but often not better than a good flat readout.

#### Extensive versus intensive — the mistake that costs you silently

**Readout** means the final step that turns $n$ node vectors into **one** graph vector, so you can predict a property of the whole molecule. It must be permutation-invariant for the same reason everything else is: shuffle the atoms and the molecular weight must not change.

**The two words are from thermodynamics, and they are worth learning properly.**

| Property type | Definition | Examples | Correct readout |
|---|---|---|---|
| **Extensive** | Doubles when you double the amount of stuff | molecular weight, total energy, number of atoms, mass | **Sum** |
| **Intensive** | Unchanged when you double the amount of stuff | temperature, density, solubility, band gap, pH | **Mean** |

> **Analogy.** Two identical glasses of water, poured into one jug. The **mass** doubled — extensive. The **temperature** did not — intensive. Nothing about the water changed; the two questions simply scale differently with size.

**Now watch the failure, with numbers.** Suppose the model has learned a per-atom energy of about $-2$ units, and you have two molecules: one with 10 atoms and one with 50.

| Readout | 10-atom molecule | 50-atom molecule | Correct for total energy? |
|---|---|---|---|
| Sum | $-20$ | $-100$ | ✓ |
| Mean | $-2$ | $-2$ | ✗ — a 50-atom molecule is *not* as stable as a 10-atom one |

▸ **With mean readout, the model cannot express total energy at all — not poorly, but *at all*.** It has no access to the count. To fit the training data it will be forced to distort its per-atom predictions into nonsense, and every diagnostic will show a model that is training normally and simply performing badly. There is no error message. **This is the silent failure the text warns about, and it is why the paragraph exists.**

The reverse error is just as real: use **sum** for an intensive property like solubility and the model must learn to divide by the atom count internally, spending capacity to undo a choice you made for it.

**Sum also carries structural information that mean discards.** Sum readout retains the graph's size; mean is deliberately size-blind. Since a sum aggregator is what made GIN maximally expressive (§29.4), sum readout is the expressivity-preserving choice — the same argument, one level up.

**Hierarchical pooling, in one line each:**

- **DiffPool (Differentiable Pooling)** — learn a soft matrix $S$ assigning each node to one of $k$ clusters, then build a smaller graph whose nodes are those clusters. The graph analogue of a CNN's stride-2 downsampling.
- **TopK / SAGPool (Self-Attention Graph Pooling)** — score every node, keep the top $k$, throw the rest away. Cheaper, and harsher.
- **EdgePool** — repeatedly contract the highest-scoring edge, merging its two endpoints. Like an image pyramid, but the pyramid is built from the data.

The honest verdict in the text ("often not better than a good flat readout") is worth taking seriously: a sum or mean over all nodes has **no parameters**, cannot overfit, and is frequently the strongest baseline. Hierarchical pooling is one of several places in machine learning where the more sophisticated method has struggled to justify itself.

#### Examples and non-examples: choosing a readout

**✅ Correct pairings of property and readout**

| Property to predict | Extensive or intensive? | Readout | Sanity check |
|---|---|---|---|
| Total molecular energy | Extensive | **Sum** | Two copies of a molecule have twice the energy |
| Molecular weight | Extensive | **Sum** | Two copies weigh twice as much |
| Number of rotatable bonds | Extensive | **Sum** | It is a count |
| Solubility (log S) | Intensive | **Mean** | Two copies dissolve just as readily as one |
| HOMO–LUMO gap | Intensive | **Mean** | A property of the electronic structure, not the amount |
| "Does this molecule contain a nitro group?" | Neither — it is existential | **Max** | Presence of one atom should trigger it |

**❌ Near-misses — readouts that look reasonable and quietly cripple the model**

| Choice | What goes wrong | The tell |
|---|---|---|
| Mean readout for total energy | The model has **no access to the atom count** and cannot express the target at all | Predictions cluster near one value regardless of molecule size |
| Sum readout for solubility | The model must learn to divide by $n$ internally, spending capacity undoing your choice | Error grows systematically with molecule size |
| Max readout for a graded property | Only the single most extreme node reaches the output | Almost all gradient flows to one atom per molecule |
| Sum readout with unnormalized features on a 100,000-node graph | The graph vector's magnitude dwarfs anything a downstream layer expects | Loss is enormous at step 0 and never recovers |
| Concatenating all node vectors and using a dense layer | Not permutation-invariant; also requires fixed $n$ | It "works" on a benchmark where every graph has the same size, then fails on anything real |

▸ **The boundary:** ask one question — *if I duplicated this entire graph, should the answer double?* Yes means sum. No means mean. "It should just be true somewhere" means max. That question takes five seconds and prevents the most expensive silent failure in the chapter.

> **Common misconception.** *"Pooling in a GNN is like pooling in a CNN — a downsampling detail."* In a CNN, max-pooling over a $2\times2$ window is a mild, local, well-understood resolution reduction, and swapping it for average pooling changes almost nothing. In a GNN, the readout is the **only** place the graph becomes a single object, and the choice determines which functions the model can represent at all. A CNN with the wrong pooling loses a percent; a GNN with the wrong readout cannot express the target. **The word is shared; the stakes are not.**

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

#### Graph transformers, decoded

**The move in one sentence: stop using the graph as the wiring diagram, and start using it as a hint.**

A message-passing GNN can only send information along edges. A graph transformer lets every node attend to every other node — the graph structure is no longer a constraint on *who talks to whom*, only a **feature** describing what their relationship is. Structure moves from the architecture into the input.

**Why the four structural signals exist, and what each supplies:**

| Signal | What it is | What it tells a node |
|---|---|---|
| **Laplacian eigenvectors** | The first $k$ eigenvectors of $L$, one coordinate per node | "Here is roughly where you sit in the graph's global shape" |
| **Random-walk encodings** | The diagonal of $(D^{-1}A)^k$ for several $k$ | "Here is how likely you are to return home in $k$ steps" — i.e. your local cycle structure |
| **Shortest-path bias** | Add a learned scalar $b_{\text{dist}(i,j)}$ to the attention logit | "This node is 4 hops away; discount it accordingly" |
| **Degree centrality** | $d_i$ as a feature | "You are a hub" / "you are a leaf" |

**The sign ambiguity, and why it needs handling.** If $v$ is an eigenvector, so is $-v$ — flipping every entry gives an equally valid eigenvector with the same eigenvalue. Standard eigensolvers pick a sign essentially arbitrarily, so **the same graph can produce positional encodings that are the exact negatives of each other on two different runs.** The usual fix is to randomly flip the sign of each eigenvector during training, forcing the model to learn a function that is invariant to the flip. It is the same style of solution as the random node identifiers in §29.4: when a symmetry cannot be removed, average over it.

**Reading the shortest-path bias.** Graphormer adds a learned number to the pre-softmax attention score, indexed by the graph distance between the two nodes:

$$\text{score}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}} + b_{\,\mathrm{dist}(i,j)}$$

So there is one learned scalar for "1 hop apart," another for "2 hops," and so on. **A negative bias for large distances recovers ordinary message passing as a special case** — set $b$ to $-\infty$ for everything beyond 1 hop and you have masked attention to the neighbourhood. This is exactly T5's relative position bias with graph distance substituted for token distance, which is why the text points at Chapter 12.

**Now the trade, made concrete.** A molecule with $n=50$ atoms and about 52 bonds:

| | Message passing | Full attention |
|---|---|---|
| Pairs computed per layer | ~52 (one per edge) | $50^2 = 2{,}500$ |
| Max hops between any two nodes | up to the graph diameter | **1, always** |
| Structural prior | strong (only bonded atoms interact) | weak (structure is a soft hint) |
| Data required | less | more |

▸ **What you buy: over-squashing disappears, because nothing is far away, and over-smoothing loses its grip, because the mixing is learned rather than fixed.** What you pay: $O(n^2)$ cost, and the loss of the sparsity prior — the built-in assumption that *bonded atoms interact and non-bonded ones mostly don't*, which happens to be true and which the model now has to rediscover from data.

**This is the identical trade as ViT versus CNN in Chapter 28**, on a different symmetry group: a strong architectural prior wins in the small-data regime, a weak prior plus scale wins when data is abundant. Molecules are usually the small-data regime — QM9 has around 134,000 molecules, which is minuscule compared with any image or text corpus — which is why hybrids rather than pure transformers currently lead.

▸ **GraphGPS is the hybrid, and the recipe generalizes beyond graphs: local layers for efficiency and inductive bias, global layers for reach.** The same pattern appears in long-context language models (sliding-window attention interleaved with full attention), in vision (convolutional stem, transformer body), and in speech. **When two mechanisms have complementary failure modes, alternating them is usually stronger than choosing one.**

> **Common misconception.** *"Graph transformers are strictly more powerful, so message passing is obsolete."* Full attention removes a *constraint*, and removing a constraint is only an improvement when the constraint was wrong. On molecules, the sparsity prior — bonded atoms interact strongly, non-bonded ones mostly don't — is chemically correct, and a message-passing model gets it for free while a transformer must rediscover it from a dataset of 134,000 molecules. That is why the leaders on molecular benchmarks are hybrids rather than pure transformers, and why the same argument settled out the same way for ViT versus CNN at small scale (Ch. 28). **A weaker prior wins only when you can pay for it in data.**

> **Common misconception.** *"Laplacian eigenvectors are the graph's positional encoding, the same way sinusoids are a sequence's."* The analogy is real and it leaks in one specific place: **eigenvectors have an arbitrary sign, and eigenvalues with multiplicity have an arbitrary basis.** Run the same eigensolver twice on the same graph and you can get encodings that are exact negatives of each other. Sinusoidal positional encodings have no such ambiguity — position 7 is position 7 on every run. The standard fix, random sign flipping during training, is not a tuning detail; without it the model is being handed an input that changes identity between runs.

#### Examples and non-examples: structural signals a graph transformer can use

**✅ Signals that  encode structure without breaking permutation symmetry**

| Signal | What it tells a node | Cost |
|---|---|---|
| First $k$ Laplacian eigenvectors | Roughly where you sit in the graph's global shape | One partial eigendecomposition per graph, precomputed |
| Diagonal of $(D^{-1}A)^k$ for several $k$ | Your local cycle structure — return probability in $k$ steps | $k$ sparse multiplies |
| Shortest-path distance as an attention bias | "This node is 4 hops away; discount it" | All-pairs shortest paths, feasible for molecules |
| Degree as a node feature | "You are a hub" / "you are a leaf" | Free |

**❌ Near-misses — encodings that look structural and aren't safe**

| Looks like it | Why it fails | What it actually is |
|---|---|---|
| The node's index $i$, embedded | Index is your data loader's decision, not the graph's | Training on the filing system |
| Row $i$ of the adjacency matrix as a feature vector | Its length is $n$ and its entries are ordered by node index | Not permutation-equivariant, and not transferable between graphs |
| Raw Laplacian eigenvectors with no sign handling | Sign is arbitrary per run | An input whose identity is not well defined |
| A random vector per node, used once and cached | Fixes the arbitrariness by freezing it, which reintroduces index dependence | A per-node embedding table in disguise |
| A random vector per node, resampled every pass | Invariant **in expectation**, not per-run | An honest bargain, and the one §29.4 recommends |

▸ **The boundary:** a structural encoding is legitimate when it is **computed from the graph itself** and would come out the same after any relabelling — up to a symmetry you explicitly average over. Anything derived from the node's index is contraband, however useful it looks on a benchmark.

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

#### Equivariance in 3-D, decoded

Everything so far treated a graph as pure connectivity. For a molecule that is not enough: **the same bonds arranged in different shapes are different substances.** So each node now carries a position $\vec r_i \in \mathbb{R}^3$, and a new symmetry group arrives.

**The groups, spelled out.**

| Group | Full name | Contains |
|---|---|---|
| SO(3) | Special Orthogonal group, 3-D | Rotations |
| SE(3) | Special Euclidean group | Rotations **and** translations |
| E(3) | Euclidean group | Rotations, translations **and** reflections |

**Why the reflection distinction is not pedantry.** A molecule and its mirror image can be  different compounds — this is **chirality**, and your two hands are the everyday example: same bones, same joints, not superimposable. Chirality routinely decides whether a drug works. **Thalidomide is the notorious case**: its two mirror forms have different biological activity, one sedative and one teratogenic, and it was marketed as a mixture of both. (The frequently-told version — "they should have sold only the safe one" — is not right: thalidomide interconverts between the two forms inside the body, so a pure preparation would not have been safe. The lesson about chirality mattering stands; the lesson about how to fix it does not.) A model that is E(3)-equivariant treats the two mirror forms as identical; one that is only SE(3)-equivariant can tell them apart. **Choose E(3) and you have built into your architecture the assumption that left and right hands are the same object.**

**"Energy invariant, forces equivariant," decoded.** Rotate a molecule in empty space:

- Its **energy** is one number and it does not change. Physics has no preferred orientation. → **invariant**.
- The **force on each atom** is an arrow, and the arrows rotate with the molecule. → **equivariant**: $\vec F \to R\vec F$.

▸ **And the two are not independent, which is the elegant part: force is the negative gradient of energy, $\vec F_i = -\partial E/\partial \vec r_i$.** Differentiating an invariant scalar with respect to a vector *automatically* produces an equivariant vector. So a network that predicts a rotation-invariant energy and obtains forces by autodiff gets force equivariance for free, and gets energy conservation for free as well — a machine-learned force field built this way cannot violate conservation of energy, because it is a gradient by construction.

**Strategy 1 — invariant features only.** Never look at coordinates; look only at quantities that don't care about orientation.

| Descriptor | What it is | Invariant to |
|---|---|---|
| Distance $\lVert \vec r_i - \vec r_j\rVert$ | how far apart two atoms are | rotation, translation, reflection |
| Angle $\angle(i,j,k)$ | the bend at atom $j$ | rotation, translation, reflection |
| Dihedral | the twist about the $j$–$k$ bond in a chain $i$–$j$–$k$–$l$ | rotation, translation — **but flips sign under reflection** |

▸ **The dihedral row is the one to notice: dihedrals are the descriptors that can see chirality**, because a signed twist reverses in a mirror. SchNet uses distances alone, DimeNet adds angles, GemNet adds dihedrals — the progression is literally a march up this table, each step buying more geometric discrimination at more cost.

**Why "distance-only descriptors are provably incomplete."** There exist  different 3-D configurations with identical sets of pairwise distances. Once that happens, no model that consumes only distances can distinguish them — the same style of blindness as the WL bound, one dimension up. It is not a limitation of SchNet; it is a limitation of the input.

**Strategy 2 — EGNN, the minimal equivariant network.** Read the two equations:

$$m_{ij}=\phi_e\big(h_i,h_j,\lVert\vec r_i-\vec r_j\rVert^2\big),\qquad \vec r_i \leftarrow \vec r_i + \sum_{j\ne i}(\vec r_i-\vec r_j)\,\phi_r(m_{ij})$$

- The message $m_{ij}$ depends on positions **only through the squared distance** — an invariant scalar. So $m_{ij}$ is invariant, and $\phi_r(m_{ij})$ is just a number.
- The position update adds up **difference vectors $(\vec r_i - \vec r_j)$, each scaled by that invariant number.**

▸ **Now the magic, in one line of algebra.** Rotate everything by $R$. Each difference becomes $R(\vec r_i - \vec r_j)$, distances are unchanged so the scalars $\phi_r$ are unchanged, and the sum becomes $R\sum_j(\vec r_i-\vec r_j)\phi_r$ — **the same update, rotated.** Equivariance is not enforced by a penalty or learned from augmentation; it is a consequence of the only two operations allowed. **You cannot write down a non-equivariant EGNN.**

> **Analogy.** Giving directions using only "walk 30 metres toward the church" rather than "walk 30 metres north." Spin the whole town on a turntable and your instructions still work perfectly, because they were phrased in terms of *relationships between things in the town* rather than an external reference frame. Equivariant architectures are exactly this discipline: never name an absolute direction, only ratios and differences of things already inside the system.

**Strategy 3 — spherical harmonics and tensor fields.** The heavy machinery. Represent features not as plain vectors but as **irreducible representations of SO(3)** — the mathematical catalogue of "all the fundamentally different ways an object can transform under rotation." A scalar (unchanged), a vector (rotates), a rank-2 tensor (rotates twice over), and so on. **Clebsch–Gordan tensor products** are the rules for combining two such objects to get a third that still transforms correctly. Maximum expressivity, best force-field accuracy, and a substantial jump in implementation complexity.

**Putting numbers on "worth the trouble."** The claim is roughly an **order of magnitude** in data efficiency: an equivariant model reaching a target accuracy on perhaps 1,000 structures where a non-equivariant one needs 10,000. Each structure costs a quantum-chemistry calculation measured in CPU-hours. **The symmetry prior is not saving you training time; it is saving you a supercomputer allocation.**

▸ **And the deeper reason it works is worth stating plainly.** A non-equivariant model must spend capacity learning that physics has no preferred orientation — a fact that is *exactly, universally true* and that no amount of data can teach it better than "always." Every parameter spent memorizing rotations is a parameter not spent on chemistry. **This is the strongest available demonstration in the book that the right prior beats more data**, precisely because the prior here is not an approximation or a heuristic; it is a law.

#### Examples and non-examples: operations you are allowed to perform on 3-D coordinates

This is the most mechanical example-set in the chapter, and the most useful: given a candidate layer, you can decide by inspection whether it is equivariant.

**✅ Rotation-equivariant / invariant operations**

| Operation | Output type | Why it is safe |
|---|---|---|
| $\lVert \vec r_i - \vec r_j\rVert$ | scalar (**invariant**) | Rotation preserves lengths |
| $(\vec r_i - \vec r_j)\cdot(\vec r_i-\vec r_k)$ | scalar (**invariant**) | Dot products are preserved: $(Ru)\cdot(Rv)=u\cdot v$ |
| $s \cdot \vec v$ for an invariant scalar $s$ | vector (**equivariant**) | Scaling commutes with rotation |
| $\vec v_1 + \vec v_2$ for two equivariant vectors | vector (**equivariant**) | $Rv_1+Rv_2 = R(v_1+v_2)$ |
| $\vec u \times \vec v$ | vector (**equivariant**, sign-flips under reflection) | The cross product is how chirality enters |
| $-\partial E/\partial \vec r_i$ where $E$ is invariant | vector (**equivariant**) | The gradient of an invariant scalar is automatically equivariant |

**❌ Near-misses — operations that break equivariance, some of them subtly**

| Operation | Why it breaks | Symptom |
|---|---|---|
| $\mathrm{MLP}(\vec r_i)$ applied to raw coordinates | The MLP has no idea $R\vec r$ means the same thing as $\vec r$ | Rotate the input molecule, get a different energy |
| Adding a learned constant vector $\vec b$ to positions | $\vec b$ names an absolute direction in space | Introduces a preferred "up" that physics does not have |
| $\mathrm{ReLU}(\vec v)$, elementwise | Clipping is per-coordinate, and coordinates are frame-dependent | A vector's clipped version depends on how you drew the axes |
| LayerNorm over the three coordinates of $\vec r_i$ | Subtracting the mean of $(x,y,z)$ mixes axes in a frame-dependent way | Same failure, harder to spot in code |
| $\lVert\vec r_i\rVert$ (distance from the origin) | Invariant to rotation but **not to translation** — it names a special point | Move the whole molecule and the features change |
| Concatenating $[\,\vec r_i \Vert h_i\,]$ and feeding it to an MLP | The MLP again sees raw coordinates | The classic way an "equivariant" implementation quietly isn't |
| Rotation augmentation instead of a constraint | Approximately equivariant, never exactly | Predictions differ by a few percent between orientations of the same molecule |

▸ **The boundary:** you may build **scalars from invariant quantities** (lengths, dot products, angles) and **vectors only as invariant-scalar multiples of difference vectors** you already had. The moment a raw coordinate reaches a nonlinearity, equivariance is gone. That single rule explains the shape of the EGNN update and is the fastest way to audit an implementation.

**A 30-second test you can run on any 3-D model.** Take one molecule, compute the prediction, then apply a random rotation matrix $R$ to all coordinates and recompute. An energy prediction must agree to numerical precision — differences at the $10^{-6}$ level are floating point, differences at the $10^{-2}$ level mean your model is not equivariant. For forces, check that the new prediction equals $R$ times the old one. **This test catches the LayerNorm-on-coordinates bug and the concatenation bug immediately**, and it is worth writing before the training loop rather than after.

> **Common misconception.** *"Rotation augmentation gives you equivariance, so the special architecture is unnecessary."* Augmentation gives you *approximate* equivariance on the orientations you sampled, learned at the cost of parameters and data. A constraint gives you *exact* equivariance on every orientation, including ones you never saw, for free. The empirical gap is roughly an order of magnitude in data efficiency on molecular property prediction — and every training structure is a quantum-chemistry calculation costing CPU-hours. The misconception is tempting because augmentation  does work and is far easier to implement. **It works; it just costs you a supercomputer allocation.**

> **Common misconception.** *"E(3)-equivariance is the stronger, better choice because it respects more symmetries."* More symmetry means more you have promised to ignore. E(3) includes reflections, so an E(3)-equivariant model treats a molecule and its mirror image as **the same object** — which is exactly wrong when chirality determines whether a drug is active. For chiral chemistry you want SE(3), the smaller group. **The right symmetry group is the one your data actually has, not the largest one you can name.**

> **Where equivariant networks came from.** The idea that symmetry is the organizing principle of physical law is **Emmy Noether's**. Her 1918 theorem — proved at Göttingen, where she lectured for years without pay or a formal position because the university would not appoint a woman — says that **every continuous symmetry of a physical system corresponds to a conserved quantity**: rotational symmetry gives conservation of angular momentum, translational symmetry gives conservation of momentum, time symmetry gives conservation of energy. It is among the most consequential theorems in physics. Equivariant neural networks are the engineering realization of the same insight: build the symmetry in and the conservation laws come along. **Taco Cohen and Max Welling** brought group equivariance into deep learning with *Group Equivariant Convolutional Networks* in 2016; **Nathaniel Thomas, Tess Smidt and colleagues** introduced Tensor Field Networks in 2018, importing spherical harmonics and **Clebsch–Gordan** coefficients from quantum angular-momentum theory. Alfred Clebsch and Paul Gordan themselves were 19th-century algebraists working on invariant theory, with no notion of quantum mechanics — the coefficients acquired their names decades later when physicists found the same combinatorics in the addition of angular momenta.

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

#### The practical warnings, decoded

**Scaffold splitting, and why random splitting lies to you.** A molecule's **scaffold** is its core skeleton with the decorations stripped off — the ring system that a medicinal chemist would recognize as "the same series." Real drug discovery programs make dozens of close analogues of one scaffold.

> **Analogy.** Testing whether a student has learned arithmetic by putting $7\times8$ on the practice sheet and $8\times7$ on the exam. They will score brilliantly and you will have measured nothing. **A random split over a molecular dataset does exactly this**, because near-duplicates of the same scaffold land on both sides of the line.

▸ **A scaffold split forces the test set to contain skeletons the model has never seen, which is the question you actually care about: "will this work on a new chemical series?"** Random splits inflate reported accuracy by 10–20 points. This is Chapter 3's grouped cross-validation point in chemical clothing — the group is the scaffold — and the text's observation that it is "violated constantly in published chemistry ML" is not a rhetorical flourish. When you read a molecular ML paper, the split protocol is the first thing to check, before the architecture.

**Why a fingerprint plus gradient boosting is a serious baseline.** An **ECFP (Extended-Connectivity FingerPrint)** enumerates every substructure within a few bonds of every atom, hashes each into a bit vector, and hands you a fixed-length binary summary — typically 1,024 or 2,048 bits. It is deterministic, has no parameters, takes milliseconds, and dates from 2010.

▸ **Note what an ECFP is really doing: it is the WL algorithm, run for a few rounds, with the colours hashed into bit positions.** The classical cheminformatics fingerprint and the modern expressivity theory are the same procedure, invented independently on either side of a disciplinary wall. Once you see that, the empirical result stops being embarrassing: **on many molecular benchmarks, a fingerprint plus gradient boosting (Ch. 23) beats a GNN because it is computing the same features with less variance and no training instability.** Run it first. If your GNN cannot beat it, the GNN is not adding anything.

**Speed of learned force fields.** The text's "$10^5$–$10^6\times$" is worth converting: a **DFT (density functional theory)** calculation that takes an hour becomes something like 40 milliseconds. That is the difference between simulating a nanosecond of molecular motion and simulating a microsecond — and biology's interesting events happen on the longer timescale. **The speedup does not make existing calculations faster; it makes a different class of question askable at all.**

**Why full-batch training fails on large graphs.** The problem is not the parameters, it is the **activations**. Every node's hidden vector at every layer must be held for the backward pass. For $10^8$ nodes, $d=256$, 3 layers, in fp32:

$$10^8 \times 256 \times 3 \times 4\ \text{bytes} \approx 300\ \text{GB}$$

▸ **And that is before gradients, and it is for one step.** GraphSAGE's neighbour sampling attacks exactly this: sample 25 neighbours then 10 of theirs, and one node's computation touches ~250 others regardless of graph size. **Cluster-GCN** takes the other route — partition the graph into dense clusters and train on one cluster at a time, so most edges stay inside a batch. Both are the same admission: on a graph of that size, "the batch" cannot be the graph.

#### Examples and non-examples: honest and dishonest molecular evaluations

**✅ Evaluation protocols that measure what you care about**

| Protocol | What it answers |
|---|---|
| Scaffold split | "Will this work on a chemical series the model has never seen?" |
| Time split (train on pre-2019 compounds, test on later ones) | "Will this work on the compounds we make next?" |
| Held-out assay or target | "Does this transfer to a new biological question?" |
| Repeated splits with reported standard deviation | "Is the 1.5-point gap real?" — Ch. 3's whole subject |
| Comparison against ECFP + gradient boosting | "Is the graph machinery adding anything at all?" |

**❌ Near-misses — protocols that look rigorous and inflate the number**

| Protocol | Why it lies | Typical inflation |
|---|---|---|
| Random split on a medicinal-chemistry dataset | Near-duplicate analogues of the same scaffold land on both sides | 10–20 points |
| Random split with 5-fold cross-validation | Five folds of the same leak. More folds, same bias, tighter error bars around the wrong number | 10–20 points |
| Reporting the best of many seeds | Selection on noise (Ch. 3) | However large the seed variance is |
| Tuning hyperparameters on the test set | The test set has become a training set | Unbounded |
| Comparing against a GNN baseline only | The strongest baseline is often not a GNN | Hides the case where the whole approach is unnecessary |

▸ **The boundary:** a split is honest when **the test set contains something the training set  does not** — a new scaffold, a later date, a different assay. Randomness alone guarantees none of that, because randomness has no idea what a scaffold is.

> **Common misconception.** *"A GNN beats gradient boosting on molecules because it learns the structure rather than using hand-crafted features."* On many standard molecular benchmarks it does not beat gradient boosting, and the reason is instructive: an ECFP fingerprint **is** WL colour refinement with hashed colours, so the "hand-crafted" features are computing very nearly what the GNN's message passing computes — with zero variance, no training instability, and no hyperparameters. **Run the fingerprint baseline first.** If the GNN cannot beat it, the GNN is not adding structure knowledge; it is adding a training loop.

> **Where the applications came from.** **AlphaFold2** (John Jumper, Demis Hassabis and colleagues at DeepMind) essentially solved single-chain protein structure prediction at the CASP14 assessment in late 2020, achieving accuracy comparable to experimental methods on most targets — a problem that had been open since Christian Anfinsen's 1972 Nobel lecture proposed that a protein's sequence determines its structure. Its Evoformer module is a graph-attention system running over pairs of residues. Hassabis and Jumper shared the 2024 Nobel Prize in Chemistry with **David Baker** for computational protein design and structure prediction. On the industrial side, **PinSage** — GraphSAGE deployed at Pinterest on a graph of roughly three billion nodes — was among the first demonstrations that graph neural networks could run at web scale at all, and it is the reason neighbour sampling is standard rather than exotic.

---

## Did you know?

- **The word "graph" was borrowed from chemistry — and has now been lent back.** James Joseph Sylvester introduced "graph" in the network sense in an 1878 note in *Nature*, explicitly by analogy with the structural diagrams chemists drew for molecules. A century and a half later, the main application of graph neural networks is predicting the properties of molecules. The word made a complete round trip.

- **Graph theory begins with a walking puzzle.** Leonhard Euler's 1736 paper on the Seven Bridges of Königsberg proved no route crosses all seven bridges exactly once. His decisive move was to declare distances and shapes irrelevant and keep only what-connects-to-what — the same abstraction that lets a GNN ignore how you drew the picture.

- **The graph Laplacian is an electrical engineering artifact from 1847.** Gustav Kirchhoff wrote it down analysing circuits and proved that a determinant of it counts the network's spanning trees. It is still sometimes called the Kirchhoff matrix.

- **The Weisfeiler–Lehman test is misspelled in nearly every paper that cites it.** Andrei Leman has stated that his name transliterates as "Leman," not "Lehman." Both spellings appear throughout the literature and refer to the same 1968 paper.

- **Boris Weisfeiler disappeared in Chile in 1985.** The co-author of the algorithm that bounds every message-passing GNN was a Penn State mathematician who vanished while hiking alone in the south of the country. Chilean judicial investigations later linked the case to the Colonia Dignidad settlement and Pinochet-era security services. It has never been resolved.

- **Raising $k$ in $k$-WL does not eventually solve graph isomorphism.** Cai, Fürer and Immerman constructed in 1992 pairs of graphs that defeat $k$-WL for every fixed $k$. So "just use higher-order GNNs" is not a road to completeness — it is a treadmill you pay $O(n^k)$ to run on.

- **Chebyshev polynomials are written $T_k$ because of a French spelling.** Chebyshev's surname has been transliterated from Cyrillic in dozens of ways; the $T$ comes from *Tchebychef*. He derived the polynomials in the 1850s while studying steam-engine linkages — asking how close a jointed mechanism could come to drawing a straight line.

- **A GCN layer was already a known tool for erasing detail.** It is Laplacian smoothing, which computer graphics had used for years to smooth bumpy 3-D meshes. Framing it that way is what made over-smoothing obvious in retrospect.

- **The standard fix for over-squashing uses the machinery that proved the Poincaré conjecture.** Graph rewiring by discrete Ricci flow adapts Richard Hamilton's 1982 tool, which Grigori Perelman used in 2002–2003 to settle a Millennium Prize problem — and then declined both the Fields Medal and the million dollars.

- **The most-used benchmark graph in the field is a record of a 1970s karate club falling out.** Wayne Zachary, an anthropologist, spent 1970–72 observing a university karate club that split in two after a dispute between the instructor and the administrator. His 34-node network became the standard toy dataset. Network scientists award a semi-serious prize, the "Zachary Karate Club Club," to the last person to have used it in a conference talk.

- **A transformer is exactly a GNN on a complete graph.** Which also explains, from the other direction, why transformers need positional encodings at all: attention is permutation-invariant by construction, so a bare transformer  cannot distinguish "the cat sat" from "sat cat the."

- **Adding one node fixes the distance problem in almost any graph.** A "virtual node" wired to everything puts every pair of nodes two hops apart, at a cost of $n$ extra edges. It is the cheapest trick in the chapter and often beats far more sophisticated rewiring schemes.

- **A 2010 cheminformatics fingerprint is secretly the Weisfeiler–Lehman algorithm.** ECFP hashes each atom's neighbourhood, then neighbourhoods-of-neighbourhoods, into bits — which is WL colour refinement with a hash function. Two communities invented the same procedure without knowing it, and the older one still wins a lot of benchmarks.

---

## Check for Understanding

**GNNs are the architecture you get by demanding permutation equivariance, which forces every layer into the aggregate-then-update message-passing form — and that form caps their expressivity at the 1-WL test, so they cannot count triangles; depth is limited from one side by over-smoothing (repeated averaging collapses all nodes to the dominant eigenvector) and from the other by over-squashing (an exponentially large receptive field compressed into a fixed vector), which is why graph transformers and hybrid local/global models exist.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What is the difference between invariant and equivariant, and why does a graph need both?** (Use the class-photograph test: "how many people?" versus "who is in red?")
2. **Why can't you just feed an adjacency matrix into a fully-connected network?** What exactly would the model learn instead of the graph?
3. **What does the graph Laplacian compute at each node,** in one sentence about neighbours and disagreement?
4. **Why does restricting a filter to a polynomial of degree $K$ automatically make it $K$-hop local?** (The answer is about what powers of the adjacency matrix count.)
5. **Describe a GCN layer as a sentence about neighbours,** without writing the formula. Then say what would go wrong if you removed the self-loop.
6. **Why must the aggregator ignore the order of its inputs, and why is concatenation therefore forbidden?**
7. **Why can't a message-passing GNN tell a triangular prism from $K_{3,3}$?** What information does the architecture simply never receive?
8. **Why sum rather than mean?** Give the three-neighbour counting example that mean fails.
9. **Explain over-smoothing and over-squashing to someone who has confused them,** and say which one pushes you toward more layers.
10. **Why is choosing sum versus mean readout a modelling decision rather than a detail?** Use molecular weight versus solubility.
11. **Why is a transformer a GNN on a complete graph,** and what does that tell you about why transformers need positional encodings?
12. **Why does building rotation equivariance into an architecture beat teaching it with rotated training data?**
13. **Why does a random train/test split inflate molecular benchmark scores,** and what is a scaffold?

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 30 — Double Descent, Grokking & the NTK](30-double-descent-grokking-ntk.md)
