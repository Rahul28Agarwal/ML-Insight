# Demystifying Dimensionality Reduction: From PCA to UMAP with Real-World Cases

> A practical guide to understanding dimensionality reduction techniques and their real-world applications.


## 📖 Introduction

**Def**: Dimensionality reduction (DR) transforms high-dimensional data (e.g., 1000 features) into a lower-dimensional space (e.g., 3 features) while preserving critical patterns. 

**Analogy**: Like compressing a high-resolution photo into a smaller file without losing recognizability.
* **Key Vocabulary**
    - **Feature**: An individual measurable property or characteristic of a dataset.
    - **Dimension**: Each feature represents a dimension in the data space.

## 🧠 When to Use Dimensionality Reduction

- Training is slow due to many features

- Features are highly correlated (e.g., > 0.8)

- Model accuracy plateaus with more features

![alt text](dr_use.png)

## 🔥 Why It Matters

- **Curse of Dimensionality**: As feature count increases, data becomes sparse, models overfit, and computations become expensive.

- **Improved Generalization**: Simplifies learning, potentially leading to better accuracy and generalization.

- **Visualization**: Makes it feasible to plot and interpret high-dimensional data.
- **Noise Reduction**: Removes irrelevant/noisy features, improving signal-to-noise ratio.

## ⚙️ Main Approaches for dimensionality reduction. 

### 1. Feature Selection
- **Definition**: Choosing a subset of the original features (columns) that are most relevant to the task.
- **Goal**: Remove redundant, irrelevant, or noisy features while keeping the data interpretable.
- **Techniques**:
    - **Filter Methods**: Use statistical measures (e.g., variance threshold, correlation-based selection).
    - **Wrapper Methods**: Evaluate different feature subsets using model performance (e.g., recursive feature elimination).
    - **Embedded Methods**: Feature selection integrated into model training (e.g., LASSO, tree-based models).
- **Pros**: Simple, interpretable, fast.
- **Cons**: May miss important feature interactions.

### 2. Feature Extraction (Projection)
- **Definition:** Transforming the original features into a new set of features (components) that capture the most important information.
- **Goal:** Compress data into fewer dimensions, often by combining features in ways that maximize variance or preserve structure.
- **Techniques**:
    - **Linear Methods**:
        - **Principal Component Analysis (PCA)**: Projects data onto axes of maximum variance.
        - **Linear Discriminant Analysis (LDA)**: Maximizes class separability (supervised).
        - **Independent Component Analysis (ICA)**: Finds statistically independent components.
        - **Non-Negative Matrix Factorization (NMF)**: Factorizes data into non-negative components.
    - **Nonlinear Methods**:
        - **t-SNE**: Preserves local similarities for visualization.
        - **UMAP**: Balances local/global structure, faster than t-SNE.
        - **Isomap, LLE**: Preserve manifold structure.
        - **Autoencoders**: Neural networks learn compressed representations.
- **Pros**: Captures more structure, powerful.
- **Cons**: Less interpretable, can be complex.

```mermaid
flowchart LR
    A[Dimensionality Reduction]
    A --> B[Feature Selection]
    A --> C[Feature Extraction Projection]

    B --> B1[Filter Methods]
    B --> B2[Wrapper Methods]
    B --> B3[Embedded Methods]

    C --> C1[Linear Methods]
    C --> C2[Nonlinear Methods]

    B1 --> D1[Variance Threshold]
    B1 --> D2[Correlation-based]
    B2 --> D3[Recursive Feature Elimination]
    B3 --> D4[LASSO, Tree-based]

    C1 --> E1[PCA]
    C1 --> E2[Linear Discriminant Analysis]
    C2 --> E3[t-SNE]
    C2 --> E4[UMAP]
    C2 --> E5[Autoencoders]

```

## 🔍 Choosing the Right Technique

```mermaid
flowchart TD
A[High-Dimensional Data] --> B{Is Interpretability Important?}
B -- Yes --> C[Feature Selection]
B -- No --> D{Data Structure}
D -- Linear --> E[PCA, LDA]
D -- Nonlinear --> F{Goal}
F -- Visualization --> G[t-SNE, UMAP]
F -- Compression --> H[Autoencoders, LLE]
```

## 📊 Comparison Table

| Technique       | Type       | Mechanism / Goal                     | Strengths                                | Weaknesses                               |
|----------------|------------|--------------------------------------|-------------------------------------------|-------------------------------------------|
| **PCA** (Principal Component Analysis) | Linear     | Maximize variance via orthogonal projection | Fast, interpretable, unsupervised         | Not effective for nonlinear relationships |
| **LDA** (Linear Discriminant Analysis) | Linear     | Maximize class separability (supervised)    | Good for classification, interpretable    | Requires class labels, assumes linearity  |
| **ICA** (Independent Component Analysis) | Linear  | Extract statistically independent components | Reveals hidden factors in data            | Sensitive to noise, assumes independence  |
| **Kernel PCA** | Nonlinear  | Nonlinear extension of PCA via kernel trick | Captures complex structure                | Computationally expensive                 |
| **t-SNE** (t-Distributed Stochastic Neighbor Embedding) | Nonlinear | Preserve local similarities using probability distributions | Great for visualization, cluster discovery | Poor for global structure, slow on large datasets |
| **UMAP** (Uniform Manifold Approximation and Projection) | Nonlinear | Preserve both local and global structure   | Fast, scalable, preserves structure better than t-SNE | Sensitive to parameters, less interpretable |
| **Isomap** | Nonlinear  | Preserve geodesic (manifold) distances      | Captures global manifold structure        | Sensitive to noise, computationally heavy |
| **LLE** (Locally Linear Embedding) | Nonlinear  | Preserve local linear relationships         | Unfolds nonlinear manifolds effectively   | Sensitive to noise and parameters         |
| **Autoencoders** | Nonlinear  | Learn compressed representations via neural networks | Flexible, handles complex patterns, supports denoising | Requires large data, hard to interpret     |
| **NMF** (Non-negative Matrix Factorization) | Linear  | Factorize into non-negative components      | Interpretable, parts-based representation | Only for non-negative data                |
| **Random Projection** | Linear     | Use random matrices to reduce dimensionality | Extremely fast, preserves distances approximately | Loss of interpretability                  |
| **Laplacian Eigenmaps** | Nonlinear  | Preserve local neighborhood graph structure | Good for local clustering, spectral methods | Sensitive to graph construction choices   |
| **Diffusion Maps** | Nonlinear  | Use diffusion process to model manifold geometry | Robust to noise, captures connectivity    | Computationally intensive                 |
| **Sammon Mapping** | Nonlinear  | Preserve pairwise distances emphasizing small distances | Good local structure visualization        | Slow, sensitive to initialization         |
| **Feature Selection** (Filters, Wrappers, Embedded) | N/A        | Select relevant features using stats or models | Simple, improves interpretability         | May miss interactions, not always optimal |


## 🧪 Case Studies

### 1. 📷 Image Compression on Social Media

- **Problem**: High-res images consume storage/bandwidth

- **Solution**: PCA compresses pixel data, retaining only principal components

- **Impact**: Smaller files, faster uploads, preserved quality

### 2. 🚦 Smart City Traffic Management (Automotus)

- **Problem**: Video data from city sensors is huge

- **Solution**: PCA reduces data to meaningful traffic patterns

- **Impact**: 20% accuracy boost, 35% size reduction, lower labeling cost

## 📚 References

- [Encord: Dimensionality Reduction Techniques](https://encord.com/blog/dimentionality-reduction-techniques-machine-learning/)

- [Datacamp: Understanding Dimensionality Reduction](https://www.datacamp.com/tutorial/understanding-dimensionality-reduction)

---

*Authored by Rahul Aggarwal — Data Scientist passionate about turning high-dimensional chaos into insights.*