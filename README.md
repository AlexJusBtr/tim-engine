# TIM: Sinusoidal String Embedding & Fuzzy Search Engine

TIM (Trigonometric Inverted Matcher) is an experimental, GPU-accelerated fuzzy string search engine written in C# and OpenCL (`Cloo`). It is designed for fast, typo-tolerant lookup across large lists of domains, hashes, or symbolic strings without requiring pre-trained neural embeddings or expensive $O(L^2)$ edit-distance (Levenshtein) matrices.

## How It Works

Instead of comparing strings character-by-character at query time, TIM projects strings into compact (16-to-32-dimensional) unit vectors using **Random Fourier Features (RFF)** and **continuous positional phase encoding**:

1. **Sinusoidal Positional Encoding:**
   Each character at index `pos` is mapped into a $D$-dimensional vector:
   $$v_d = \sum_{\text{pos}} w_{\text{pos}} \cdot \sin\left(\omega_d \cdot c + \phi_d + \frac{\pi \cdot \text{pos}}{L_{\max}}\right)$$
   - **High-frequency character projection ($\omega_d \cdot c$):** Random base frequencies $\omega_d$ ensure that different characters produce nearly orthogonal contributions, making the vector sensitive to character identity.
   - **Low-frequency positional phase ($\pi \cdot \text{pos} / L_{\max}$):** Because the position offset is added outside the base frequency multiplier, shifting a character left or right by 1–2 indices (due to an insertion or deletion typo) only slightly perturbs the sine angle. As a result, similarity degrades smoothly across both character substitutions and positional shifts.

2. **OpenCL Batch Encoding:**
   During index construction (`BuildIndex`), large string datasets are flattened and dispatched to an OpenCL GPU kernel (`encodeKernel`) that computes and L2-normalizes all embedding vectors in parallel.

3. **Top-Dimension Inverted Index (Coarse Filtering):**
   For each normalized embedding vector, the engine extracts the top-$K$ dimensions by absolute magnitude and registers the record in an inverted bucket index. At query time, only records sharing a minimum number of dominant dimensions with the query vector are evaluated.

4. **Cosine Similarity Ranking:**
   Because all vectors are pre-normalized to unit length during encoding, candidate scoring requires only a fast $O(D)$ dot product on the CPU.

## Current Limitations & Tradeoffs

- **Low Dimensionality (16–32D):** Because strings are compressed into at most 32 floats, long strings suffer from superposition crowding (hash collisions in vector space). Increasing dimensions to 128–256 improves recall on longer strings at the cost of memory bandwidth.
- **Coarse Inverted Index Buckets:** With 32 total dimensions, inverted-index buckets are broad and act as a coarse filter rather than a true sublinear graph index (such as HNSW).
- **CPU Candidate Scoring:** Batch encoding is offloaded to the GPU via OpenCL, while candidate filtering and dot-product scoring currently execute on the CPU.

## Roadmap

- Move candidate dot-product scoring and top-$K$ reduction into a persistent GPU buffer so the dataset stays in VRAM across queries.
- Benchmark recall and latency directly against trigram ($n$-gram) inverted indexes and CPU/GPU brute-force Levenshtein distance.
- Experiment with $k$-mer / codon-level sinusoidal encoding for short DNA/RNA sequence fragment matching.
