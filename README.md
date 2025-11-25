# ndindex

**Automated Spectral Index Discovery for Remote Sensing**

ndindex discovers optimal spectral indices for binary classification of multispectral satellite imagery. It systematically explores normalized difference combinations (like NDVI) across all band pairs and identifies the minimal index set achieving your target accuracy.

## Why ndindex?

Traditional spectral indices like NDVI use fixed band combinations. But what if a different combination works better for your specific classification problem? ndindex answers this by:

- **Exploring all possibilities** — Tests every band pair combination, not just established indices
- **Finding minimal solutions** — Identifies the fewest indices needed for accurate classification
- **Producing deployment-ready equations** — Outputs coefficients that work directly on raw band values

**Example result:** For Kochia weed detection using Sentinel-2, ndindex found that just 4 spectral indices achieve 97% accuracy.

## Features

- 🔍 **Automatic band detection** — Finds spectral bands (b1, b2, ...) in your CSV
- 📊 **Comprehensive index space** — Generates degree 1 and 2 polynomial indices
- 🎯 **Dual selection methods** — Compares RFE and SelectKBest, picks the best
- 📈 **Visual diagnostics** — Accuracy curves, overfitting analysis, diminishing returns
- 📝 **Publication-ready output** — Equations with absorbed standardization

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ndindex.git
cd ndindex
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- pandas ≥ 1.5.0
- numpy ≥ 1.24.0
- scikit-learn ≥ 1.2.0
- matplotlib ≥ 3.5.0

## Quick Start

```bash
python ndindex.py
```

The interactive prompt will guide you through:

1. **Input CSV files** — Your spectral data with band columns and class labels
2. **Band selection** — Auto-detect or manually specify which bands to use
3. **Parameters** — Max indices to test, accuracy threshold, test split

### Input Data Format

Your CSV should have:
- Spectral band columns named `b1`, `b2`, `b3`, etc.
- A target column with exactly 2 classes (e.g., "Kochia" vs "Other")

```csv
b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,Type
0.0892,0.0756,0.0723,0.0614,0.1245,0.2847,0.3215,0.3156,0.3389,0.1845,Kochia
0.0521,0.0612,0.0589,0.0534,0.0987,0.2156,0.2534,0.2489,0.2612,0.1423,Other
...
```

## Output

ndindex generates two files in your output directory:

### `spectral_indices.txt`

Classification equations for each model size:

```
======================================================================
4-INDEX MODEL (Test Accuracy: 0.9700)
======================================================================
f(x) = -2.341562
       +8.234521 * ND_b4_b6
       -3.127845 * ND_b3_b8
       +2.891234 * ND_b4_b6_sq
       -1.456789 * ND_b2_b7_ND_b4_b9_prod

Decision Rule: f(x) > 0 => Kochia | f(x) <= 0 => Other
```

### `index_analysis.png`

Four-panel visualization:
- Test accuracy vs number of indices
- Overfitting gap (train-test difference)  
- Train vs test accuracy curves
- Marginal improvement per additional index

## How It Works

### Index Generation

ndindex builds a polynomial feature space from normalized differences:

**Degree 1 (like NDVI):**
```
ND_ij = (b_i - b_j) / (b_i + b_j + ε)
```

**Degree 2:**
```
(ND_ij)²  and  ND_ij × ND_kl
```

For 10 bands, this creates 1,080 candidate indices.

### Index Selection

Two methods are compared at each model size:

1. **RFE (Recursive Feature Elimination)** — Wrapper method using SVM coefficient magnitudes
2. **SelectKBest** — Filter method using ANOVA F-statistic

The method achieving higher test accuracy is selected.

### Standardization Absorption

Training uses standardized indices for numerical stability. The final equations absorb this standardization:

```
w̃_j = w_j / σ_j
w̃_0 = w_0 - Σ(w_j × μ_j / σ_j)
```

This lets you apply equations directly to raw band values without needing training statistics.

## Example Use Cases

- **Weed detection** — Identify invasive species in agricultural imagery
- **Land cover classification** — Distinguish vegetation types
- **Water quality** — Detect algal blooms or sediment
- **Crop health** — Monitor stress or disease

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
