# Physics-SR Framework

## Three-Stage Physics-Informed Symbolic Regression Framework v4.7

**Author:** Zhengze Zhang  
**Affiliation:** Department of Statistics, Columbia University  
**Advisors:** Prof. Tian Zheng, Dr. Kara Lamb  
**Date:** January 2026

---

## Overview

Physics-SR is a comprehensive framework for discovering interpretable mathematical equations from scientific data. It combines physics-informed preprocessing with state-of-the-art symbolic regression methods, featuring a novel **5-layer augmented feature library** and **dual-track model selection** to achieve superior equation recovery compared to baseline approaches.

### Key Innovations (v4.7)

- **5-Layer Augmented Library**: Physics-guided term generation with Layer 0 inverse terms from power-law symmetry detection
- **Dual-Track Selection**: Parallel PySR coefficient refinement and E-WSINDy sparse regression with adaptive selection
- **Three-Layer Uncertainty Quantification**: Structural, parametric, and predictive UQ via bootstrap
- **Intercept Fix**: Critical correction for non-zero mean data preventing catastrophic R² failures

### Target Applications

- **Atmospheric Microphysics**: Discovering warm rain process equations (autoconversion, accretion)
- **Physics Equation Recovery**: Recovering known physical laws from noisy experimental data
- **Scientific Discovery**: Finding interpretable relationships in high-dimensional scientific datasets

---

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Variable Selection & Preprocessing               │
│                              (10-30 seconds)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  1.1 Buckingham Pi Dimensional Analysis                                     │
│  1.2 PAN+SR Nonlinear Variable Screening (RF Permutation Importance)        │
│  1.3 Power-Law Symmetry Detection (Log-Log Regression) → Layer 0 trigger    │
│  1.4 Adaptive Interaction Discovery (Enumeration / TreeSHAP)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STAGE 2: Structure-Guided Discovery (v4.7 Dual-Track)          │
│                              (60-120 seconds)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  2.1 PySR Structure Exploration (Pareto front extraction)                   │
│  2.2 Structure Parsing (SymPy term extraction)                              │
│  2.3 5-Layer Augmented Library Construction:                                │
│      • Layer 0 [PowLaw]: Inverse terms from symmetry analysis (1/r, 1/r²)   │
│      • Layer 1 [PySR]: Exact terms from PySR Pareto front                   │
│      • Layer 2 [Var]: Variable substitution variants                        │
│      • Layer 3 [Poly]: Polynomial baseline terms                            │
│      • Layer 4 [Op]: Operator-guided simple terms                           │
│  2.4 Dual-Track Selection:                                                  │
│      • Track 1: PySR Coefficient Refinement (curve_fit)                     │
│      • Track 2: E-WSINDy Sparse Selection (fit_intercept=True)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: Validation & Uncertainty Quantification          │
│                              (20-40 seconds)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  3.1 Model Selection (K-Fold CV + EBIC)                                     │
│  3.2 Physics Verification (Dimensional consistency, physical bounds)        │
│  3.3 Three-Layer Bootstrap UQ (Structural, Parametric, Predictive)          │
│  3.4 Statistical Inference (Hypothesis testing, p-values)                   │
└─────────────────────────────────────────────────────────────────────────────┘

TOTAL RUNTIME: 90-180 seconds on Google Colab Pro
```

---

## Project Structure

```
Physics-Informed-Symbolic-Regression/
├── algorithms/                        # Core algorithm notebooks
│   ├── 00_Core.ipynb                 # DataClasses, utilities, constants
│   ├── 01_BuckinghamPi.ipynb         # Dimensional analysis
│   ├── 02_VariableScreening.ipynb    # RF permutation importance screening
│   ├── 03_SymmetryAnalysis.ipynb     # Power-law detection → Layer 0
│   ├── 04_InteractionDiscovery.ipynb # Pairwise interaction detection
│   ├── 05_FeatureLibrary.ipynb       # 5-layer augmented library builder
│   ├── 06_PySR.ipynb                 # PySR + structure parsing
│   ├── 07_EWSINDy_STLSQ.ipynb        # E-WSINDy sparse regression
│   ├── 08_AdaptiveLasso.ipynb        # Adaptive Lasso verification
│   ├── 09_ModelSelection.ipynb       # CV + EBIC model selection
│   ├── 10_PhysicsVerification.ipynb  # Physics constraint checking
│   ├── 11_UQ_Inference.ipynb         # Bootstrap UQ + inference
│   └── 12_Full_Pipeline.ipynb        # Complete integrated pipeline
│
├── benchmark/                         # Benchmark experiments
│   ├── DataGen.ipynb                 # AI Feynman test data generation
│   ├── Experiments.ipynb             # Experiment execution
│   ├── Analysis.ipynb                # Results visualization
│   ├── data/                         # Generated datasets
│   └── results/                      # Experiment outputs
│       ├── figures/                  # Plots
│       └── tables/                   # LaTeX tables
│
├── Physics_SR_Framework_v4.7_Complete.md  # Full algorithm specification
├── requirements.txt                  # Python dependencies
├── setup_colab.sh                    # Colab setup script
├── SETUP_GUIDE.md                    # Detailed setup instructions
└── README.md                         # This file
```

---

## Quick Start

### Option 1: Google Colab (Recommended)

1. **Open in Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Open notebook → GitHub
   - Enter: `https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression`
   - Select the notebook you want to run

2. **Setup Environment:**
   ```python
   # Run this cell first in any notebook
   !git clone https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression.git
   %cd Physics-Informed-Symbolic-Regression
   !bash setup_colab.sh
   ```

3. **Run Benchmark:**
   ```python
   # 1. Generate test data
   %run benchmark/DataGen.ipynb
   
   # 2. Run experiments
   %run benchmark/Experiments.ipynb
   
   # 3. Analyze results
   %run benchmark/Analysis.ipynb
   ```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression.git
cd Physics-Informed-Symbolic-Regression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
python -c "import pysr; pysr.install()"  # Install Julia backend

# Run notebooks
jupyter notebook
```

---

## Usage

### Basic Usage

```python
# Import the pipeline
%run algorithms/12_Full_Pipeline.ipynb

# Prepare user inputs with dimensional information
user_inputs = UserInputs(
    variable_dimensions={
        'q1': [0, 0, 1, 0],   # Charge dimension [A·s]
        'q2': [0, 0, 1, 0],   # Charge dimension
        'r': [0, 1, 0, 0],    # Length dimension [m]
    },
    target_dimensions=[1, 1, -2, 0],  # Force [kg·m/s²]
    physical_bounds={'target': {'min': None, 'max': None}}
)

# Run pipeline
pipeline = PhysicsSRPipeline()
results = pipeline.run(X, y, feature_names, user_inputs)

# Get discovered equation
print(f"Equation: {results['final_equation']}")
print(f"Test R²: {results['test_r2']:.4f}")
print(f"Selected method: {results['selected_method']}")
```

### Understanding the 5-Layer Library

```python
# Stage 1 power-law detection finds r^(-2) exponent
# This automatically triggers Layer 0 generation:

# Layer 0 [PowLaw]: 1/r, 1/r², q1*q2/r²  ← from symmetry analysis
# Layer 1 [PySR]:   terms from PySR Pareto front
# Layer 2 [Var]:    variable substitution variants
# Layer 3 [Poly]:   x1, x2, x1², x1*x2, ...
# Layer 4 [Op]:     sin(x1), exp(x2), ...
```

---

## Algorithm Details

### Stage 1: Variable Selection

| Component | Method | Purpose |
|-----------|--------|---------|
| 1.1 Buckingham Pi | Null space enumeration | Reduce to dimensionless groups |
| 1.2 Variable Screening | RF Permutation Importance | Filter irrelevant variables |
| 1.3 Symmetry Detection | Log-log regression | Detect power-law, estimate exponents |
| 1.4 Interaction Discovery | Enumeration / TreeSHAP | Find stable variable interactions |

**Why RF instead of iRF?** iRF's Random Intersection Trees (RIT) have combinatorial complexity for path extraction. Direct pairwise enumeration is 1500x faster with 100% recall, since E-WSINDy filters false positives anyway.

### Stage 2: Dual-Track Selection (v4.7)

```
                    ┌─────────────────────────────┐
                    │  5-Layer Augmented Library  │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
    ┌─────────────────────┐               ┌─────────────────────┐
    │  Track 1: PySR      │               │  Track 2: E-WSINDy  │
    │  Coefficient Refine │               │  Sparse Selection   │
    │  (curve_fit)        │               │  (fit_intercept=T)  │
    └─────────┬───────────┘               └─────────┬───────────┘
              │                                     │
              └───────────────┬─────────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Selection Logic:   │
                    │  if PySR R² ≥ 0.70: │
                    │    prefer PySR      │
                    │  else:              │
                    │    pick best R²     │
                    └─────────────────────┘
```

### Stage 3: Three-Layer UQ

| Layer | Type | Output |
|-------|------|--------|
| 1 | Structural UQ | Term inclusion probability across bootstrap |
| 2 | Parametric UQ | Coefficient confidence intervals |
| 3 | Predictive UQ | Prediction intervals for new data |

---

## Benchmark Equations

The benchmark uses AI Feynman equations covering different functional forms:

| Equation | Formula | Type | Challenge |
|----------|---------|------|-----------|
| Coulomb (I.12.2) | F = q₁q₂/(4πε₀r²) | Inverse-square | Requires Layer 0 inverse terms |
| Cosines (I.29.16) | y = √(x₁² + x₂² - 2x₁x₂cos(θ)) | Trigonometric | Nested sqrt + cos |
| Barometric (I.40.1) | P = P₀·exp(-mgz/kT) | Exponential | Requires exp operator |
| DotProduct (I.11.19) | y = x₁y₁ + x₂y₂ + x₃y₃ | Polynomial | High-dimensional interaction |

---

## Key References

- **Buckingham Pi**: Buckingham, E. (1914). On physically similar systems. *Physical Review*, 4(4), 345.
- **SINDy**: Brunton, S. L., et al. (2016). Discovering governing equations from data. *PNAS*, 113(15), 3932-3937.
- **Weak-SINDy**: Messenger, D. A., & Bortz, D. M. (2021). Weak SINDy for partial differential equations. *JCP*, 443, 110525.
- **PySR**: Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. arXiv:2305.01582.
- **AI Feynman**: Udrescu, S. M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*, 6(16), eaay2631.
- **EBIC**: Chen, J., & Chen, Z. (2008). Extended Bayesian information criteria for model selection. *Biometrika*, 95(3), 759-771.

---

## Citation

```bibtex
@software{zhang2026physicssr,
  author = {Zhang, Zhengze},
  title = {Physics-SR: Three-Stage Physics-Informed Symbolic Regression Framework},
  version = {4.7},
  year = {2026},
  url = {https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Prof. Tian Zheng (Columbia University) - Advisor
- Dr. Kara Lamb (Columbia University, LEAP Center) - Advisor
- The PySR and SINDy communities for foundational work

---

## Contact

**Zhengze Zhang**  
Department of Statistics, Columbia University  
Email: zz3239@columbia.edu
