# Physics Informed SR Framework

## Three-Stage Physics-Informed Symbolic Regression Framework

**Author:** Zhengze Zhang  
**Affiliation:** Department of Statistics, Columbia University  
**Advisors:** Prof. Tian Zheng, Dr. Kara Lamb  
**Date:** January 2026

---

## Overview

Physics-SR is a comprehensive framework for discovering interpretable mathematical equations from scientific data. It combines physics-informed preprocessing with state-of-the-art symbolic regression methods to achieve superior variable selection and equation recovery compared to baseline approaches.

### Key Features

- **Stage 1: Variable Selection** - Buckingham Pi dimensional analysis, PAN+SR screening, symmetry detection
- **Stage 2: Structure Discovery** - PySR genetic programming, E-WSINDy sparse regression, Adaptive Lasso
- **Stage 3: Validation & UQ** - Cross-validation model selection, physics verification, bootstrap uncertainty quantification

### Performance Highlights

| Metric | Physics-SR | PySR-Only | LASSO+PySR |
|--------|------------|-----------|------------|
| Variable Selection F1 | **0.90** | 0.60 | 0.75 |
| Test R² | **0.95** | 0.85 | 0.90 |
| Noise Robustness | **High** | Low | Medium |

---

## Project Structure

```
Physics-Informed-Symbolic-Regression/
├── algorithms/                    # Core algorithm notebooks
│   ├── 00_Core.ipynb             # DataClasses, utilities
│   ├── 01_BuckinghamPi.ipynb     # Dimensional analysis
│   ├── 02_VariableScreening.ipynb # PAN+SR screening
│   ├── 03_SymmetryAnalysis.ipynb # Power-law detection
│   ├── 04_InteractionDiscovery.ipynb # iRF interactions
│   ├── 05_FeatureLibrary.ipynb   # Feature expansion
│   ├── 06_PySR.ipynb             # Genetic programming
│   ├── 07_EWSINDy_STLSQ.ipynb    # Sparse regression
│   ├── 08_AdaptiveLasso.ipynb    # Adaptive Lasso
│   ├── 09_ModelSelection.ipynb   # CV + EBIC
│   ├── 10_PhysicsVerification.ipynb # Physics checks
│   ├── 11_UQ_Inference.ipynb     # Bootstrap UQ
│   └── 12_Full_Pipeline.ipynb    # Complete integration
│
├── benchmark/                     # Benchmark experiments
│   ├── DataGen.ipynb             # Test data generation
│   ├── Experiments.ipynb         # Experiment execution
│   ├── Analysis.ipynb            # Results visualization
│   ├── data/                     # Generated datasets
│   └── results/                  # Experiment results
│       ├── figures/              # Plots
│       └── tables/               # LaTeX tables
│
├── requirements.txt              # Python dependencies
├── setup_colab.sh               # Colab setup script
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
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
   - Run `benchmark/DataGen.ipynb` to generate test data
   - Run `benchmark/Experiments.ipynb` to execute experiments
   - Run `benchmark/Analysis.ipynb` to visualize results

### Option 2: Local Installation

1. **Clone Repository:**
   ```bash
   git clone https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression.git
   cd Physics-Informed-Symbolic-Regression
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -c "import pysr; pysr.install()"  # Install Julia backend
   ```

4. **Run Notebooks:**
   ```bash
   jupyter notebook
   ```

---

## Usage

### Basic Usage

```python
# Import the pipeline
%run algorithms/12_Full_Pipeline.ipynb

# Prepare user inputs
user_inputs = UserInputs(
    variable_dimensions={
        'x1': [0, 1, 0, 0],   # Length dimension
        'x2': [1, 0, 0, 0],   # Mass dimension
        'x3': [0, 0, -1, 0],  # Time^-1 dimension
    },
    target_dimensions=[1, 1, -2, 0],  # Force = kg*m/s^2
    physical_bounds={'target': {'min': 0, 'max': None}}
)

# Run pipeline
pipeline = PhysicsSRPipeline()
results = pipeline.run(X, y, feature_names, user_inputs)

# Get final equation
print(f"Discovered equation: {results['final_equation']}")
```

### Benchmark Experiments

```python
# 1. Generate test data
%run benchmark/DataGen.ipynb
generator = BenchmarkDataGenerator()
generator.generate_all_datasets()

# 2. Run experiments
%run benchmark/Experiments.ipynb
runner = ExperimentRunner()
results = runner.run_all_experiments()

# 3. Analyze results
%run benchmark/Analysis.ipynb
```

---

## Test Equations

The benchmark includes 4 physics equations of varying complexity:

| Equation | Formula | Type | Difficulty |
|----------|---------|------|------------|
| KK2000 | $y = 1350 \cdot q_c^{2.47} \cdot N_d^{-1.79}$ | Power-law | Medium |
| Newton | $F = G \cdot m_1 \cdot m_2 / r^2$ | Rational | Easy |
| Ideal Gas | $P = n \cdot R \cdot T / V$ | Rational | Easy |
| Damped Osc. | $x = A \cdot e^{-\gamma t} \cdot \cos(\omega t + \phi)$ | Nested | Hard |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{zhang2026physicssr,
  author = {Zhang, Zhengze},
  title = {Physics-SR: Three-Stage Physics-Informed Symbolic Regression Framework},
  year = {2026},
  url = {https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Prof. Tian Zheng (Columbia University) - Advisor
- Dr. Kara Lamb (Columbia University, LEAP Center) - Advisor
- The PySR and SINDy communities for their foundational work

---

## Contact

**Zhengze Zhang**  
Department of Statistics, Columbia University  
Email: zz3039@columbia.edu
