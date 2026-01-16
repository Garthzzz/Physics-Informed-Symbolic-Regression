# Three-Stage Physics-Informed Symbolic Regression Framework

## Complete Algorithm Specification v4.7

**Author:** Zhengze Zhang  
**Affiliation:** Department of Statistics, Columbia University  
**Advisors:** Prof. Tian Zheng, Dr. Kara Lamb  
**Version:** 4.7 (Dual-Track Selection + Intercept Fix + 5-Layer Library)  
**Last Updated:** January 2026

---

## Changelog from v4.1 to v4.7

| Section | Change Type | Description |
|---------|-------------|-------------|
| Section 4.3 | **Enhanced** | 5-Layer Library Construction (new Layer 0: [PowLaw] for inverse terms) |
| Section 4.4 | **Major Redesign** | Dual-Track Selection: PySR Refinement vs E-WSINDy |
| Section 4.4 | **Critical Fix** | E-WSINDy now uses `fit_intercept=True` and stores intercept |
| Section 7 | **Updated** | New v4.7 constants for dual-track thresholds |
| Section 8 | **Added** | 8.13-8.15 Implementation notes for v4.7 features |
| DataClasses | **Enhanced** | Stage2Results has 5 new fields for dual-track results |

## Changelog from v4.0 to v4.1

| Section | Change Type | Description |
|---------|-------------|-------------|
| Section 4.1 | **Enhanced** | Optimized PySR parameters for 180s runtime budget |
| Section 7 | **Updated** | Added time budget allocation table |
| Section 8.9-8.12 | **New** | Computational optimization implementation notes |
| Section 9 | **New** | Complete Computational Optimization Guide |
| Overview | **Updated** | Realistic time estimates for Colab Pro |

## Changelog from v3.0 to v4.0

| Section | Change Type | Description |
|---------|-------------|-------------|
| Stage 2 | **Major Redesign** | Structure-Guided Feature Library Construction |
| 2.2 | **New** | Structure Parsing from PySR Pareto Front |
| 2.3 | **New** | Augmented Library Construction (4 Layers) |
| 2.4 | **Modified** | E-WSINDy now operates on augmented library |
| Section 6 | **Updated** | End-to-End pseudocode reflects new Stage 2 |
| Section 8 | **Added** | 8.6-8.8 New implementation notes for v4.0 |

---

## Table of Contents

1. [Overview Pipeline](#1-overview-pipeline)
2. [User-Defined Inputs (Prerequisites)](#2-user-defined-inputs-prerequisites)
3. [Stage 1: Variable Selection & Preprocessing](#3-stage-1-variable-selection--preprocessing)
4. [Stage 2: Structure-Guided Discovery](#4-stage-2-structure-guided-discovery)
5. [Stage 3: Validation & Uncertainty Quantification](#5-stage-3-validation--uncertainty-quantification)
6. [End-to-End Algorithm Pseudocode](#6-end-to-end-algorithm-pseudocode)
7. [Component Summary](#7-component-summary)
8. [Implementation Notes for Special Components](#8-implementation-notes-for-special-components)
9. [Computational Optimization Guide](#9-computational-optimization-guide)

---

## 1. Overview Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER-DEFINED INPUTS (Prerequisites)                  │
│                                                                             │
│  * Variable dimension dictionary: {var_name: [M, L, T, Theta]}              │
│  * Physical bounds constraints: {target: {min: 0, max: None}, ...}          │
│  * Variable name mapping: {column_name: physical_meaning}                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT DATA                                      │
│  * Raw variables: X = [x1, x2, ..., xn] with units                          │
│  * Target: y (e.g., dqr/dt)                                                 │
│  * Variable metadata: units, physical bounds, known constraints              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: VARIABLE SELECTION & PREPROCESSING               │
│                          (10-30 seconds on Colab Pro)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1.1 Buckingham Pi Dimensional Analysis                                     │
│  1.2 PAN+SR Nonlinear Variable Screening (RF Permutation Importance)        │
│  1.3 Power-Law Symmetry Detection (Log-Log Regression)                      │
│  1.4 Adaptive Interaction Discovery (TreeSHAP/Enumeration)                  │
│                                                                             │
│  Output: Reduced variable set + estimated exponents for Layer 0             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STAGE 2: STRUCTURE-GUIDED DISCOVERY (v4.7 Redesign)            │
│                         (60-120 seconds on Colab Pro)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2.1 PySR Structure Exploration (Mode-based: fast/standard/thorough)        │
│      * Run PySR with optimized parameters for 180s budget                   │
│      * Extract Pareto front of complexity-accuracy tradeoffs                │
│                                                                             │
│  2.2 Structure Parsing                                                      │
│      * Parse Pareto equations via SymPy                                     │
│      * Extract exact functional terms with evaluation functions             │
│      * Detect operators {sin, cos, exp, log, sqrt, ...}                     │
│                                                                             │
│  2.3 Augmented Library Construction (5 Layers - v4.7)                       │
│      * Layer 0: [PowLaw] Power-law guided inverse terms (NEW v4.1)          │
│      * Layer 1: [PySR] Exact terms from PySR Pareto front                   │
│      * Layer 2: [Var] Variant terms (variable substitution)                 │
│      * Layer 3: [Poly] Polynomial baseline (always included)                │
│      * Layer 4: [Op] Operator-guided simple terms                           │
│                                                                             │
│  2.4 Dual-Track Selection (NEW v4.7)                                        │
│      * Track 1: PySR Coefficient Refinement via curve_fit                   │
│      * Track 2: E-WSINDy Sparse Selection with fit_intercept=True           │
│      * Selection based on PySR_TRUST_THRESHOLD (0.70)                       │
│                                                                             │
│  2.5 Adaptive Lasso Verification (optional, time-budget dependent)          │
│                                                                             │
│  Output: Candidate equation with intercept + source attribution             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: VALIDATION & UNCERTAINTY QUANTIFICATION          │
│                          (20-40 seconds on Colab Pro)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  3.1 Model Selection                                                        │
│      * K-Fold Cross-Validation                                              │
│      * EBIC (Extended BIC) for high-dimensional selection                   │
│                                                                             │
│  3.2 Physics Verification                                                   │
│      * Dimensional consistency check (unit validation)                      │
│      * Physical bounds check (non-negativity, etc.)                         │
│                                                                             │
│  3.3 Uncertainty Quantification (Three-Layer)                               │
│      * Layer 1: Structural UQ (inclusion probability via bootstrap)         │
│      * Layer 2: Parametric UQ (coefficient CI via bootstrap)                │
│      * Layer 3: Predictive UQ (prediction intervals)                        │
│                                                                             │
│  3.4 Formal Statistical Inference                                           │
│      * Hypothesis testing for term significance                             │
│      * P-values and confidence levels                                       │
│                                                                             │
│  Output: Final equation with full uncertainty characterization              │
└─────────────────────────────────────────────────────────────────────────────┘

TOTAL RUNTIME BUDGET: 90-180 seconds on Google Colab Pro (2-4 cores)
```

---

## 2. User-Defined Inputs (Prerequisites)

Before running the framework, users must prepare the following inputs. These cannot be automatically inferred and require domain knowledge.

### 2.1 Variable Dimension Dictionary

**Purpose:** Required for Buckingham Pi analysis and dimensional consistency checking.

**Format:** Dictionary mapping variable names to their dimensional exponents [M, L, T, Theta] (Mass, Length, Time, Temperature).

**Example (Warm Rain Microphysics):**

```python
variable_dimensions = {
    # Mixing ratios (dimensionless mass fractions)
    'q_c': [0, 0, 0, 0],      # cloud water mixing ratio [kg/kg]
    'q_r': [0, 0, 0, 0],      # rain water mixing ratio [kg/kg]
    
    # Number concentrations
    'N_c': [0, -3, 0, 0],     # cloud droplet number [m^-3]
    'N_d': [0, -3, 0, 0],     # droplet number [m^-3]
    
    # Size parameters
    'r_c': [0, 1, 0, 0],      # cloud droplet radius [m]
    'r_r': [0, 1, 0, 0],      # rain drop radius [m]
    'r_eff': [0, 1, 0, 0],    # effective radius [m]
    
    # Densities and masses
    'rho_w': [1, -3, 0, 0],   # water density [kg/m^3]
    'rho_a': [1, -3, 0, 0],   # air density [kg/m^3]
    'LWC': [1, -3, 0, 0],     # liquid water content [kg/m^3]
}

target_dimensions = [0, 0, -1, 0]  # Rate: s^-1
```

### 2.2 Physical Bounds Constraints

**Purpose:** Ensure discovered equations respect physical constraints.

**Format:** Dictionary specifying min/max bounds for variables and target.

**Example:**

```python
physical_bounds = {
    'target': {'min': 0, 'max': None},      # Autoconversion rate >= 0
    'q_c': {'min': 0, 'max': 0.01},         # Mixing ratio in [0, 0.01]
    'q_r': {'min': 0, 'max': 0.01},
    'N_d': {'min': 1e6, 'max': 1e12},       # Realistic droplet concentrations
    'r_eff': {'min': 1e-6, 'max': 100e-6},  # 1-100 micrometers
}
```

### 2.3 Variable Name Mapping (Optional)

**Purpose:** Map data column names to standardized physical variable names.

**Example:**

```python
variable_mapping = {
    'cloud_water_mixing_ratio': 'q_c',
    'rain_water_mixing_ratio': 'q_r',
    'droplet_number_concentration': 'N_d',
    'effective_radius_meters': 'r_eff',
}
```

---

## 3. Stage 1: Variable Selection & Preprocessing

### 3.1 Buckingham Pi Dimensional Analysis

**Mathematical Foundation:**

The Buckingham Pi theorem states that for a physical equation involving $n$ variables with $k$ fundamental dimensions, the equation can be rewritten using $(n-k)$ dimensionless groups.

If a physical law relates $n$ dimensional variables:
$$f(q_1, q_2, \ldots, q_n) = 0$$

and these variables span $k$ independent dimensions (e.g., M, L, T, Θ), then the law can be expressed as:
$$F(\pi_1, \pi_2, \ldots, \pi_{n-k}) = 0$$

where each $\pi_i$ is a dimensionless group.

**Algorithm:**

```
Input: variable_dimensions: Dict[str, List[float]]
Output: pi_groups, pi_exponents, X_transformed

1. Construct dimensional matrix D ∈ R^(k×n)
   D[i,j] = exponent of dimension i for variable j

2. Compute rank k' = rank(D)
   Number of pi-groups = n - k'

3. Find null space basis vectors e such that D·e = 0
   - Enumerate integer solutions with |exp| <= max_exponent
   - Score by complexity: 10*(nonzero count) + sum(|exp|) + 0.1*max(|exp|)

4. Select (n-k') linearly independent vectors with lowest complexity

5. Form pi-groups: π_j = ∏_i q_i^{e_ij}

6. Transform data: X_transformed = compute_pi_values(X, pi_exponents)
```

**Implementation Class:** `BuckinghamPiAnalyzer` (01_BuckinghamPi.ipynb)

### 3.2 PAN+SR Nonlinear Variable Screening

**Problem with LARS:**

LARS selects features based on linear correlation:
$$\hat{j} = \arg\max_j |\text{corr}(X_j, r)|$$

This fails for nonlinear dependencies. For example, if $y = \sin(x)$ with symmetric $x \in [-\pi, \pi]$, then $\text{corr}(x, y) \approx 0$.

**Solution: Random Forest Permutation Importance**

Permutation importance measures how much the prediction error increases when a feature is randomly shuffled:

$$I_j = \frac{\text{MSE}_{\text{permuted}_j} - \text{MSE}_{\text{baseline}}}{\text{MSE}_{\text{baseline}}}$$

**Algorithm:**

```
Input: X, y, feature_names
Output: selected_features, importance_scores

1. Fit Random Forest with n_estimators=500, max_features='sqrt'
2. Compute permutation importance for each feature (n_permutations=10)
3. Normalize importance scores to sum to 1
4. Select features with normalized importance > threshold (default: 0.01)
```

**Implementation Class:** `PANSRVariableScreener` (02_VariableScreening.ipynb)

### 3.3 Power-Law Symmetry Detection

**Mathematical Foundation:**

If $y = C \cdot x_1^{\alpha_1} \cdot x_2^{\alpha_2} \cdots x_p^{\alpha_p}$, then taking logarithms:

$$\log(y) = \log(C) + \alpha_1 \log(x_1) + \alpha_2 \log(x_2) + \cdots + \alpha_p \log(x_p)$$

This is a linear regression problem in log-space. High $R^2$ indicates the relationship follows a power-law form.

**Algorithm (Simplified):**

```
Input: X, y, feature_names
Output: is_power_law, estimated_exponents, power_law_r2

1. Filter valid data (positive values only for log transform)
2. Apply log transformation: log(X), log(y)
3. Fit multivariate linear regression in log space
4. Compute R² to determine if power-law holds
5. If R² > threshold (default: 0.90):
   - Set is_power_law = True
   - Extract exponent estimates (including negative for inverse terms)
```

**Key v4.1 Enhancement:** Estimated exponents with negative values (e.g., r^-2) trigger Layer 0 inverse term generation in the feature library.

**Implementation Class:** `SymmetryAnalyzer` (03_SymmetryAnalysis.ipynb)

### 3.4 Adaptive Interaction Discovery

**Purpose:** Discover stable variable interactions to guide feature library construction.

**Algorithm:**

```
Input: X, y, feature_names
Output: stable_interactions, interaction_stability

1. Choose method based on dataset size:
   - If n_features <= 10: Use enumeration (all pairwise)
   - If n_features > 10: Use TreeSHAP interaction values

2. For enumeration method:
   - Generate all pairwise interaction candidates
   - Fit RF with interaction terms
   - Score by permutation importance
   - Apply stability selection (bootstrap)

3. For TreeSHAP method:
   - Fit GradientBoosting or XGBoost
   - Compute SHAP interaction values
   - Aggregate to get stable interactions

4. Filter interactions with stability > threshold (default: 0.5)
```

**Implementation Class:** `AdaptiveInteractionDiscoverer` (04_InteractionDiscovery.ipynb)

---

## 4. Stage 2: Structure-Guided Discovery

### 4.1 PySR Structure Exploration

**Purpose:** Search for symbolic expressions via evolutionary algorithms.

**v4.1 Mode-Based Configuration:**

| Mode | niterations | maxsize | timeout | Use Case |
|------|-------------|---------|---------|----------|
| fast | 20 | 18 | 60s | Quick prototyping |
| standard | 40 | 20 | 100s | **Default** |
| thorough | 80 | 25 | 150s | Final runs |

**Algorithm:**

```
Input: X, y, feature_names, mode='standard'
Output: best_equation, pareto_front, elapsed_time

1. Configure PySR with mode-based parameters
2. Set operators: {+, -, *, /, ^, sin, cos, exp, log, sqrt}
3. Run symbolic regression with timeout
4. Extract Pareto front (complexity vs accuracy)
5. Select best equation by loss
```

**Implementation Class:** `PySRDiscoverer` (06_PySR.ipynb)

### 4.2 Structure Parsing

**Purpose:** Extract exact functional terms from PySR equations for library construction.

**Algorithm:**

```
Input: pareto_equations, feature_names, X
Output: unique_terms, detected_operators, term_map

1. For each equation in Pareto front:
   a. Parse via SymPy: sympify(equation_str)
   b. Expand and extract additive terms
   c. For each term:
      - Create evaluation function via lambdify
      - Test numerical stability
      - Store (expr, name, eval_func)

2. Detect operators used: {'sin', 'cos', 'exp', 'log', 'sqrt', ...}

3. Deduplicate terms by symbolic equivalence

4. Create term_map: term -> source_equation
```

**Implementation Class:** `StructureParser` (06_PySR.ipynb)

### 4.3 Augmented Library Construction (5 Layers - v4.7)

**Purpose:** Build comprehensive feature library combining physics-guided terms with PySR discoveries.

**v4.7 Five-Layer Architecture:**

| Layer | Tag | Priority | Description |
|-------|-----|----------|-------------|
| 0 | `[PowLaw]` | **Highest** | Power-law inverse terms from Stage 1 symmetry |
| 1 | `[PySR]` | High | Exact terms from PySR Pareto front |
| 2 | `[Var]` | Medium | Variant terms via variable substitution |
| 3 | `[Poly]` | Baseline | Polynomial terms (always included) |
| 4 | `[Op]` | Safety net | Operator-guided simple terms |

**Layer 0 Detail (NEW v4.1):**

When Stage 1 symmetry analysis detects negative exponents (e.g., r^-2 in Coulomb's law), Layer 0 automatically adds inverse terms that standard polynomial libraries lack:

```python
# If estimated_exponents['r'] = -2.0
# Layer 0 adds: 1/r, 1/r^2
```

This is **critical** for equations like:
- Coulomb's law: F = k·q₁·q₂/r²
- Gravitational force: F = G·m₁·m₂/r²
- Barometric formula: P = P₀·exp(-mgh/kT)

**Algorithm:**

```
Input: X, feature_names, parsed_terms, detected_operators, 
       pysr_r2, estimated_exponents
Output: Phi_aug, library_names, library_info

1. Layer 0: Power-Law Guided Terms
   For each (var, exp) in estimated_exponents:
       If exp < 0:
           Add inverse terms: 1/var, 1/var^2, ...

2. Layer 1: PySR Exact Terms
   For each (expr, name, func) in parsed_terms:
       Evaluate func(X) and add to library
       Tag as [PySR]

3. Layer 2: Variant Terms (if pysr_r2 < 0.95)
   For each PySR term:
       Generate variants by variable substitution
       Tag as [Var]

4. Layer 3: Polynomial Baseline
   Generate polynomial terms up to max_degree
   Include: x_i, x_i^2, x_i*x_j, ...
   Tag as [Poly]

5. Layer 4: Operator-Guided Terms
   For each operator in detected_operators:
       Generate simple applications: sin(x_i), exp(x_i), ...
   Tag as [Op]

6. Concatenate all layers: Phi_aug = [L0 | L1 | L2 | L3 | L4]

7. Remove duplicate/constant/NaN columns
```

**Implementation Class:** `AugmentedLibraryBuilder` (05_FeatureLibrary.ipynb)

### 4.4 Dual-Track Selection (NEW v4.7)

**Purpose:** Choose between refined PySR equation and E-WSINDy sparse selection based on PySR quality.

**v4.7 Dual-Track Architecture:**

```
                    ┌─────────────────────────────────────┐
                    │         PySR Equation               │
                    │         (R² = pysr_r2)              │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │                                     │
                    ▼                                     ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │   Track 1: Refine     │           │   Track 2: E-WSINDy   │
        │   PySR Coefficients   │           │   Sparse Selection    │
        │   via curve_fit       │           │   with intercept      │
        └───────────┬───────────┘           └───────────┬───────────┘
                    │                                   │
                    │ pysr_refined_r2                   │ ewsindy_r2
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         Selection Logic             │
                    │  (based on PYSR_TRUST_THRESHOLD)    │
                    └─────────────────────────────────────┘
```

**Selection Logic:**

```python
PYSR_TRUST_THRESHOLD = 0.70

if pysr_r2 >= PYSR_TRUST_THRESHOLD and curve_fit_success:
    # HIGH TRUST: Compare refined PySR vs E-WSINDy
    if pysr_refined_r2 >= ewsindy_r2 * 0.95:  # Allow 5% margin
        final_method = 'pysr_refined'
    else:
        final_method = 'ewsindy'
else:
    # LOW TRUST or curve_fit failed
    if curve_fit_success and pysr_refined_r2 > ewsindy_r2:
        final_method = 'pysr_refined'
    else:
        final_method = 'ewsindy'
```

**Track 1: PySR Coefficient Refinement**

```
Input: pysr_equation, X, y, feature_names
Output: refined_equation, r2, success

1. Parse equation and extract numeric constants
2. Create parametric function: replace constants with parameters
3. Use scipy.optimize.curve_fit to optimize parameters
4. Compute R² of refined equation
5. Return refined equation string with optimized coefficients
```

**Track 2: E-WSINDy with Intercept (v4.7 Critical Fix)**

```
Input: Phi_aug, y, library_names
Output: coefficients, support, intercept, r_squared

1. Identify term sources by tag: [PySR], [PowLaw], [Poly], [Op]
2. Initial Ridge fit with fit_intercept=True
3. Adaptive term selection based on pysr_r2:
   - If pysr_r2 >= 0.95: Select top MAX_PYSR_TERMS PySR terms
   - Else: Select proportional to pysr_r2, fill with Poly terms
4. Final Ridge fit with fit_intercept=True
5. Store intercept separately (CRITICAL for predictions)
6. Analyze selection sources
```

**v4.7 Critical Fix: Intercept Handling**

```python
# CORRECT (v4.7):
ridge_final = Ridge(alpha=1e-8, fit_intercept=True)
ridge_final.fit(X_selected, y)
coefficients = ridge_final.coef_
intercept = ridge_final.intercept_  # MUST store this!

# Predictions:
y_pred = Phi @ coefficients + intercept  # MUST add intercept!
```

**Implementation Classes:**
- `refine_pysr_coefficients()` (12_Full_Pipeline.ipynb)
- `EWSINDySTLSQ` (07_EWSINDy_STLSQ.ipynb)

### 4.5 Adaptive Lasso Verification (Optional)

**Purpose:** Provide secondary verification using Adaptive Lasso's oracle property.

**Algorithm:**

```
Input: Phi_aug, y, library_names
Output: alasso_coefficients, alasso_support, alasso_r2

1. Initial OLS/Ridge fit to get γ weights
2. Adaptive Lasso: minimize ||y - Φξ||² + λ Σ w_j|ξ_j|
   where w_j = 1/|ξ_j^OLS|^γ
3. Cross-validate to select λ
4. Extract support and coefficients
```

**Conditional Execution:** Skip if time budget is tight (checked via `TimeBudgetManager`).

**Implementation Class:** `AdaptiveLassoSelector` (08_AdaptiveLasso.ipynb)

---

## 5. Stage 3: Validation & Uncertainty Quantification

### 5.1 Model Selection

**K-Fold Cross-Validation:**

```
1. Split data into K folds (default: K=5)
2. For each fold:
   a. Train on K-1 folds
   b. Evaluate on held-out fold
3. Report mean and std of CV scores
```

**Extended BIC (EBIC):**

$$\text{EBIC}_\gamma = n \log(\text{RSS}/n) + k \log(n) + 2\gamma k \log(p)$$

where:
- n = sample size
- k = number of selected features
- p = total features in library
- γ = complexity penalty (default: 0.5)

**Implementation Class:** `ModelSelector` (09_ModelSelection.ipynb)

### 5.2 Physics Verification

**Dimensional Consistency Check:**

```
For each selected term in equation:
    1. Parse term structure
    2. Compute dimensional exponents
    3. Verify: dim(term) = dim(target) within tolerance
```

**Physical Bounds Check:**

```
For predictions y_pred:
    1. Check y_pred >= bounds['target']['min']
    2. Check y_pred <= bounds['target']['max']
    3. Report fraction of violations
```

**Implementation Class:** `PhysicsVerifier` (10_PhysicsVerification.ipynb)

### 5.3 Three-Layer Uncertainty Quantification

**Layer 1: Structural UQ (Inclusion Probability)**

```
For b = 1 to B bootstrap samples:
    1. Resample (X_b, y_b) with replacement
    2. Run sparse regression
    3. Record which terms are selected

Inclusion probability = (# times selected) / B
```

**Layer 2: Parametric UQ (Coefficient CI)**

```
For b = 1 to B bootstrap samples:
    1. Resample with replacement
    2. Fit model with fixed support
    3. Record coefficient estimates

95% CI = [2.5th percentile, 97.5th percentile]
```

**Layer 3: Predictive UQ (Prediction Intervals)**

```
For new point x*:
    1. Use bootstrap coefficient samples
    2. Generate B predictions
    3. Compute prediction interval from distribution
```

**Adaptive Bootstrap Count (v4.1):**

```python
remaining_time = budget - elapsed
n_bootstrap = max(50, min(int(remaining_time * 5), 200))
```

**Implementation Class:** `BootstrapUQ` (11_UQ_Inference.ipynb)

### 5.4 Formal Statistical Inference

**Hypothesis Testing:**

For each coefficient ξ_j:
- H₀: ξ_j = 0
- H₁: ξ_j ≠ 0

Test statistic from bootstrap distribution:
$$z_j = \frac{\hat{\xi}_j}{\text{SE}_{\text{bootstrap}}(\hat{\xi}_j)}$$

P-value from standard normal approximation.

---

## 6. End-to-End Algorithm Pseudocode

```python
def physics_sr_pipeline(X, y, feature_names, user_inputs, max_time=180):
    """
    Three-Stage Physics-Informed Symbolic Regression (v4.7)
    """
    start_time = time.time()
    budget_manager = TimeBudgetManager(max_time)
    
    # =========================================================================
    # STAGE 1: Variable Selection & Preprocessing (~10-30s)
    # =========================================================================
    
    # 1.1 Buckingham Pi Analysis
    if user_inputs.variable_dimensions:
        pi_analyzer = BuckinghamPiAnalyzer(max_exponent=4)
        pi_result = pi_analyzer.analyze(user_inputs.variable_dimensions)
    
    # 1.2 PAN+SR Variable Screening
    screener = PANSRVariableScreener(importance_threshold=0.01)
    screening_result = screener.screen(X, y, feature_names)
    
    # 1.3 Power-Law Symmetry Detection
    symmetry_analyzer = SymmetryAnalyzer(r2_threshold=0.90)
    symmetry_result = symmetry_analyzer.analyze(X, y, feature_names)
    estimated_exponents = symmetry_result.get('estimated_exponents', {})
    
    # 1.4 Adaptive Interaction Discovery
    interaction_discoverer = AdaptiveInteractionDiscoverer(stability_threshold=0.5)
    interaction_result = interaction_discoverer.discover(X, y, feature_names)
    
    budget_manager.record_stage('stage1', time.time() - start_time)
    
    # =========================================================================
    # STAGE 2: Structure-Guided Discovery (~60-120s)
    # =========================================================================
    
    # 2.1 PySR Structure Exploration
    pysr_timeout = budget_manager.allocate_pysr_time()
    pysr_discoverer = PySRDiscoverer(mode='standard', timeout=pysr_timeout)
    pysr_result = pysr_discoverer.discover(X, y, feature_names)
    pysr_r2 = pysr_result.get('best_r2', 0.0)
    pysr_eq = pysr_result.get('best_equation', '')
    
    # 2.2 Structure Parsing
    parser = StructureParser()
    pareto_equations = pysr_discoverer.get_pareto_equations()
    parsed_terms, detected_operators, term_map = parser.parse_pareto_equations(
        pareto_equations, feature_names, X
    )
    
    # 2.3 Augmented Library Construction (5 Layers)
    library_builder = AugmentedLibraryBuilder(max_poly_degree=3)
    Phi_aug, library_names, library_info = library_builder.build(
        X, feature_names,
        parsed_terms=parsed_terms,
        detected_operators=detected_operators,
        pysr_r2=pysr_r2,
        estimated_exponents=estimated_exponents  # For Layer 0
    )
    
    # 2.4 Dual-Track Selection (v4.7)
    
    # Track 1: Refine PySR coefficients
    if pysr_eq and pysr_r2 > 0:
        pysr_refined_eq, pysr_refined_r2, curve_fit_success = refine_pysr_coefficients(
            pysr_eq, X, y, feature_names
        )
    else:
        pysr_refined_eq, pysr_refined_r2, curve_fit_success = None, 0.0, False
    
    # Track 2: E-WSINDy with intercept
    ewsindy = EWSINDySTLSQ(threshold=0.1)
    ewsindy._pysr_r2 = pysr_r2  # Pass for adaptive selection
    ewsindy_result = ewsindy.fit(Phi_aug, y, library_names=library_names)
    ewsindy_r2 = ewsindy_result['r_squared']
    intercept = ewsindy_result['intercept']  # CRITICAL: Store intercept!
    
    # Selection logic
    PYSR_TRUST_THRESHOLD = 0.70
    if pysr_r2 >= PYSR_TRUST_THRESHOLD and curve_fit_success:
        if pysr_refined_r2 >= ewsindy_r2 * 0.95:
            final_method = 'pysr_refined'
            final_r2 = pysr_refined_r2
        else:
            final_method = 'ewsindy'
            final_r2 = ewsindy_r2
    else:
        if curve_fit_success and pysr_refined_r2 > ewsindy_r2:
            final_method = 'pysr_refined'
            final_r2 = pysr_refined_r2
        else:
            final_method = 'ewsindy'
            final_r2 = ewsindy_r2
    
    # 2.5 Adaptive Lasso (optional)
    if not budget_manager.should_skip_optional(min_required=20):
        alasso = AdaptiveLassoSelector(gamma=1.0)
        alasso_result = alasso.fit(Phi_aug, y, library_names)
    
    budget_manager.record_stage('stage2', time.time() - start_time - budget_manager.elapsed['stage1'])
    
    # =========================================================================
    # STAGE 3: Validation & UQ (~15-45s)
    # =========================================================================
    
    support = ewsindy_result['support']
    coefficients = ewsindy_result['coefficients']
    
    # 3.1 Model Selection
    model_selector = ModelSelector()
    cv_scores = model_selector.cross_validate(Phi_aug[:, support], y, cv=5)
    ebic_score = model_selector.compute_ebic(Phi_aug[:, support], y, gamma=0.5)
    
    # 3.2 Physics Verification
    verifier = PhysicsVerifier()
    dim_check = verifier.check_dimensions(
        coefficients, support, library_names, user_inputs.variable_dimensions
    )
    bounds_check = verifier.check_bounds(
        Phi_aug[:, support] @ coefficients[support] + intercept,  # Add intercept!
        user_inputs.physical_bounds
    )
    
    # 3.3 Bootstrap UQ (adaptive count)
    n_bootstrap = budget_manager.allocate_bootstrap_count()
    uq = BootstrapUQ(n_bootstrap=n_bootstrap)
    bootstrap_results = uq.run(Phi_aug[:, support], y)
    
    # 3.4 Statistical Inference
    hypothesis_tests = uq.compute_hypothesis_tests(bootstrap_results)
    
    budget_manager.record_stage('stage3', time.time() - start_time - 
                                 budget_manager.elapsed['stage1'] - 
                                 budget_manager.elapsed['stage2'])
    
    # =========================================================================
    # Compile Results
    # =========================================================================
    
    return {
        'equation': format_equation(coefficients, support, library_names, intercept),
        'coefficients': coefficients[support],
        'intercept': intercept,  # v4.7: Always include intercept
        'support': support,
        'library_names': [library_names[i] for i in np.where(support)[0]],
        'r_squared': final_r2,
        'final_method': final_method,  # v4.7: Track which method was selected
        'cv_scores': cv_scores,
        'ebic_score': ebic_score,
        'bootstrap_results': bootstrap_results,
        'hypothesis_tests': hypothesis_tests,
        'physics_verification': {
            'dimensional_check': dim_check,
            'bounds_check': bounds_check
        },
        'timing': budget_manager.get_report(),
        'library_info': library_info,
        'selection_analysis': ewsindy_result.get('selection_analysis', {})
    }
```

---

## 7. Component Summary

### 7.1 Global Configuration Constants (v4.7)

```python
# Stage 1 defaults
DEFAULT_MAX_EXPONENT = 4              # Buckingham pi search range
DEFAULT_IMPORTANCE_THRESHOLD = 0.01   # Variable screening threshold
DEFAULT_POWERLAW_R2_THRESHOLD = 0.9   # Power-law detection R² threshold
DEFAULT_STABILITY_THRESHOLD = 0.5     # Interaction stability threshold

# Stage 2 defaults
DEFAULT_MAX_POLY_DEGREE = 3           # Feature library polynomial degree
DEFAULT_STLSQ_THRESHOLD = 0.1         # STLSQ sparsity threshold
DEFAULT_STLSQ_MAX_ITER = 20           # STLSQ maximum iterations
DEFAULT_ALASSO_GAMMA = 1.0            # Adaptive Lasso gamma parameter

# Stage 3 defaults
DEFAULT_CV_FOLDS = 5                  # Cross-validation folds
DEFAULT_EBIC_GAMMA = 0.5              # EBIC gamma parameter
DEFAULT_N_BOOTSTRAP = 200             # Number of bootstrap samples
DEFAULT_CONFIDENCE_LEVEL = 0.95       # Confidence interval level

# v4.1 Computational Optimization Constants
DEFAULT_RUNTIME_BUDGET = 180          # Total runtime budget (seconds)
DEFAULT_PYSR_TIMEOUT = 100            # PySR timeout (seconds)
DEFAULT_PROCS = 2                     # Number of parallel processes

# v4.7 Dual-Track Selection Constants (NEW)
DEFAULT_PYSR_TRUST_THRESHOLD = 0.70   # Above this, trust PySR structure
DEFAULT_PYSR_SKIP_THRESHOLD = 0.95    # Above this, HIGH TRUST mode
DEFAULT_MAX_PYSR_TERMS = 12           # Cap on PySR terms in HIGH TRUST
DEFAULT_MAX_TOTAL_TERMS = 15          # Cap on total selected terms
```

### 7.2 DataClasses

**Stage2Results (v4.7 Enhanced):**

```python
@dataclass
class Stage2Results:
    # PySR results
    pysr_equations: Optional[List[str]] = None
    pysr_pareto: Optional[pd.DataFrame] = None
    best_pysr_equation: Optional[str] = None
    best_pysr_r2: Optional[float] = None
    pysr_elapsed_time: Optional[float] = None
    pysr_model: Optional[Any] = None
    
    # Structure Parsing results
    parsed_terms: Optional[List[Tuple]] = None
    detected_operators: Optional[set] = None
    term_to_equation_map: Optional[Dict] = None
    
    # Augmented Library
    augmented_library: Optional[np.ndarray] = None
    library_names: Optional[List[str]] = None
    library_info: Optional[Dict] = None
    library_builder: Optional[Any] = None
    
    # E-WSINDy results
    ewsindy_coefficients: Optional[np.ndarray] = None
    ewsindy_support: Optional[np.ndarray] = None
    ewsindy_equation: Optional[str] = None
    ewsindy_r2: Optional[float] = None
    selection_analysis: Optional[Dict] = None
    
    # v4.7 NEW: Intercept and Dual-Track fields
    ewsindy_intercept: Optional[float] = None        # CRITICAL for predictions
    final_method: Optional[str] = None               # 'pysr_refined' or 'ewsindy'
    pysr_refined_equation: Optional[str] = None
    pysr_refined_r2: Optional[float] = None
    curve_fit_success: Optional[bool] = None
    
    # Adaptive Lasso results (optional)
    alasso_coefficients: Optional[np.ndarray] = None
    alasso_support: Optional[np.ndarray] = None
    alasso_r2: Optional[float] = None
    
    # Timing
    timing: Optional[Dict[str, float]] = None
```

### 7.3 Time Budget Allocation

| Stage | Component | Typical Time | Budget % |
|-------|-----------|--------------|----------|
| 1 | Buckingham Pi | 0.1-0.5s | 0.3% |
| 1 | Variable Screening | 5-10s | 5% |
| 1 | Symmetry Analysis | 0.5-2s | 1% |
| 1 | Interaction Discovery | 5-15s | 8% |
| 2 | PySR | 60-100s | **55%** |
| 2 | Structure Parsing | 1-2s | 1% |
| 2 | Library Construction | 2-5s | 2% |
| 2 | E-WSINDy | 3-8s | 4% |
| 2 | Adaptive Lasso | 3-5s | 2% |
| 3 | Model Selection | 2-5s | 2% |
| 3 | Physics Verification | 0.5-1s | 0.5% |
| 3 | Bootstrap UQ | 10-30s | 15% |
| 3 | Statistical Inference | 1-2s | 1% |
| **Total** | | **90-180s** | **100%** |

---

## 8. Implementation Notes for Special Components

### 8.1 Buckingham Pi: Null Space Computation

The null space of the dimensional matrix can be found via SVD:

```python
U, S, Vh = np.linalg.svd(D)
null_space = Vh[rank(D):, :].T
```

However, we prefer integer solutions for interpretability. The algorithm enumerates all integer vectors with bounded exponents and checks if $D \cdot e = 0$.

### 8.2 Symmetry Analysis: Simplified Approach

The framework uses **simplified power-law detection** rather than full Hessian-based separability tests:

- Skip noise-sensitive Hessian computation
- Focus on robust log-log regression
- Extract exponent estimates for PySR initialization

This trade-off prioritizes robustness over theoretical completeness.

### 8.3 PySR Configuration for Colab Pro

Optimized parameters for 180s budget on 2-core Colab Pro:

```python
pysr_params = {
    'niterations': 40,
    'maxsize': 20,
    'maxdepth': 10,
    'populations': 15,
    'population_size': 33,
    'ncycles_per_iteration': 400,
    'timeout_in_seconds': 100,
    'procs': 2,
    'precision': 32,  # Float32 for memory efficiency
    'turbo': True
}
```

### 8.4 E-WSINDy: Weak Form Theory

**Strong form** (noise-sensitive):
$$\frac{\partial q}{\partial t} = f(q, \nabla q, \nabla^2 q)$$

**Weak form** (noise-robust): Multiply by test function $\psi$ and integrate by parts:
$$\int \psi \cdot \nabla^2 q \, dx = -\int \nabla\psi \cdot \nabla q \, dx + \text{boundary terms}$$

**Result:** Derivatives transferred from noisy data $q$ to smooth test function $\psi$.

Typical improvement: **50-1000x noise robustness** over finite differences.

### 8.5 Adaptive Lasso: Oracle Property

The Adaptive Lasso achieves the oracle property under certain conditions:

$$\hat{\xi}^{AL} = \arg\min_\xi \left\{ ||y - \Phi\xi||^2 + \lambda \sum_j w_j |\xi_j| \right\}$$

where $w_j = 1/|\hat{\xi}_j^{OLS}|^\gamma$.

With $\gamma > 0$, large coefficients are penalized less, leading to consistent variable selection.

### 8.6 Structure Parsing: SymPy Integration

```python
from sympy import sympify, expand, Add, symbols

def parse_equation(eq_str, feature_names):
    # Create symbols
    syms = {name: symbols(name) for name in feature_names}
    
    # Parse and expand
    expr = sympify(eq_str, locals=syms)
    expanded = expand(expr)
    
    # Extract additive terms
    if isinstance(expanded, Add):
        terms = list(expanded.args)
    else:
        terms = [expanded]
    
    return terms
```

### 8.7 5-Layer Library: Layer 0 Power-Law Terms (v4.1)

```python
def build_layer0_powerlaw(X, feature_names, estimated_exponents):
    """
    Build Layer 0: Power-law guided inverse terms.
    
    When symmetry analysis detects negative exponents (e.g., r^-2),
    add inverse terms that polynomial libraries lack.
    """
    terms = []
    names = []
    
    for i, name in enumerate(feature_names):
        if name in estimated_exponents:
            exp = estimated_exponents[name]
            if exp < -0.5:  # Significant negative exponent
                x = X[:, i]
                
                # Add 1/x
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv1 = np.where(np.abs(x) > 1e-10, 1.0 / x, 0.0)
                terms.append(inv1)
                names.append(f'[PowLaw] 1/{name}')
                
                # Add 1/x^2 if exp < -1.5
                if exp < -1.5:
                    inv2 = np.where(np.abs(x) > 1e-10, 1.0 / x**2, 0.0)
                    terms.append(inv2)
                    names.append(f'[PowLaw] 1/{name}^2')
    
    return terms, names
```

### 8.8 Source Attribution Analysis

```python
def analyze_selection_sources(support, library_names):
    """Analyze where selected terms originated."""
    sources = {
        'from_powerlaw': 0,  # Layer 0
        'from_pysr': 0,      # Layer 1
        'from_variant': 0,   # Layer 2
        'from_poly': 0,      # Layer 3
        'from_op': 0,        # Layer 4
        'from_unknown': 0,
        'total_selected': 0
    }
    
    for idx in np.where(support)[0]:
        name = library_names[idx]
        if name.startswith('[PowLaw]'):
            sources['from_powerlaw'] += 1
        elif name.startswith('[PySR]'):
            sources['from_pysr'] += 1
        elif name.startswith('[Var]'):
            sources['from_variant'] += 1
        elif name.startswith('[Poly]'):
            sources['from_poly'] += 1
        elif name.startswith('[Op]'):
            sources['from_op'] += 1
        else:
            sources['from_unknown'] += 1
        sources['total_selected'] += 1
    
    return sources
```

### 8.9 TimeBudgetManager (v4.1)

```python
class TimeBudgetManager:
    """Adaptive time allocation for computational optimization."""
    
    def __init__(self, total_budget: float = 180.0):
        self.total_budget = total_budget
        self.start_time = time.time()
        self.elapsed = {}
    
    def allocate_pysr_time(self) -> int:
        """Allocate time for PySR based on remaining budget."""
        elapsed = time.time() - self.start_time
        remaining = self.total_budget - elapsed
        
        # Reserve 40s for Stage 3
        pysr_budget = max(30, int(remaining - 40))
        return min(pysr_budget, 120)  # Cap at 120s
    
    def allocate_bootstrap_count(self) -> int:
        """Allocate bootstrap count based on remaining time."""
        elapsed = time.time() - self.start_time
        remaining = self.total_budget - elapsed
        
        # ~5 bootstraps per second
        return max(50, min(int(remaining * 5), 200))
    
    def should_skip_optional(self, min_required: float = 20) -> bool:
        """Check if optional components should be skipped."""
        elapsed = time.time() - self.start_time
        remaining = self.total_budget - elapsed
        return remaining < min_required
```

### 8.10 Float32 Precision (v4.1)

```python
def convert_to_float32(X: np.ndarray, y: np.ndarray):
    """Convert to Float32 for memory efficiency."""
    X_32 = X.astype(np.float32)
    y_32 = y.astype(np.float32)
    return X_32, y_32
```

Memory savings: ~50% reduction (4 bytes vs 8 bytes per float).

### 8.11 Memory Cleanup (v4.1)

```python
def cleanup_memory():
    """Force garbage collection after heavy computation."""
    import gc
    gc.collect()
```

Call after PySR and bootstrap UQ stages.

### 8.12 Parallel Bootstrap (v4.1)

```python
from joblib import Parallel, delayed

def parallel_bootstrap(Phi, y, B=200, n_jobs=2):
    """Parallelized bootstrap with joblib."""
    def single_bootstrap(b):
        idx = np.random.choice(len(y), len(y), replace=True)
        return fit_model(Phi[idx], y[idx])
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_bootstrap)(b) for b in range(B)
    )
    return results
```

### 8.13 v4.7: PySR Coefficient Refinement

```python
from scipy.optimize import curve_fit

def refine_pysr_coefficients(equation_str, X, y, feature_names):
    """
    Refine PySR equation coefficients using curve_fit.
    
    1. Parse equation and extract numeric constants
    2. Replace constants with parameters: _p0, _p1, ...
    3. Build evaluation function via lambdify
    4. Use scipy.optimize.curve_fit to optimize
    5. Return refined equation with optimized values
    """
    # Create parametric function
    parametric_eq, initial_params = create_parametric_function(equation_str, feature_names)
    
    # Build evaluation function
    eval_func = build_eval_function(parametric_eq, feature_names, len(initial_params))
    
    # Run curve_fit
    popt, pcov = curve_fit(
        eval_func, X, y,
        p0=initial_params,
        bounds=([-1e6]*len(initial_params), [1e6]*len(initial_params)),
        maxfev=5000,
        method='trf'
    )
    
    # Compute R²
    y_pred = eval_func(X, *popt)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    
    # Build refined equation string
    refined_eq = parametric_eq
    for i, p in enumerate(popt):
        refined_eq = refined_eq.replace(f'_p{i}', f'{p:.6f}')
    
    return refined_eq, r2, True
```

### 8.14 v4.7: E-WSINDy Intercept Handling (CRITICAL)

```python
class EWSINDySTLSQ:
    def fit(self, feature_library, y, library_names=None):
        # ... term selection logic ...
        
        # CRITICAL: Always use fit_intercept=True
        ridge_final = Ridge(alpha=1e-8, fit_intercept=True)
        ridge_final.fit(X_selected, y)
        
        final_coef[final_support] = ridge_final.coef_
        intercept = ridge_final.intercept_  # MUST store this!
        
        # Predictions MUST include intercept
        y_pred = feature_library @ final_coef + intercept
        
        return {
            'coefficients': final_coef,
            'support': final_support,
            'intercept': intercept,  # CRITICAL: Include in output
            'r_squared': compute_r2(y, y_pred),
            # ...
        }
```

**Why Intercept Matters:**

Without intercept, if the true equation has a non-zero mean:
- Predictions are biased by ~mean(y)
- R² becomes negative (worse than predicting mean)

Example: DotProduct with intercept=27:
- Without intercept: R² = -9.87
- With intercept: R² = 0.96

### 8.15 v4.7: Dual-Track Selection Logic

```python
def select_final_method(pysr_r2, pysr_refined_r2, ewsindy_r2, 
                        curve_fit_success, trust_threshold=0.70):
    """
    Select between PySR-refined and E-WSINDy based on quality.
    
    Decision tree:
    1. If PySR R² >= trust_threshold AND curve_fit succeeded:
       - HIGH TRUST mode
       - Prefer PySR if within 5% of E-WSINDy
    2. Else:
       - LOW TRUST mode
       - Choose strictly better method
    """
    if pysr_r2 >= trust_threshold and curve_fit_success:
        # HIGH TRUST: PySR found good structure
        if pysr_refined_r2 >= ewsindy_r2 * 0.95:
            return 'pysr_refined', pysr_refined_r2
        else:
            return 'ewsindy', ewsindy_r2
    else:
        # LOW TRUST: Be conservative
        if curve_fit_success and pysr_refined_r2 > ewsindy_r2:
            return 'pysr_refined', pysr_refined_r2
        else:
            return 'ewsindy', ewsindy_r2
```

---

## 9. Computational Optimization Guide

### 9.1 Profiling Results (Colab Pro, 500 samples)

| Component | Before v4.1 | After v4.1 | Speedup |
|-----------|-------------|------------|---------|
| Stage 1 | 15-25s | 10-20s | 1.3x |
| PySR | 120-180s | 60-100s | 1.5x |
| Library Construction | 5-10s | 2-5s | 2x |
| E-WSINDy | 5-10s | 3-5s | 1.5x |
| Bootstrap UQ | 30-60s | 15-30s | 2x |
| **Total** | **180-300s** | **90-160s** | **~2x** |

### 9.2 Memory Optimization

- Float32 precision: 50% memory reduction
- Lazy variant generation: Skip if pysr_r2 > 0.95
- Garbage collection after heavy stages
- Sparse matrix where applicable

### 9.3 PySR Mode Selection

| Budget | Recommended Mode | Expected Quality |
|--------|------------------|------------------|
| < 90s | fast | R² ~ 0.6-0.8 |
| 90-150s | standard | R² ~ 0.7-0.9 |
| > 150s | thorough | R² ~ 0.8-0.95 |

### 9.4 Benchmark Results (AI Feynman Equations)

| Equation | True Form | Physics-SR v4.7 | Baseline PySR |
|----------|-----------|-----------------|---------------|
| Coulomb I.12.2 | k·q₁·q₂/r² | **R²=0.9994** | R²=0.65 |
| Cosines I.29.16 | x₁+x₂·cos(θ₁-θ₂) | R²=0.26 | R²=0.21 |
| Barometric I.40.1 | n₀·exp(-mgx/kT) | **R²=0.85** | R²=0.84 |
| DotProduct I.11.19 | x₁y₁+x₂y₂+x₃y₃ | **R²=0.91** | R²=0.78 |

**Key Insight:** Physics-SR v4.7 significantly outperforms baseline PySR, especially for:
- Inverse relationships (Coulomb: +52%)
- Linear with noise (DotProduct: +17%)

---

## References

1. Buckingham, E. (1914). On physically similar systems; illustrations of the use of dimensional equations. *Physical Review*, 4(4), 345.

2. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932-3937.

3. Basu, S., Kumbier, K., Brown, J. B., & Yu, B. (2018). Iterative random forests to discover predictive and stable high-order interactions. *PNAS*, 115(8), 1943-1948.

4. Meinshausen, N., & Buhlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society B*, 72(4), 417-473.

5. Chen, J., & Chen, Z. (2008). Extended Bayesian information criteria for model selection with large model spaces. *Biometrika*, 95(3), 759-771.

6. Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476), 1418-1429.

7. Messenger, D. A., & Bortz, D. M. (2021). Weak SINDy: Galerkin-based data-driven model selection. *Multiscale Modeling & Simulation*, 19(3), 1474-1497.

8. Udrescu, S. M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*, 6(16), eaay2631.

9. Fasel, U., Kutz, J. N., Brunton, B. W., & Brunton, S. L. (2022). Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise limit. *Proceedings of the Royal Society A*, 478(2260), 20210904.

10. Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. *arXiv preprint arXiv:2305.01582*.

11. Shojaee, P., et al. (2024). Transformer-based Planning for Symbolic Regression. *NeurIPS 2023*.

12. Li, Z., et al. (2024). LLM-SR: Scientific Equation Discovery via Programming with Large Language Models. *arXiv preprint*.

---

**Document Version:** 4.7  
**Last Modified:** January 2026  
**Status:** Complete Algorithm Specification with Dual-Track Selection + Intercept Fix + 5-Layer Library

---

*This document is part of the Three-Stage SR Framework project at Columbia University.*  
*Contact: zz3239@columbia.edu*
