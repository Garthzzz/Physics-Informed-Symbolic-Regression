# Physics-SR v3.0 完整操作指南

## Complete Setup and Execution Guide

**Author:** Zhengze Zhang  
**Date:** January 2026

---

## 目录

1. [文件清单和结构](#1-文件清单和结构)
2. [GitHub Desktop 上传指南](#2-github-desktop-上传指南)
3. [VSCode + Google Colab 连接指南](#3-vscode--google-colab-连接指南)
4. [运行完整实验流程](#4-运行完整实验流程)
5. [故障排除](#5-故障排除)

---

## 1. 文件清单和结构

### 1.1 完整文件列表 (24个文件)

```
Physics-Informed-Symbolic-Regression/
│
├── algorithms/                           # 13个算法notebook
│   ├── 00_Core.ipynb                    # 核心数据结构
│   ├── 01_BuckinghamPi.ipynb            # 量纲分析
│   ├── 02_VariableScreening.ipynb       # 变量筛选
│   ├── 03_SymmetryAnalysis.ipynb        # 对称性分析
│   ├── 04_InteractionDiscovery.ipynb    # 交互发现
│   ├── 05_FeatureLibrary.ipynb          # 特征库构建
│   ├── 06_PySR.ipynb                    # PySR符号回归
│   ├── 07_EWSINDy_STLSQ.ipynb           # E-WSINDy稀疏回归
│   ├── 08_AdaptiveLasso.ipynb           # 自适应Lasso
│   ├── 09_ModelSelection.ipynb          # 模型选择
│   ├── 10_PhysicsVerification.ipynb     # 物理验证
│   ├── 11_UQ_Inference.ipynb            # 不确定性量化
│   └── 12_Full_Pipeline.ipynb           # 完整流水线
│
├── benchmark/                            # 3个执行notebook
│   ├── DataGen.ipynb                    # 数据生成
│   ├── Experiments.ipynb                # 实验执行
│   ├── Analysis.ipynb                   # 结果分析
│   ├── data/                            # 数据目录
│   │   └── .gitkeep                     # 保持目录结构
│   └── results/                         # 结果目录
│       ├── .gitkeep
│       ├── figures/
│       │   └── .gitkeep
│       └── tables/
│           └── .gitkeep
│
├── requirements.txt                      # Python依赖
├── setup_colab.sh                       # Colab设置脚本
├── .gitignore                           # Git忽略规则
└── README.md                            # 项目说明
```

### 1.2 文件大小估计

| 类别 | 文件数 | 总大小 |
|------|--------|--------|
| algorithms/*.ipynb | 13 | ~500 KB |
| benchmark/*.ipynb | 3 | ~210 KB |
| 配置文件 | 4 | ~10 KB |
| **总计** | **20+** | **~720 KB** |

---

## 2. GitHub Desktop 上传指南

### Step 1: 在本地创建文件夹结构

1. 在你的电脑上创建一个新文件夹，命名为 `Physics-Informed-Symbolic-Regression`

2. 按照上面的结构创建子文件夹：
   ```
   Physics-Informed-Symbolic-Regression/
   ├── algorithms/
   └── benchmark/
       ├── data/
       └── results/
           ├── figures/
           └── tables/
   ```

3. 将下载的文件放入对应目录

### Step 2: 使用GitHub Desktop创建Repository

1. **打开 GitHub Desktop**

2. **创建新Repository:**
   - 点击 `File` → `New Repository...`
   - Name: `Physics-Informed-Symbolic-Regression`
   - Description: `Three-Stage Physics-Informed Symbolic Regression Framework`
   - Local Path: 选择你刚才创建的 `Physics-Informed-Symbolic-Regression` 文件夹的**父目录**
   - 勾选 `Initialize this repository with a README` (可选，因为我们已有README)
   - Git Ignore: 选择 `None` (我们已有.gitignore)
   - License: 选择 `MIT License`
   - 点击 `Create Repository`

3. **如果文件夹已存在:**
   - 点击 `File` → `Add Local Repository...`
   - 选择 `Physics-Informed-Symbolic-Regression` 文件夹
   - 如果提示不是git repository，点击 `create a repository`

### Step 3: 提交文件

1. **查看Changes面板:**
   - 左侧会显示所有新文件
   - 确保所有需要的文件都被勾选

2. **写Commit信息:**
   - Summary: `Initial commit: Physics-SR Framework v3.0`
   - Description (可选): 
     ```
     - 13 algorithm notebooks (Stage 1-3)
     - 3 benchmark notebooks (DataGen, Experiments, Analysis)
     - Configuration files (requirements.txt, setup_colab.sh)
     ```

3. **点击 `Commit to main`**

### Step 4: 推送到GitHub

1. **点击 `Publish repository`**
   - 取消勾选 `Keep this code private` (如果你想公开)
   - Organization: 选择你的用户名
   - 点击 `Publish Repository`

2. **验证上传:**
   - 在浏览器打开 `https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression`
   - 确认所有文件都已上传

### Step 5: 后续更新

每次修改文件后：
1. GitHub Desktop 会自动检测变化
2. 在左侧面板查看修改的文件
3. 写 commit message
4. 点击 `Commit to main`
5. 点击 `Push origin`

---

## 3. VSCode + Google Colab 连接指南

### 方案 A: 使用 Google Colab 直接运行 (推荐)

这是最简单的方案，适合大多数情况。

#### Step 1: 打开 Colab

1. 访问 [Google Colab](https://colab.research.google.com/)
2. 确保已登录你的 Google 账号

#### Step 2: 从 GitHub 打开 Notebook

1. 点击 `File` → `Open notebook`
2. 选择 `GitHub` 标签
3. 输入你的仓库URL: `https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression`
4. 选择要打开的notebook (如 `benchmark/DataGen.ipynb`)

#### Step 3: 运行设置单元

在任何notebook的第一个单元格运行：

```python
# ==============================================================================
# COLAB SETUP - Run this cell first!
# ==============================================================================
import sys
import os

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    # Clone repository if not already done
    if not os.path.exists('/content/Physics-Informed-Symbolic-Regression'):
        !git clone https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression.git
        %cd /content/Physics-Informed-Symbolic-Regression
    else:
        %cd /content/Physics-Informed-Symbolic-Regression
        !git pull  # Get latest changes
    
    # Install dependencies
    !pip install -q pysr sympy scikit-learn scipy seaborn tqdm
    
    # Initialize PySR (takes 2-3 minutes first time)
    import pysr
    try:
        pysr.install()
    except:
        print("PySR already installed")
    
    print("=" * 50)
    print("Setup complete! Ready to run experiments.")
    print("=" * 50)
```

---

### 方案 B: VSCode 连接 Colab (高级)

如果你想在 VSCode 中编辑并运行 Colab，有几种方法：

#### 方法 1: VSCode 的 Jupyter 扩展 + Colab 内核

1. **安装 VSCode 扩展:**
   - 打开 VSCode
   - 安装 `Jupyter` 扩展 (by Microsoft)
   - 安装 `Python` 扩展 (by Microsoft)

2. **在 Colab 中获取连接 URL:**
   
   在 Colab 中运行：
   ```python
   from google.colab import drive
   from google.colab import runtime
   
   # Get the URL for connecting
   from google.colab.output import eval_js
   print(eval_js("google.colab.kernel.proxyPort(8888)"))
   ```

3. **注意:** 这种方法比较复杂且不稳定，建议使用方案 A 或方法 2

#### 方法 2: 使用 colab-ssh (推荐的 VSCode + Colab 方案)

1. **在 Colab 中安装 colab-ssh:**
   ```python
   !pip install colab-ssh
   from colab_ssh import launch_ssh
   launch_ssh("YOUR_NGROK_TOKEN", password="your_password")
   ```

2. **获取 ngrok token:**
   - 访问 https://ngrok.com/
   - 注册并获取 authtoken

3. **在 VSCode 中:**
   - 安装 `Remote - SSH` 扩展
   - 按 `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`
   - 输入 Colab 显示的 SSH 地址

4. **缺点:** 需要额外设置，且 ngrok 免费版有限制

---

### 方案 C: 本地 VSCode + 同步到 Colab

这是**最推荐**的工作流程，因为它结合了本地编辑的便利性和 Colab 的计算能力：

#### 工作流程:

```
┌─────────────────────────────────────────────────────────┐
│                    你的开发工作流                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 本地 VSCode 编辑代码                                  │
│         │                                               │
│         ▼                                               │
│  2. Git push 到 GitHub                                  │
│         │                                               │
│         ▼                                               │
│  3. Colab 中 git pull 获取最新代码                        │
│         │                                               │
│         ▼                                               │
│  4. 在 Colab 中运行实验                                   │
│         │                                               │
│         ▼                                               │
│  5. 保存结果到 Google Drive 或下载                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 详细步骤:

**Step 1: 本地 VSCode 设置**

1. 克隆你的仓库到本地：
   ```bash
   git clone https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression.git
   cd Physics-Informed-Symbolic-Regression
   ```

2. 在 VSCode 中打开：
   ```bash
   code .
   ```

3. 安装推荐扩展：
   - Jupyter
   - Python
   - GitLens

**Step 2: 编辑并推送**

1. 在 VSCode 中编辑 notebook
2. 保存文件
3. 使用 VSCode 内置 Git 或 GitHub Desktop 提交并推送

**Step 3: Colab 中拉取运行**

在 Colab 中运行：
```python
# 如果是第一次
!git clone https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression.git
%cd Physics-Informed-Symbolic-Regression

# 如果已经clone过，拉取最新
%cd /content/Physics-Informed-Symbolic-Regression
!git pull
```

---

## 4. 运行完整实验流程

### 完整执行步骤 (在 Colab 中)

#### Step 1: 环境设置 (运行一次)

```python
# Cell 1: Clone and setup
import os
if not os.path.exists('/content/Physics-Informed-Symbolic-Regression'):
    !git clone https://github.com/Garthzzz/Physics-Informed-Symbolic-Regression.git
%cd /content/Physics-Informed-Symbolic-Regression

# Cell 2: Install dependencies
!pip install -q pysr sympy scikit-learn scipy seaborn tqdm joblib

# Cell 3: Initialize PySR (takes 2-3 minutes)
import pysr
pysr.install()

print("Environment ready!")
```

#### Step 2: 运行 DataGen.ipynb (生成测试数据)

**预计时间: ~5 分钟**

1. 打开 `benchmark/DataGen.ipynb`
2. 运行所有单元格 (`Runtime` → `Run all`)
3. 验证生成的数据：
   ```python
   !ls -la benchmark/data/
   # 应该看到 24 个 .npz 文件
   ```

**输出文件:**
```
benchmark/data/
├── eq1_kk2000_n500_noise0.00_dummy0.npz
├── eq1_kk2000_n500_noise0.00_dummy5.npz
├── eq1_kk2000_n500_noise0.05_dummy0.npz
├── eq1_kk2000_n500_noise0.05_dummy5.npz
├── ... (共24个文件)
```

#### Step 3: 运行 Experiments.ipynb (执行实验)

**预计时间: ~4-5 小时** (Colab Pro 推荐)

1. 打开 `benchmark/Experiments.ipynb`
2. **重要:** 确保 Colab 不会超时
   - 使用 Colab Pro 获得更长运行时间
   - 定期检查运行状态
3. 运行所有单元格

**进度监控:**
```
Core experiments: 96 runs
[=================>              ] 50/96 (52%) ETA: 2h 15m
```

**输出文件:**
```
benchmark/results/
├── experiment_results.csv    # 主结果表
├── experiment_results.pkl    # 详细结果
└── checkpoint.pkl            # 检查点 (实验完成后删除)
```

#### Step 4: 运行 Analysis.ipynb (分析结果)

**预计时间: ~5 分钟**

1. 打开 `benchmark/Analysis.ipynb`
2. 运行所有单元格
3. 查看生成的图表和表格

**输出文件:**
```
benchmark/results/
├── figures/
│   ├── fig1_f1_vs_noise.png
│   ├── fig2_f1_vs_dummy.png
│   ├── fig3_r2_comparison.png
│   ├── fig4_dims_benefit.png
│   ├── fig5_runtime.png
│   ├── fig6_heatmap.png
│   ├── fig7_sample_size.png
│   └── fig8_per_equation.png
├── tables/
│   ├── table1_main_results.tex
│   ├── table2_var_selection.tex
│   ├── table3_prediction_by_eq.tex
│   └── table4_dims_benefit.tex
└── summary_table.csv
```

#### Step 5: 保存结果到 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# 复制结果到 Drive
!cp -r /content/Physics-Informed-Symbolic-Regression/benchmark/results /content/drive/MyDrive/physics_sr_results
print("Results saved to Google Drive!")
```

---

## 5. 故障排除

### 问题 1: PySR 安装失败

**症状:** `pysr.install()` 报错

**解决方案:**
```python
# 方法1: 重新安装
!pip uninstall -y pysr
!pip install pysr
import pysr
pysr.install()

# 方法2: 使用指定版本
!pip install pysr==0.11.0
```

### 问题 2: Colab 超时断开

**症状:** 长时间运行后连接断开

**解决方案:**
1. 使用 Colab Pro ($9.99/月)
2. 使用检查点功能（Experiments.ipynb 已内置）
3. 定期保存中间结果到 Drive

**恢复运行:**
```python
# Experiments.ipynb 会自动检测 checkpoint.pkl
# 重新运行会从断点继续
```

### 问题 3: 内存不足 (OOM)

**症状:** `ResourceExhaustedError`

**解决方案:**
```python
# 减少batch大小
PHYSICS_SR_CONFIG['n_bootstrap'] = 30  # 从50降到30

# 清理内存
import gc
gc.collect()
```

### 问题 4: Git clone/pull 失败

**症状:** `Permission denied` 或 `Repository not found`

**解决方案:**
1. 确保仓库是 public
2. 或使用 token 认证：
   ```python
   !git clone https://YOUR_TOKEN@github.com/Garthzzz/Physics-Informed-Symbolic-Regression.git
   ```

### 问题 5: 文件路径错误

**症状:** `FileNotFoundError`

**解决方案:**
```python
# 检查当前目录
!pwd

# 确保在正确目录
%cd /content/Physics-Informed-Symbolic-Regression

# 检查文件是否存在
!ls -la algorithms/
!ls -la benchmark/
```

---

## 附录: 快速参考

### Colab 快捷键

| 操作 | 快捷键 |
|------|--------|
| 运行当前单元格 | `Ctrl+Enter` |
| 运行并移到下一个 | `Shift+Enter` |
| 运行所有单元格 | `Ctrl+F9` |
| 中断执行 | `Ctrl+M I` |
| 添加代码单元格 | `Ctrl+M B` |

### 预计运行时间

| 阶段 | 时间 |
|------|------|
| 环境设置 | 3-5 分钟 |
| DataGen | 5 分钟 |
| Experiments | 4-5 小时 |
| Analysis | 5 分钟 |
| **总计** | **~5 小时** |

### 输出文件汇总

| 文件 | 位置 | 用途 |
|------|------|------|
| `*.npz` | benchmark/data/ | 测试数据 |
| `experiment_results.csv` | benchmark/results/ | 主结果 |
| `fig*.png` | benchmark/results/figures/ | 可视化 |
| `table*.tex` | benchmark/results/tables/ | LaTeX表格 |

---

**准备好了吗？** 按照上述步骤操作，你将能够完整运行 Physics-SR v3.0 benchmark！

如有问题，请检查故障排除部分或在 GitHub 上提出 issue。
