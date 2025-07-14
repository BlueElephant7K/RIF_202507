# Gene Expression Analysis for Clinical Pregnancy Prediction

基因表达数据用于临床妊娠预测的机器学习分析框架

A comprehensive machine learning framework for analyzing gene expression data to predict clinical pregnancy outcomes.

## 📋 Overview / 概述

This project provides a complete machine learning pipeline for analyzing gene expression data to predict clinical pregnancy outcomes (Y1) and live birth outcomes (Y2). The framework includes 9 different machine learning algorithms with rigorous stability analysis and comprehensive visualization.

本项目提供了一个完整的机器学习管道，用于分析基因表达数据以预测临床妊娠结局(Y1)和活产结局(Y2)。该框架包含9种不同的机器学习算法，具有严格的稳定性分析和全面的可视化功能。

## 🛠 Installation / 安装

### Prerequisites / 环境要求
- Python 3.11+
- Poetry (推荐) 或 pip

### Method 1: Poetry (Recommended) / 方法1: Poetry（推荐）
```bash
# Install Poetry if not already installed / 如果未安装Poetry，先安装Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository / 克隆仓库
git clone <repository-url>
cd RIF_202507

# Install all dependencies using Poetry / 使用Poetry安装所有依赖
poetry install

# Activate the virtual environment / 激活虚拟环境
poetry shell

# Or run commands with poetry run / 或使用poetry run运行命令
poetry run python model/NB_stability_analysis.py
```

### Method 2: pip (Alternative) / 方法2: pip（替代方案）
```bash
# Clone the repository / 克隆仓库
git clone <repository-url>
cd RIF_202507

# Create virtual environment / 创建虚拟环境
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies / 安装依赖
pip install pandas>=2.3.1 numpy>=2.3.1 scipy>=1.16.0 scikit-learn>=1.7.0
pip install torch>=2.7.1 torchvision>=0.22.1 torchaudio>=2.7.1
pip install matplotlib>=3.10.3 seaborn>=0.13.2 jupyter>=1.1.1
pip install xgboost>=3.0.2 lightgbm>=4.6.0
```

### Current Dependencies / 当前依赖项
The project uses the following packages (defined in `pyproject.toml`):
项目使用以下包（在`pyproject.toml`中定义）：

```toml
pandas = ">=2.3.1,<3.0.0"
numpy = ">=2.3.1,<3.0.0"
scipy = ">=1.16.0,<2.0.0"
scikit-learn = ">=1.7.0,<2.0.0"
torch = ">=2.7.1,<3.0.0"
torchvision = ">=0.22.1,<0.23.0"
torchaudio = ">=2.7.1,<3.0.0"
matplotlib = ">=3.10.3,<4.0.0"
seaborn = ">=0.13.2,<0.14.0"
jupyter = ">=1.1.1,<2.0.0"
xgboost = ">=3.0.2,<4.0.0"
lightgbm = ">=4.6.0,<5.0.0"
```

## 📁 Data Setup / 数据设置

### Raw Data Placement / 原始数据放置
Place your gene expression data file in the following location:
将基因表达数据文件放置在以下位置：

```
raw_data/filtered_已经log2.csv
```

### Data Format Requirements / 数据格式要求
The CSV file should contain:
CSV文件应包含：

- **Gene expression columns**: Numerical values (log2 transformed)
  **基因表达列**: 数值型数据（log2转换后）
- **Clinical outcome columns**: 
  **临床结局列**:
  - `临床妊娠结局` (Clinical pregnancy outcome) → Y1
  - `活产结局` (Live birth outcome) → Y2
- **Clinical feature columns**:
  **临床特征列**:
  - `体重指数` (BMI)
  - `基础内分泌FSH` (Basal FSH)
  - `基础内分泌AMH` (Basal AMH)
  - `移植胚胎数` (Number of transferred embryos)

## 📊 File Structure / 文件结构

### 📓 Jupyter Notebooks / Jupyter笔记本
Located in root directory / 位于根目录：

| File | Purpose | 用途 |
|------|---------|------|
| `data_analysis.ipynb` | Basic data exploration and visualization | 基础数据探索和可视化 |
| `correlation_analysis.ipynb` | Gene-outcome correlation analysis | 基因-结局相关性分析 |

### 🤖 Machine Learning Models / 机器学习模型
Located in `model/` directory / 位于`model/`目录：

| File | Algorithm | Description | 算法描述 |
|------|-----------|-------------|----------|
| `NB_stability_analysis.py` | Naive Bayes | Probabilistic classification with 3 variants | 概率分类，3个变体 |
| `LR_stability_analysis.py` | Logistic Regression | Linear classification with regularization | 线性分类，带正则化 |
| `LinearRegression_stability_analysis.py` | Linear Regression | Regression baseline with 6 variants | 回归基线，6个变体 |
| `SVM_stability_analysis.py` | Support Vector Machine | Kernel methods with 4 variants | 核方法，4个变体 |
| `RandomForest_stability_analysis.py` | Random Forest | Tree ensemble with feature importance | 树集成，特征重要性 |
| `XGBoost_stability_analysis.py` | XGBoost | Gradient boosting with 3 variants | 梯度提升，3个变体 |
| `LightGBM_stability_analysis.py` | LightGBM | Fast gradient boosting | 快速梯度提升 |
| `LDA_QDA_stability_analysis.py` | LDA/QDA | Discriminant analysis with dimensionality reduction | 判别分析，降维 |
| `KNN_stability_analysis.py` | K-Nearest Neighbors | Instance-based learning with distance metrics | 实例学习，距离度量 |

### 📁 Results Structure / 结果结构
```
result_num/          # Numerical results / 数值结果
├── naive_bayes/     # Naive Bayes results / 朴素贝叶斯结果
├── logistic_regression/  # Logistic Regression results / 逻辑回归结果
├── linear_regression/    # Linear Regression results / 线性回归结果
├── svm/             # SVM results / SVM结果
├── random_forest/   # Random Forest results / 随机森林结果
├── xgboost/         # XGBoost results / XGBoost结果
├── lightgbm/        # LightGBM results / LightGBM结果
├── lda_qda/         # LDA/QDA results / LDA/QDA结果
└── knn/             # KNN results / KNN结果

result_pic/          # Visualizations / 可视化结果
├── naive_bayes/     # Naive Bayes plots / 朴素贝叶斯图表
├── logistic_regression/  # Logistic Regression plots / 逻辑回归图表
├── linear_regression/    # Linear Regression plots / 线性回归图表
├── svm/             # SVM plots / SVM图表
├── random_forest/   # Random Forest plots / 随机森林图表
├── xgboost/         # XGBoost plots / XGBoost图表
├── lightgbm/        # LightGBM plots / LightGBM图表
├── lda_qda/         # LDA/QDA plots / LDA/QDA图表
└── knn/             # KNN plots / KNN图表
```

## 🚀 Usage / 使用方法

### 1. Basic Data Analysis / 基础数据分析

#### Using Poetry (Recommended) / 使用Poetry（推荐）
```bash
# Activate Poetry environment / 激活Poetry环境
poetry shell

# Run Jupyter notebooks for initial exploration
# 运行Jupyter笔记本进行初始探索
jupyter notebook data_analysis.ipynb
jupyter notebook correlation_analysis.ipynb

# Or use poetry run / 或使用poetry run
poetry run jupyter notebook data_analysis.ipynb
```

#### Using pip / 使用pip
```bash
# Activate virtual environment / 激活虚拟环境
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run Jupyter notebooks / 运行Jupyter笔记本
jupyter notebook data_analysis.ipynb
jupyter notebook correlation_analysis.ipynb
```

### 2. Machine Learning Analysis / 机器学习分析

#### Using Poetry (Recommended) / 使用Poetry（推荐）
```bash
# Activate Poetry environment / 激活Poetry环境
poetry shell

# Run specific algorithm / 运行特定算法
python model/NB_stability_analysis.py
python model/LR_stability_analysis.py
python model/LightGBM_stability_analysis.py
python model/XGBoost_stability_analysis.py
# ... etc

# Or use poetry run for each / 或为每个使用poetry run
poetry run python model/NB_stability_analysis.py
poetry run python model/LightGBM_stability_analysis.py

# Run all algorithms / 运行所有算法
for script in model/*_stability_analysis.py; do
    echo "Running $script"
    poetry run python "$script"
done
```

#### Using pip / 使用pip
```bash
# Activate virtual environment / 激活虚拟环境
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run specific algorithm / 运行特定算法
python model/NB_stability_analysis.py
python model/LR_stability_analysis.py
python model/LightGBM_stability_analysis.py
# ... etc

# Run all algorithms / 运行所有算法
for script in model/*_stability_analysis.py; do
    echo "Running $script"
    python "$script"
done
```

### 3. Algorithm Comparison / 算法对比
Each algorithm automatically analyzes both Y1 and Y2 outcomes and provides:
每个算法都会自动分析Y1和Y2结局，并提供：

- **Performance metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
  **性能指标**: 准确率、精确率、召回率、F1分数、AUC-ROC
- **Stability analysis**: 20 iterations with different random splits
  **稳定性分析**: 20次不同随机分割的迭代
- **Feature stability**: Biomarker discovery and ranking
  **特征稳定性**: 生物标记发现和排序
- **Model-specific insights**: Algorithm-specific analysis
  **模型特定洞察**: 算法特定分析

## 📊 Final Reports / 最终报告

### Generated Files per Algorithm / 每个算法生成的文件

For each algorithm, the following files are generated:
对于每个算法，生成以下文件：

#### 📈 Numerical Results / 数值结果
Located in `result_num/{algorithm}/`:
位于 `result_num/{算法}/`：

| File Type | Description | 描述 |
|-----------|-------------|------|
| `*_analysis_Y1_*.csv` | Detailed performance results for Y1 | Y1的详细性能结果 |
| `*_analysis_Y2_*.csv` | Detailed performance results for Y2 | Y2的详细性能结果 |
| `feature_stability_Y1_*.csv` | Feature selection stability for Y1 | Y1的特征选择稳定性 |
| `feature_stability_Y2_*.csv` | Feature selection stability for Y2 | Y2的特征选择稳定性 |
| `core_biomarkers_Y1_*.csv` | High-stability biomarkers for Y1 | Y1的高稳定性生物标记 |
| `core_biomarkers_Y2_*.csv` | High-stability biomarkers for Y2 | Y2的高稳定性生物标记 |
| `*_report_Y1_*.txt` | Summary report for Y1 | Y1的总结报告 |
| `*_report_Y2_*.txt` | Summary report for Y2 | Y2的总结报告 |
| `complete_stability_data_*.pkl` | Complete analysis data (Python pickle) | 完整分析数据 |

#### 🎨 Visualizations / 可视化
Located in `result_pic/{algorithm}/`:
位于 `result_pic/{算法}/`：

| Plot Type | Description | 描述 |
|-----------|-------------|------|
| Performance plots | Accuracy distributions, metric comparisons | 性能图表：准确率分布、指标对比 |
| Stability plots | Model stability, feature selection patterns | 稳定性图表：模型稳定性、特征选择模式 |
| Algorithm-specific plots | Specialized visualizations per algorithm | 算法特定图表：每个算法的专门可视化 |

### 📋 Comprehensive Summary / 综合总结

#### Key Performance Indicators / 关键性能指标
Each algorithm provides:
每个算法提供：

1. **Best Model Identification**: Top-performing variant
   **最佳模型识别**: 表现最佳的变体
2. **Performance Level**: Excellent/Good/Moderate/Poor classification
   **性能水平**: 优秀/良好/中等/差 分类
3. **Stability Score**: Coefficient of variation across iterations
   **稳定性评分**: 跨迭代的变异系数
4. **Biomarker Discovery**: High-stability genes (≥80% selection rate)
   **生物标记发现**: 高稳定性基因（≥80%选择率）

#### Algorithm-Specific Insights / 算法特定洞察

| Algorithm | Unique Insights | 独特洞察 |
|-----------|----------------|---------|
| **Naive Bayes** | Probabilistic classification, feature independence | 概率分类，特征独立性 |
| **Logistic Regression** | Linear decision boundaries, coefficient interpretation | 线性决策边界，系数解释 |
| **Linear Regression** | Baseline performance, dual classification/regression | 基线性能，分类/回归双重评估 |
| **SVM** | Support vector analysis, kernel comparison | 支持向量分析，核函数对比 |
| **Random Forest** | Feature importance ranking, OOB validation | 特征重要性排序，袋外验证 |
| **XGBoost** | Gradient boosting, early stopping analysis | 梯度提升，早停分析 |
| **LightGBM** | Speed vs accuracy trade-off, memory efficiency | 速度vs准确率权衡，内存效率 |
| **LDA/QDA** | Dimensionality reduction, discriminant functions | 降维，判别函数 |
| **KNN** | Distance metrics, neighbor analysis | 距离度量，邻居分析 |

## 🔬 Methodology / 方法学

### Rigorous Validation / 严格验证
- **Proper train/test splitting**: Feature selection ONLY on training data
  **正确的训练/测试分割**: 特征选择仅在训练数据上进行
- **Stability testing**: 20 iterations with different random seeds
  **稳定性测试**: 20次不同随机种子的迭代
- **No data leakage**: Strict separation of training and testing
  **无数据泄露**: 严格的训练和测试分离
- **Comprehensive metrics**: Multiple performance measures
  **综合指标**: 多种性能度量

### Feature Selection / 特征选择
- **Method**: SelectKBest with ANOVA F-test
  **方法**: 使用ANOVA F检验的SelectKBest
- **Count**: Top 1000 features per iteration
  **数量**: 每次迭代选择前1000个特征
- **Stability analysis**: Feature selection frequency across iterations
  **稳定性分析**: 跨迭代的特征选择频率

## 🎯 Interpretation Guide / 解释指南

### Performance Levels / 性能水平
- **Excellent**: >80% accuracy (优秀：准确率>80%)
- **Good**: 70-80% accuracy (良好：准确率70-80%)
- **Moderate**: 60-70% accuracy (中等：准确率60-70%)
- **Poor**: <60% accuracy (差：准确率<60%)

### Biomarker Confidence / 生物标记置信度
- **Core biomarkers**: ≥80% selection rate (核心生物标记：≥80%选择率)
- **Stable biomarkers**: ≥50% selection rate (稳定生物标记：≥50%选择率)
- **Candidate biomarkers**: ≥30% selection rate (候选生物标记：≥30%选择率)

## 🔧 Troubleshooting / 故障排除

### Common Issues / 常见问题

1. **Poetry Installation Issues** / Poetry安装问题:
   ```bash
   # If Poetry is not installed / 如果Poetry未安装
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Update Poetry / 更新Poetry
   poetry self update
   
   # Check Poetry version / 检查Poetry版本
   poetry --version
   ```

2. **Missing packages** / 缺少包:
   ```bash
   # Using Poetry / 使用Poetry
   poetry install
   poetry add xgboost lightgbm
   
   # Using pip / 使用pip
   pip install xgboost lightgbm
   ```

3. **Virtual Environment Issues** / 虚拟环境问题:
   ```bash
   # Using Poetry / 使用Poetry
   poetry shell  # Activate environment / 激活环境
   poetry env info  # Check environment info / 检查环境信息
   poetry env remove python  # Remove current env / 删除当前环境
   poetry install  # Recreate environment / 重新创建环境
   
   # Using pip / 使用pip
   deactivate  # Exit current environment / 退出当前环境
   rm -rf venv  # Remove virtual environment / 删除虚拟环境
   python -m venv venv  # Create new environment / 创建新环境
   ```

4. **Memory issues** / 内存问题:
   - Reduce `N_FEATURES` from 1000 to 500
   - 将 `N_FEATURES` 从1000减少到500

5. **Slow execution** / 执行缓慢:
   - Reduce `N_ITERATIONS` from 20 to 10
   - 将 `N_ITERATIONS` 从20减少到10

6. **File not found** / 文件未找到:
   - Ensure `raw_data/filtered_已经log2.csv` exists
   - 确保 `raw_data/filtered_已经log2.csv` 存在

7. **Permission Issues** / 权限问题:
   ```bash
   # Check file permissions / 检查文件权限
   ls -la raw_data/
   
   # Fix permissions if needed / 如需要修复权限
   chmod 644 raw_data/filtered_已经log2.csv
   ```

## 🔍 Poetry Commands Reference / Poetry命令参考

Useful Poetry commands for this project:
本项目的有用Poetry命令：

```bash
# Project management / 项目管理
poetry show                    # List installed packages / 列出已安装的包
poetry show --tree            # Show dependency tree / 显示依赖树
poetry check                  # Check pyproject.toml validity / 检查pyproject.toml有效性

# Environment management / 环境管理
poetry env list               # List virtual environments / 列出虚拟环境
poetry env info               # Show environment information / 显示环境信息
poetry shell                  # Activate virtual environment / 激活虚拟环境

# Package management / 包管理
poetry add <package>          # Add new package / 添加新包
poetry remove <package>       # Remove package / 删除包
poetry update                 # Update all packages / 更新所有包
poetry install                # Install all dependencies / 安装所有依赖

# Running scripts / 运行脚本
poetry run python <script>    # Run Python script / 运行Python脚本
poetry run jupyter notebook   # Run Jupyter notebook / 运行Jupyter笔记本
```

## 📞 Support / 支持

For questions or issues, please check:
如有问题，请检查：

1. **Poetry setup**: Ensure Poetry is correctly installed and configured
   **Poetry设置**: 确保Poetry正确安装和配置
2. **Data format**: Ensure CSV file matches expected format
   **数据格式**: 确保CSV文件符合预期格式
3. **Dependencies**: All required packages are installed via `poetry install`
   **依赖项**: 通过`poetry install`安装所有必需的包
4. **Paths**: Raw data file is in correct location
   **路径**: 原始数据文件在正确位置
5. **Virtual environment**: Ensure you're in the correct Poetry environment
   **虚拟环境**: 确保您在正确的Poetry环境中

## 📚 Citation / 引用

If you use this framework in your research, please cite:
如果您在研究中使用此框架，请引用：

```
Gene Expression Analysis Framework for Clinical Pregnancy Prediction
[Your Institution/Publication Details]
```

---

## 🌟 Features Highlights / 特色功能

✅ **9 Machine Learning Algorithms** / 9种机器学习算法
✅ **Rigorous Stability Testing** / 严格稳定性测试  
✅ **Comprehensive Visualization** / 综合可视化
✅ **Biomarker Discovery** / 生物标记发现
✅ **No Data Leakage** / 无数据泄露
✅ **Automated Report Generation** / 自动报告生成
✅ **Multi-outcome Analysis** / 多结局分析
✅ **Algorithm Comparison** / 算法对比

---

*Last updated: 2025-01-14* / *最后更新：2025年1月14日*