# Precog Quant Task 2026 - Algorithmic Trading Pipeline

**Author:** [Your Name]  
**Task:** Quantitative Trading Strategy Development  
**Institution:** Precog Research Group, IIIT

---

## 📋 Project Overview

This repository contains an end-to-end algorithmic trading pipeline for a universe of anonymized stocks. The project transforms raw OHLCV price data into a systematic trading strategy that maximizes risk-adjusted returns.

### Task Completion Status

| Part | Task | Status |
|------|------|--------|
| 1 | Feature Engineering & Data Cleaning | ⏳ In Progress |
| 2 | Model Training & Strategy Formulation | ⏳ In Progress |
| 3 | Backtesting & Performance Analysis | ⏳ In Progress |
| 4 | Statistical Arbitrage Overlay | ⏳ In Progress |

---

## 🗂️ Directory Structure

```
Precog Task/
├── .github/
│   └── instructions/          # Task instructions
├── data/
│   ├── raw/                   # Original daily_prices.csv (NOT committed)
│   └── processed/             # Cleaned & engineered features (NOT committed)
├── notebooks/
│   ├── 01_data_cleaning_feature_engineering.ipynb
│   ├── 02_model_training_strategy.ipynb
│   ├── 03_backtesting_performance.ipynb
│   └── 04_statistical_arbitrage.ipynb
├── src/
│   ├── data/
│   │   ├── cleaning.py        # Data quality checks & cleaning logic
│   │   └── features.py        # Feature engineering functions
│   ├── models/
│   │   ├── predictors.py      # Model architectures
│   │   └── ensemble.py        # Ensemble methods
│   ├── backtesting/
│   │   ├── engine.py          # Backtesting simulation engine
│   │   └── metrics.py         # Performance metrics calculation
│   └── utils/
│       └── visualization.py   # Plotting & visualization helpers
├── outputs/
│   ├── figures/               # Generated plots & visualizations
│   ├── models/                # Saved model checkpoints
│   └── results/               # Performance logs & metrics
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Kaggle account (for dataset download)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Precog Task"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle:
   - Visit: https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026
   - Download `daily_prices.csv`
   - Place it in `data/raw/`

### Running the Project

Execute notebooks in order:

```bash
# 1. Data Cleaning & Feature Engineering
jupyter notebook notebooks/01_data_cleaning_feature_engineering.ipynb

# 2. Model Training & Strategy Development
jupyter notebook notebooks/02_model_training_strategy.ipynb

# 3. Backtesting & Performance Analysis
jupyter notebook notebooks/03_backtesting_performance.ipynb

# 4. Statistical Arbitrage Analysis
jupyter notebook notebooks/04_statistical_arbitrage.ipynb
```

---

## 📊 Methodology

### Part 1: Feature Engineering & Data Cleaning

**Approach:**
- Data quality assessment (missing values, outliers, anomalies)
- Feature extraction capturing market dynamics:
  - Technical indicators (momentum, volatility, volume)
  - Statistical features (rolling statistics, z-scores)
  - [Add your specific approaches here]

**Key Decisions:**
- [Document your cleaning strategies]
- [Justify feature selection]

### Part 2: Model Training & Strategy Formulation

**Approach:**
- Prediction target: [Classification/Regression]
- Model architecture(s): [List models used]
- Ensemble methods: [If applicable]
- Signal generation logic: [How predictions → trades]

**Key Decisions:**
- [Rationale for model choice]
- [Handling non-stationarity]
- [Risk management approach]

### Part 3: Backtesting & Performance Analysis

**Simulation Parameters:**
- Initial Capital: $1,000,000
- Transaction Costs: 10 bps per trade
- Universe: [Specify stocks traded]

**Performance Metrics:**
- Sharpe Ratio (annualized): [Value]
- Maximum Drawdown: [Value]
- Average Drawdown: [Value]
- Portfolio Turnover: [Value]
- Total Return: [Value]

**Analysis:**
- [Transaction cost impact]
- [Failure modes identified]
- [When/why strategy underperforms]

### Part 4: Statistical Arbitrage Overlay

**Approach:**
- Pair/group selection methodology: [Describe]
- Cointegration analysis: [Methods used]
- Lead-lag relationships: [Findings]

**Key Findings:**
- [Identified asset relationships]
- [Mathematical justification]
- [Integration with main strategy]

---

## 📈 Results Summary

### Out-of-Sample Performance (Test Period: [Start] - [End])

| Metric | Strategy | Benchmark | Difference |
|--------|----------|-----------|------------|
| Sharpe Ratio | [X.XX] | [X.XX] | [+/-X.XX] |
| Max Drawdown | [X.XX%] | [X.XX%] | [+/-X.XX%] |
| Total Return | [X.XX%] | [X.XX%] | [+/-X.XX%] |
| Turnover | [X.XX] | [X.XX] | [+/-X.XX] |

### Key Insights

1. **What Worked:**
   - [Insight 1]
   - [Insight 2]

2. **What Didn't Work:**
   - [Challenge 1]
   - [Challenge 2]

3. **Hypotheses for Future Improvement:**
   - [Hypothesis 1]
   - [Hypothesis 2]

---

## 📚 References & Literature

1. [Key paper/resource 1]
2. [Key paper/resource 2]
3. [Key paper/resource 3]

---

## 🔧 Dependencies

See [requirements.txt](requirements.txt) for full list. Key libraries:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib`, `seaborn` - Visualization
- `statsmodels` - Statistical analysis
- [Add others as used]

---

## 📝 Notes & Future Work

- [Any limitations encountered]
- [Ideas for future extensions]
- [Computational constraints faced]

---

## 📧 Contact

For questions about this implementation:
- Email: [Your email]
- GitHub: [Your GitHub username]

---

**Disclaimer:** This project is for educational and research purposes as part of the Precog Research Group recruitment process.
