# Git Setup & GitHub Repository Instructions

## 📝 Prerequisites
- Git installed on your system (download from https://git-scm.com/)
- GitHub account created (https://github.com/)

---

## 🚀 Initial Setup - Creating Your GitHub Repository

### Step 1: Initialize Local Git Repository

```bash
# Navigate to your project directory
cd "c:\Users\ponna\OneDrive\Desktop\Precog Task"

# Initialize git repository
git init

# Configure your identity (if not already done globally)
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Step 2: Create Initial Commit

```bash
# Check status of files
git status

# Add all files to staging
git add .

# Create your first commit
git commit -m "Initial commit: Project structure setup for Precog Quant Task 2026"
```

### Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `precog-quant-task-2026` (or your preferred name)
3. Description: "End-to-end algorithmic trading pipeline for Precog recruitment task"
4. Set to **Private** (recommended for recruitment tasks)
5. **DO NOT** initialize with README, .gitignore, or license (you already have these)
6. Click "Create repository"

### Step 4: Link Local Repository to GitHub

```bash
# Add GitHub remote (replace <username> with your GitHub username)
git remote add origin https://github.com/<username>/precog-quant-task-2026.git

# Verify remote was added
git remote -v

# Push your code to GitHub
git branch -M main
git push -u origin main
```

---

## 💾 Working with Checkpoints (Push & Pull Workflow)

### Creating Checkpoints (Committing & Pushing)

```bash
# After completing work on a part/feature:

# 1. Check what files have changed
git status

# 2. Add specific files (recommended for clarity)
git add notebooks/01_data_cleaning_feature_engineering.ipynb
git add src/data/cleaning.py
git add src/data/features.py

# OR add all changed files
git add .

# 3. Create a meaningful commit message
git commit -m "Part 1: Implement data cleaning and feature engineering

- Added missing value handling logic
- Implemented momentum and volatility features
- Generated correlation analysis plots"

# 4. Push to GitHub
git push origin main
```

### Best Practice Commit Messages

```bash
# Part 1 checkpoint
git commit -m "Part 1 Complete: Data cleaning & feature engineering with 50+ features"

# Part 2 checkpoint
git commit -m "Part 2: Implement ensemble model with XGBoost + LightGBM"

# Part 3 checkpoint
git commit -m "Part 3: Backtesting engine complete, Sharpe 1.8 on test set"

# Part 4 checkpoint
git commit -m "Part 4: Statistical arbitrage - cointegration analysis complete"

# Bug fixes
git commit -m "Fix: Correct forward-looking bias in feature calculation"

# Performance improvement
git commit -m "Optimize: Vectorize feature computation, 10x speedup"
```

### Pulling Updates (if working from multiple machines)

```bash
# Before starting work, pull latest changes
git pull origin main

# If you have local changes and want to pull
git stash              # Temporarily store local changes
git pull origin main   # Get remote updates
git stash pop          # Reapply your local changes
```

---

## 🌿 Branching Strategy (Recommended for Clean Workflow)

### Working on Different Parts

```bash
# Create a branch for each major part
git checkout -b part1-feature-engineering
# Work on Part 1...
git add .
git commit -m "Part 1: Complete feature engineering"
git push origin part1-feature-engineering

# Switch to main and merge when ready
git checkout main
git merge part1-feature-engineering

# Create branch for Part 2
git checkout -b part2-model-training
# Work on Part 2...
```

### Experimental Features

```bash
# Try risky/experimental approaches without affecting main code
git checkout -b experiment-transformer-model
# Experiment...

# If it works, merge it
git checkout main
git merge experiment-transformer-model

# If it doesn't work, just delete the branch
git branch -D experiment-transformer-model
```

---

## 📊 Checkpoint Recommendations

### Suggested Checkpoint Schedule

| Checkpoint | Description | When to Commit |
|------------|-------------|----------------|
| **CP1** | Project setup | After directory structure created |
| **CP2** | Data loading & EDA | After initial data exploration |
| **CP3** | Data cleaning complete | After handling missing values & outliers |
| **CP4** | Feature engineering v1 | After creating first set of features |
| **CP5** | Feature engineering v2 | After optimization/adding more features |
| **CP6** | Model baseline | After first working model |
| **CP7** | Model optimization | After hyperparameter tuning |
| **CP8** | Ensemble model | After combining multiple models |
| **CP9** | Backtesting engine | After backtester implemented |
| **CP10** | Performance analysis | After completing Part 3 |
| **CP11** | Statistical arbitrage | After completing Part 4 |
| **CP12** | Final submission | After README & report complete |

---

## 🔍 Useful Git Commands

### Checking Status & History

```bash
# See what's changed
git status

# See commit history
git log --oneline --graph --all

# See what changed in last commit
git show

# See changes in a specific file
git diff notebooks/01_data_cleaning_feature_engineering.ipynb
```

### Undoing Changes

```bash
# Discard changes in a file (before staging)
git checkout -- filename.py

# Unstage a file (after git add)
git reset HEAD filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes) - CAREFUL!
git reset --hard HEAD~1
```

### Viewing Differences

```bash
# See uncommitted changes
git diff

# See changes between commits
git diff main..part1-feature-engineering

# See files changed in last commit
git diff HEAD~1 HEAD --name-only
```

---

## ⚠️ Important Notes

### Do NOT Commit Large Files

Your `.gitignore` is configured to exclude:
- Large datasets in `data/raw/` and `data/processed/`
- Model files in `outputs/models/`
- CSV/Parquet files

If you need to track model performance:
```bash
# Commit only summary files, not large binaries
git add outputs/results/performance_summary.csv
git add outputs/figures/cumulative_pnl.png
```

### Handling Large Notebooks

Jupyter notebooks can get large. Consider:
1. Clear outputs before committing:
   ```bash
   # In Jupyter: Cell > All Output > Clear
   ```
2. Or use `nbstripout` to auto-strip outputs:
   ```bash
   pip install nbstripout
   nbstripout --install
   ```

### Syncing Across Devices

```bash
# On Machine A (after work)
git add .
git commit -m "Progress checkpoint"
git push origin main

# On Machine B (before work)
git pull origin main
```

---

## 🎯 Quick Reference

```bash
# Daily workflow
git status                          # Check status
git add .                           # Stage changes
git commit -m "Meaningful message"  # Commit
git push origin main                # Push to GitHub

# Before starting work
git pull origin main                # Get latest changes

# Create branch for experiments
git checkout -b experiment-name     # Create & switch to branch
git checkout main                   # Switch back to main
git merge experiment-name           # Merge branch into main
```

---

## 📧 Need Help?

- Git documentation: https://git-scm.com/doc
- GitHub guides: https://guides.github.com/
- Resolve merge conflicts: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts

---

**Pro Tip:** Commit often, push regularly. It's better to have many small checkpoints than to lose work!
