#!/bin/bash

echo "🐍 Setting up MLS Next Pro Predictor with Python 3.12"
echo "===================================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Current Python version: $python_version"

# Install dependencies that work with Python 3.12
echo "📦 Installing Python 3.12 compatible dependencies..."

# Core dependencies (these should work fine)
pip install -r requirements.txt

# AutoML options for Python 3.12
echo "🤖 Installing AutoML libraries compatible with Python 3.12..."

# Option 1: AutoGluon (works with Python 3.12)
echo "Installing AutoGluon..."
pip install autogluon

# Option 2: If AutoGluon is too heavy, use sklearn + optuna
if [ $? -ne 0 ]; then
    echo "AutoGluon installation failed, trying lighter alternatives..."
    pip install scikit-learn optuna lightgbm xgboost
fi

# Test imports
echo "🧪 Testing installations..."
python3 << EOF
import sys
print(f"Python version: {sys.version}")

try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon installed successfully!")
except ImportError:
    print("❌ AutoGluon not available")
    try:
        import sklearn
        import optuna
        print("✅ Sklearn + Optuna installed as fallback!")
    except ImportError:
        print("❌ No ML libraries available")

try:
    import pandas
    import numpy
    import fastapi
    print("✅ Core dependencies installed!")
except ImportError as e:
    print(f"❌ Missing core dependency: {e}")
EOF

echo ""
echo "Setup complete! Your options:"
echo "1. Use AutoGluon (recommended) - already integrated in the updated code"
echo "2. Use Sklearn RandomForest (fallback) - also integrated"
echo "3. Downgrade to Python 3.11 if you specifically need PyCaret"