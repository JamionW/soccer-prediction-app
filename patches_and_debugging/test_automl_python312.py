#!/usr/bin/env python3
"""
Test AutoML functionality with Python 3.12
Works with AutoGluon or sklearn fallback
"""

import sys
print(f"Python version: {sys.version}")

# Test which ML libraries are available
ml_library = None
try:
    from autogluon.tabular import TabularPredictor
    ml_library = "AutoGluon"
    print("✅ AutoGluon is available!")
except ImportError:
    print("❌ AutoGluon not found, trying sklearn...")
    try:
        from sklearn.ensemble import RandomForestRegressor
        ml_library = "Scikit-learn"
        print("✅ Sklearn is available as fallback!")
    except ImportError:
        print("❌ No ML libraries found!")

if ml_library:
    print(f"\n🤖 Using {ml_library} for AutoML")
    
    # Quick test with dummy data
    import pandas as pd
    import numpy as np
    
    # Create sample training data
    print("\n📊 Creating sample soccer data...")
    n_samples = 100
    
    data = pd.DataFrame({
        'is_home': np.random.randint(0, 2, n_samples),
        'team_xgf_per_game': np.random.uniform(0.8, 2.0, n_samples),
        'team_xga_per_game': np.random.uniform(0.8, 2.0, n_samples),
        'opp_xgf_per_game': np.random.uniform(0.8, 2.0, n_samples),
        'opp_xga_per_game': np.random.uniform(0.8, 2.0, n_samples),
        'team_form_points': np.random.uniform(0, 3, n_samples),
        'rest_days': np.random.randint(3, 14, n_samples),
        'is_chattanooga': np.random.randint(0, 2, n_samples),
        'rivalry_intensity': np.random.uniform(0, 2, n_samples),
        'goals': np.random.poisson(1.5, n_samples)  # Target variable
    })
    
    print(f"Created {len(data)} sample matches")
    
    if ml_library == "AutoGluon":
        print("\n🏃 Training AutoGluon model...")
        from autogluon.tabular import TabularPredictor
        
        predictor = TabularPredictor(
            label='goals',
            problem_type='regression',
            path='./autogluon_test'
        )
        
        predictor.fit(
            data.iloc[:80],  # Train on 80%
            time_limit=60,   # 1 minute for quick test
            presets='medium_quality_faster_train',
            verbosity=0
        )
        
        # Test prediction
        test_data = data.iloc[80:].drop('goals', axis=1)
        predictions = predictor.predict(test_data)
        
        print(f"✅ Model trained! Made {len(predictions)} predictions")
        print(f"   Average predicted goals: {predictions.mean():.2f}")
        
    elif ml_library == "Scikit-learn":
        print("\n🏃 Training Sklearn RandomForest model...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        X = data.drop('goals', axis=1)
        y = data['goals']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        print(f"✅ Model trained! Made {len(predictions)} predictions")
        print(f"   Average predicted goals: {predictions.mean():.2f}")
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Top 5 most important features:")
        print(importances.head())
    
    print("\n✅ AutoML is working with Python 3.12!")
    print(f"   You can use {ml_library} in your simulations")
    
else:
    print("\n❌ No ML libraries available. Please install:")
    print("   pip install autogluon")
    print("   OR")
    print("   pip install scikit-learn lightgbm xgboost")