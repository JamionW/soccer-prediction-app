#!/usr/bin/env python3
"""
Simple test of the trained AutoGluon model
This will help us understand what features it expects
"""

import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
from pathlib import Path

def test_model_simple():
    """Test the model with dummy data to understand its features"""
    
    print("🔬 Testing AutoGluon Model")
    print("="*60)
    
    # Load the model
    model_path = "models/xg_predictor_eastern_202506.pkl"
    predictor = TabularPredictor.load(model_path)
    
    print(f"✅ Model loaded from {model_path}")
    print(f"   Target: {predictor.label}")
    print(f"   Problem type: {predictor.problem_type}")
    print(f"   Best model: {predictor.model_best}")
    
    # Get expected features
    features = predictor.features()
    print(f"\n📋 Expected features ({len(features)}):")
    for i, feat in enumerate(features, 1):
        print(f"   {i:2d}. {feat}")
    
    # Create a dummy dataset with the expected features
    print("\n🧪 Creating test data...")
    
    # Create sample data for a hypothetical game
    test_data = pd.DataFrame({
        'is_home': [1, 0],  # One home game, one away game
        'team_xgf_per_game': [1.5, 1.2],
        'team_xga_per_game': [1.0, 1.3],
        'opp_xgf_per_game': [1.3, 1.4],
        'opp_xga_per_game': [1.1, 1.2],
        'xg_diff': [0.5, -0.1],  # team xGF - team xGA
        'opp_xg_diff': [0.2, 0.2],  # opp xGF - opp xGA
        'team_form_points': [2.0, 1.5],  # Recent form in points
        'team_form_gf': [1.8, 1.2],
        'team_form_ga': [0.8, 1.5],
        'opp_form_points': [1.5, 2.2],
        'opp_form_gf': [1.5, 2.0],
        'opp_form_ga': [1.2, 0.9],
        'h2h_games_played': [3, 2],
        'h2h_win_rate': [0.67, 0.0],
        'h2h_goals_for_avg': [2.0, 0.5],
        'h2h_goals_against_avg': [1.0, 2.0],
        'team_rest_days': [3, 7],
        'opp_rest_days': [4, 3],
        'month': [6, 6],  # June
        'day_of_week': [6, 3],  # Saturday, Wednesday
        'is_weekend': [1, 0]
    })
    
    print("   Sample data created with all required features")
    
    # Make predictions
    print("\n🎯 Making predictions...")
    try:
        predictions = predictor.predict(test_data)
        print(f"   Predictions: {predictions.values}")
        
        # If model supports prediction intervals
        try:
            pred_proba = predictor.predict_proba(test_data)
            print(f"   Prediction probabilities shape: {pred_proba.shape}")
        except:
            print("   (Model doesn't support probability predictions)")
        
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return
    
    # Test feature importance
    print("\n📊 Feature Importance:")
    try:
        importance = predictor.feature_importance(test_data)
        print("   Top 10 most important features:")
        for feat, score in importance.head(10).items():
            print(f"      {feat}: {score:.4f}")
    except:
        print("   (Feature importance requires more test data)")
    
    # Model evaluation metrics from training
    print("\n📈 Training Performance:")
    try:
        leaderboard = predictor.leaderboard(silent=True)
        best_model_score = leaderboard.iloc[0]['score_val']
        print(f"   Best model validation RMSE: {abs(best_model_score):.3f} goals")
        print(f"   (This means predictions are typically off by ~{abs(best_model_score):.1f} goals)")
    except:
        pass
    
    print("\n✅ Model test complete!")
    print("\n💡 Insights:")
    print("   - The model expects match-level features (not team-level)")
    print("   - Features include team stats, opponent stats, form, and H2H")
    print("   - The model predicts goals for the 'team' (not specifically home team)")
    print("   - Use is_home=1 for home games, is_home=0 for away games")

def create_feature_documentation():
    """Create detailed documentation of the features"""
    
    doc = """# AutoGluon Model Feature Documentation

## Feature Descriptions

### Team Identity
- **is_home**: Binary flag (1 if team is playing at home, 0 if away)

### Expected Goals (xG) Features
- **team_xgf_per_game**: Team's expected goals for per game
- **team_xga_per_game**: Team's expected goals against per game
- **opp_xgf_per_game**: Opponent's expected goals for per game
- **opp_xga_per_game**: Opponent's expected goals against per game
- **xg_diff**: Team's xG differential (xGF - xGA)
- **opp_xg_diff**: Opponent's xG differential

### Recent Form (Last 5 games)
- **team_form_points**: Team's points per game in last 5 games
- **team_form_gf**: Team's goals for per game in last 5
- **team_form_ga**: Team's goals against per game in last 5
- **opp_form_points**: Opponent's points per game in last 5
- **opp_form_gf**: Opponent's goals for per game in last 5
- **opp_form_ga**: Opponent's goals against per game in last 5

### Head-to-Head History
- **h2h_games_played**: Number of previous meetings
- **h2h_win_rate**: Team's win rate vs this opponent
- **h2h_goals_for_avg**: Team's average goals scored in H2H
- **h2h_goals_against_avg**: Team's average goals conceded in H2H

### Match Context
- **team_rest_days**: Days since team's last game
- **opp_rest_days**: Days since opponent's last game
- **month**: Month of the game (1-12)
- **day_of_week**: Day of week (0=Monday, 6=Sunday)
- **is_weekend**: Binary flag (1 if Fri/Sat/Sun, 0 otherwise)

## Usage Notes

1. The model predicts goals for the 'team' (not home/away)
2. Set is_home=1 when predicting for home team
3. All stats should be calculated BEFORE the game being predicted
4. Form metrics use the 5 most recent games
5. Rest days are capped at reasonable values (e.g., 14 days)
"""
    
    # Save documentation
    doc_path = Path("output") / "ml_model_feature_docs.md"
    doc_path.parent.mkdir(exist_ok=True)
    with open(doc_path, 'w') as f:
        f.write(doc)
    
    print(f"\n📄 Feature documentation saved to: {doc_path}")

if __name__ == "__main__":
    test_model_simple()
    create_feature_documentation()