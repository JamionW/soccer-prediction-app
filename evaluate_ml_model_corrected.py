#!/usr/bin/env python3
"""
Evaluate the AutoGluon model using the SAME feature extraction as training
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.tabular import TabularPredictor

# Import your actual predictor to use its feature extraction
from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

async def evaluate_ml_model():
    """Evaluate ML model on test data using proper feature extraction"""
    
    print("\n🔬 MLS Next Pro ML Model Evaluation")
    print("="*70)
    
    await connect()
    db_manager = DatabaseManager(database)
    
    try:
        # Load the AutoGluon model directly
        model_path = "models/xg_predictor_eastern_202506.pkl"
        ml_model = TabularPredictor.load(model_path)
        print(f"✅ Loaded model from {model_path}")
        
        # Get Eastern Conference data
        sim_data = await db_manager.get_data_for_simulation('eastern', 2025)
        
        # Create a predictor instance to use its feature extraction
        predictor = MLSNPRegSeasonPredictor(
            conference='eastern',
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages={'league_avg_xgf': 1.3, 'league_avg_xga': 1.2},
            use_automl=False  # We don't need to train, just use feature extraction
        )
        
        # Prepare test data from completed games
        completed_games = [g for g in sim_data['games_data'] if g.get('is_completed')]
        print(f"\n📊 Found {len(completed_games)} completed games")
        
        # Filter to only conference games
        conference_games = []
        for game in completed_games:
            if (game['home_team_id'] in sim_data['conference_teams'] and 
                game['away_team_id'] in sim_data['conference_teams']):
                conference_games.append(game)
        
        print(f"   Conference games: {len(conference_games)}")
        
        # Pre-process all game dates to ensure consistency
        for game in completed_games:
            if 'date' in game and hasattr(game['date'], 'strftime'):
                # Convert datetime to string for consistency
                game['date_str'] = game['date'].strftime('%Y-%m-%d')
            else:
                game['date_str'] = str(game['date']).split()[0]
        
        # Create a wrapper for the predictor that handles date conversion
        original_calculate_form = predictor._calculate_form
        
        def patched_calculate_form(team_id, before_date, games, n_games=5):
            # Convert before_date to match game date format
            import pandas as pd
            
            # Ensure before_date is properly formatted
            if before_date:
                before_date_dt = pd.to_datetime(before_date)
            else:
                return original_calculate_form(team_id, before_date, games, n_games)
            
            # Filter games with proper date comparison
            filtered_games = []
            for g in games:
                if not g.get('is_completed'):
                    continue
                    
                game_date = g.get('date')
                if game_date:
                    game_date_dt = pd.to_datetime(game_date) if not hasattr(game_date, 'year') else game_date
                    
                    # Only include games before the specified date
                    if game_date_dt < before_date_dt:
                        if g['home_team_id'] == team_id or g['away_team_id'] == team_id:
                            filtered_games.append(g)
                            if len(filtered_games) >= n_games:
                                break
            
            # Calculate form from filtered games
            if not filtered_games:
                return {
                    'points_per_game': 0.0,
                    'goals_for_per_game': 0.0,
                    'goals_against_per_game': 0.0,
                    'games_count': 0
                }
            
            # Calculate metrics
            total_points = 0
            total_goals_for = 0
            total_goals_against = 0
            
            for game in filtered_games:
                if game['home_team_id'] == team_id:
                    goals_for = game.get('home_score', 0)
                    goals_against = game.get('away_score', 0)
                    
                    # Points calculation
                    if game.get('went_to_shootout', False):
                        home_pens = game.get('home_penalties', 0)
                        away_pens = game.get('away_penalties', 0)
                        total_points += 2 if home_pens > away_pens else 1
                    elif goals_for > goals_against:
                        total_points += 3
                    elif goals_for == goals_against:
                        total_points += 1
                else:
                    goals_for = game.get('away_score', 0)
                    goals_against = game.get('home_score', 0)
                    
                    if game.get('went_to_shootout', False):
                        home_pens = game.get('home_penalties', 0)
                        away_pens = game.get('away_penalties', 0)
                        total_points += 2 if away_pens > home_pens else 1
                    elif goals_for > goals_against:
                        total_points += 3
                    elif goals_for == goals_against:
                        total_points += 1
                
                total_goals_for += goals_for
                total_goals_against += goals_against
            
            n = len(filtered_games)
            return {
                'points_per_game': total_points / n,
                'goals_for_per_game': total_goals_for / n,
                'goals_against_per_game': total_goals_against / n,
                'games_count': n
            }
        
        # Monkey patch the method
        predictor._calculate_form = patched_calculate_form
        
        # Extract features for each game
        features_list = []
        actual_goals = []
        game_info = []
        
        print("\n🔄 Extracting features for each game...")
        for game in conference_games:
            # Get games before this one for historical features
            game_date = game['date']
            
            # Ensure game_date is a datetime object
            if isinstance(game_date, str):
                from datetime import datetime
                try:
                    game_date = datetime.strptime(game_date.split()[0], '%Y-%m-%d')
                except:
                    continue
            
            games_before = [g for g in completed_games if g['date'] < game_date]
            
            # Skip if not enough history
            if len(games_before) < 10:
                continue
            
            # Extract features for home team
            # Ensure game_date is a string in YYYY-MM-DD format
            if hasattr(game_date, 'strftime'):
                game_date_str = game_date.strftime('%Y-%m-%d')
            else:
                game_date_str = str(game_date).split()[0]  # Get just the date part
            
            home_features = predictor._extract_features(
                team_id=game['home_team_id'],
                opponent_id=game['away_team_id'],
                is_home=True,
                game_date=game_date_str,
                games_before=games_before
            )
            
            # Extract features for away team
            away_features = predictor._extract_features(
                team_id=game['away_team_id'],
                opponent_id=game['home_team_id'],
                is_home=False,
                game_date=game_date_str,
                games_before=games_before
            )
            
            # Store features and actual results
            features_list.append(home_features)
            actual_goals.append(game['home_score'])
            game_info.append({
                'home_team': sim_data['conference_teams'].get(game['home_team_id'], 'Unknown'),
                'away_team': sim_data['conference_teams'].get(game['away_team_id'], 'Unknown'),
                'home_score': game['home_score'],
                'away_score': game['away_score']
            })
            
            features_list.append(away_features)
            actual_goals.append(game['away_score'])
            game_info.append({
                'home_team': sim_data['conference_teams'].get(game['home_team_id'], 'Unknown'),
                'away_team': sim_data['conference_teams'].get(game['away_team_id'], 'Unknown'),
                'home_score': game['home_score'],
                'away_score': game['away_score']
            })
        
        print(f"   Extracted features for {len(features_list)} team-game combinations")
        
        # Create DataFrame with features
        X_test = pd.DataFrame(features_list)
        
        # Remove league average features if they're in there (not used by model)
        if 'league_avg_xgf' in X_test.columns:
            X_test = X_test.drop(['league_avg_xgf', 'league_avg_xga'], axis=1)
        
        # Make predictions
        print("\n🎯 Making predictions...")
        predictions = ml_model.predict(X_test)
        
        # Calculate metrics
        y_test = np.array(actual_goals)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        print(f"\n📊 Test Set Statistics:")
        print(f"   Games evaluated: {len(features_list) // 2}")
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Actual goals range: {min(y_test)} - {max(y_test)}")
        print(f"   Predicted goals range: {predictions.min():.2f} - {predictions.max():.2f}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   R² Score: {r2:.4f} ({r2*100:.1f}% of variance explained)")
        print(f"   Mean Squared Error (MSE): {mse:.4f}")
        print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"   Mean Absolute Error (MAE): {mae:.4f}")
        
        # Residual analysis
        residuals = y_test - predictions
        print(f"\n📊 Residual Analysis:")
        print(f"   Mean residual: {np.mean(residuals):.4f}")
        print(f"   Std deviation: {np.std(residuals):.4f}")
        print(f"   Min residual: {residuals.min():.4f}")
        print(f"   Max residual: {residuals.max():.4f}")
        
        # Interpretation
        print(f"\n🎯 Model Interpretation:")
        if r2 > 0.3:
            print(f"   ✅ Good model fit for soccer prediction (R² > 0.3)")
        else:
            print(f"   ⚠️  Moderate model fit (R² = {r2:.3f})")
        
        print(f"   • On average, predictions are off by {mae:.2f} goals")
        print(f"   • 68% of predictions are within {rmse:.2f} goals of actual")
        print(f"   • The model explains {r2*100:.1f}% of goal-scoring variance")
        
        # Create visualizations
        print("\n📊 Creating diagnostic plots...")
        create_diagnostic_plots(y_test, predictions, residuals)
        
        # Save detailed results
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions,
            'residual': residuals,
            'abs_error': np.abs(residuals)
        })
        results_df.to_csv('output/ml_evaluation_results.csv', index=False)
        print(f"💾 Detailed results saved to: output/ml_evaluation_results.csv")
        
        # Show some example predictions
        print("\n📋 Sample Predictions (first 10):")
        print(f"{'Actual':<10}{'Predicted':<12}{'Error':<10}{'Team Type'}")
        print("-" * 45)
        for i in range(min(10, len(y_test))):
            team_type = "Home" if i % 2 == 0 else "Away"
            print(f"{y_test[i]:<10}{predictions[i]:<12.2f}{residuals[i]:<10.2f}{team_type}")
        
        # Feature importance
        print("\n🎯 Feature Importance (if available):")
        try:
            importance = ml_model.feature_importance(X_test)
            print("   Top 10 most important features:")
            for i, (feat, score) in enumerate(importance.head(10).items(), 1):
                print(f"   {i:2d}. {feat:<25} {score:.4f}")
        except:
            print("   (Feature importance calculation failed - need more data)")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()

def create_diagnostic_plots(y_test, predictions, residuals):
    """Create diagnostic plots for model evaluation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ML Model Diagnostic Plots', fontsize=16)
    
    # 1. Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(y_test, predictions, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add perfect prediction line
    max_val = max(max(y_test), max(predictions))
    ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect predictions')
    
    # Add trend line
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    ax1.plot(y_test, p(y_test), "b-", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax1.set_xlabel('Actual Goals')
    ax1.set_ylabel('Predicted Goals')
    ax1.set_title('Predicted vs Actual Goals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual Plot
    ax2 = axes[0, 1]
    ax2.scatter(predictions, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Goals')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # Add confidence bands
    std_resid = np.std(residuals)
    ax2.axhline(y=2*std_resid, color='orange', linestyle=':', alpha=0.7, label='±2 SD')
    ax2.axhline(y=-2*std_resid, color='orange', linestyle=':', alpha=0.7)
    ax2.legend()
    
    # 3. Residual Distribution
    ax3 = axes[1, 0]
    n, bins, patches = ax3.hist(residuals, bins=20, density=True, alpha=0.7, edgecolor='black')
    
    # Add normal distribution overlay
    mu, std = residuals.mean(), residuals.std()
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mu) / std) ** 2))
    ax3.plot(x, p, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
    
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Density')
    ax3.set_title('Residual Distribution')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.legend()
    
    # 4. Q-Q Plot
    ax4 = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Normal Q-Q Plot')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/ml_diagnostic_plots.png', dpi=300, bbox_inches='tight')
    print("   Diagnostic plots saved to: output/ml_diagnostic_plots.png")
    plt.close()
    
    # Additional plot: Error by goal count
    plt.figure(figsize=(10, 6))
    goal_errors = {}
    for actual, error in zip(y_test, np.abs(residuals)):
        actual = int(actual)
        if actual not in goal_errors:
            goal_errors[actual] = []
        goal_errors[actual].append(error)
    
    goals = sorted(goal_errors.keys())
    mean_errors = [np.mean(goal_errors[g]) for g in goals]
    std_errors = [np.std(goal_errors[g]) for g in goals]
    
    plt.errorbar(goals, mean_errors, yerr=std_errors, fmt='o-', capsize=5, capthick=2)
    plt.xlabel('Actual Goals Scored')
    plt.ylabel('Mean Absolute Error')
    plt.title('Prediction Error by Goal Count')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/error_by_goals.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import timezone_patch  # Apply timezone fix
    asyncio.run(evaluate_ml_model())