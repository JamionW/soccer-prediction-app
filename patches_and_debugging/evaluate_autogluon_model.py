import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import asyncio
from datetime import datetime
from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoGluonEvaluator:
    def __init__(self, model_path: str, conference: str = "eastern"):
        self.model_path = model_path
        self.conference = conference
        self.predictor = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.test_df = None
        
    def load_model(self):
        """Load the AutoGluon predictor"""
        self.predictor = TabularPredictor.load(self.model_path)
        logger.info(f"✅ Loaded AutoGluon predictor from {self.model_path}")
        
        # Get model info
        model_info = self.predictor.info()
        logger.info(f"   Problem type: {model_info.get('problem_type', 'Unknown')}")
        logger.info(f"   Eval metric: {model_info.get('eval_metric', 'Unknown')}")
        logger.info(f"   Number of models: {len(self.predictor.model_names())}")
        
    async def prepare_test_data(self, db_manager: DatabaseManager):
        """Prepare test data from completed games"""
        # Get completed games
        games = await db_manager.get_games_for_season(2025, self.conference)
        completed_games = [g for g in games if g.get('is_completed')]
        
        logger.info(f"Found {len(completed_games)} completed games for evaluation")
        
        # Get team performance data
        conference_teams = await db_manager.get_conference_teams(self.conference, 2025)
        
        features_list = []
        
        for game in completed_games:
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            
            # Skip if not conference game
            if home_id not in conference_teams or away_id not in conference_teams:
                continue
            
            # Extract features for this game
            features = await self._extract_features_for_game(
                game, home_id, away_id, completed_games, db_manager
            )
            
            if features is not None:
                features_list.append(features)
        
        if features_list:
            # Create DataFrame with proper column names
            self.test_df = pd.DataFrame(features_list)
            self.y_test = self.test_df['home_goals'].values
            
            # Remove target from features
            feature_cols = [col for col in self.test_df.columns if col != 'home_goals']
            self.X_test = self.test_df[feature_cols]
            
            logger.info(f"Prepared {len(features_list)} games for evaluation")
            logger.info(f"Features: {list(self.X_test.columns)}")
        else:
            raise ValueError("No valid test data found")
    
    async def _extract_features_for_game(self, game, home_id, away_id, all_games, db_manager):
        """Extract features for a single game - matching your training features"""
        game_date = game['date']
        
        # Calculate team stats up to this game
        home_stats = self._calculate_team_stats(home_id, all_games, game_date)
        away_stats = self._calculate_team_stats(away_id, all_games, game_date)
        
        if home_stats['games'] < 3 or away_stats['games'] < 3:
            return None  # Not enough data
        
        # Get xG data if available
        home_xg = await db_manager.get_or_fetch_team_xg(home_id, 2025)
        away_xg = await db_manager.get_or_fetch_team_xg(away_id, 2025)
        
        features = {
            # Home team features
            'home_avg_goals_for': home_stats['avg_goals_for'],
            'home_avg_goals_against': home_stats['avg_goals_against'],
            'home_win_rate': home_stats['win_rate'],
            'home_points_per_game': home_stats['points_per_game'],
            'home_recent_form': home_stats['recent_form'],
            'home_games_played': home_stats['games'],
            
            # Away team features
            'away_avg_goals_for': away_stats['avg_goals_for'],
            'away_avg_goals_against': away_stats['avg_goals_against'],
            'away_win_rate': away_stats['win_rate'],
            'away_points_per_game': away_stats['points_per_game'],
            'away_recent_form': away_stats['recent_form'],
            'away_games_played': away_stats['games'],
            
            # Goal differentials
            'home_goal_diff': home_stats['avg_goals_for'] - home_stats['avg_goals_against'],
            'away_goal_diff': away_stats['avg_goals_for'] - away_stats['avg_goals_against'],
            
            # Form difference
            'form_difference': home_stats['recent_form'] - away_stats['recent_form'],
            
            # Rest days
            'home_days_rest': min(home_stats['days_since_last'], 14),
            'away_days_rest': min(away_stats['days_since_last'], 14),
            
            # Target variable
            'home_goals': game['home_score']
        }
        
        # Add xG features if available
        if home_xg.get('games_played', 0) > 0:
            features['home_xg_for'] = home_xg.get('x_goals_for', 0) / home_xg['games_played']
            features['home_xg_against'] = home_xg.get('x_goals_against', 0) / home_xg['games_played']
        
        if away_xg.get('games_played', 0) > 0:
            features['away_xg_for'] = away_xg.get('x_goals_for', 0) / away_xg['games_played']
            features['away_xg_against'] = away_xg.get('x_goals_against', 0) / away_xg['games_played']
        
        return features
    
    def _calculate_team_stats(self, team_id, games, before_date):
        """Calculate team statistics before a certain date"""
        team_games = []
        for g in games:
            if g['date'] >= before_date or not g['is_completed']:
                continue
            if g['home_team_id'] == team_id or g['away_team_id'] == team_id:
                team_games.append(g)
        
        if not team_games:
            return {'games': 0}
        
        # Sort by date
        team_games.sort(key=lambda x: x['date'])
        
        # Calculate stats
        goals_for = 0
        goals_against = 0
        wins = 0
        draws = 0
        points = 0
        
        for g in team_games:
            if g['home_team_id'] == team_id:
                gf = g['home_score']
                ga = g['away_score']
            else:
                gf = g['away_score']
                ga = g['home_score']
            
            goals_for += gf
            goals_against += ga
            
            if gf > ga:
                wins += 1
                points += 3
            elif gf == ga:
                draws += 1
                points += 1
        
        # Recent form (last 5 games)
        recent_games = team_games[-5:]
        recent_points = 0
        for g in recent_games:
            if g['home_team_id'] == team_id:
                if g['home_score'] > g['away_score']:
                    recent_points += 3
                elif g['home_score'] == g['away_score']:
                    recent_points += 1
            else:
                if g['away_score'] > g['home_score']:
                    recent_points += 3
                elif g['away_score'] == g['home_score']:
                    recent_points += 1
        
        # Days since last game
        last_game_date = team_games[-1]['date']
        days_since = (before_date - last_game_date).days
        
        return {
            'games': len(team_games),
            'avg_goals_for': goals_for / len(team_games),
            'avg_goals_against': goals_against / len(team_games),
            'win_rate': wins / len(team_games),
            'points_per_game': points / len(team_games),
            'recent_form': recent_points / (len(recent_games) * 3) if recent_games else 0.5,
            'days_since_last': days_since
        }
    
    def evaluate_model(self):
        """Evaluate the AutoGluon model"""
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test data available. Run prepare_test_data first.")
        
        # Make predictions
        self.predictions = self.predictor.predict(self.X_test)
        
        # Calculate metrics
        r2 = r2_score(self.y_test, self.predictions)
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.predictions)
        
        # Calculate residuals
        residuals = self.y_test - self.predictions
        
        metrics = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'residuals': residuals,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals)
        }
        
        return metrics
    
    def get_feature_importance(self):
        """Get feature importance from AutoGluon"""
        try:
            importance = self.predictor.feature_importance(self.X_test)
            return importance
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return None
    
    def get_model_leaderboard(self):
        """Get the leaderboard of all models"""
        try:
            leaderboard = self.predictor.leaderboard(self.X_test, y=self.y_test, silent=True)
            return leaderboard
        except Exception as e:
            logger.warning(f"Could not get leaderboard: {e}")
            return None
    
    def print_evaluation_report(self, metrics):
        """Print a detailed evaluation report"""
        print("\n" + "="*70)
        print("AUTOGLUON MODEL EVALUATION REPORT")
        print("="*70)
        print(f"\nModel: {self.model_path}")
        print(f"Test Set Size: {len(self.y_test)} games")
        
        # Model composition
        print(f"\n🤖 MODEL COMPOSITION:")
        model_names = self.predictor.model_names()
        print(f"   Total models in ensemble: {len(model_names)}")
        print(f"   Best model: {self.predictor.model_best}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   R² Score: {metrics['r2_score']:.4f}")
        print(f"   Mean Squared Error (MSE): {metrics['mse']:.4f}")
        print(f"   Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"   Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        
        print(f"\n📈 RESIDUAL ANALYSIS:")
        print(f"   Mean Residual: {metrics['mean_residual']:.4f}")
        print(f"   Std Dev of Residuals: {metrics['std_residual']:.4f}")
        print(f"   Min Residual: {metrics['residuals'].min():.4f}")
        print(f"   Max Residual: {metrics['residuals'].max():.4f}")
        
        # Feature importance
        importance = self.get_feature_importance()
        if importance is not None:
            print(f"\n🎯 TOP 10 FEATURE IMPORTANCES:")
            for i, (feature, score) in enumerate(importance.head(10).items()):
                print(f"   {i+1}. {feature}: {score:.4f}")
        
        # Model leaderboard
        leaderboard = self.get_model_leaderboard()
        if leaderboard is not None:
            print(f"\n🏆 MODEL LEADERBOARD (Top 5):")
            print(leaderboard[['model', 'score_test', 'pred_time_test', 'fit_time']].head())
        
        print(f"\n🎯 INTERPRETATION:")
        if metrics['r2_score'] > 0.7:
            print(f"   ✅ Excellent model fit (R² > 0.7)")
        elif metrics['r2_score'] > 0.5:
            print(f"   ✅ Good model fit (R² > 0.5)")
        elif metrics['r2_score'] > 0.3:
            print(f"   ⚠️  Moderate model fit (R² > 0.3)")
        else:
            print(f"   ❌ Poor model fit (R² < 0.3)")
        
        print(f"\n   The model explains {metrics['r2_score']*100:.1f}% of the variance in goals scored.")
        print(f"   On average, predictions are off by {metrics['mae']:.2f} goals.")
    
    def plot_diagnostics(self, output_dir="output"):
        """Create diagnostic plots"""
        if self.predictions is None:
            raise ValueError("No predictions available. Run evaluate_model first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Predicted vs Actual
        ax1 = axes[0, 0]
        ax1.scatter(self.y_test, self.predictions, alpha=0.6)
        max_val = max(max(self.y_test), max(self.predictions))
        ax1.plot([0, max_val], [0, max_val], 'r--', lw=2)
        ax1.set_xlabel('Actual Goals')
        ax1.set_ylabel('Predicted Goals')
        ax1.set_title('Predicted vs Actual Goals')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residual Plot
        ax2 = axes[0, 1]
        residuals = self.y_test - self.predictions
        ax2.scatter(self.predictions, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Goals')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution')
        ax3.axvline(x=0, color='r', linestyle='--')
        
        # 4. Feature Importance (if available)
        ax4 = axes[1, 1]
        importance = self.get_feature_importance()
        if importance is not None:
            top_features = importance.head(10)
            ax4.barh(range(len(top_features)), top_features.values)
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features.index)
            ax4.set_xlabel('Importance')
            ax4.set_title('Top 10 Feature Importances')
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f"autogluon_model_diagnostics_{self.conference}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Diagnostic plots saved to: {output_path}")
        
        plt.close()

async def main():
    """Main evaluation function"""
    print("\n🔬 MLS Next Pro AutoGluon Model Evaluation")
    print("="*70)
    
    # Connect to database
    await connect()
    db_manager = DatabaseManager(database)
    await db_manager.initialize()
    
    try:
        # Evaluate Eastern Conference model
        evaluator = AutoGluonEvaluator(
            model_path="models/xg_predictor_eastern_202506.pkl",
            conference="eastern"
        )
        
        # Load model
        evaluator.load_model()
        
        # Prepare test data
        await evaluator.prepare_test_data(db_manager)
        
        # Evaluate
        metrics = evaluator.evaluate_model()
        
        # Print report
        evaluator.print_evaluation_report(metrics)
        
        # Create plots
        evaluator.plot_diagnostics()
        
        # Save detailed results
        results_df = pd.DataFrame({
            'actual': evaluator.y_test,
            'predicted': evaluator.predictions,
            'residual': metrics['residuals']
        })
        results_df.to_csv('output/autogluon_evaluation_results.csv', index=False)
        print(f"\n💾 Detailed results saved to: output/autogluon_evaluation_results.csv")
        
    finally:
        await disconnect()

if __name__ == "__main__":
    asyncio.run(main())