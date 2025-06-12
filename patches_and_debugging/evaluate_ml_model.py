import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import asyncio
from datetime import datetime
from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelEvaluator:
    def __init__(self, model_path: str, conference: str = "eastern"):
        self.model_path = model_path
        self.conference = conference
        self.model = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        
    def load_model(self):
        """Load the trained model from pickle file"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model from {self.model_path}")
        
    async def prepare_test_data(self, db_manager: DatabaseManager):
        """Prepare test data from completed games"""
        # Get completed games
        games = await db_manager.get_games_for_season(2025, self.conference)
        completed_games = [g for g in games if g.get('is_completed')]
        
        logger.info(f"Found {len(completed_games)} completed games for evaluation")
        
        # Get team performance data
        conference_teams = await db_manager.get_conference_teams(self.conference, 2025)
        
        features_list = []
        targets = []
        
        for game in completed_games:
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            
            # Skip if not conference game
            if home_id not in conference_teams or away_id not in conference_teams:
                continue
            
            # Get team stats up to this game
            features = await self._extract_features_for_game(
                game, home_id, away_id, completed_games, db_manager
            )
            
            if features is not None:
                features_list.append(features)
                # Target is actual home goals
                targets.append(game['home_score'])
        
        if features_list:
            self.X_test = np.array(features_list)
            self.y_test = np.array(targets)
            logger.info(f"Prepared {len(features_list)} games for evaluation")
        else:
            raise ValueError("No valid test data found")
    
    async def _extract_features_for_game(self, game, home_id, away_id, all_games, db_manager):
        """Extract features for a single game"""
        # This should match your training feature extraction
        # Example features (adjust based on your actual implementation):
        
        game_date = game['date']
        
        # Calculate team stats up to this game
        home_stats = self._calculate_team_stats(home_id, all_games, game_date)
        away_stats = self._calculate_team_stats(away_id, all_games, game_date)
        
        if home_stats['games'] < 3 or away_stats['games'] < 3:
            return None  # Not enough data
        
        features = [
            # Home team features
            home_stats['avg_goals_for'],
            home_stats['avg_goals_against'],
            home_stats['win_rate'],
            home_stats['recent_form'],
            home_stats['home_avg_goals'],
            
            # Away team features
            away_stats['avg_goals_for'],
            away_stats['avg_goals_against'],
            away_stats['win_rate'],
            away_stats['recent_form'],
            away_stats['away_avg_goals'],
            
            # Head-to-head
            self._calculate_h2h_stats(home_id, away_id, all_games, game_date),
            
            # Days since last game
            home_stats['days_since_last'],
            away_stats['days_since_last']
        ]
        
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
        home_goals = []
        away_goals = []
        
        for g in team_games:
            if g['home_team_id'] == team_id:
                gf = g['home_score']
                ga = g['away_score']
                home_goals.append(gf)
            else:
                gf = g['away_score']
                ga = g['home_score']
                away_goals.append(gf)
            
            goals_for += gf
            goals_against += ga
            if gf > ga:
                wins += 1
        
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
            'recent_form': recent_points / (len(recent_games) * 3),  # Normalized
            'home_avg_goals': np.mean(home_goals) if home_goals else 0,
            'away_avg_goals': np.mean(away_goals) if away_goals else 0,
            'days_since_last': min(days_since, 30)  # Cap at 30 days
        }
    
    def _calculate_h2h_stats(self, home_id, away_id, games, before_date):
        """Calculate head-to-head statistics"""
        h2h_games = []
        for g in games:
            if g['date'] >= before_date or not g['is_completed']:
                continue
            if (g['home_team_id'] == home_id and g['away_team_id'] == away_id) or \
               (g['home_team_id'] == away_id and g['away_team_id'] == home_id):
                h2h_games.append(g)
        
        if not h2h_games:
            return 0  # No H2H history
        
        # Calculate goal difference for home team
        total_diff = 0
        for g in h2h_games:
            if g['home_team_id'] == home_id:
                total_diff += g['home_score'] - g['away_score']
            else:
                total_diff += g['away_score'] - g['home_score']
        
        return total_diff / len(h2h_games)
    
    def evaluate_model(self):
        """Calculate model evaluation metrics"""
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test data available. Run prepare_test_data first.")
        
        # Make predictions
        self.predictions = self.model.predict(self.X_test)
        
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
    
    def print_evaluation_report(self, metrics):
        """Print a detailed evaluation report"""
        print("\n" + "="*70)
        print("ML MODEL EVALUATION REPORT")
        print("="*70)
        print(f"\nModel: {self.model_path}")
        print(f"Test Set Size: {len(self.y_test)} games")
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
        ax1.plot([0, max(self.y_test)], [0, max(self.y_test)], 'r--', lw=2)
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
        
        # 4. Q-Q Plot
        ax4 = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f"ml_model_diagnostics_{self.conference}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Diagnostic plots saved to: {output_path}")
        
        plt.close()

async def main():
    """Main evaluation function"""
    print("\n🔬 MLS Next Pro ML Model Evaluation")
    print("="*70)
    
    # Connect to database
    await connect()
    db_manager = DatabaseManager(database)
    await db_manager.initialize()
    
    try:
        # Evaluate Eastern Conference model
        evaluator = MLModelEvaluator(
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
        results_df.to_csv('output/ml_model_evaluation_results.csv', index=False)
        print(f"\n💾 Detailed results saved to: output/ml_model_evaluation_results.csv")
        
    finally:
        await disconnect()

if __name__ == "__main__":
    asyncio.run(main())