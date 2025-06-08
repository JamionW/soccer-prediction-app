#!/usr/bin/env python3
"""
Train the AutoML model directly without going through the API
This helps you understand and debug the training process
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def train_model_for_conference(conference: str = 'eastern'):
    """
    Train an AutoML model for a specific conference
    """
    print(f"\n🎯 Training AutoML model for {conference.upper()} Conference")
    print("=" * 60)
    
    try:
        # 1. Connect to Railway database
        print("\n1️⃣ Connecting to Railway database...")
        await connect()
        print("✅ Connected!")
        
        # 2. Initialize database manager
        db_manager = DatabaseManager(database)
        
        # 3. Fetch all data needed for the conference
        print(f"\n2️⃣ Fetching {conference} conference data...")
        sim_data = await db_manager.get_data_for_simulation(conference, 2025)
        
        print(f"   Teams: {len(sim_data['conference_teams'])}")
        print(f"   Total games: {len(sim_data['games_data'])}")
        
        # Count completed games (needed for training)
        completed_games = [g for g in sim_data['games_data'] if g.get('is_completed')]
        print(f"   Completed games: {len(completed_games)} (needed for training)")
        
        if len(completed_games) < 50:
            print("\n⚠️  Warning: Less than 50 completed games. Model may not train well.")
            print("   Consider loading historical data or waiting for more games.")
        
        # 4. Calculate league averages
        print("\n3️⃣ Calculating league averages...")
        league_averages = {
            'league_avg_xgf': 1.2,  # You could calculate this from actual data
            'league_avg_xga': 1.2
        }
        
        # 5. Create predictor (this triggers training)
        print(f"\n4️⃣ Initializing predictor (this will train the model)...")
        print("   This may take a few minutes on first run...")
        
        predictor = MLSNPRegSeasonPredictor(
            conference=conference,
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages=league_averages,
            use_automl=True  # Force AutoML
        )
        
        # 6. Check if model was trained
        if predictor.use_automl and predictor.ml_model:
            print("\n✅ Model trained successfully!")
            
            # Get model info
            model_path = predictor.model_path
            print(f"   Model saved at: {model_path}")
            
            # If using AutoGluon, we can get more info
            if hasattr(predictor.ml_model, 'get_model_best'):
                print(f"   Best model type: {predictor.ml_model.get_model_best()}")
            
            # 7. Test the model with a sample prediction
            print("\n5️⃣ Testing model with sample prediction...")
            
            # Get two teams for a test matchup
            team_ids = list(sim_data['conference_teams'].keys())
            if len(team_ids) >= 2:
                test_game = {
                    'home_team_id': team_ids[0],
                    'away_team_id': team_ids[1],
                    'date': '2025-06-15'
                }
                
                # Run multiple simulations to see the distribution
                results = []
                for _ in range(100):
                    h_goals, a_goals, shootout = predictor._simulate_game(test_game)
                    results.append((h_goals, a_goals))
                
                avg_home = sum(r[0] for r in results) / len(results)
                avg_away = sum(r[1] for r in results) / len(results)
                
                print(f"\n   Test Matchup: {sim_data['conference_teams'][team_ids[0]]} vs {sim_data['conference_teams'][team_ids[1]]}")
                print(f"   Average predicted score: {avg_home:.2f} - {avg_away:.2f}")
                print(f"   Home wins: {sum(1 for r in results if r[0] > r[1])}%")
                print(f"   Draws: {sum(1 for r in results if r[0] == r[1])}%")
                print(f"   Away wins: {sum(1 for r in results if r[0] < r[1])}%")
                
            # 8. Show feature importance (if available)
            print("\n6️⃣ Analyzing feature importance...")
            
            # Get a sample of features to see what's being used
            if team_ids:
                sample_features = predictor._extract_features(
                    team_id=team_ids[0],
                    opponent_id=team_ids[1] if len(team_ids) > 1 else team_ids[0],
                    is_home=True,
                    game_date='2025-06-15',
                    games_before=sim_data['games_data']
                )
                
                print(f"\n   Total features extracted: {len(sample_features)}")
                print("\n   Sample features:")
                for i, (feat, value) in enumerate(list(sample_features.items())[:10]):
                    print(f"     {feat}: {value:.3f}")
                
                # Check for Chattanooga-specific features
                cfc_features = [k for k in sample_features.keys() if 'chattanooga' in k]
                if cfc_features:
                    print(f"\n   Chattanooga FC specific features found: {len(cfc_features)}")
                    for feat in cfc_features:
                        print(f"     - {feat}")
            
        else:
            print("\n❌ Model training failed or was disabled")
            print("   Check the logs above for errors")
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()
        print("\n✅ Done!")


async def main():
    """Main function to train models"""
    
    # Train Eastern Conference model
    await train_model_for_conference('eastern')
    
    # Optionally train Western Conference model too
    print("\n" + "="*60)
    print("Would you like to train the Western Conference model too? (y/n)")
    if input().lower() == 'y':
        await train_model_for_conference('western')


if __name__ == "__main__":
    print("🚀 MLS Next Pro AutoML Model Trainer")
    print("=====================================")
    asyncio.run(main())