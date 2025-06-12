# File: quick_test.py
# A minimal test to isolate the simulation issue

import asyncio
import logging
from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

logging.basicConfig(level=logging.INFO)

async def quick_test():
    """Quick test of simulation without ML."""
    try:
        await connect()
        db_manager = DatabaseManager(database)
        
        print("Loading data...")
        sim_data = await db_manager.get_data_for_simulation("eastern", 2025)
        
        # Simple league averages
        league_averages = {"league_avg_xgf": 1.2, "league_avg_xga": 1.2}
        
        print("Creating predictor...")
        predictor = MLSNPRegSeasonPredictor(
            conference="eastern",
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages=league_averages
        )
        
        print(f"Remaining games: {len(predictor.remaining_games)}")
        print(f"Conference teams: {len(predictor.conference_teams)}")
        
        # Test single game simulation
        if predictor.remaining_games:
            test_game = predictor.remaining_games[0]
            print(f"Testing single game: {test_game['home_team_id']} vs {test_game['away_team_id']}")
            
            result = predictor._simulate_game(test_game)
            print(f"Single game result: {result}")
            
            if result is None:
                print("❌ _simulate_game returned None!")
                return
            
            # Test small simulation
            print("Running 10 simulations...")
            summary_df, _, _, _ = predictor.run_simulations(n_simulations=10)
            
            if summary_df is not None:
                print("✅ Small simulation successful!")
                print(summary_df[['Team', 'Current Points']].head())
            else:
                print("❌ Simulation returned None")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await disconnect()

if __name__ == "__main__":
    asyncio.run(quick_test())