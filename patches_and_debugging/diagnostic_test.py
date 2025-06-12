# File: diagnostic_test.py
# Let's see what's actually in your predictor object

import asyncio
import logging
from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

logging.basicConfig(level=logging.INFO)

async def diagnostic_test():
    """Diagnose what's missing in the predictor initialization."""
    try:
        await connect()
        db_manager = DatabaseManager(database)
        
        print("🔍 DIAGNOSTIC TEST - Predictor Initialization")
        print("=" * 60)
        
        print("Loading data...")
        sim_data = await db_manager.get_data_for_simulation("eastern", 2025)
        league_averages = {"league_avg_xgf": 1.2, "league_avg_xga": 1.2}
        
        print("Creating predictor...")
        predictor = MLSNPRegSeasonPredictor(
            conference="eastern",
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages=league_averages
        )
        
        print("\n🔎 PREDICTOR ATTRIBUTES CHECK:")
        print("-" * 40)
        
        # Check all expected attributes
        attributes_to_check = [
            'conference',
            'conference_teams', 
            'team_names',
            'games_data',
            'team_performance',
            'league_avg_xgf',
            'league_avg_xga',
            'current_standings',
            'remaining_games'
        ]
        
        for attr in attributes_to_check:
            if hasattr(predictor, attr):
                value = getattr(predictor, attr)
                if isinstance(value, (list, dict, set)):
                    print(f"✅ {attr}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"✅ {attr}: {value}")
            else:
                print(f"❌ {attr}: MISSING!")
        
        print(f"\n📊 DATA SUMMARY:")
        print(f"Conference: {predictor.conference}")
        
        if hasattr(predictor, 'conference_teams'):
            print(f"Conference teams: {len(predictor.conference_teams)}")
            
        if hasattr(predictor, 'games_data'):
            total_games = len(predictor.games_data)
            completed = sum(1 for g in predictor.games_data if g.get('is_completed', False))
            print(f"Total games: {total_games}")
            print(f"Completed games: {completed}")
            print(f"Incomplete games: {total_games - completed}")
        
        if hasattr(predictor, 'remaining_games'):
            print(f"Remaining games to simulate: {len(predictor.remaining_games)}")
            if predictor.remaining_games:
                sample_game = predictor.remaining_games[0]
                print(f"Sample remaining game: {sample_game.get('home_team_id')} vs {sample_game.get('away_team_id')}")
        else:
            print("❌ remaining_games attribute is missing!")
            
        if hasattr(predictor, 'current_standings'):
            print(f"Teams in current standings: {len(predictor.current_standings)}")
            if predictor.current_standings:
                # Show a sample team's standings
                sample_team_id = list(predictor.current_standings.keys())[0]
                sample_standings = predictor.current_standings[sample_team_id]
                print(f"Sample standings entry: {sample_standings}")
        else:
            print("❌ current_standings attribute is missing!")
            
        # Try to call the missing methods manually if they exist
        print(f"\n🔧 METHOD CHECK:")
        print("-" * 40)
        
        methods_to_check = [
            '_calculate_current_standings',
            '_filter_remaining_games',
            '_simulate_game',
            'run_simulations'
        ]
        
        for method in methods_to_check:
            if hasattr(predictor, method):
                print(f"✅ {method}: exists")
            else:
                print(f"❌ {method}: MISSING!")
        
        # If remaining_games is missing, try to create it manually
        if not hasattr(predictor, 'remaining_games'):
            print(f"\n🔨 TRYING TO FIX remaining_games manually...")
            try:
                if hasattr(predictor, '_filter_remaining_games'):
                    predictor.remaining_games = predictor._filter_remaining_games()
                    print(f"✅ Manually set remaining_games: {len(predictor.remaining_games)} games")
                else:
                    # Create it manually
                    remaining = []
                    for game in predictor.games_data:
                        if not game.get("is_completed") and \
                           game.get("home_team_id") in predictor.conference_teams and \
                           game.get("away_team_id") in predictor.conference_teams:
                            remaining.append(game)
                    predictor.remaining_games = remaining
                    print(f"✅ Manually created remaining_games: {len(remaining)} games")
                    
            except Exception as e:
                print(f"❌ Failed to fix remaining_games: {e}")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await disconnect()

if __name__ == "__main__":
    asyncio.run(diagnostic_test())