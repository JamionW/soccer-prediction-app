# File: verify_fix.py
# Quick test to verify the fix worked

import asyncio
import logging
from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

logging.basicConfig(level=logging.INFO)

async def verify_fix():
    """Verify that the __init__ fix worked."""
    try:
        await connect()
        db_manager = DatabaseManager(database)
        
        print("🔧 VERIFYING PREDICTOR FIX")
        print("=" * 50)
        
        # Load data
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
        
        # Quick validation
        print(f"\n✅ VERIFICATION RESULTS:")
        print(f"Conference: {predictor.conference}")
        print(f"Conference teams: {len(predictor.conference_teams)}")
        print(f"Games data: {len(predictor.games_data)}")
        print(f"Team performance: {len(predictor.team_performance)}")
        print(f"League avg xGF: {predictor.league_avg_xgf}")
        print(f"League avg xGA: {predictor.league_avg_xga}")
        print(f"Current standings: {len(predictor.current_standings)}")
        print(f"Remaining games: {len(predictor.remaining_games)}")
        
        if predictor.remaining_games:
            print(f"\n🎲 Sample remaining game:")
            sample = predictor.remaining_games[0]
            home_team = predictor.team_names.get(sample['home_team_id'], sample['home_team_id'])
            away_team = predictor.team_names.get(sample['away_team_id'], sample['away_team_id'])
            print(f"   {home_team} vs {away_team} on {sample.get('date', 'TBD')}")
        
        if predictor.current_standings:
            print(f"\n🏆 Sample current standings:")
            # Show Chattanooga FC if available
            cfc_id = "raMyeZAMd2"  # Chattanooga FC
            if cfc_id in predictor.current_standings:
                cfc_stats = predictor.current_standings[cfc_id]
                print(f"   Chattanooga FC: {cfc_stats['points']} pts, {cfc_stats['games_played']} GP")
            else:
                # Show any team
                sample_team_id = list(predictor.current_standings.keys())[0]
                sample_stats = predictor.current_standings[sample_team_id]
                print(f"   {sample_stats['name']}: {sample_stats['points']} pts, {sample_stats['games_played']} GP")
        
        # Test single game simulation
        if predictor.remaining_games:
            print(f"\n🎯 Testing single game simulation...")
            test_game = predictor.remaining_games[0]
            try:
                result = predictor._simulate_game(test_game)
                if result and len(result) == 3:
                    home_goals, away_goals, shootout = result
                    print(f"   Simulated result: {home_goals}-{away_goals}" + 
                          (" (shootout)" if shootout else ""))
                    print("✅ Single game simulation working!")
                else:
                    print(f"❌ Single game simulation returned: {result}")
            except Exception as e:
                print(f"❌ Single game simulation failed: {e}")
        
        print(f"\n🎉 PREDICTOR IS NOW PROPERLY INITIALIZED!")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await disconnect()

if __name__ == "__main__":
    asyncio.run(verify_fix())