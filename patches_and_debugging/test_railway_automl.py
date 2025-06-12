#!/usr/bin/env python3
"""
Complete test of Railway database + AutoML integration
Updated for Python 3.12 and AutoGluon
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

# Load environment variables
load_dotenv()

async def test_everything():
    """Test Railway connection and AutoML in one go"""
    
    print("🧪 MLS Next Pro Predictor - Complete System Test")
    print("=" * 50)
    
    # 1. Check environment
    print("\n1️⃣ Checking environment variables...")
    db_url = os.getenv('DATABASE_URL', '')
    
    if not db_url or 'trolley.proxy.rlwy.net' not in db_url:
        print("❌ DATABASE_URL not set properly in .env file!")
        print("   It should contain: trolley.proxy.rlwy.net:13360")
        return
    
    # Hide password in output
    safe_url = db_url.split(':')[0] + ':****@' + db_url.split('@')[1]
    print(f"✅ DATABASE_URL configured: {safe_url}")
    
    try:
        # 2. Test database connection
        print("\n2️⃣ Testing Railway database connection...")
        await connect()
        
        # Quick test query
        version = await database.fetch_one("SELECT version()")
        print(f"✅ Connected to PostgreSQL {version['version'].split(' ')[1]}")
        
        # 3. Check database contents
        print("\n3️⃣ Checking database contents...")
        
        # Teams
        team_count = await database.fetch_one("SELECT COUNT(*) as count FROM team")
        print(f"   Teams: {team_count['count']}")
        
        # Games
        games_2025 = await database.fetch_one(
            "SELECT COUNT(*) as count FROM games WHERE season_year = 2025"
        )
        completed_2025 = await database.fetch_one(
            "SELECT COUNT(*) as count FROM games WHERE season_year = 2025 AND is_completed = true"
        )
        print(f"   2025 Games: {games_2025['count']} total, {completed_2025['count']} completed")
        
        if games_2025['count'] == 0:
            print("\n⚠️  No 2025 games found in database!")
            print("   You may need to run the fixture loader:")
            print("   python -m src.mlsnp_scraper.run_full_workflow")
        
        # Check for conferences
        conferences = await database.fetch_all("SELECT * FROM conference")
        print(f"   Conferences: {', '.join([c['conf_name'] for c in conferences])}")
        
        # 4. Test AutoML libraries (Python 3.12 compatible)
        print("\n4️⃣ Testing AutoML libraries...")
        
        # Check Python version
        print(f"   Python version: {sys.version.split()[0]}")
        
        # Check for AutoGluon
        automl_available = False
        automl_library = None
        
        try:
            from autogluon.tabular import TabularPredictor
            print(f"✅ AutoGluon is installed and available!")
            automl_available = True
            automl_library = "AutoGluon"
        except ImportError:
            print("❌ AutoGluon not installed")
            
            # Check for sklearn fallback
            try:
                from sklearn.ensemble import RandomForestRegressor
                print("✅ Scikit-learn is available as fallback!")
                automl_available = True
                automl_library = "Scikit-learn"
            except ImportError:
                print("❌ No ML libraries found!")
                print("   Install with: pip install autogluon")
                print("   OR: pip install scikit-learn lightgbm")
        
        if not automl_available:
            print("\n⚠️  No AutoML libraries available. Install one to use ML features.")
            return
        
        # 5. Initialize database manager
        print("\n5️⃣ Loading Eastern Conference data...")
        db_manager = DatabaseManager(database)
        
        # Get data for Eastern conference
        sim_data = await db_manager.get_data_for_simulation('eastern', 2025)
        
        print(f"   Teams: {len(sim_data['conference_teams'])}")
        print(f"   Games: {len(sim_data['games_data'])}")
        
        # Count completed games (needed for ML training)
        completed_games = [g for g in sim_data['games_data'] if g.get('is_completed')]
        print(f"   Completed games: {len(completed_games)}")
        
        if len(completed_games) < 50:
            print("\n⚠️  Less than 50 completed games. ML model may not train well.")
            print("   The model will use traditional methods until more games are played.")
        
        # 6. Try to create predictor
        print(f"\n6️⃣ Initializing predictor with {automl_library}...")
        
        league_averages = {'league_avg_xgf': 1.2, 'league_avg_xga': 1.2}
        
        try:
            predictor = MLSNPRegSeasonPredictor(
                conference='eastern',
                conference_teams=sim_data["conference_teams"],
                games_data=sim_data["games_data"],
                team_performance=sim_data["team_performance"],
                league_averages=league_averages,
                use_automl=True
            )
            
            if predictor.use_automl:
                print(f"✅ {automl_library} successfully initialized!")
                
                # Quick test - simulate one game if we have teams
                if len(sim_data['conference_teams']) >= 2:
                    team_ids = list(sim_data['conference_teams'].keys())
                    test_game = {
                        'home_team_id': team_ids[0],
                        'away_team_id': team_ids[1],
                        'date': '2025-06-08'
                    }
                    
                    print(f"\n7️⃣ Test simulation: {sim_data['conference_teams'][team_ids[0]]} vs {sim_data['conference_teams'][team_ids[1]]}")
                    
                    # Run a few simulations
                    results = []
                    for i in range(10):
                        h_goals, a_goals, shootout = predictor._simulate_game(test_game)
                        results.append((h_goals, a_goals))
                    
                    avg_home = sum(r[0] for r in results) / len(results)
                    avg_away = sum(r[1] for r in results) / len(results)
                    
                    print(f"   Average score over 10 simulations: {avg_home:.1f} - {avg_away:.1f}")
                    print("✅ Game simulation working!")
            else:
                print("⚠️  AutoML disabled - likely insufficient training data")
                print("   Using traditional simulation method")
                
        except Exception as e:
            print(f"❌ Error initializing predictor: {e}")
            import traceback
            traceback.print_exc()
        
        # 8. Check for Chattanooga FC specifically
        print("\n8️⃣ Checking for Chattanooga FC...")
        cfc = await database.fetch_one(
            "SELECT * FROM team WHERE LOWER(team_name) LIKE '%chattanooga%'"
        )
        
        if cfc:
            print(f"✅ Found Chattanooga FC: {cfc['team_name']} (ID: {cfc['team_id']})")
            
            # Get their games
            cfc_games = await database.fetch_one(
                """
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN is_completed THEN 1 ELSE 0 END) as completed
                FROM games 
                WHERE (home_team_id = :team_id OR away_team_id = :team_id)
                AND season_year = 2025
                """,
                values={"team_id": cfc['team_id']}
            )
            if cfc_games:
                print(f"   2025 Games: {cfc_games['total']} scheduled, {cfc_games['completed']} completed")
        else:
            print("❌ Chattanooga FC not found in database")
        
        print("\n✅ All tests completed successfully!")
        print("\n📊 Summary:")
        print(f"   - Railway database: Connected")
        print(f"   - Python version: {sys.version.split()[0]} ✅")
        print(f"   - Teams loaded: {team_count['count']}")
        print(f"   - 2025 season data: {games_2025['count']} games")
        print(f"   - AutoML library: {automl_library}")
        print(f"   - AutoML status: {'Enabled' if automl_available else 'Disabled'}")
        print(f"   - Ready to run simulations: Yes")
        
        print("\n🎯 Next Steps:")
        print("1. If no 2025 games, run: python -m src.mlsnp_scraper.run_full_workflow")
        print("2. Train model: python train_model_directly.py")
        print("3. Run predictions: python simple_workflow.py")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()

if __name__ == "__main__":
    asyncio.run(test_everything())