#!/usr/bin/env python3
"""
Simple workflow showing the complete process:
1. Connect to Railway
2. Train model (if needed)
3. Run predictions
4. See results
"""

import asyncio
import json
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

async def run_complete_workflow():
    """
    Complete workflow from data to predictions
    """
    print("🏃 Complete MLS Next Pro Prediction Workflow")
    print("=" * 60)
    
    conference = 'eastern'  # Focus on Eastern conference
    
    try:
        # Step 1: Connect to Railway
        print("\n1️⃣ Connecting to Railway database...")
        await connect()
        db_manager = DatabaseManager(database)
        print("✅ Connected!")
        
        # Step 2: Get data
        print(f"\n2️⃣ Loading {conference} conference data...")
        sim_data = await db_manager.get_data_for_simulation(conference, 2025)
        
        print(f"   Teams: {len(sim_data['conference_teams'])}")
        for team_id, team_name in list(sim_data['conference_teams'].items())[:5]:
            print(f"     - {team_name}")
        print("     ...")
        
        # Step 3: Initialize predictor (trains model if needed)
        print("\n3️⃣ Initializing predictor with AutoML...")
        print("   (First run will train the model - this takes a few minutes)")
        
        predictor = MLSNPRegSeasonPredictor(
            conference=conference,
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages={'league_avg_xgf': 1.2, 'league_avg_xga': 1.2},
            use_automl=True
        )
        
        if predictor.use_automl:
            print("✅ AutoML model ready!")
        else:
            print("⚠️  Using traditional method (AutoML not available)")
        
        # Step 4: Run a mini simulation
        print("\n4️⃣ Running mini simulation (100 iterations)...")
        summary_df, final_ranks, _, qual_data = predictor.run_simulations(n_simulations=100)
        
        # Step 5: Show results
        print("\n5️⃣ Results - Top 10 Teams by Playoff Probability:")
        print("-" * 80)
        print(f"{'Team':<30} {'Current Pts':>12} {'Avg Final Pts':>14} {'Playoff %':>12}")
        print("-" * 80)
        
        for idx, row in summary_df.head(10).iterrows():
            print(f"{row['Team']:<30} {row['Current Points']:>12} "
                  f"{row['Average Points']:>14.1f} {row['Playoff Qualification %']:>11.1f}%")
        
        # Find Chattanooga FC specifically
        print("\n6️⃣ Chattanooga FC Analysis:")
        cfc_row = summary_df[summary_df['Team'].str.contains('Chattanooga', case=False)]
        if not cfc_row.empty:
            cfc = cfc_row.iloc[0]
            print(f"   Current Points: {cfc['Current Points']}")
            print(f"   Current Rank: {int(cfc['Average Final Rank'])}")
            print(f"   Projected Final Points: {cfc['Average Points']:.1f}")
            print(f"   Playoff Probability: {cfc['Playoff Qualification %']:.1f}%")
            
            # Get the team_id for more analysis
            cfc_id = cfc['_team_id']
            
            # Show remaining games
            remaining_games = [g for g in predictor.remaining_games 
                             if g['home_team_id'] == cfc_id or g['away_team_id'] == cfc_id]
            
            print(f"\n   Remaining games ({len(remaining_games)}):")
            for game in remaining_games[:5]:
                if game['home_team_id'] == cfc_id:
                    opponent = sim_data['conference_teams'].get(game['away_team_id'], 'Unknown')
                    print(f"     vs {opponent} (H)")
                else:
                    opponent = sim_data['conference_teams'].get(game['home_team_id'], 'Unknown')
                    print(f"     @ {opponent} (A)")
            if len(remaining_games) > 5:
                print(f"     ... and {len(remaining_games) - 5} more")
        
        # Save results
        output_file = f"output/simulation_results_{conference}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = {
            'conference': conference,
            'n_simulations': 100,
            'run_date': datetime.now().isoformat(),
            'model_type': 'AutoML' if predictor.use_automl else 'Traditional',
            'teams': summary_df.to_dict('records')
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Feature importance check
        if predictor.use_automl and hasattr(predictor.ml_model, 'feature_importance'):
            print("\n7️⃣ Top Feature Importance (if available):")
            try:
                importance = predictor.ml_model.feature_importance()
                print(importance.head(10))
            except:
                print("   Feature importance not available for this model type")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await disconnect()
        print("\n✅ Workflow complete!")


if __name__ == "__main__":
    print("🚀 Starting MLS Next Pro Prediction Workflow")
    print("This will:")
    print("  1. Connect to Railway database")
    print("  2. Train AutoML model (if needed)")
    print("  3. Run 100 simulations")
    print("  4. Show results")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    asyncio.run(run_complete_workflow())