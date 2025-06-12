#!/usr/bin/env python3
"""
Recover from interrupted ML training and fix the simulation issue
"""

import asyncio
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Apply timezone fix
import timezone_patch

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

async def recover_and_retrain():
    """Clean up and retrain the ML model properly"""
    
    print("🔧 ML Training Recovery Tool")
    print("=" * 60)
    
    # Step 1: Clean up old model files
    print("\n1️⃣ Cleaning up interrupted training files...")
    
    model_dir = Path("models")
    if model_dir.exists():
        for model_file in model_dir.glob("*.pkl"):
            print(f"   Removing: {model_file}")
            model_file.unlink()
        print("   ✅ Cleaned up old models")
    else:
        model_dir.mkdir(exist_ok=True)
        print("   ✅ Created models directory")
    
    # Also clean up any AutoGluon directories
    autogluon_dir = Path("autogluon_test")
    if autogluon_dir.exists():
        import shutil
        shutil.rmtree(autogluon_dir)
        print("   ✅ Cleaned up AutoGluon test directory")
    
    try:
        await connect()
        db_manager = DatabaseManager(database)
        
        # Step 2: Test with traditional method first
        print("\n2️⃣ Testing predictions with traditional method...")
        
        sim_data = await db_manager.get_data_for_simulation('eastern', 2025)
        
        # Force traditional method
        predictor = MLSNPRegSeasonPredictor(
            conference='eastern',
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages={'league_avg_xgf': 1.3, 'league_avg_xga': 1.2},
            use_automl=False  # Force traditional
        )
        
        # Quick test
        print("   Running 100 test simulations...")
        summary_df, _, _, _ = predictor.run_simulations(n_simulations=100)
        print("   ✅ Traditional method works!")
        
        # Show Chattanooga's position
        cfc_row = summary_df[summary_df['Team'].str.contains('Chattanooga', case=False)]
        if not cfc_row.empty:
            cfc = cfc_row.iloc[0]
            rank = cfc_row.index[0] + 1
            print(f"\n   Chattanooga FC (Traditional Method):")
            print(f"   Current: {rank} place, {cfc['Current Points']} points")
            print(f"   Projected: {cfc.get('Average Points', 0):.1f} points")
            print(f"   Playoff %: {cfc['Playoff Qualification %']:.1f}%")
        
        # Step 3: Now try ML with fresh training
        print("\n3️⃣ Training fresh ML model...")
        print("   This will take 2-3 minutes. DO NOT INTERRUPT!")
        
        # Create new predictor with ML
        ml_predictor = MLSNPRegSeasonPredictor(
            conference='eastern',
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages={'league_avg_xgf': 1.3, 'league_avg_xga': 1.2},
            use_automl=True  # Enable ML
        )
        
        if ml_predictor.use_automl:
            print("   ✅ ML model trained successfully!")
            
            # Test it works
            print("\n4️⃣ Testing ML predictions...")
            ml_summary_df, _, _, _ = ml_predictor.run_simulations(n_simulations=1000)
            
            print("\n" + "="*85)
            print("EASTERN CONFERENCE ML PREDICTIONS (Based on Actual Performance)")
            print("="*85)
            print(f"{'Rank':<6}{'Team':<35}{'Current':<10}{'Projected':<12}{'Playoff %':<12}")
            print("-"*85)
            
            for idx, row in ml_summary_df.head(10).iterrows():
                rank = idx + 1
                team_name = row['Team'][:33]
                current_pts = row['Current Points']
                projected = row.get('Average Points', 0)
                playoff_pct = row['Playoff Qualification %']
                
                marker = "→" if "Chattanooga" in team_name else " "
                print(f"{marker}{rank:<5}{team_name:<35}{current_pts:<10}{projected:<11.1f}{playoff_pct:>6.1f}%")
                
                if rank == 8:
                    print("-"*85 + " ← Playoff Line")
            
            # Chattanooga focus
            cfc_ml = ml_summary_df[ml_summary_df['Team'].str.contains('Chattanooga', case=False)]
            if not cfc_ml.empty:
                cfc = cfc_ml.iloc[0]
                rank = cfc_ml.index[0] + 1
                
                print(f"\n🔵 CHATTANOOGA FC ML ANALYSIS:")
                print(f"   Current Position: {rank} of 15")
                print(f"   Current Points: {cfc['Current Points']} (from 13 games)")
                print(f"   ML Projected Final: {cfc.get('Average Points', 0):.1f} points")
                print(f"   Playoff Probability: {cfc['Playoff Qualification %']:.1f}%")
                
                # Compare to pre-season
                print(f"\n   📊 Season Progression:")
                print(f"   Pre-season prediction: 4th, 43.1 pts, 92.3% playoffs")
                print(f"   Current projection: {rank}, {cfc.get('Average Points', 0):.1f} pts, {cfc['Playoff Qualification %']:.1f}% playoffs")
                
                if cfc.get('Average Points', 0) > 50:
                    print(f"\n   🎉 EXCEEDING EXPECTATIONS!")
            
            # Save results
            ml_summary_df.to_csv("output/ml_predictions_recovered.csv", index=False)
            print(f"\n💾 Results saved to: output/ml_predictions_recovered.csv")
            
        else:
            print("   ❌ ML training failed. Using traditional method is fine!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()

async def quick_standings_check():
    """Quick check of current standings"""
    
    print("\n\n📊 Current Eastern Conference Standings")
    print("=" * 50)
    
    try:
        await connect()
        
        standings = await database.fetch_all("""
            WITH team_records AS (
                SELECT 
                    t.team_id,
                    t.team_name,
                    COUNT(CASE WHEN g.is_completed THEN 1 END) as games_played,
                    SUM(CASE 
                        WHEN g.is_completed = false THEN 0
                        WHEN (g.home_team_id = t.team_id AND g.home_score > g.away_score) OR
                             (g.away_team_id = t.team_id AND g.away_score > g.home_score)
                        THEN 3
                        WHEN g.is_completed AND g.home_score = g.away_score AND g.went_to_shootout
                        THEN CASE
                            WHEN (g.home_team_id = t.team_id AND g.home_penalties > g.away_penalties) OR
                                 (g.away_team_id = t.team_id AND g.away_penalties > g.home_penalties)
                            THEN 2
                            ELSE 1
                        END
                        WHEN g.is_completed AND g.home_score = g.away_score
                        THEN 1
                        ELSE 0
                    END) as points,
                    SUM(CASE 
                        WHEN g.home_team_id = t.team_id THEN g.home_score
                        WHEN g.away_team_id = t.team_id THEN g.away_score
                        ELSE 0
                    END) as goals_for,
                    SUM(CASE 
                        WHEN g.home_team_id = t.team_id THEN g.away_score
                        WHEN g.away_team_id = t.team_id THEN g.home_score
                        ELSE 0
                    END) as goals_against
                FROM team t
                JOIN team_affiliations ta ON t.team_id = ta.team_id
                LEFT JOIN games g ON (g.home_team_id = t.team_id OR g.away_team_id = t.team_id) 
                    AND g.season_year = 2025
                WHERE ta.conference_id = 1 AND ta.is_current = true
                GROUP BY t.team_id, t.team_name
            )
            SELECT 
                team_name,
                games_played,
                points,
                goals_for,
                goals_against,
                goals_for - goals_against as goal_diff,
                CASE WHEN games_played > 0 THEN ROUND(points::numeric / games_played, 2) ELSE 0 END as ppg
            FROM team_records
            WHERE games_played > 0
            ORDER BY points DESC, goal_diff DESC, goals_for DESC
        """)
        
        print(f"{'Pos':<4}{'Team':<30}{'GP':<4}{'Pts':<5}{'GF':<4}{'GA':<4}{'GD':<5}{'PPG':<5}")
        print("-" * 65)
        
        for i, team in enumerate(standings):
            pos = i + 1
            marker = "→" if "Chattanooga" in team['team_name'] else " "
            print(f"{marker}{pos:<3}{team['team_name']:<30}{team['games_played']:<4}{team['points']:<5}"
                  f"{team['goals_for']:<4}{team['goals_against']:<4}{team['goal_diff']:>+4} {team['ppg']:<5}")
            
            if pos == 8:
                print("-" * 65 + " ← Playoff Line")
        
    finally:
        await disconnect()

if __name__ == "__main__":
    print("🚀 ML Training Recovery Tool")
    print("=" * 60)
    print("This will:")
    print("1. Clean up the interrupted training")
    print("2. Test traditional predictions work")
    print("3. Retrain the ML model properly")
    print("4. Show updated predictions")
    print("\nPress Enter to continue...")
    input()
    
    asyncio.run(recover_and_retrain())
    asyncio.run(quick_standings_check())