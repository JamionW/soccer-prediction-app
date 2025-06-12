#!/usr/bin/env python3
"""
Explore and test feature engineering for the ML model
This helps you understand what features are being created and their values
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

async def explore_features():
    """
    Explore the features being generated for different teams and matchups
    """
    print("🔍 Feature Engineering Explorer")
    print("=" * 60)
    
    try:
        # Connect to database
        await connect()
        db_manager = DatabaseManager(database)
        
        # Get Eastern conference data
        print("\nFetching Eastern Conference data...")
        sim_data = await db_manager.get_data_for_simulation('eastern', 2025)
        
        # Find Chattanooga FC
        chattanooga_id = None
        for team_id, team_name in sim_data['conference_teams'].items():
            if 'chattanooga' in team_name.lower():
                chattanooga_id = team_id
                print(f"\n✅ Found Chattanooga FC: {team_name} (ID: {team_id})")
                break
        
        if not chattanooga_id:
            print("❌ Chattanooga FC not found! Using first team instead...")
            chattanooga_id = list(sim_data['conference_teams'].keys())[0]
        
        # Create a temporary predictor to access feature extraction
        predictor = MLSNPRegSeasonPredictor(
            conference='eastern',
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages={'league_avg_xgf': 1.2, 'league_avg_xga': 1.2},
            use_automl=False  # Don't train, just use for feature extraction
        )
        
        # Test different matchups
        print("\n📊 Analyzing features for different matchups:")
        print("-" * 60)
        
        # Find some interesting opponents
        opponents = []
        for team_id, team_name in sim_data['conference_teams'].items():
            if team_id != chattanooga_id:
                if 'huntsville' in team_name.lower():
                    opponents.insert(0, (team_id, team_name, "RIVAL"))  # Priority
                elif 'atlanta' in team_name.lower():
                    opponents.append((team_id, team_name, "Regional"))
                elif len(opponents) < 5:
                    opponents.append((team_id, team_name, "Normal"))
        
        # Analyze each matchup
        all_features = []
        
        for opp_id, opp_name, matchup_type in opponents[:5]:
            print(f"\n🆚 {sim_data['conference_teams'][chattanooga_id]} vs {opp_name} ({matchup_type})")
            
            # Get features for home game
            home_features = predictor._extract_features(
                team_id=chattanooga_id,
                opponent_id=opp_id,
                is_home=True,
                game_date='2025-07-04',  # Summer game
                games_before=sim_data['games_data']
            )
            
            # Get features for away game
            away_features = predictor._extract_features(
                team_id=chattanooga_id,
                opponent_id=opp_id,
                is_home=False,
                game_date='2025-07-04',
                games_before=sim_data['games_data']
            )
            
            # Store for comparison
            home_features['matchup'] = f"vs {opp_name} (H)"
            away_features['matchup'] = f"@ {opp_name} (A)"
            all_features.append(home_features)
            all_features.append(away_features)
            
            # Show key differences
            print("\n  Key feature differences (Home vs Away):")
            important_features = [
                'is_home', 'is_chattanooga', 'rivalry_intensity',
                'finley_stadium_advantage', 'chattanooga_heat_advantage',
                'opponent_travel_fatigue', 'expected_attendance_impact'
            ]
            
            for feat in important_features:
                if feat in home_features:
                    home_val = home_features.get(feat, 0)
                    away_val = away_features.get(feat, 0)
                    if home_val != away_val:
                        print(f"    {feat}: {home_val:.2f} (H) vs {away_val:.2f} (A)")
        
        # Create DataFrame for analysis
        df = pd.DataFrame(all_features)
        
        print("\n📈 Feature Statistics Across All Matchups:")
        print("-" * 60)
        
        # Find features that vary the most
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        feature_variance = df[numeric_cols].var().sort_values(ascending=False)
        
        print("\nMost variable features (these differentiate matchups):")
        for feat, var in feature_variance.head(10).items():
            if feat != 'matchup' and var > 0.01:
                print(f"  {feat}: variance = {var:.3f}")
        
        # Chattanooga-specific features
        cfc_features = [col for col in df.columns if 'chattanooga' in col]
        if cfc_features:
            print(f"\n🔵 Chattanooga FC Specific Features ({len(cfc_features)}):")
            for feat in cfc_features:
                unique_vals = df[feat].unique()
                print(f"  {feat}: {unique_vals}")
        
        # Save detailed feature analysis
        output_file = "output/chattanooga_features_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed feature analysis saved to: {output_file}")
        
        # Feature correlation analysis
        print("\n🔗 Feature Correlations with Team Performance:")
        if 'team_xgf_per_game' in df.columns:
            correlations = df[numeric_cols].corrwith(df['team_xgf_per_game']).sort_values(ascending=False)
            print("\nFeatures most correlated with offensive performance:")
            for feat, corr in correlations.head(10).items():
                if feat != 'team_xgf_per_game' and abs(corr) > 0.1:
                    print(f"  {feat}: {corr:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await disconnect()


async def test_feature_modifications():
    """
    Test how modifying features affects predictions
    """
    print("\n\n🧪 Testing Feature Impact on Predictions")
    print("=" * 60)
    
    try:
        await connect()
        db_manager = DatabaseManager(database)
        
        # Get data
        sim_data = await db_manager.get_data_for_simulation('eastern', 2025)
        
        # Create predictor with trained model
        predictor = MLSNPRegSeasonPredictor(
            conference='eastern',
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages={'league_avg_xgf': 1.2, 'league_avg_xga': 1.2},
            use_automl=True
        )
        
        if not predictor.use_automl:
            print("❌ AutoML model not available. Train it first!")
            return
        
        # Find Chattanooga
        chattanooga_id = None
        for team_id, team_name in sim_data['conference_teams'].items():
            if 'chattanooga' in team_name.lower():
                chattanooga_id = team_id
                break
        
        if chattanooga_id:
            # Test impact of different features
            opponent_id = [tid for tid in sim_data['conference_teams'].keys() if tid != chattanooga_id][0]
            
            print(f"\nTesting feature impacts for Chattanooga FC vs {sim_data['conference_teams'][opponent_id]}")
            
            # Baseline prediction
            base_game = {
                'home_team_id': chattanooga_id,
                'away_team_id': opponent_id,
                'date': '2025-06-15'
            }
            
            # Run baseline simulations
            baseline_results = []
            for _ in range(100):
                h, a, _ = predictor._simulate_game(base_game)
                baseline_results.append(h - a)  # Goal difference
            
            baseline_avg = sum(baseline_results) / len(baseline_results)
            print(f"\nBaseline average goal difference: {baseline_avg:.2f}")
            
            # Now test with modified features
            # You would need to modify the predictor's _extract_features method
            # or create a wrapper to test different scenarios
            
            print("\n💡 To test feature impacts:")
            print("1. Modify _extract_features() in reg_season_predictor.py")
            print("2. Add print statements to see feature values")
            print("3. Run simulations with different conditions")
            print("4. Compare results to baseline")
            
    finally:
        await disconnect()


if __name__ == "__main__":
    print("🚀 MLS Next Pro Feature Engineering Explorer")
    print("=" * 60)
    asyncio.run(explore_features())
    
    print("\n\nWould you like to test feature impact on predictions? (y/n)")
    if input().lower() == 'y':
        asyncio.run(test_feature_modifications())