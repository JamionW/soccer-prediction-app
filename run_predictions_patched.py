#!/usr/bin/env python3
"""
Run predictions with a patched ASA integration
This works around the DataFrame issue
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# IMPORTANT: Apply the patch BEFORE importing DatabaseManager
from src.common.database_manager import DatabaseManager
import pandas as pd

# Monkey patch the problematic method
original_get_or_fetch_team_xg = DatabaseManager.get_or_fetch_team_xg

async def get_or_fetch_team_xg_patched(self, team_id: str, season_year: int):
    """Patched version that handles missing ASA data gracefully"""
    
    # First try the database
    query = """
        SELECT * FROM team_xg_history 
        WHERE team_id = :team_id AND season_year = :season_year
        ORDER BY date_captured DESC
        LIMIT 1
    """
    
    xg_data = await self.db.fetch_one(query, values={
        "team_id": team_id, 
        "season_year": season_year
    })
    
    if xg_data:
        return dict(xg_data)
    
    # Since ASA doesn't have 2025 MLSNP data, return defaults
    # This allows the simulation to run with equal team strengths
    return {
        "team_id": team_id,
        "games_played": 0,
        "x_goals_for": 0.0,
        "x_goals_against": 0.0
    }

# Apply the patch
DatabaseManager.get_or_fetch_team_xg = get_or_fetch_team_xg_patched

# Now import the rest
from src.common.database import database, connect, disconnect
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor

async def run_eastern_conference_predictions():
    """Run Eastern Conference predictions"""
    
    print("🎯 MLS Next Pro Eastern Conference Predictions")
    print("=" * 60)
    print("Using schedule-based predictions (all teams equal strength)")
    print()
    
    try:
        await connect()
        db_manager = DatabaseManager(database)
        
        # Get Eastern Conference data
        print("Loading Eastern Conference data...")
        sim_data = await db_manager.get_data_for_simulation('eastern', 2025)
        
        print(f"✅ Teams: {len(sim_data['conference_teams'])}")
        print(f"✅ Games: {len(sim_data['games_data'])}")
        
        # Show teams
        print("\nEastern Conference Teams:")
        for i, (team_id, team_name) in enumerate(sim_data['conference_teams'].items()):
            if i < 5:
                print(f"  - {team_name}")
        print(f"  ... and {len(sim_data['conference_teams']) - 5} more")
        
        # Create predictor
        print("\nInitializing predictor...")
        predictor = MLSNPRegSeasonPredictor(
            conference='eastern',
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages={'league_avg_xgf': 1.2, 'league_avg_xga': 1.2},
            use_automl=False  # No ML until we have completed games
        )
        
        # Run simulations
        print("\nRunning 5000 season simulations...")
        print("(This will take about 1-2 minutes)")
        
        summary_df, final_ranks, _, qual_data = predictor.run_simulations(n_simulations=5000)
        
        # Show results
        print("\n" + "="*85)
        print("EASTERN CONFERENCE PREDICTED FINAL STANDINGS")
        print("="*85)
        print(f"{'Rank':<6}{'Team':<35}{'Playoff %':<12}{'Avg Pts':<10}{'Best':<8}{'Worst':<8}")
        print("-"*85)
        
        # Color coding for output
        for idx, row in summary_df.iterrows():
            rank = idx + 1
            team_name = row['Team'][:33]
            playoff_pct = row['Playoff Qualification %']
            avg_points = row.get('Average Points', 0)
            
            # Determine status
            if playoff_pct >= 99.5:
                status = "✓"  # Clinched
            elif playoff_pct <= 0.5:
                status = "✗"  # Eliminated
            elif rank <= 8:
                status = "↑"  # Currently in
            else:
                status = "↓"  # Currently out
            
            print(f"{rank:<6}{team_name:<35}{playoff_pct:>6.1f}%    {avg_points:>6.1f}")
            
            if rank == 8:
                print("-"*85 + " ← Playoff Line")
        
        # Chattanooga FC Focus
        print("\n" + "="*70)
        print("CHATTANOOGA FC DETAILED ANALYSIS")
        print("="*70)
        
        cfc_row = summary_df[summary_df['Team'].str.contains('Chattanooga', case=False)]
        if not cfc_row.empty:
            cfc = cfc_row.iloc[0]
            cfc_rank = cfc_row.index[0] + 1
            cfc_id = cfc['_team_id']
            
            print(f"\nProjected Final Position: {cfc_rank} of {len(summary_df)}")
            print(f"Playoff Probability: {cfc['Playoff Qualification %']:.1f}%")
            print(f"Average Points: {cfc.get('Average Points', 0):.1f}")
            
            # Key games analysis
            remaining_games = [g for g in predictor.remaining_games 
                             if g['home_team_id'] == cfc_id or g['away_team_id'] == cfc_id]
            
            print(f"\nSchedule Analysis ({len(remaining_games)} games):")
            
            # Count home/away
            home_games = sum(1 for g in remaining_games if g['home_team_id'] == cfc_id)
            away_games = len(remaining_games) - home_games
            
            print(f"  Home Games: {home_games}")
            print(f"  Away Games: {away_games}")
            
            # Show rivals
            print("\nKey Matchups:")
            rivals = ['Huntsville', 'Atlanta', 'Crown', 'Carolina']
            for rival in rivals:
                rival_games = [g for g in remaining_games 
                              if rival.lower() in sim_data['conference_teams'].get(
                                  g['home_team_id'] if g['away_team_id'] == cfc_id else g['away_team_id'], ''
                              ).lower()]
                if rival_games:
                    print(f"  vs {rival}: {len(rival_games)} game(s)")
            
            # Probability distribution
            print("\nFinish Probability Distribution:")
            rank_probs = {}
            for rank_list in final_ranks[cfc_id]:
                rank_probs[rank_list] = rank_probs.get(rank_list, 0) + 1
            
            for rank in range(1, 16):
                prob = (rank_probs.get(rank, 0) / len(final_ranks[cfc_id])) * 100
                if prob > 1:  # Only show significant probabilities
                    bar = "█" * int(prob / 2)
                    print(f"  {rank:>2}{'st' if rank==1 else 'nd' if rank==2 else 'rd' if rank==3 else 'th'}: {bar} {prob:.1f}%")
            
            # Scenarios
            print("\nWhat Chattanooga FC Needs:")
            if cfc['Playoff Qualification %'] > 90:
                print("  ✅ Strong position - maintain current form")
            elif cfc['Playoff Qualification %'] > 50:
                print("  ⚡ Good position - win home games, split away games")
            elif cfc['Playoff Qualification %'] > 20:
                print("  ⚠️  Challenging position - need strong home record")
            else:
                print("  🚨 Difficult position - need exceptional performance")
                
        # Save detailed results
        output_file = "output/eastern_conference_predictions_detailed.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Western Conference option
        print("\n" + "="*60)
        print("Run Western Conference predictions too? (y/n)")
        if input().lower() == 'y':
            await run_western_conference_predictions(db_manager)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()


async def run_western_conference_predictions(db_manager):
    """Run Western Conference predictions"""
    
    print("\n🎯 Running Western Conference predictions...")
    
    sim_data = await db_manager.get_data_for_simulation('western', 2025)
    
    predictor = MLSNPRegSeasonPredictor(
        conference='western',
        conference_teams=sim_data["conference_teams"],
        games_data=sim_data["games_data"],
        team_performance=sim_data["team_performance"],
        league_averages={'league_avg_xgf': 1.2, 'league_avg_xga': 1.2},
        use_automl=False
    )
    
    summary_df, _, _, _ = predictor.run_simulations(n_simulations=5000)
    
    print("\n" + "="*85)
    print("WESTERN CONFERENCE PREDICTED FINAL STANDINGS")
    print("="*85)
    print(f"{'Rank':<6}{'Team':<35}{'Playoff %':<12}{'Avg Pts':<10}")
    print("-"*85)
    
    for idx, row in summary_df.head(15).iterrows():
        rank = idx + 1
        team_name = row['Team'][:33]
        playoff_pct = row['Playoff Qualification %']
        avg_points = row.get('Average Points', 0)
        
        print(f"{rank:<6}{team_name:<35}{playoff_pct:>6.1f}%    {avg_points:>6.1f}")
        
        if rank == 8:
            print("-"*85 + " ← Playoff Line")
    
    summary_df.to_csv("output/western_conference_predictions_detailed.csv", index=False)


if __name__ == "__main__":
    print("🚀 MLS Next Pro Season Predictor (2025)")
    print("=" * 60)
    print("This will simulate the entire 2025 season")
    print("Note: Using equal team strengths until games are played")
    print()
    
    asyncio.run(run_eastern_conference_predictions())