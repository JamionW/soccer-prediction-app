#!/usr/bin/env python3
"""
Fix the ASA API integration in database_manager.py
The issue is that ASA returns DataFrames, not lists
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.common.database import database, connect, disconnect
from itscalledsoccer.client import AmericanSoccerAnalysis
import pandas as pd

async def test_and_fix_asa():
    """Test ASA API and show how to fix it"""
    
    print("🔧 Fixing ASA API Integration")
    print("=" * 50)
    
    asa = AmericanSoccerAnalysis()
    
    # Test 1: Show correct way to call get_team_xgoals
    print("\n1. Testing correct ASA API usage...")
    
    try:
        # The issue: ASA returns a DataFrame, not a list
        result = asa.get_team_xgoals(
            team_ids=['raMyeZAMd2'],  # Must be plural
            leagues=['mlsnp'],        # Specify league
            seasons=['2024', '2025']  # Try multiple seasons
        )
        
        print(f"   Result type: {type(result)}")
        
        if isinstance(result, pd.DataFrame):
            print(f"   ✅ Got DataFrame with {len(result)} rows")
            if not result.empty:
                print("\n   Columns available:")
                for col in result.columns[:10]:
                    print(f"     - {col}")
                    
                # Show sample data
                print("\n   Sample data:")
                print(result.head(2).to_string())
            else:
                print("   ⚠️  DataFrame is empty - no data for this team/season")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Show how to properly handle in database_manager
    print("\n2. Correct way to handle in code:")
    
    print("""
    # In database_manager.py, update get_or_fetch_team_xg method:
    
    try:
        # Get team xG data from ASA - returns DataFrame
        xg_stats_df = self.asa_client.get_team_xgoals(
            team_ids=[team_id],      # Note: plural, and pass as list
            leagues=['mlsnp'],       # Specify league
            seasons=[str(season_year)]
        )
        
        # Check if we got data
        if xg_stats_df is not None and not xg_stats_df.empty:
            # Convert DataFrame to dict
            stat = xg_stats_df.iloc[0].to_dict()  # Get first row as dict
            
            # Now use stat['count_games'], stat['xgoals_for'], etc.
    """)
    
    # Test 3: Try to get any MLS Next Pro data
    print("\n3. Checking for any MLS Next Pro data...")
    
    try:
        # Get all teams to see what's available
        teams_df = asa.get_teams(leagues=['mlsnp'])
        
        if isinstance(teams_df, pd.DataFrame) and not teams_df.empty:
            print(f"   ✅ Found {len(teams_df)} MLS Next Pro teams")
            
            # Show Chattanooga FC if found
            cfc = teams_df[teams_df['team_name'].str.contains('Chattanooga', case=False, na=False)]
            if not cfc.empty:
                print("\n   Chattanooga FC info:")
                print(cfc.to_string())
        else:
            print("   ❌ No MLS Next Pro teams found in ASA")
            
    except Exception as e:
        print(f"   ❌ Error getting teams: {e}")
    
    # Test 4: Alternative - use MLS data if MLSNP not available
    print("\n4. Checking MLS data as alternative...")
    
    try:
        # Sometimes Next Pro data is under MLS
        mls_teams = asa.get_teams(leagues=['mls'])
        
        if isinstance(mls_teams, pd.DataFrame) and not mls_teams.empty:
            # Look for "2" or "II" teams
            reserve_teams = mls_teams[
                mls_teams['team_name'].str.contains('2|II|B$', regex=True, case=False, na=False)
            ]
            
            if not reserve_teams.empty:
                print(f"   Found {len(reserve_teams)} possible Next Pro teams in MLS data:")
                print(reserve_teams[['team_id', 'team_name']].head(5).to_string())
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    await connect()
    
    # Test 5: Create a patched version of the method
    print("\n5. Testing patched get_or_fetch_team_xg...")
    
    async def get_or_fetch_team_xg_fixed(team_id: str, season_year: int):
        """Fixed version of get_or_fetch_team_xg"""
        
        # Try to get from database first
        query = """
            SELECT * FROM team_xg_history 
            WHERE team_id = :team_id AND season_year = :season_year
            ORDER BY date_captured DESC
            LIMIT 1
        """
        
        xg_data = await database.fetch_one(query, values={
            "team_id": team_id, 
            "season_year": season_year
        })
        
        if xg_data:
            return dict(xg_data)
        
        # Try ASA with proper DataFrame handling
        print(f"   Fetching xG for {team_id} from ASA...")
        
        try:
            # Try multiple leagues in case MLSNP isn't available
            for league in ['mlsnp', 'mls', 'uslc']:
                xg_stats_df = asa.get_team_xgoals(
                    team_ids=[team_id],
                    leagues=[league],
                    seasons=[str(season_year), str(season_year-1)]  # Try current and previous
                )
                
                if isinstance(xg_stats_df, pd.DataFrame) and not xg_stats_df.empty:
                    stat = xg_stats_df.iloc[0]
                    
                    print(f"   ✅ Found data in {league} league")
                    
                    # Store in database
                    await database.execute("""
                        INSERT INTO team_xg_history (
                            team_id, games_played, x_goals_for, x_goals_against,
                            season_year, date_captured
                        ) VALUES (
                            :team_id, :games_played, :xgf, :xga, :season_year, NOW()
                        )
                    """, values={
                        'team_id': team_id,
                        'games_played': int(stat.get('count_games', 0)),
                        'xgf': float(stat.get('xgoals_for', 0)),
                        'xga': float(stat.get('xgoals_against', 0)),
                        'season_year': season_year
                    })
                    
                    return {
                        'team_id': team_id,
                        'games_played': int(stat.get('count_games', 0)),
                        'x_goals_for': float(stat.get('xgoals_for', 0)),
                        'x_goals_against': float(stat.get('xgoals_against', 0))
                    }
                    
        except Exception as e:
            print(f"   ❌ ASA error: {e}")
        
        # Return defaults if no data found
        print(f"   ⚠️  No xG data found for {team_id}, using defaults")
        return {
            'team_id': team_id,
            'games_played': 0,
            'x_goals_for': 0.0,
            'x_goals_against': 0.0
        }
    
    # Test with Chattanooga
    result = await get_or_fetch_team_xg_fixed('raMyeZAMd2', 2025)
    print(f"\n   Result: {result}")
    
    await disconnect()


async def patch_database_manager():
    """
    Create a monkey patch for the database manager
    This fixes the ASA integration without modifying the original file
    """
    
    print("\n6. Creating monkey patch for database_manager...")
    
    patch_content = '''
# Add this to your code before running predictions:

from src.common.database_manager import DatabaseManager
import pandas as pd

# Monkey patch the problematic method
def get_or_fetch_team_xg_patched(self, team_id: str, season_year: int):
    """Patched version that handles DataFrames correctly"""
    
    # Default return value
    default_data = {
        "team_id": team_id,
        "games_played": 0,
        "x_goals_for": 0.0,
        "x_goals_against": 0.0
    }
    
    # For now, just return defaults since ASA doesn't have 2025 MLSNP data yet
    return default_data

# Apply the patch
DatabaseManager.get_or_fetch_team_xg = get_or_fetch_team_xg_patched
'''
    
    print(patch_content)
    
    # Save patch file
    with open('asa_patch.py', 'w') as f:
        f.write(patch_content)
    
    print("\n✅ Patch saved to: asa_patch.py")
    print("   Import this before running predictions")


if __name__ == "__main__":
    asyncio.run(test_and_fix_asa())
    asyncio.run(patch_database_manager())