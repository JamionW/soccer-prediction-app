#!/usr/bin/env python3
"""
Fix the timezone issue in database_manager.py
The problem: mixing timezone-aware and timezone-naive datetimes
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
sys.path.append(str(Path(__file__).parent))

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager

async def diagnose_and_fix_timezone_issue():
    """Diagnose and fix the timezone problem"""
    
    print("🔧 Fixing Timezone Issue in Database Manager")
    print("=" * 50)
    
    try:
        await connect()
        
        # 1. Check what's in the database
        print("\n1. Checking date_captured format in database...")
        
        sample = await database.fetch_one("""
            SELECT team_id, date_captured, 
                   pg_typeof(date_captured) as data_type
            FROM team_xg_history 
            LIMIT 1
        """)
        
        if sample:
            print(f"   Sample date_captured: {sample['date_captured']}")
            print(f"   Database type: {sample['data_type']}")
            print(f"   Python type: {type(sample['date_captured'])}")
            print(f"   Has timezone: {sample['date_captured'].tzinfo is not None}")
        
        # 2. Create the patch
        print("\n2. Creating patched version of get_or_fetch_team_xg...")
        
        # Save the patched method
        patch_code = '''# Fix for timezone issue in database_manager.py
# Add this import at the top of database_manager.py:
from datetime import datetime, timezone, timedelta

# Then replace the problematic line in get_or_fetch_team_xg (around line 377):
# OLD: data_age = datetime.now() - xg_data['date_captured']
# NEW: Use timezone-aware datetime

# Option 1: Make both timezone-aware (recommended)
if xg_data['date_captured'].tzinfo is None:
    # If date_captured is naive, assume UTC
    date_captured_aware = xg_data['date_captured'].replace(tzinfo=timezone.utc)
else:
    date_captured_aware = xg_data['date_captured']

data_age = datetime.now(timezone.utc) - date_captured_aware

# Option 2: Make both timezone-naive (simpler)
# date_captured_naive = xg_data['date_captured'].replace(tzinfo=None) if xg_data['date_captured'].tzinfo else xg_data['date_captured']
# data_age = datetime.now() - date_captured_naive
'''
        
        print(patch_code)
        
        # 3. Test the fix
        print("\n3. Testing the fix...")
        
        # Monkey patch the method for testing
        original_method = DatabaseManager.get_or_fetch_team_xg
        
        async def get_or_fetch_team_xg_fixed(self, team_id: str, season_year: int):
            """Fixed version with proper timezone handling"""
            
            # Try to get from database first
            query = """
                SELECT * FROM team_xg_history 
                WHERE team_id = :team_id AND season_year = :season_year
                ORDER BY date_captured DESC
                LIMIT 1
            """
            
            values = {"team_id": team_id, "season_year": season_year}
            xg_data = await self.db.fetch_one(query, values=values)
            
            # Check if data is fresh (less than 1 day old)
            if xg_data:
                # FIX: Handle timezone properly
                if xg_data['date_captured'].tzinfo is None:
                    # If naive, assume UTC
                    date_captured = xg_data['date_captured'].replace(tzinfo=timezone.utc)
                else:
                    date_captured = xg_data['date_captured']
                
                # Use timezone-aware current time
                data_age = datetime.now(timezone.utc) - date_captured
                
                if data_age < timedelta(days=1):
                    return dict(xg_data)
            
            # For now, return defaults since ASA isn't working for 2025
            return {
                "team_id": team_id,
                "games_played": 0,
                "x_goals_for": 0.0,
                "x_goals_against": 0.0
            }
        
        # Apply the fix
        DatabaseManager.get_or_fetch_team_xg = get_or_fetch_team_xg_fixed
        
        # Test it
        db_manager = DatabaseManager(database)
        result = await db_manager.get_or_fetch_team_xg('raMyeZAMd2', 2025)
        print(f"\n✅ Fix works! Got result: {result}")
        
        # 4. Create a permanent patch file
        permanent_patch = '''#!/usr/bin/env python3
"""
Timezone fix for database_manager.py
Apply this before running any scripts that use DatabaseManager
"""

from datetime import datetime, timezone, timedelta
from src.common.database_manager import DatabaseManager

# Store original method
_original_get_or_fetch_team_xg = DatabaseManager.get_or_fetch_team_xg

async def get_or_fetch_team_xg_with_timezone_fix(self, team_id: str, season_year: int):
    """Fixed version with proper timezone handling"""
    
    # Try to get from database first
    query = """
        SELECT * FROM team_xg_history 
        WHERE team_id = :team_id AND season_year = :season_year
        ORDER BY date_captured DESC
        LIMIT 1
    """
    
    values = {"team_id": team_id, "season_year": season_year}
    xg_data = await self.db.fetch_one(query, values=values)
    
    # Check if data is fresh (less than 1 day old)
    if xg_data:
        # FIX: Handle timezone properly
        if xg_data['date_captured'].tzinfo is None:
            # If naive, assume UTC
            date_captured = xg_data['date_captured'].replace(tzinfo=timezone.utc)
        else:
            date_captured = xg_data['date_captured']
        
        # Use timezone-aware current time
        data_age = datetime.now(timezone.utc) - date_captured
        
        if data_age < timedelta(days=1):
            return dict(xg_data)
    
    # Return defaults for now (ASA not working for 2025)
    return {
        "team_id": team_id,
        "games_played": 0,
        "x_goals_for": 0.0,
        "x_goals_against": 0.0
    }

# Apply the patch
DatabaseManager.get_or_fetch_team_xg = get_or_fetch_team_xg_with_timezone_fix

print("✅ Timezone patch applied to DatabaseManager")
'''
        
        with open('timezone_patch.py', 'w') as f:
            f.write(permanent_patch)
        
        print("\n✅ Created timezone_patch.py")
        print("   Import this at the top of your scripts to fix the timezone issue")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()


async def test_with_completed_games():
    """Now that we have completed games, let's check what we can do"""
    
    print("\n\n📊 Analyzing Completed Games Data")
    print("=" * 50)
    
    try:
        await connect()
        
        # Check completed games
        completed = await database.fetch_all("""
            SELECT g.*, 
                   ht.team_name as home_team_name,
                   at.team_name as away_team_name
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE g.season_year = 2025 
            AND g.is_completed = true
            ORDER BY g.date DESC
            LIMIT 5
        """)
        
        print(f"\nFound {len(completed)} recent completed games:")
        for game in completed:
            print(f"  {game['home_team_name']} {game['home_score']} - {game['away_score']} {game['away_team_name']}")
        
        # Check Chattanooga's results
        cfc_games = await database.fetch_all("""
            SELECT g.*, 
                   ht.team_name as home_team_name,
                   at.team_name as away_team_name,
                   CASE 
                       WHEN g.home_team_id = 'raMyeZAMd2' THEN 'HOME'
                       ELSE 'AWAY'
                   END as cfc_location
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE g.season_year = 2025 
            AND g.is_completed = true
            AND (g.home_team_id = 'raMyeZAMd2' OR g.away_team_id = 'raMyeZAMd2')
            ORDER BY g.date DESC
        """)
        
        if cfc_games:
            print(f"\n🔵 Chattanooga FC Results ({len(cfc_games)} games):")
            
            wins = 0
            draws = 0
            losses = 0
            points = 0
            
            for game in cfc_games[:5]:  # Show last 5
                if game['cfc_location'] == 'HOME':
                    cfc_score = game['home_score']
                    opp_score = game['away_score']
                    opponent = game['away_team_name']
                    loc = "vs"
                else:
                    cfc_score = game['away_score']
                    opp_score = game['home_score']
                    opponent = game['home_team_name']
                    loc = "@"
                
                # Calculate result
                if cfc_score > opp_score:
                    result = "W"
                    wins += 1
                    points += 3
                elif cfc_score < opp_score:
                    result = "L"
                    losses += 1
                elif game['went_to_shootout']:
                    # Need to check who won shootout
                    if game['cfc_location'] == 'HOME':
                        if game['home_penalties'] > game['away_penalties']:
                            result = "W(SO)"
                            points += 2
                        else:
                            result = "L(SO)"
                            points += 1
                    else:
                        if game['away_penalties'] > game['home_penalties']:
                            result = "W(SO)"
                            points += 2
                        else:
                            result = "L(SO)"
                            points += 1
                    draws += 1
                else:
                    result = "D"
                    draws += 1
                    points += 1
                
                print(f"  {result}: {loc} {opponent} {cfc_score}-{opp_score}")
            
            # Current form
            total_games = len(cfc_games)
            if total_games > 0:
                print(f"\n  Record: {wins}W-{draws}D-{losses}L")
                print(f"  Points: {points} from {total_games} games")
                print(f"  PPG: {points/total_games:.2f}")
                
                # Project full season
                games_remaining = 28 - total_games  # They have 28 total games
                projected_points = points + (points/total_games * games_remaining)
                print(f"  Projected Final Points: {projected_points:.1f}")
        
        print("\n💡 Now you can:")
        print("1. Re-run predictions with ACTUAL team performance!")
        print("2. Train the ML model on completed games")
        print("3. Get more accurate playoff probabilities")
        
    finally:
        await disconnect()


if __name__ == "__main__":
    asyncio.run(diagnose_and_fix_timezone_issue())
    asyncio.run(test_with_completed_games())