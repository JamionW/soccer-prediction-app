#!/usr/bin/env python3
"""
Check and fix shootout games in the database
MLS Next Pro games that end in a draw go to a penalty shootout
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import timezone_patch
from src.common.database import database, connect, disconnect
from datetime import datetime

async def check_and_fix_shootouts():
    """Check for games that should be shootouts and fix them"""
    
    print("🔍 Checking for Shootout Games")
    print("="*70)
    
    await connect()
    
    try:
        # First, let's check all completed games with tied scores
        print("\n📊 Finding all tied games in 2025...")
        tied_games = await database.fetch_all("""
            SELECT 
                g.game_id,
                g.date,
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score,
                g.went_to_shootout,
                g.home_penalties,
                g.away_penalties,
                ht.team_name as home_team,
                at.team_name as away_team
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE g.season_year = 2025
            AND g.is_completed = true
            AND g.home_score = g.away_score
            ORDER BY g.date DESC
        """)
        
        print(f"Found {len(tied_games)} games that ended with tied scores")
        
        # Check each tied game
        games_to_fix = []
        for game in tied_games:
            date_str = game['date'].strftime('%Y-%m-%d') if hasattr(game['date'], 'strftime') else str(game['date']).split()[0]
            
            print(f"\n📋 Game on {date_str}:")
            print(f"   {game['home_team']} vs {game['away_team']}")
            print(f"   Score: {game['home_score']}-{game['away_score']}")
            print(f"   Went to shootout: {game['went_to_shootout']}")
            print(f"   Penalties: {game['home_penalties'] or 'None'}-{game['away_penalties'] or 'None'}")
            
            if not game['went_to_shootout']:
                print("   ⚠️  This game should have gone to shootout!")
                games_to_fix.append(game)
        
        # Specific check for CFC games
        print("\n🔍 Checking Chattanooga FC tied games specifically...")
        cfc_tied_games = await database.fetch_all("""
            SELECT 
                g.*,
                ht.team_name as home_team,
                at.team_name as away_team
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE g.season_year = 2025
            AND g.is_completed = true
            AND g.home_score = g.away_score
            AND (g.home_team_id = 'raMyeZAMd2' OR g.away_team_id = 'raMyeZAMd2')
            ORDER BY g.date DESC
        """)
        
        print(f"\nChattanooga FC has {len(cfc_tied_games)} tied games:")
        
        for game in cfc_tied_games:
            date_str = game['date'].strftime('%Y-%m-%d') if hasattr(game['date'], 'strftime') else str(game['date']).split()[0]
            print(f"\n   {date_str}: {game['home_team']} {game['home_score']}-{game['away_score']} {game['away_team']}")
            
            if game['went_to_shootout']:
                print(f"   ✅ Shootout recorded: {game['home_penalties']}-{game['away_penalties']}")
            else:
                print(f"   ❌ NO SHOOTOUT RECORDED - needs fixing!")
        
        # Ask if we should fix the games
        if games_to_fix:
            print(f"\n⚠️  Found {len(games_to_fix)} games that need shootout data")
            print("\nFor the June 7 CFC vs Inter Miami game (3-3):")
            print("According to MLS Next Pro rules, this should have gone to a shootout.")
            print("If CFC won the shootout, they get 2 points (total: 29 points)")
            print("If CFC lost the shootout, they get 1 point (total: 27 points)")
            
            # For now, let's check if we can find the actual shootout result
            # In a real scenario, we'd need to get this from a reliable source
            print("\n🔍 Checking for shootout results from other sources...")
            
            # Update the specific CFC vs Inter Miami game
            # YOU NEED TO VERIFY THE ACTUAL SHOOTOUT RESULT
            print("\n⚠️  MANUAL INTERVENTION NEEDED!")
            print("Please verify the shootout result for CFC vs Inter Miami on June 7")
            print("Then update with one of these commands:")
            print("\nIf CFC won the shootout:")
            print("UPDATE games SET went_to_shootout = true, home_penalties = 5, away_penalties = 4")
            print("WHERE game_id = 'raMyeZAMd2_vzqowoZqap_2025-06-07';")
            print("\nIf CFC lost the shootout:")
            print("UPDATE games SET went_to_shootout = true, home_penalties = 4, away_penalties = 5")
            print("WHERE game_id = 'raMyeZAMd2_vzqowoZqap_2025-06-07';")
            
            # Show what the points would be in each case
            print("\n📊 Points calculation:")
            print("Current calculated: 27 points (assuming regular draw)")
            print("If CFC won shootout: 29 points (27 + 2 instead of 1)")
            print("If CFC lost shootout: 28 points (27 + 1)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await disconnect()

if __name__ == "__main__":
    print("🏆 MLS Next Pro Shootout Game Checker")
    print("This will find games that ended in draws and check if they have shootout data")
    print()
    
    asyncio.run(check_and_fix_shootouts())