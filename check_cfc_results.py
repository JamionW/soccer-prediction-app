#!/usr/bin/env python3
"""
Check Chattanooga FC's actual games and verify points calculation
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import timezone_patch
from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from datetime import datetime

async def check_chattanooga_games():
    """Check all CFC games and calculate points"""
    
    print("🔍 Checking Chattanooga FC Results")
    print("="*70)
    
    await connect()
    db_manager = DatabaseManager(database)
    
    try:
        # Get CFC's team ID
        cfc_id = 'raMyeZAMd2'  # From your constants
        
        # Get all CFC games from database
        print("\n📊 Getting games from database...")
        db_games = await database.fetch_all("""
            SELECT 
                game_id,
                date,
                home_team_id,
                away_team_id,
                home_score,
                away_score,
                went_to_shootout,
                home_penalties,
                away_penalties,
                is_completed,
                status
            FROM games
            WHERE season_year = 2025
            AND (home_team_id = :team_id OR away_team_id = :team_id)
            ORDER BY date
        """, values={"team_id": cfc_id})
        
        print(f"Found {len(db_games)} total games for Chattanooga FC")
        
        # Also fetch from ASA API to compare
        print("\n🌐 Fetching latest from ASA API...")
        try:
            games_from_asa = db_manager.asa_client.get_games(
                team_id=[cfc_id],
                leagues=['mlsnp'],
                season_name=['2025']
            )
            print(f"ASA API returned {len(games_from_asa)} games")
            
            # Update any missing games
            for asa_game in games_from_asa:
                await db_manager.store_game(asa_game)
            
        except Exception as e:
            print(f"⚠️  Could not fetch from ASA: {e}")
        
        # Calculate points from completed games
        print("\n📋 Completed Games:")
        print("-"*90)
        print(f"{'Date':<12} {'Home':<25} {'Away':<25} {'Score':<10} {'Result':<15} {'Pts'}")
        print("-"*90)
        
        total_points = 0
        wins = 0
        draws = 0
        losses = 0
        shootout_wins = 0
        shootout_losses = 0
        goals_for = 0
        goals_against = 0
        
        for game in db_games:
            if not game['is_completed']:
                continue
            
            # Determine if CFC was home or away
            is_home = game['home_team_id'] == cfc_id
            
            # Get team names
            if is_home:
                home_name = "Chattanooga FC"
                away_name = await get_team_name(db_manager, game['away_team_id'])
                cfc_score = game['home_score']
                opp_score = game['away_score']
            else:
                home_name = await get_team_name(db_manager, game['home_team_id'])
                away_name = "Chattanooga FC"
                cfc_score = game['away_score']
                opp_score = game['home_score']
            
            # Format date
            date_str = game['date'].strftime('%Y-%m-%d') if hasattr(game['date'], 'strftime') else str(game['date']).split()[0]
            
            # Calculate points
            points_earned = 0
            result = ""
            
            if game['went_to_shootout']:
                # Shootout game
                home_pens = game.get('home_penalties', 0)
                away_pens = game.get('away_penalties', 0)
                
                if is_home:
                    if home_pens > away_pens:
                        points_earned = 2
                        shootout_wins += 1
                        result = "W (SO)"
                    else:
                        points_earned = 1
                        shootout_losses += 1
                        result = "L (SO)"
                else:
                    if away_pens > home_pens:
                        points_earned = 2
                        shootout_wins += 1
                        result = "W (SO)"
                    else:
                        points_earned = 1
                        shootout_losses += 1
                        result = "L (SO)"
                draws += 1  # Shootouts count as draws
            else:
                # Regular game
                if cfc_score > opp_score:
                    points_earned = 3
                    wins += 1
                    result = "W"
                elif cfc_score < opp_score:
                    points_earned = 0
                    losses += 1
                    result = "L"
                else:
                    points_earned = 1
                    draws += 1
                    result = "D"
            
            total_points += points_earned
            goals_for += cfc_score
            goals_against += opp_score
            
            # Print game info
            score_str = f"{game['home_score']}-{game['away_score']}"
            if game['went_to_shootout']:
                score_str += f" ({game.get('home_penalties', 0)}-{game.get('away_penalties', 0)})"
            
            print(f"{date_str:<12} {home_name:<25} {away_name:<25} {score_str:<10} {result:<15} {points_earned}")
        
        print("-"*90)
        
        # Summary
        games_played = wins + draws + losses
        print(f"\n📊 SUMMARY:")
        print(f"   Games Played: {games_played}")
        print(f"   Record: {wins}W-{draws}D-{losses}L")
        print(f"   Shootouts: {shootout_wins} wins, {shootout_losses} losses")
        print(f"   Goals: {goals_for} for, {goals_against} against ({goals_for - goals_against:+d} GD)")
        print(f"   Points: {total_points}")
        print(f"   PPG: {total_points/games_played:.2f}" if games_played > 0 else "")
        
        # Check database standings table
        print("\n🔍 Checking standings table...")
        standings = await database.fetch_one("""
            SELECT points, games_played, wins, draws, losses, goals_for, goals_against
            FROM standings_history
            WHERE team_id = :team_id AND season_year = 2025
            ORDER BY date_captured DESC
            LIMIT 1
        """, values={"team_id": cfc_id})
        
        if standings:
            print(f"   Database standings show: {standings['points']} points")
            if standings['points'] != total_points:
                print(f"   ⚠️  MISMATCH! Calculated {total_points} but database shows {standings['points']}")
        
        # Show upcoming games
        print("\n📅 Upcoming Games:")
        upcoming_count = 0
        for game in db_games:
            if game['is_completed']:
                continue
            
            upcoming_count += 1
            if upcoming_count > 5:
                break
            
            date_str = game['date'].strftime('%m/%d') if hasattr(game['date'], 'strftime') else str(game['date']).split()[0]
            
            if game['home_team_id'] == cfc_id:
                opp_name = await get_team_name(db_manager, game['away_team_id'])
                print(f"   {date_str}: vs {opp_name} (H)")
            else:
                opp_name = await get_team_name(db_manager, game['home_team_id'])
                print(f"   {date_str}: @ {opp_name} (A)")
        
        if upcoming_count > 5:
            print(f"   ... and {upcoming_count - 5} more")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await disconnect()

async def get_team_name(db_manager, team_id):
    """Get team name from database"""
    team = await db_manager.db.fetch_one(
        "SELECT team_name FROM team WHERE team_id = :team_id",
        values={"team_id": team_id}
    )
    return team['team_name'] if team else f"Team {team_id}"

if __name__ == "__main__":
    print("🏆 Chattanooga FC 2025 Season Results Checker")
    print("This will:")
    print("  - Show all completed games")
    print("  - Calculate actual points")
    print("  - Compare with database")
    print("  - Update from ASA if needed")
    print()
    
    asyncio.run(check_chattanooga_games())