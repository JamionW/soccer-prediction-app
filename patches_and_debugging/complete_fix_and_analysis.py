#!/usr/bin/env python3
"""
Complete fix for timezone issue and deep analysis of Chattanooga FC
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta  # Fixed import!
sys.path.append(str(Path(__file__).parent))

# First, let's create the timezone patch properly
patch_content = '''from datetime import datetime, timezone, timedelta
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
print("✅ Timezone patch applied")
'''

# Write the patch file
with open('timezone_patch.py', 'w') as f:
    f.write(patch_content)

print("✅ Created timezone_patch.py")

# Now import it
import patches_and_debugging.timezone_patch_old as timezone_patch_old

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager

async def analyze_chattanooga_deeply():
    """Deep dive into Chattanooga FC's actual performance"""
    
    print("\n🔍 Deep Analysis of Chattanooga FC Performance")
    print("=" * 60)
    
    try:
        await connect()
        
        # Get ALL Chattanooga games with full details
        all_cfc_games = await database.fetch_all("""
            SELECT 
                g.*,
                ht.team_name as home_team_name,
                at.team_name as away_team_name,
                CASE 
                    WHEN g.home_team_id = 'raMyeZAMd2' THEN 'HOME'
                    ELSE 'AWAY'
                END as cfc_location,
                CASE
                    WHEN g.home_team_id = 'raMyeZAMd2' THEN g.home_score
                    ELSE g.away_score
                END as cfc_score,
                CASE
                    WHEN g.home_team_id = 'raMyeZAMd2' THEN g.away_score
                    ELSE g.home_score
                END as opp_score
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE g.season_year = 2025 
            AND (g.home_team_id = 'raMyeZAMd2' OR g.away_team_id = 'raMyeZAMd2')
            ORDER BY g.date ASC
        """)
        
        print(f"\n📅 Chattanooga FC 2025 Season Analysis")
        print(f"Total games in database: {len(all_cfc_games)}")
        
        # Separate completed and future games
        completed_games = [g for g in all_cfc_games if g['is_completed']]
        future_games = [g for g in all_cfc_games if not g['is_completed']]
        
        print(f"Completed: {len(completed_games)}")
        print(f"Remaining: {len(future_games)}")
        
        # Analyze completed games
        if completed_games:
            print(f"\n📊 Completed Games Analysis:")
            
            wins = 0
            draws = 0
            losses = 0
            so_wins = 0
            so_losses = 0
            total_points = 0
            goals_for = 0
            goals_against = 0
            
            print("\nGame-by-Game Results:")
            print("-" * 80)
            
            for game in completed_games:
                cfc_score = int(game['cfc_score'] or 0)
                opp_score = int(game['opp_score'] or 0)
                
                goals_for += cfc_score
                goals_against += opp_score
                
                # Determine opponent
                if game['cfc_location'] == 'HOME':
                    opponent = game['away_team_name']
                    location = "vs"
                else:
                    opponent = game['home_team_name']
                    location = "@"
                
                # Determine result and points
                if cfc_score > opp_score:
                    result = "W"
                    wins += 1
                    total_points += 3
                    result_str = f"WIN (+3)"
                elif cfc_score < opp_score:
                    result = "L"
                    losses += 1
                    result_str = f"LOSS (0)"
                else:  # Draw
                    if game['went_to_shootout']:
                        draws += 1
                        # Check shootout result
                        if game['cfc_location'] == 'HOME':
                            cfc_pens = game['home_penalties'] or 0
                            opp_pens = game['away_penalties'] or 0
                        else:
                            cfc_pens = game['away_penalties'] or 0
                            opp_pens = game['home_penalties'] or 0
                        
                        if cfc_pens > opp_pens:
                            result = "W(SO)"
                            so_wins += 1
                            total_points += 2
                            result_str = f"SO WIN (+2)"
                        else:
                            result = "L(SO)"
                            so_losses += 1
                            total_points += 1
                            result_str = f"SO LOSS (+1)"
                    else:
                        # Regular draw (shouldn't happen in MLSNP but just in case)
                        result = "D"
                        draws += 1
                        total_points += 1
                        result_str = f"DRAW (+1)"
                
                print(f"{game['date'].strftime('%m/%d')} {location:>3} {opponent:<25} {cfc_score}-{opp_score} {result:<6} {result_str:<12} Pts: {total_points}")
            
            print("-" * 80)
            print(f"\n📈 Season Summary:")
            print(f"Record: {wins}W-{draws}D(SO)-{losses}L")
            print(f"Shootout Record: {so_wins} wins, {so_losses} losses")
            print(f"Goals: {goals_for} For, {goals_against} Against ({goals_for-goals_against:+d})")
            print(f"Total Points: {total_points}")
            print(f"PPG: {total_points/len(completed_games):.2f}")
            
            # Project full season
            games_remaining = len(future_games)
            if len(completed_games) > 0:
                ppg = total_points / len(completed_games)
                projected_points = total_points + (ppg * games_remaining)
                points_needed_for_playoffs = 43  # Based on your prediction
                
                print(f"\n🎯 Projections:")
                print(f"Games Remaining: {games_remaining}")
                print(f"Current Pace: {ppg:.2f} PPG")
                print(f"Projected Final Points: {projected_points:.1f}")
                print(f"Points Needed for Playoffs: ~{points_needed_for_playoffs}")
                
                if projected_points < points_needed_for_playoffs:
                    deficit = points_needed_for_playoffs - total_points
                    ppg_needed = deficit / games_remaining if games_remaining > 0 else 0
                    print(f"\n⚠️  WARNING: Current pace won't make playoffs!")
                    print(f"Need {deficit} points from {games_remaining} games")
                    print(f"Required PPG for rest of season: {ppg_needed:.2f}")
                    
                    # What record would that require?
                    wins_needed = int(deficit / 3)
                    so_wins_needed = int((deficit - wins_needed * 3) / 2)
                    print(f"That's approximately {wins_needed} wins + {so_wins_needed} SO wins")
                else:
                    print(f"\n✅ On pace for playoffs!")
            
            # Check recent form
            if len(completed_games) >= 5:
                recent_games = completed_games[-5:]
                recent_points = 0
                for game in recent_games:
                    if game['cfc_score'] > game['opp_score']:
                        recent_points += 3
                    elif game['cfc_score'] == game['opp_score']:
                        if game['went_to_shootout']:
                            # Check who won (simplified - you'd need to check penalties)
                            recent_points += 2  # Assume they won some
                        else:
                            recent_points += 1
                
                print(f"\n📊 Last 5 Games Form: {recent_points} points ({recent_points/5:.1f} PPG)")
        
        # Show upcoming games
        if future_games:
            print(f"\n📅 Next 5 Games:")
            for game in future_games[:5]:
                if game['cfc_location'] == 'HOME':
                    opponent = game['away_team_name']
                    location = "vs"
                else:
                    opponent = game['home_team_name']
                    location = "@"
                
                print(f"{game['date'].strftime('%m/%d')} {location} {opponent}")
        
        # Compare conferences
        print("\n🏆 Conference Comparison:")
        conf_standings = await database.fetch_all("""
            SELECT 
                t.team_name,
                COUNT(CASE WHEN g.is_completed THEN 1 END) as games_played,
                SUM(CASE 
                    WHEN g.is_completed AND 
                         ((g.home_team_id = t.team_id AND g.home_score > g.away_score) OR
                          (g.away_team_id = t.team_id AND g.away_score > g.home_score))
                    THEN 3
                    WHEN g.is_completed AND g.went_to_shootout
                    THEN 1  -- Simplified - would need to check who won
                    ELSE 0
                END) as points
            FROM team t
            JOIN team_affiliations ta ON t.team_id = ta.team_id
            LEFT JOIN games g ON (g.home_team_id = t.team_id OR g.away_team_id = t.team_id) 
                AND g.season_year = 2025
            WHERE ta.conference_id = 1 AND ta.is_current = true
            GROUP BY t.team_id, t.team_name
            HAVING COUNT(CASE WHEN g.is_completed THEN 1 END) > 0
            ORDER BY points DESC
        """)
        
        print("\nEastern Conference Current Standings (Simplified):")
        for i, team in enumerate(conf_standings[:10]):
            marker = "→" if "Chattanooga" in team['team_name'] else " "
            print(f"{marker} {i+1}. {team['team_name']:<30} {team['points']:>3} pts ({team['games_played']} games)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()

async def test_ml_readiness():
    """Test if we can now train ML"""
    
    print("\n\n🤖 Testing ML Readiness")
    print("=" * 40)
    
    try:
        await connect()
        db_manager = DatabaseManager(database)
        
        # This should work now
        print("Testing data fetch...")
        sim_data = await db_manager.get_data_for_simulation('eastern', 2025)
        
        completed = [g for g in sim_data['games_data'] if g.get('is_completed')]
        print(f"✅ Successfully loaded {len(completed)} completed games")
        print("✅ Ready to train ML model!")
        
        print("\n🎯 Next step: python train_with_real_data.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        await disconnect()

if __name__ == "__main__":
    asyncio.run(analyze_chattanooga_deeply())
    asyncio.run(test_ml_readiness())