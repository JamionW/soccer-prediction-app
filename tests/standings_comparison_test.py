import asyncio
import csv
import io
import logging
import sys
import os
from typing import Dict, List, Tuple
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import your predictor components
try:
    from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor
    from src.common.database import database
    from src.common.database_manager import DatabaseManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root directory.")
    print("Current directory:", os.getcwd())
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandingsComparisonTest:
    def __init__(self):
        self.real_standings = {}  # Will store parsed CSV data
        self.predictor_standings = {}  # Will store predictor results
        self.db_manager = None
        
    def parse_csv_standings(self, csv_content: str) -> Dict[str, Dict]:
        """
        Parse the current_table.txt file which contains two CSV tables.
        Returns a dictionary with team ASA_IDs as keys.
        """
        standings = {}
        
        # Split the content by the markdown headers
        sections = csv_content.split('```csv')
        
        for section in sections[1:]:  # Skip the first section (header)
            # Clean up the section and get just the CSV part
            csv_data = section.split('```')[0].strip()
            if not csv_data:
                continue
                
            # Parse the CSV
            reader = csv.DictReader(io.StringIO(csv_data))
            
            for row in reader:
                asa_id = row.get('ASA_ID', '').strip()
                if not asa_id:
                    continue
                    
                # Parse the stats, handling potential formatting issues
                try:
                    # Parse Home and Away records (format: W-L-T)
                    home_record = row['Home'].split('-')
                    away_record = row['Away'].split('-')
                    
                    home_wins = int(home_record[0])
                    home_losses = int(home_record[1]) 
                    home_ties = int(home_record[2])
                    
                    away_wins = int(away_record[0])
                    away_losses = int(away_record[1])
                    away_ties = int(away_record[2])
                    
                    # CORRECTED: W = regulation wins, not total wins!
                    regulation_wins = int(row['W'])  # This is regulation wins directly
                    shootout_wins = int(row['SOW'])
                    total_wins = regulation_wins + shootout_wins  # Calculate total wins
                    total_losses = int(row['L'])
                    total_ties = int(row['T'])
                    
                    standings[asa_id] = {
                        'team_name': row['Club'].strip(),
                        'points': int(row['PointsPTS']),
                        'games_played': int(row['GP']),
                        'total_wins': total_wins,  # regulation + shootout wins
                        'regulation_wins': regulation_wins,  # From W column directly
                        'shootout_wins': shootout_wins,
                        'losses': total_losses,
                        'ties': total_ties,  # Games that went to shootout
                        'goals_for': int(row['GF']),
                        'goals_against': int(row['GA']),
                        'goal_difference': int(row['GD']),
                        'rank': int(row['Rank']),
                        # Additional breakdown for debugging
                        'home_record': f"{home_wins}-{home_losses}-{home_ties}",
                        'away_record': f"{away_wins}-{away_losses}-{away_ties}",
                        'home_wins': home_wins,
                        'home_losses': home_losses,
                        'home_ties': home_ties,
                        'away_wins': away_wins,
                        'away_losses': away_losses,
                        'away_ties': away_ties
                    }
                    
                except (ValueError, KeyError, IndexError) as e:
                    logger.warning(f"Error parsing row for {row.get('Club', 'Unknown')}: {e}")
                    continue
        
        logger.info(f"Parsed {len(standings)} teams from CSV data")
        return standings
    
    async def get_predictor_standings(self, conference: str, season_year: int) -> Dict[str, Dict]:
        """
        Get standings as calculated by your predictor.
        """
        # Initialize database manager
        self.db_manager = DatabaseManager(database)
        await self.db_manager.initialize()
        
        # Get data for simulation (this includes calculating current standings)
        sim_data = await self.db_manager.get_data_for_simulation(conference, season_year)
        
        # Create predictor instance
        predictor = MLSNPRegSeasonPredictor(
            conference=conference,
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages={"league_avg_xgf": 1.2, "league_avg_xga": 1.2}  # Default values
        )
        
        # Get the calculated standings
        return predictor.current_standings
    
    def compare_standings(self, real: Dict, calculated: Dict) -> List[Dict]:
        """
        Compare real standings with calculated standings and return discrepancies.
        """
        discrepancies = []
        
        for asa_id, real_stats in real.items():
            if asa_id not in calculated:
                discrepancies.append({
                    'team_id': asa_id,
                    'team_name': real_stats['team_name'],
                    'issue': 'MISSING_FROM_PREDICTOR',
                    'real': real_stats,
                    'calculated': None
                })
                continue
            
            calc_stats = calculated[asa_id]
            team_discrepancies = []
            
            # CORRECTED: Compare the right metrics
            # CSV "W" = total wins, predictor tracks regulation wins + shootout wins separately
            calc_total_wins = calc_stats.get('wins', 0) + calc_stats.get('shootout_wins', 0)
            
            comparisons = [
                ('points', 'points', real_stats['points'], calc_stats.get('points', 0)),
                ('games_played', 'games_played', real_stats['games_played'], calc_stats.get('games_played', 0)),
                ('total_wins', 'total_wins', real_stats['total_wins'], calc_total_wins),
                ('regulation_wins', 'regulation_wins', real_stats['regulation_wins'], calc_stats.get('wins', 0)),
                ('shootout_wins', 'shootout_wins', real_stats['shootout_wins'], calc_stats.get('shootout_wins', 0)),
                ('losses', 'losses', real_stats['losses'], calc_stats.get('losses', 0)),
                ('ties', 'ties_to_shootout', real_stats['ties'], calc_stats.get('draws', 0)),
                ('goals_for', 'goals_for', real_stats['goals_for'], calc_stats.get('goals_for', 0)),
                ('goals_against', 'goals_against', real_stats['goals_against'], calc_stats.get('goals_against', 0)),
                ('goal_difference', 'goal_difference', real_stats['goal_difference'], calc_stats.get('goal_difference', 0))
            ]
            
            for metric_name, calc_name, real_val, calc_val in comparisons:
                if real_val != calc_val:
                    team_discrepancies.append({
                        'metric': metric_name,
                        'real': real_val,
                        'calculated': calc_val,
                        'difference': calc_val - real_val
                    })
            
            if team_discrepancies:
                discrepancies.append({
                    'team_id': asa_id,
                    'team_name': real_stats['team_name'],
                    'issue': 'STAT_MISMATCH',
                    'real': real_stats,
                    'calculated': calc_stats,
                    'calculated_total_wins': calc_total_wins,  # Add this for debugging
                    'discrepancies': team_discrepancies
                })
        
        # Check for teams in predictor but not in real data
        for team_id in calculated:
            if team_id not in real:
                discrepancies.append({
                    'team_id': team_id,
                    'team_name': calculated[team_id].get('name', 'Unknown'),
                    'issue': 'EXTRA_IN_PREDICTOR',
                    'real': None,
                    'calculated': calculated[team_id]
                })
        
        return discrepancies
    
    def print_detailed_report(self, discrepancies: List[Dict]):
        """
        Print a detailed report of all discrepancies found.
        """
        print("\n" + "="*80)
        print("STANDINGS COMPARISON REPORT")
        print("="*80)
        
        if not discrepancies:
            print("🎉 NO DISCREPANCIES FOUND! All standings match perfectly.")
            return
        
        print(f"Found {len(discrepancies)} teams with discrepancies:\n")
        
        for i, team_issue in enumerate(discrepancies, 1):
            print(f"{i}. {team_issue['team_name']} ({team_issue['team_id']})")
            print(f"   Issue: {team_issue['issue']}")
            
            if team_issue['issue'] == 'STAT_MISMATCH':
                print("   Discrepancies:")
                for disc in team_issue['discrepancies']:
                    print(f"     {disc['metric']}: Real={disc['real']}, Calculated={disc['calculated']}, Diff={disc['difference']:+d}")
                
                # Show detailed comparison
                real = team_issue['real']
                calc = team_issue['calculated']
                calc_total_wins = team_issue.get('calculated_total_wins', calc.get('wins', 0) + calc.get('shootout_wins', 0))
                
                print(f"   CSV Data:        {real['points']}pts, {real['games_played']}GP, {real['total_wins']}W({real['regulation_wins']}reg+{real['shootout_wins']}SO), {real['losses']}L, {real['ties']}T")
                print(f"                    Home: {real['home_record']}, Away: {real['away_record']}")
                print(f"   Predictor Data:  {calc.get('points', 0)}pts, {calc.get('games_played', 0)}GP, {calc_total_wins}W({calc.get('wins', 0)}reg+{calc.get('shootout_wins', 0)}SO), {calc.get('losses', 0)}L, {calc.get('draws', 0)}T")
                
                # Verify points calculation for CSV data  
                # MLS Next Pro: 3pts for reg win, 2pts for SO win, 1pt for SO loss, 0pts for reg loss
                shootout_losses = real['ties'] - real['shootout_wins']  # Games that went to SO but were lost
                expected_points = (real['regulation_wins'] * 3) + (real['shootout_wins'] * 2) + (shootout_losses * 1)
                if expected_points != real['points']:
                    print(f"   ⚠️  CSV points verification: Expected {expected_points}, got {real['points']} (difference: {real['points'] - expected_points})")
                    print(f"       Calculation: {real['regulation_wins']}reg×3 + {real['shootout_wins']}SO_wins×2 + {shootout_losses}SO_losses×1 = {expected_points}")
                else:
                    print(f"   ✅ CSV points calculation verified: {real['regulation_wins']}×3 + {real['shootout_wins']}×2 + {shootout_losses}×1 = {real['points']}")

            
            elif team_issue['issue'] == 'MISSING_FROM_PREDICTOR':
                print(f"   CSV data: {team_issue['real']}")
            
            elif team_issue['issue'] == 'EXTRA_IN_PREDICTOR':
                print(f"   Predictor data: {team_issue['calculated']}")
            
            print()
        
        # Summary statistics
        stat_mismatches = [d for d in discrepancies if d['issue'] == 'STAT_MISMATCH']
        if stat_mismatches:
            print("MOST COMMON ISSUES:")
            metric_counts = {}
            for team in stat_mismatches:
                for disc in team['discrepancies']:
                    metric_counts[disc['metric']] = metric_counts.get(disc['metric'], 0) + 1
            
            for metric, count in sorted(metric_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {metric}: {count} teams affected")

    async def debug_specific_team(self, team_id: str, season_year: int = 2025):
        """
        Debug a specific team's game-by-game calculation to find the issue.
        """
        print(f"\n=== DEBUGGING TEAM {team_id} ===")
        
        # Get all games for this team with more details
        games_query = """
            SELECT g.*, ht.team_name as home_team_name, at.team_name as away_team_name
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE g.season_year = :season_year 
            AND (g.home_team_id = :team_id OR g.away_team_id = :team_id)
            AND g.is_completed = true
            ORDER BY g.date
        """
        
        games = await self.db_manager.db.fetch_all(
            games_query, 
            values={"season_year": season_year, "team_id": team_id}
        )
        
        print(f"Found {len(games)} completed games for team {team_id}")
        print("\nGame-by-game breakdown:")
        print("Date       | Matchup                    | Score | Shootout | Result | Points | ASA? | GameID")
        print("-" * 100)
        
        total_points = 0
        reg_wins = 0
        so_wins = 0
        losses = 0
        draws = 0
        
        for game in games:
            game_dict = dict(game)
            home_id = game_dict['home_team_id']
            away_id = game_dict['away_team_id']
            home_score = game_dict['home_score']
            away_score = game_dict['away_score']
            went_to_shootout = game_dict['went_to_shootout']
            home_pens = game_dict.get('home_penalties', 0) or 0
            away_pens = game_dict.get('away_penalties', 0) or 0
            asa_loaded = game_dict.get('asa_loaded', False)
            game_id = game_dict.get('game_id', 'N/A')
            
            # Determine if team was home or away
            is_home = (home_id == team_id)
            opponent = away_id if is_home else home_id
            opponent_name = game_dict['away_team_name'] if is_home else game_dict['home_team_name']
            
            team_score = home_score if is_home else away_score
            opp_score = away_score if is_home else home_score
            
            # Determine result
            if went_to_shootout:
                draws += 1
                if is_home:
                    if home_pens > away_pens:
                        result = "SO Win"
                        so_wins += 1
                        total_points += 2
                    else:
                        result = "SO Loss"
                        total_points += 1
                else:
                    if away_pens > home_pens:
                        result = "SO Win"
                        so_wins += 1
                        total_points += 2
                    else:
                        result = "SO Loss"
                        total_points += 1
                shootout_text = f"SO {home_pens}-{away_pens}"
            else:
                shootout_text = "No"
                if team_score > opp_score:
                    result = "Reg Win"
                    reg_wins += 1
                    total_points += 3
                elif team_score < opp_score:
                    result = "Reg Loss"
                    losses += 1
                else:
                    result = "Reg Draw"
                    draws += 1
                    total_points += 1
            
            # Format the output
            date_str = game_dict['date'].strftime('%Y-%m-%d') if game_dict['date'] else 'Unknown'
            matchup = f"{'vs' if is_home else '@'} {opponent_name[:15]:<15}"
            score_str = f"{team_score}-{opp_score}"
            asa_str = "ASA" if asa_loaded else "Manual"
            
            print(f"{date_str} | {matchup} | {score_str:^5} | {shootout_text:^8} | {result:^7} | +{2 if result=='SO Win' else 3 if result=='Reg Win' else 1 if 'Loss' not in result else 0} | {asa_str:^6} | {game_id[:12]}")
            
            # Flag suspicious games
            if went_to_shootout and (home_pens == 0 and away_pens == 0):
                print(f"    ⚠️  WARNING: Shootout game with no penalty data!")
            if not asa_loaded:
                print(f"    ⚠️  WARNING: Game not loaded from ASA - might be unofficial!")
        
        print("-" * 100)
        print(f"TOTALS: {total_points} points, {reg_wins} reg wins, {so_wins} SO wins, {losses} losses, {draws} ties")
        
        # Check for non-ASA games
        non_asa_games = [dict(g) for g in games if not dict(g).get('asa_loaded', False)]
        if non_asa_games:
            print(f"\n⚠️  Found {len(non_asa_games)} games NOT from ASA:")
            for game in non_asa_games:
                print(f"   - {game['game_id']}: {game.get('date', 'No date')}")
            print("   These might be scrimmages, cup games, or other non-league matches!")
        
        # Calculate home/away breakdowns
        home_reg_wins = away_reg_wins = 0
        home_so_wins = away_so_wins = 0  
        home_losses = away_losses = 0
        home_ties = away_ties = 0
        
        for game in games:
            game_dict = dict(game)
            home_id = game_dict['home_team_id']
            went_to_shootout = game_dict['went_to_shootout']
            home_score = game_dict['home_score']
            away_score = game_dict['away_score']
            home_pens = game_dict.get('home_penalties', 0) or 0
            away_pens = game_dict.get('away_penalties', 0) or 0
            
            is_home = (home_id == team_id)
            team_score = home_score if is_home else away_score
            opp_score = away_score if is_home else home_score
            
            if went_to_shootout:
                if is_home:
                    home_ties += 1
                    if home_pens > away_pens:
                        home_so_wins += 1
                else:
                    away_ties += 1  
                    if away_pens > home_pens:
                        away_so_wins += 1
            else:
                if team_score > opp_score:  # Regulation win
                    if is_home:
                        home_reg_wins += 1
                    else:
                        away_reg_wins += 1
                elif team_score < opp_score:  # Regulation loss
                    if is_home:
                        home_losses += 1
                    else:
                        away_losses += 1
        
        print(f"\nHOME/AWAY BREAKDOWN:")
        print(f"Home: {home_reg_wins + home_so_wins}-{home_losses}-{home_ties} ({home_reg_wins}reg wins, {home_so_wins}SO wins)")
        print(f"Away: {away_reg_wins + away_so_wins}-{away_losses}-{away_ties} ({away_reg_wins}reg wins, {away_so_wins}SO wins)")
        
        return {
            'points': total_points,
            'regulation_wins': reg_wins,
            'shootout_wins': so_wins,
            'losses': losses,
            'draws': draws,
            'total_wins': reg_wins + so_wins,
            'home_record': f"{home_reg_wins + home_so_wins}-{home_losses}-{home_ties}",
            'away_record': f"{away_reg_wins + away_so_wins}-{away_losses}-{away_ties}",
            'home_reg_wins': home_reg_wins,
            'home_so_wins': home_so_wins,
            'away_reg_wins': away_reg_wins, 
            'away_so_wins': away_so_wins,
            'non_asa_games': len([dict(g) for g in games if not dict(g).get('asa_loaded', False)])
        }

    async def run_test(self, season_year: int = 2025):
        """
        Run the complete standings comparison test.
        """
        print(f"Starting standings comparison test for {season_year} season...")
        
        try:
            # Connect to database
            await database.connect()
            
            # Read and parse the CSV data
            with open('current_table.txt', 'r', encoding='utf-8') as f:
                csv_content = f.read()
            
            self.real_standings = self.parse_csv_standings(csv_content)
            print(f"Loaded real standings for {len(self.real_standings)} teams")
            
            # Get predictor standings for both conferences
            predictor_standings = {}
            
            for conference in ['eastern', 'western']:
                print(f"\nProcessing {conference} conference...")
                conf_standings = await self.get_predictor_standings(conference, season_year)
                predictor_standings.update(conf_standings)
            
            print(f"Calculated predictor standings for {len(predictor_standings)} teams")
            
            # Compare the standings
            discrepancies = self.compare_standings(self.real_standings, predictor_standings)
            
            # Print detailed report
            self.print_detailed_report(discrepancies)
            
            # Debug a specific team if there are discrepancies
            if discrepancies:
                print(f"\n{'='*80}")
                print("GAME-BY-GAME DEBUGGING")
                print("="*80)
                print("To debug a specific team's games, you can call:")
                print("debug_result = await test.debug_specific_team('TEAM_ASA_ID')")
                print("\nExample teams with issues:")
                for d in discrepancies[:3]:  # Show first 3 teams
                    if d['issue'] == 'STAT_MISMATCH':
                        print(f"  {d['team_name']}: {d['team_id']}")
                
                # Automatically debug the first team with issues
                first_problem_team = None
                for d in discrepancies:
                    if d['issue'] == 'STAT_MISMATCH':
                        first_problem_team = d['team_id']
                        break
                
                if first_problem_team:
                    print(f"\nAutomatically debugging first problematic team: {first_problem_team}")
                    debug_result = await self.debug_specific_team(first_problem_team, season_year)
                    
                    # Compare with CSV data
                    real_data = self.real_standings.get(first_problem_team, {})
                    print(f"\nComparison for {first_problem_team}:")
                    print(f"Manual calculation: {debug_result}")
                    print(f"CSV data:           {real_data}")
                    
                    # Show specific comparisons
                    print(f"\nDetailed comparison:")
                    print(f"Points:       Manual={debug_result['points']}, CSV={real_data.get('points', 'N/A')}")
                    print(f"Reg Wins:     Manual={debug_result['regulation_wins']}, CSV={real_data.get('regulation_wins', 'N/A')}")
                    print(f"SO Wins:      Manual={debug_result['shootout_wins']}, CSV={real_data.get('shootout_wins', 'N/A')}")
                    print(f"Home Record:  Manual={debug_result['home_record']}, CSV={real_data.get('home_record', 'N/A')}")
                    print(f"Away Record:  Manual={debug_result['away_record']}, CSV={real_data.get('away_record', 'N/A')}")
                    
                    # Check if manual calculation matches CSV
                    matches_csv = (
                        debug_result['points'] == real_data.get('points', 0) and
                        debug_result['regulation_wins'] == real_data.get('regulation_wins', 0) and
                        debug_result['shootout_wins'] == real_data.get('shootout_wins', 0)
                    )
                    
                    if matches_csv:
                        print(f"\n✅ Manual calculation matches CSV data!")
                        print(f"   This suggests the issue is in your predictor's _calculate_current_standings method.")
                    else:
                        print(f"\n❌ Manual calculation doesn't match CSV data!")
                        print(f"   This suggests there may be missing or incorrect game data in your database.")

            
        except FileNotFoundError:
            print("ERROR: current_table.txt file not found. Make sure it's in the current directory.")
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Test failed: {e}", exc_info=True)
        finally:
            await database.disconnect()

async def main():
    """
    Main function to run the standings comparison test.
    """
    test = StandingsComparisonTest()
    await test.run_test()

if __name__ == "__main__":
    asyncio.run(main())