import logging
import sys
import io
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor
from src.common.database import database
from src.common.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/run_predictor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


async def calculate_league_averages(db_manager: DatabaseManager, season_year: int) -> Dict[str, float]:
    """
    Calculate league-wide averages for all teams for a given season.
    Added safety guards against division by zero.
    """
    # Get all teams' performance data
    try:
        all_teams_xg = await db_manager.db.fetch_all("""
            SELECT 
                team_id,
                x_goals_for,
                x_goals_against,
                games_played,
                date_captured
            FROM team_xg_history
            WHERE season_year = :season_year
            AND games_played > 0
            ORDER BY team_id, date_captured DESC
        """, values={"season_year": season_year})
    except Exception as e:
        logger.error(f"Error fetching team xG data: {e}")
        return {"league_avg_xgf": 1.2, "league_avg_xga": 1.2, "total_teams": 0, "total_games": 0}
    
    if not all_teams_xg:
        logger.warning(f"No xG data found for season {season_year}")
        return {"league_avg_xgf": 1.2, "league_avg_xga": 1.2, "total_teams": 0, "total_games": 0}
    
    # Group by team and take most recent data point
    team_latest = {}
    for row in all_teams_xg:
        team_id = row['team_id']
        if team_id not in team_latest:
            team_latest[team_id] = dict(row)
    
    # Calculate weighted averages with validation
    total_xgf = sum(float(t['x_goals_for'] or 0) for t in team_latest.values())
    total_xga = sum(float(t['x_goals_against'] or 0) for t in team_latest.values())
    total_games = sum(int(t['games_played']) for t in team_latest.values())
    
    # Safety guards
    if total_games > 0 and total_xgf > 0 and total_xga > 0:
        league_avg_xgf = total_xgf / total_games
        league_avg_xga = total_xga / total_games
    else:
        logger.warning(f"Using fallback averages. Games: {total_games}, xGF: {total_xgf}, xGA: {total_xga}")
        league_avg_xgf = 1.2
        league_avg_xga = 1.2
    
    # Ensure reasonable bounds
    league_avg_xgf = max(min(league_avg_xgf, 3.0), 0.5)
    league_avg_xga = max(min(league_avg_xga, 3.0), 0.5)
    
    logger.info(f"League averages: xGF={league_avg_xgf:.3f}/game, xGA={league_avg_xga:.3f}/game ({len(team_latest)} teams, {total_games} total games)")
    
    return {
        "league_avg_xgf": league_avg_xgf,
        "league_avg_xga": league_avg_xga,
        "total_teams": len(team_latest),
        "total_games": total_games
    }


async def main():
    logger.info("Starting MLS Next Pro Predictor standalone execution...")
    try:
        # Connect to database
        logger.info("Connecting to database...")
        await database.connect()
        db_manager = DatabaseManager(database)
        await db_manager.initialize() # Ensure conferences and other initial data are set up

        # Prompt user for season year
        try:
            season_year = int(input("Enter season year to simulate (e.g., 2025): "))
        except ValueError:
            logger.error("Invalid season year. Please enter a number (e.g., 2025).")
            return

        # Ask user which conference to simulate
        conference_input = input("Which conference to simulate? (eastern/western/both): ").lower().strip()
        if conference_input not in ['eastern', 'western', 'both']:
            logger.error("Invalid conference. Please choose 'eastern', 'western', or 'both'")
            return
        
        # Get number of simulations
        try:
            n_simulations = int(input("Number of simulations (default 10000): ") or "10000")
        except ValueError:
            n_simulations = 10000

        # Update incomplete games for the specified season year
        logger.info(f"Updating incomplete games with ASA for {season_year}...")
        if conference_input == 'both':
            await db_manager.update_games_with_asa(season_year, 'eastern')
            await db_manager.update_games_with_asa(season_year, 'western')
        else:
            await db_manager.update_games_with_asa(season_year, conference_input)
        logger.info("Finished updating incomplete games.")
        
        # Calculate league averages for the specified season year
        logger.info(f"Calculating league averages for {season_year}...")
        league_averages = await calculate_league_averages(db_manager, season_year)
        logger.info(f"League averages calculated: {league_averages}")

        # Determine conferences to simulate based on user input
        conferences_to_simulate: List[Tuple[str, str]] = []  # Changed to (name, name) instead of (name, id)
        if conference_input == 'eastern':
            conferences_to_simulate.append(('Eastern', 'eastern'))
        elif conference_input == 'western':
            conferences_to_simulate.append(('Western', 'western'))
        elif conference_input == 'both':
            conferences_to_simulate.append(('Eastern', 'eastern'))
            conferences_to_simulate.append(('Western', 'western'))

        for conf_display_name, conf_name in conferences_to_simulate:
            logger.info(f"\n--- Starting simulation for {conf_display_name} Conference (Year: {season_year}) ---")

            # Convert conference name to ID (ex. 1 = Eastern, 2 = Western)
            conf_id = 1 if conf_name == 'eastern' else 2
            conference_teams_map = await db_manager.get_conference_teams(conf_id, season_year)

            if not conference_teams_map:
                logger.warning(f"No teams found for {conf_display_name} Conference. Skipping simulation for this conference.")
                continue

            logger.info(f"Found {len(conference_teams_map)} teams for {conf_display_name} Conference")
            for team_id, team_name in conference_teams_map.items():
                logger.info(f"  - {team_name} ({team_id})")

            all_games = await db_manager.get_games_for_season(season_year, include_incomplete=True)

            conference_team_ids = set(conference_teams_map.keys())
            conference_games = [
                game for game in all_games 
                if (game.get('home_team_id') in conference_team_ids and 
                    game.get('away_team_id') in conference_team_ids)
            ]
            
            logger.info(f"Total games in database: {len(all_games)}")
            logger.info(f"Conference-specific games: {len(conference_games)}")

            # Analyze game completion status
            completed_games = [g for g in conference_games if g.get('is_completed')]
            incomplete_games = [g for g in conference_games if not g.get('is_completed')]
            games_with_scores = [g for g in conference_games if g.get('home_score') is not None]
            
            logger.info(f"Completed games: {len(completed_games)}")
            logger.info(f"Incomplete games: {len(incomplete_games)}")
            logger.info(f"Games with scores: {len(games_with_scores)}")

            # Get team performance data
            team_performance_data = {}
            teams_with_data = 0
            teams_with_fallback = 0
            
            for team_id in conference_teams_map.keys():
                xg_data = await db_manager.get_or_fetch_team_xg(team_id, season_year)
                if xg_data and xg_data.get('games_played', 0) > 0:
                    team_performance_data[team_id] = xg_data
                    teams_with_data += 1
                    logger.debug(f"xG data for {conference_teams_map[team_id]}: {xg_data['games_played']} games, {xg_data['x_goals_for']:.2f} xGF")
                else:
                    # Fallback data
                    team_performance_data[team_id] = {
                        'team_id': team_id,
                        'games_played': 1,
                        'x_goals_for': league_averages['league_avg_xgf'],
                        'x_goals_against': league_averages['league_avg_xga'],
                        'goals_for': league_averages['league_avg_xgf'],
                        'goals_against': league_averages['league_avg_xga']
                    }
                    teams_with_fallback += 1
                    logger.warning(f"Using fallback data for {conference_teams_map[team_id]}")
            
            logger.info(f"Teams with xG data: {teams_with_data}, teams with fallback: {teams_with_fallback}")

            # Initialize predictor with validation
            logger.info("Initializing predictor...")
            predictor = MLSNPRegSeasonPredictor(
                conference=conf_name,
                conference_teams=conference_teams_map,
                games_data=conference_games,
                team_performance=team_performance_data,
                league_averages=league_averages
            )

            # Validate predictor state before running simulations
            logger.info("Validating predictor state...")
            logger.info(f"Conference teams: {len(predictor.conference_teams)}")
            logger.info(f"Remaining games: {len(predictor.remaining_games)}")
            logger.info(f"Current standings teams: {len(predictor.current_standings)}")
            
            # Check standings calculation
            teams_with_games = [team_id for team_id, stats in predictor.current_standings.items() 
                              if stats.get('games_played', 0) > 0]
            logger.info(f"Teams with games played in standings: {len(teams_with_games)}")

            # Run simulations
            logger.info(f"Running {n_simulations} simulations for {conf_display_name} conference...")
            summary_df, simulation_results, _, qualification_data = predictor.run_simulations(n_simulations)

            # Better result handling and display
            if not summary_df.empty:
                logger.info(f"Successfully generated predictions for {conf_display_name} conference.")
                
                # Custom formatted output instead of raw pandas
                print(f"\n{'='*80}")
                print(f"{conf_display_name} Conference Final Standings Projection")
                print(f"{'='*80}")
                print(f"{'Proj':<4} {'Team':<25} {'Curr':<4} {'Pts':<4} {'GP':<4} {'Playoff %':<10} {'Avg Final':<10}")
                print(f"{'Rank':<4} {'':<25} {'Rank':<4} {'':<4} {'':<4} {'':<10} {'Rank':<10}")
                print(f"{'-'*80}")

                for idx, (_, row) in enumerate(summary_df.iterrows(), 1):
                    team_name = row['Team'][:24]  # Truncate long names
                    current_rank = int(row['Current Rank'])
                    current_pts = int(row['Current Points'])
                    games_played = int(row['Games Played'])
                    playoff_pct = f"{row['Playoff Qualification %']:.1f}%"
                    avg_rank = f"{row['Average Final Rank']:.2f}"
                    
                    print(f"{idx:<4} {team_name:<25} {current_rank:<4} {current_pts:<4} {games_played:<4} {playoff_pct:<10} {avg_rank:<10}")

                print(f"{'-'*80}")
                print(f"Total teams: {len(summary_df)}")
                print(f"Simulations run: {n_simulations:,}")

                # Additional insights
                playoff_teams = summary_df[summary_df['Playoff Qualification %'] >= 50.0]
                eliminated_teams = summary_df[summary_df['Playoff Qualification %'] < 1.0]
                clinched_teams = summary_df[summary_df['Playoff Qualification %'] >= 99.9]

                print(f"\nPlayoff Picture:")
                print(f"  Teams likely to make playoffs (>50%): {len(playoff_teams)}")
                print(f"  Teams clinched playoffs (>99.9%): {len(clinched_teams)}")
                print(f"  Teams effectively eliminated (<1%): {len(eliminated_teams)}")

                # Save results to CSV with timestamp
                output_file = f"output/{conf_name}_season_simulation_results_{season_year}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                summary_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
                
                # Store in database option
                store_in_db = input(f"\nStore {conf_display_name} conference results in database? (y/n): ").lower().strip()
                if store_in_db == 'y':
                    try:
                        run_id = await db_manager.store_simulation_run(
                            user_id=1,  # Default for standalone
                            conference=conf_name,
                            n_simulations=n_simulations,
                            season_year=season_year
                        )
                        
                        await db_manager.store_simulation_results(
                            run_id, summary_df, simulation_results, qualification_data
                        )
                        logger.info(f"Results stored in database with run_id: {run_id}")
                    except Exception as e:
                        logger.error(f"Error storing results: {e}")
                
            else:
                logger.error(f"ERROR: Summary DataFrame is empty for {conf_display_name} conference!")
                logger.error("This indicates a critical issue with data processing.")
                
                # Debug information
                logger.error(f"Debug info:")
                logger.error(f"  - Conference teams: {len(conference_teams_map)}")
                logger.error(f"  - Games data: {len(conference_games)}")
                logger.error(f"  - Team performance data: {len(team_performance_data)}")
                logger.error(f"  - Predictor teams: {len(predictor.conference_teams) if predictor else 'None'}")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    
    finally:
        logger.info("Disconnecting from database...")
        await database.disconnect()
    
    logger.info("\nMLS Next Pro Predictor execution finished.")

if __name__ == "__main__":
    asyncio.run(main())