import logging
import sys
import io
import asyncio
import numpy as np
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
    all_teams_xg = await db_manager.db.fetch_all("""
        SELECT 
            team_id,
            x_goals_for,
            x_goals_against,
            games_played
        FROM team_xg_history
        WHERE season_year = :season_year
        AND games_played > 0
        ORDER BY team_id, date_captured DESC
    """, values={"season_year": season_year})
    
    # Group by team and take most recent data point for each team
    team_latest = {}
    for row in all_teams_xg:
        if row['team_id'] not in team_latest:
            team_latest[row['team_id']] = row
    
    # Calculate weighted averages
    total_xgf = sum(t['x_goals_for'] or 0 for t in team_latest.values())  # Handle None values
    total_xga = sum(t['x_goals_against'] or 0 for t in team_latest.values())  # Handle None values
    total_games = sum(t['games_played'] for t in team_latest.values())
    
    # SAFETY GUARDS: Ensure we never return 0 for league averages
    if total_games > 0 and total_xgf > 0 and total_xga > 0:
        league_avg_xgf = total_xgf / total_games
        league_avg_xga = total_xga / total_games
    else:
        # Fallback values when no data is available OR when xG values are all zero
        logger.warning(f"Using fallback league averages. total_games={total_games}, total_xgf={total_xgf}, total_xga={total_xga}")
        league_avg_xgf = 1.2
        league_avg_xga = 1.2
    
    # Additional safety check - ensure values are never zero
    league_avg_xgf = max(league_avg_xgf, 0.1)  # Minimum 0.1 goals per game
    league_avg_xga = max(league_avg_xga, 0.1)  # Minimum 0.1 goals per game
    
    logger.info(f"Calculated league averages: xGF={league_avg_xgf:.3f}, xGA={league_avg_xga:.3f}")
    
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
        logger.info(f"Updating incomplete games for {season_year}...")
        await db_manager.update_incomplete_games(season_year)
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

            # Get team performance data (xG) for the season for the fetched teams
            team_performance_data: Dict[str, Dict] = {}
            for team_id in conference_teams_map.keys():
                xg_data = await db_manager.get_or_fetch_team_xg(team_id, season_year)
                if xg_data and xg_data.get('games_played', 0) > 0:
                    team_performance_data[team_id] = xg_data
                    logger.debug(f"Retrieved xG data for {conference_teams_map[team_id]}: {xg_data}")
                else:
                    logger.warning(f"Could not retrieve xG data for {conference_teams_map[team_id]} (ID: {team_id}). Using fallback values.")
                    # Provide fallback data instead of skipping the team
                    team_performance_data[team_id] = {
                        'games_played': 1,
                        'x_goals_for': league_averages['league_avg_xgf'],
                        'x_goals_against': league_averages['league_avg_xga'],
                        'goals_for': league_averages['league_avg_xgf'],
                        'goals_against': league_averages['league_avg_xga']
                    }
            
            if not team_performance_data:
                logger.error(f"No team performance data available for {conf_display_name} Conference. Cannot proceed with simulation.")
                continue

            # Get all games for the specified season year and conference
            # FIXED: Pass both season_year and conference properly
            all_games_data = await db_manager.get_games_for_season(season_year, include_incomplete=True)
            logger.info(f"Total games retrieved from database: {len(all_games_data)}")
            
            # Filter games for this conference
            conference_team_ids = set(conference_teams_map.keys())
            conference_games = [
                game for game in all_games_data 
                if (game.get('home_team_id') in conference_team_ids and 
                    game.get('away_team_id') in conference_team_ids)
            ]
            
            logger.info(f"Conference games after filtering: {len(conference_games)}")
            completed_conference_games = [g for g in conference_games if g.get('is_completed')]
            incomplete_conference_games = [g for g in conference_games if not g.get('is_completed')]
            logger.info(f"Completed conference games: {len(completed_conference_games)}")
            logger.info(f"Incomplete conference games: {len(incomplete_conference_games)}")
            logger.info(f"Found {len(conference_games)} games for {conf_display_name} Conference")

            # Initialize predictor
            predictor = MLSNPRegSeasonPredictor(
                conference=conf_name,
                conference_teams=conference_teams_map,
                games_data=conference_games,  # Use filtered games
                team_performance=team_performance_data,
                league_averages=league_averages
            )

            logger.info(f"Running {n_simulations} simulations for {conf_display_name} conference...")
            summary_df, simulation_results, _, qualification_data = predictor.run_simulations(n_simulations)

            if not summary_df.empty:
                logger.info(f"Successfully generated predictions for {conf_display_name} conference.")
                print(f"\n--- {conf_display_name} Conference Final Standings Projection ---")
                print(summary_df.to_string())

                # Save results to CSV
                output_file = f"output/{conf_name}_season_simulation_results_{season_year}.csv"
                summary_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
                
                # Store in database if desired
                store_in_db = input(f"\nStore {conf_display_name} conference results in database? (y/n): ").lower().strip()
                if store_in_db == 'y':
                    # Create a mock user_id for standalone execution
                    run_id = await db_manager.store_simulation_run(
                        user_id=1,  # Default to user_id 1 for standalone
                        conference=conf_name,
                        n_simulations=n_simulations,
                        season_year=season_year
                    )
                    
                    await db_manager.store_simulation_results(
                        run_id, summary_df, simulation_results, qualification_data
                    )
                    logger.info(f"Results stored in database with run_id: {run_id}")
                
            else:
                logger.error(f"Unable to generate predictions for {conf_display_name} conference - summary DataFrame is empty.")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    
    finally:
        # Disconnect from database
        logger.info("Disconnecting from database...")
        await database.disconnect()
    
    logger.info("\nMLS Next Pro Predictor execution finished.")

if __name__ == "__main__":
    asyncio.run(main())