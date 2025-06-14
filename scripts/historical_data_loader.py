import asyncio
import logging
import sys
from typing import List, Dict, Optional
from datetime import datetime
from src.common.database import database
from src.common.database_manager import DatabaseManager

"""
Historical Data Loading Script for MLS Next Pro Predictor

This script loads historical data (2022-2025) for MLSNP teams:
- team_xg_history (Expected Goals data)
- team_ga_history (Goals Added data)

Has self._clean_2025_xg_data() so I could delete bad data while testing.
Can be changed to another year or commented out if not needed.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('historical_data_load.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Handles loading historical data from ASA API to database"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.seasons = [2022, 2023, 2024, 2025]
        self.teams_processed = 0
        self.errors_encountered = 0
        self.xg_records_loaded = 0
        self.ga_records_loaded = 0
    
    async def load_all_historical_data(self, dry_run: bool = False):
        """Main method to load all historical data"""
        run_type = "DRY RUN" if dry_run else "PRODUCTION"
        logger.info(f"Starting historical data loading process - {run_type}...")
        
        try:
            # Get MLSNP teams only
            teams = await self._get_mlsnp_teams()
            logger.info(f"Found {len(teams)} MLSNP teams to process")
            
            if not dry_run:
                # Step 1: Clean 2025 team_xg_history data
                await self._clean_2025_xg_data()
            
            # Step 2: Load team xG data for all seasons
            await self._load_team_xg_data(teams, dry_run)
            
            # Step 3: Load team Goals Added data for all seasons
            await self._load_team_ga_data(teams, dry_run)
            
            logger.info(f"Historical data loading completed successfully! ({run_type})")
            logger.info(f"Teams processed: {self.teams_processed}")
            logger.info(f"xG records loaded: {self.xg_records_loaded}")
            logger.info(f"Goals Added records loaded: {self.ga_records_loaded}")
            logger.info(f"Errors encountered: {self.errors_encountered}")
            
        except Exception as e:
            logger.error(f"Critical error during data loading: {e}", exc_info=True)
            raise
    
    async def _get_mlsnp_teams(self) -> List[Dict]:
        """Get only MLSNP teams from database"""
        query = """
            SELECT DISTINCT t.team_id, t.team_name, t.is_active
            FROM team t
            JOIN team_affiliations ta ON t.team_id = ta.team_id
            JOIN league l ON ta.league_id = l.league_id
            WHERE l.league_abbv = 'MLSNP'
            ORDER BY t.team_name
        """
        teams = await self.db_manager.db.fetch_all(query)
        return [dict(team) for team in teams]
    
    async def _clean_2025_xg_data(self):
        """Delete existing 2025 team_xg_history data"""
        logger.info("Cleaning existing 2025 team_xg_history data...")
        
        delete_query = """
            DELETE FROM team_xg_history 
            WHERE season_year = 2025
        """
        
        await self.db_manager.db.execute(delete_query)
        logger.info("✓ Cleaned 2025 team_xg_history data")
    
    async def _load_team_xg_data(self, teams: List[Dict], dry_run: bool):
        """Load team Expected Goals data for all seasons"""
        logger.info("Loading team xG data...")
        
        total_operations = len(teams) * len(self.seasons)
        current_operation = 0
        
        for team in teams:
            team_id = team['team_id']
            team_name = team['team_name']
            
            for season in self.seasons:
                current_operation += 1
                
                try:
                    logger.info(f"[{current_operation}/{total_operations}] Processing xG: {team_name} - {season}")
                    
                    xg_data = await self._fetch_team_xg_corrected(team_id, season, dry_run)
                    
                    if xg_data:
                        self.xg_records_loaded += 1
                        logger.debug(f"✓ Loaded xG data for {team_name} {season}")
                    else:
                        logger.warning(f"✗ No xG data for {team_name} {season}")
                    
                    # Small delay to be respectful to API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error loading xG data for {team_name} {season}: {e}")
                    self.errors_encountered += 1
            
            self.teams_processed += 1
    
    async def _fetch_team_xg_corrected(self, team_id: str, season_year: int, dry_run: bool) -> Optional[Dict]:
        try:
            # Get ASA client
            asa_client = self.db_manager.asa_client
            
            # Fetch from ASA API with MLSNP league filter
            stats_df = asa_client.get_team_xgoals(
                leagues=['mlsnp'],  # CRITICAL: Filter for MLSNP only
                team_ids=[team_id], 
                season_name=[str(season_year)]
            )
            
            if stats_df.empty:
                logger.debug(f"No xG data from ASA for team {team_id}, season {season_year}")
                return None

            stat = stats_df.iloc[0].to_dict()
            
            insert_values = {
                "team_id": team_id,
                "games_played": stat.get('count_games', 0),
                "shots_for": stat.get('shots_for', 0),
                "shots_against": stat.get('shots_against', 0),
                "x_goals_for": stat.get('xgoals_for', 0.0),
                "x_goals_against": stat.get('xgoals_against', 0.0),
                "season_year": season_year,
                "matchday": None  # Season summary, not matchday-specific
            }
            
            if dry_run:
                logger.info(f"DRY RUN: Would store xG data for {team_id} {season_year}:")
                logger.info(f"  games_played: {insert_values['games_played']}")
                logger.info(f"  shots_for: {insert_values['shots_for']} (was 0 in old data)")
                logger.info(f"  shots_against: {insert_values['shots_against']} (worked before)")  
                logger.info(f"  x_goals_for: {insert_values['x_goals_for']} (was 0 in old data)")
                logger.info(f"  x_goals_against: {insert_values['x_goals_against']} (worked before)")
                return insert_values
            
            # Store in database using UPSERT
            insert_query = """
                INSERT INTO team_xg_history (
                    team_id, season_year, games_played, shots_for, shots_against,
                    x_goals_for, x_goals_against, date_captured, matchday
                ) VALUES (
                    :team_id, :season_year, :games_played, :shots_for, :shots_against,
                    :x_goals_for, :x_goals_against, NOW(), :matchday
                )
                ON CONFLICT (team_id, season_year) DO UPDATE SET
                    games_played = EXCLUDED.games_played,
                    shots_for = EXCLUDED.shots_for,
                    shots_against = EXCLUDED.shots_against,
                    x_goals_for = EXCLUDED.x_goals_for,
                    x_goals_against = EXCLUDED.x_goals_against,
                    date_captured = NOW()
                RETURNING *
            """
            
            stored_data = await self.db_manager.db.fetch_one(insert_query, values=insert_values)
            return dict(stored_data)
            
        except Exception as e:
            logger.error(f"Error fetching xG data for team {team_id}, season {season_year}: {e}")
            return None
    
    async def _load_team_ga_data(self, teams: List[Dict], dry_run: bool):
        """Load team Goals Added data for all seasons"""
        logger.info("Loading team Goals Added data...")
        
        total_operations = len(teams) * len(self.seasons)
        current_operation = 0
        
        for team in teams:
            team_id = team['team_id']
            team_name = team['team_name']
            
            for season in self.seasons:
                current_operation += 1
                
                try:
                    logger.info(f"[{current_operation}/{total_operations}] Processing GA: {team_name} - {season}")
                    
                    ga_records = await self._fetch_team_ga(team_id, season, dry_run)
                    
                    if ga_records:
                        self.ga_records_loaded += len(ga_records)
                        logger.debug(f"✓ Loaded {len(ga_records)} GA records for {team_name} {season}")
                    else:
                        logger.warning(f"✗ No GA data for {team_name} {season}")
                    
                    # Small delay to be respectful to API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error loading GA data for {team_name} {season}: {e}")
                    self.errors_encountered += 1
    
    async def _fetch_team_ga(self, team_id: str, season_year: int, dry_run: bool) -> Optional[List[Dict]]:
        """
        Fetch team Goals Added data using parsing of nested data structure
        """
        try:
            # Get ASA client
            asa_client = self.db_manager.asa_client
            
            # Fetch Goals Added data from ASA API
            ga_stats_df = asa_client.get_team_goals_added(
                leagues=['mlsnp'],
                team_ids=[team_id],
                season_name=[str(season_year)]
            )
            
            if ga_stats_df.empty:
                logger.debug(f"No Goals Added data from ASA for team {team_id}, season {season_year}")
                return None
            
            row = ga_stats_df.iloc[0]
            team_minutes = int(row.get('minutes', 0))
            action_data_list = row.get('data', [])
            
            if not action_data_list:
                logger.warning(f"Empty data array for Goals Added: {team_id} {season_year}")
                return None
            
            stored_records = []
            
            # Process each action type from the data array
            for action_data in action_data_list:
                # Clean up action_type (fix any typos from ASA data)
                action_type = str(action_data.get('action_type', 'Unknown'))
                if 'Dribbl' in action_type and action_type != 'Dribbling':
                    action_type = 'Dribbling'  # Fix typos in ASA data
                
                insert_values = {
                    "team_id": team_id,
                    "action_type": action_type,
                    "num_actions_for": int(action_data.get('num_actions_for', 0)),
                    "goals_added_for": float(action_data.get('goals_added_for', 0.0)),
                    "num_actions_against": int(action_data.get('num_actions_against', 0)),
                    "goals_added_against": float(action_data.get('goals_added_against', 0.0)),
                    "season_year": season_year,
                    "matchday": None,  # Season summary
                    "minutes_played": team_minutes  # Use already-converted value
                }
                
                if dry_run:
                    logger.info(f"DRY RUN: Would store GA data for {team_id} {season_year} {action_type}: {insert_values}")
                    stored_records.append(insert_values)
                    continue
                
                # Store in database
                insert_query = """
                    INSERT INTO team_ga_history (
                        team_id, action_type, num_actions_for, goals_added_for,
                        num_actions_against, goals_added_against, date_captured,
                        season_year, matchday, minutes_played
                    ) VALUES (
                        :team_id, :action_type, :num_actions_for, :goals_added_for,
                        :num_actions_against, :goals_added_against, NOW(),
                        :season_year, :matchday, :minutes_played
                    )
                    ON CONFLICT (team_id, season_year, action_type) DO UPDATE SET
                        num_actions_for = EXCLUDED.num_actions_for,
                        goals_added_for = EXCLUDED.goals_added_for,
                        num_actions_against = EXCLUDED.num_actions_against,
                        goals_added_against = EXCLUDED.goals_added_against,
                        minutes_played = EXCLUDED.minutes_played,
                        date_captured = NOW()
                    RETURNING *
                """
                
                stored_record = await self.db_manager.db.fetch_one(insert_query, values=insert_values)
                stored_records.append(dict(stored_record))
                
                logger.debug(f"Stored GA: {team_id} {season_year} {action_type}")
            
            return stored_records
            
        except Exception as e:
            logger.error(f"Error fetching Goals Added data for team {team_id}, season {season_year}: {e}")
            return None


async def main():
    """Main execution function"""
    print("Historical Data Loader for MLS Next Pro")
    print("=" * 50)
    
    # Ask user for run type
    while True:
        run_type = input("Run type? (dry-run/production): ").lower().strip()
        if run_type in ['dry-run', 'production']:
            break
        print("Please enter 'dry-run' or 'production'")
    
    is_dry_run = (run_type == 'dry-run')
    
    if not is_dry_run:
        print("\n WARNING: This will modify your database!")
        print("- Delete all 2025 team_xg_history records")
        print("- Load fresh data for 2022-2025 seasons")
        confirm = input("Continue? (yes/no): ").lower().strip()
        if confirm != 'yes':
            print("Cancelled.")
            return 0
    
    logger.info(f"=== Historical Data Loading Script Started ({run_type.upper()}) ===")
    
    try:
        # Connect to database
        logger.info("Connecting to database...")
        await database.connect()
        
        # Initialize database manager
        db_manager = DatabaseManager(database)
        await db_manager.initialize()
        
        # Create loader and run
        loader = HistoricalDataLoader(db_manager)
        await loader.load_all_historical_data(dry_run=is_dry_run)
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}", exc_info=True)
        return 1
    
    finally:
        # Disconnect from database
        logger.info("Disconnecting from database...")
        await database.disconnect()
    
    logger.info(f"=== Historical Data Loading Script Completed ({run_type.upper()}) ===")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)