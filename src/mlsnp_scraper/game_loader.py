import asyncio
import json
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/fixture_loader.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class FixtureLoader:
    """
    Handles loading scraped fixtures into the database.
    
    This class:
    1. Reads fixture data from JSON files
    2. Validates the data
    3. Ensures teams exist in the database
    4. Loads fixtures as future games
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the fixture loader.
        
        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.db_manager = db_manager
        self.stats = {
            'fixtures_read': 0,
            'fixtures_loaded': 0,
            'fixtures_skipped': 0,
            'teams_created': 0,
            'errors': []
        }
    
    async def load_fixtures_from_file(self, filepath: str = "output/fox_sports_mlsnp_fixtures.json") -> bool:
        """
        Load fixtures from a JSON file into the database.
        
        Args:
            filepath: Path to the JSON file containing fixtures
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read the JSON file
            logger.info(f"Reading fixtures from: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                fixtures = json.load(f)
            
            self.stats['fixtures_read'] = len(fixtures)
            logger.info(f"Found {len(fixtures)} fixtures to process")
            
            if not fixtures:
                logger.warning("No fixtures found in file")
                return False
            
            # Process each fixture
            for idx, fixture in enumerate(fixtures):
                try:
                    await self._process_fixture(fixture, idx + 1)
                except Exception as e:
                    error_msg = f"Error processing fixture {idx + 1}: {e}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    continue
            
            # Log summary
            self._print_summary()
            
            return self.stats['fixtures_loaded'] > 0
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            logger.error("Make sure to run the Fox scraper first!")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading fixtures: {e}")
            return False
    
    async def _process_fixture(self, fixture: Dict, fixture_num: int):
        """
        Process a single fixture and store it in the database.
        
        Args:
            fixture: Fixture data dictionary
            fixture_num: Fixture number for logging
        """
        try:
            # Validate required fields
            required_fields = ['date', 'home_team_id', 'away_team_id', 'home_team', 'away_team']
            missing_fields = [field for field in required_fields if field not in fixture]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Log what we're processing
            logger.info(f"Processing fixture {fixture_num}: {fixture['home_team']} vs {fixture['away_team']} on {fixture['date']}")
            
            # DEBUG: Print the entire fixture data to see what we're working with
            logger.debug(f"Full fixture data: {fixture}")
            
            # Ensure both teams exist in the database
            await self._ensure_team_exists(fixture['home_team_id'], fixture['home_team'])
            await self._ensure_team_exists(fixture['away_team_id'], fixture['away_team'])
            
            # Parse the fixture date
            try:
                fixture_date = datetime.strptime(fixture['date'], '%Y-%m-%d')
                logger.debug(f"Parsed date: {fixture_date}, type: {type(fixture_date)}")
            except ValueError:
                raise ValueError(f"Invalid date format: {fixture['date']}")
            
            # Generate a unique game_id
            game_id = f"{fixture['home_team_id']}_{fixture['away_team_id']}_{fixture['date']}"
            logger.debug(f"Generated game_id: {game_id}")
            
            # Check if this game already exists
            existing_game = await self.db_manager.db.fetch_one(
                "SELECT game_id FROM games WHERE game_id = :game_id",
                values={"game_id": game_id}
            )
            
            if existing_game:
                logger.info(f"  → Game already exists in database, skipping")
                self.stats['fixtures_skipped'] += 1
                return
            
            # Parse scores from the "time" field (which actually contains scores)
            home_score, away_score, home_penalties, away_penalties = self._parse_score_data(fixture.get('score_or_status', ''))
            
            # Determine if game is completed and status
            status, is_completed = self._determine_game_status(fixture.get('score_or_status', ''))
            
            # Determine if it went to shootout
            went_to_shootout = (is_completed and home_score is not None and away_score is not None and 
                            home_score == away_score and (home_penalties is not None or away_penalties is not None))
            
            # Prepare game data for insertion - ensuring all types are correct
            game_data = {
                "game_id": game_id,
                "date": fixture_date,
                "status": status,
                "home_team_id": fixture['home_team_id'],
                "away_team_id": fixture['away_team_id'],
                "matchday": int(0),  # Explicitly convert to int
                "attendance": int(0),  # Explicitly convert to int
                "is_completed": is_completed,
                "went_to_shootout": went_to_shootout,
                "season_year": int(fixture_date.year),  # Explicitly convert to int
                "expanded_minutes": int(90),  # Explicitly convert to int
                "home_score": home_score,  # Can be None or int
                "away_score": away_score,  # Can be None or int
                "home_penalties": home_penalties,  # Can be None or int
                "away_penalties": away_penalties  # Can be None or int
            }
            
            # DEBUG: Print game_data with types to identify any remaining issues
            logger.debug("Game data with types:")
            for key, value in game_data.items():
                logger.debug(f"  {key}: {value} (type: {type(value)})")
            
            # Insert the game using the existing store_game method
            logger.debug("About to call store_game...")
            await self.db_manager.store_game(game_data)
            
            logger.info(f"  → Successfully loaded fixture into database")
            self.stats['fixtures_loaded'] += 1
            
        except Exception as e:
            logger.error(f"Error in _process_fixture: {e}", exc_info=True)
            self.stats['errors'].append(f"Fixture {fixture_num}: {str(e)}")
            # Don't re-raise to continue processing other fixtures

    def _parse_score_data(self, time_field: str) -> tuple:
        """
        Parse the "time" field which actually contains score data.
        
        Args:
            time_field: The time field from scraped data (e.g., "4-2", "POSTPONED-", "TBD")
            
        Returns:
            Tuple of (home_score, away_score, home_penalties, away_penalties)
            All values are either int or None
        """
        if not time_field or time_field in ['TBD', 'POSTPONED-', '']:
            return None, None, None, None
        
        # Check for score pattern like "4-2", "0-1", etc.
        score_pattern = r'^(\d+)-(\d+)$'
        match = re.match(score_pattern, time_field.strip())
        
        if match:
            try:
                home_score = int(match.group(1))
                away_score = int(match.group(2))
                logger.debug(f"Parsed scores: {home_score}-{away_score}")
                return home_score, away_score, None, None
            except ValueError:
                logger.warning(f"Could not convert scores to integers: {time_field}")
                return None, None, None, None
        
        # If no score pattern matches, treat as future/incomplete game
        logger.debug(f"No score pattern found in: {time_field}")
        return None, None, None, None

    def _determine_game_status(self, time_field: str) -> tuple:
        """
        Determine game status and completion based on the time field.
        
        Args:
            time_field: The time field from scraped data
            
        Returns:
            Tuple of (status, is_completed)
        """
        if not time_field:
            return "scheduled", False
        
        time_field = time_field.strip().upper()
        
        if 'POSTPONED' in time_field:
            return "postponed", False
        elif time_field in ['TBD', '']:
            return "scheduled", False
        elif re.match(r'^\d+-\d+$', time_field):
            # Has a score, so it's completed
            return "final", True
        elif ':' in time_field and ('PM' in time_field or 'AM' in time_field):
            # Looks like a future game time
            return "scheduled", False
        else:
            # Default case - might be a future game with unusual time format
            return "scheduled", False
    
    async def _ensure_team_exists(self, team_id: str, team_name: str):
        """
        Ensure a team exists in the database, creating it if necessary.
        
        Args:
            team_id: ASA team ID
            team_name: Team name
        """
        # Check if team exists
        existing_team = await self.db_manager.db.fetch_one(
            "SELECT team_id FROM team WHERE team_id = :team_id",
            values={"team_id": team_id}
        )
        
        if not existing_team:
            logger.info(f"  → Creating team: {team_name} ({team_id})")
            
            # Determine conference based on team name patterns
            eastern_teams = ['Atlanta', 'Carolina', 'Charlotte', 'Chattanooga', 'Chicago', 
                           'Cincinnati', 'Columbus', 'Crown', 'Huntsville', 'Inter Miami', 
                           'New England', 'New York', 'Orlando', 'Philadelphia', 'Rochester', 
                           'Toronto']
            
            conference_id = None
            for eastern_pattern in eastern_teams:
                if eastern_pattern.lower() in team_name.lower():
                    conference_id = 1  # Eastern Conference
                    break
            
            if conference_id is None:
                conference_id = 2  # Default to Western Conference
            
            # Create the team
            await self.db_manager.get_or_create_team(
                team_id=team_id,
                team_name=team_name,
                conference_id=conference_id
            )
            
            self.stats['teams_created'] += 1
    
    def _print_summary(self):
        """Print a summary of the loading operation."""
        print("\n" + "="*70)
        print("FIXTURE LOADING SUMMARY")
        print("="*70)
        print(f"\nFixtures read from file: {self.stats['fixtures_read']}")
        print(f"Fixtures loaded to database: {self.stats['fixtures_loaded']}")
        print(f"Fixtures skipped (already exist): {self.stats['fixtures_skipped']}")
        print(f"New teams created: {self.stats['teams_created']}")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more")
        
        print("\n" + "="*70)


async def main():
    """
    Main execution function.
    """
    logger.info("Starting fixture loader...")
    
    try:
        # Connect to database
        logger.info("Connecting to database...")
        await connect()
        
        # Initialize database manager
        db_manager = DatabaseManager(database)
        await db_manager.initialize()
        
        # Ensure conferences exist
        await db_manager.ensure_conferences_exist()
        
        # Create and run the fixture loader
        loader = FixtureLoader(db_manager)
        
        # Load fixtures from the default file
        success = await loader.load_fixtures_from_file()
        
        if success:
            logger.info("Fixture loading completed successfully!")
            return 0
        else:
            logger.error("Fixture loading failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error in fixture loader: {e}", exc_info=True)
        return 1
    finally:
        # Always disconnect from database
        logger.info("Disconnecting from database...")
        await disconnect()


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
