"""
Database Manager for MLS Next Pro Predictor
This module handles all database operations, providing a unified interface
for storing and retrieving game data, team statistics, and simulation results.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from databases import Database
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Central database manager that handles all database operations.
    
    This class follows a "database-first" approach:
    1. Always check the database for existing data.
    2. Only fetch from ASA API if data is missing or outdated.
    3. Cache all fetched data in the database for future use.
    """
    
    def __init__(self, database: Database):
        """
        Initialize the database manager.
        
        Args:
            database: The database connection object from the 'databases' library.
        """
        self.db = database
        self._asa_client = None  # Lazy loaded only when needed

    async def initialize(self):
        """
        Initialize the database manager and ensure all foundational data,
        like conferences, exists in the database. This should be called
        after the database connection is established.
        """
        logger.info("Initializing database manager...")
        await self.ensure_conferences_exist()
        logger.info("Database manager initialization complete.")
        
    @property
    def asa_client(self):
        """Lazy load ASA client only when needed"""
        if self._asa_client is None:
            from itscalledsoccer.client import AmericanSoccerAnalysis
            self._asa_client = AmericanSoccerAnalysis()
            logger.info("Initialized ASA client for data fetching.")
        return self._asa_client

    # ==================== Team & Conference Management ====================
    
    async def get_or_create_team(self, team_id: str, team_name: str, 
                                 team_abbv: str = None, conference_id: int = None) -> Dict:
        """
        Get a team from the database or create it if it doesn't exist.
        
        This method ensures consistent team data across the system.
        """
        query = "SELECT * FROM team WHERE team_id = :team_id"
        team = await self.db.fetch_one(query, values={"team_id": team_id})
        
        if team:
            return dict(team)
        
        logger.info(f"Creating new team: {team_name} ({team_id})")
        
        insert_query = """
            INSERT INTO team (team_id, team_name, team_abbv, team_short_name, is_active, created_at)
            VALUES (:team_id, :team_name, :team_abbv, :short_name, true, NOW())
            RETURNING *
        """
        
        if not team_abbv:
            team_abbv = ''.join(word[0].upper() for word in team_name.split()[:3])
        
        short_name = team_name.split()[0]
        
        values = {
            "team_id": team_id,
            "team_name": team_name,
            "team_abbv": team_abbv,
            "short_name": short_name
        }
        
        team = await self.db.fetch_one(insert_query, values=values)
        
        if conference_id:
            await self.create_team_affiliation(team_id, conference_id)
        
        return dict(team)
    
    async def create_team_affiliation(self, team_id: str, conference_id: int, 
                                      league_id: int = 1, division_id: int = None):
        """
        Create or update a team's affiliation with a conference and league.
        This is crucial for tracking which teams belong to which conference over time.
        """
        check_query = """
            SELECT * FROM team_affiliations 
            WHERE team_id = :team_id AND conference_id = :conference_id AND is_current = true
        """
        existing = await self.db.fetch_one(check_query, values={"team_id": team_id, "conference_id": conference_id})
        
        if existing:
            return dict(existing)
            
        update_query = """
            UPDATE team_affiliations SET is_current = false, end_date = CURRENT_DATE
            WHERE team_id = :team_id AND is_current = true
        """
        await self.db.execute(update_query, values={"team_id": team_id})
        
        insert_query = """
            INSERT INTO team_affiliations 
            (team_id, league_id, conference_id, division_id, start_date, is_current, created_at)
            VALUES (:team_id, :league_id, :conference_id, :division_id, CURRENT_DATE, true, NOW())
            RETURNING *
        """
        values = {
            "team_id": team_id,
            "league_id": league_id,
            "conference_id": conference_id,
            "division_id": division_id
        }
        
        affiliation = await self.db.fetch_one(insert_query, values=values)
        return dict(affiliation)

    async def get_conference_teams(self, conference_id: int, season_year: int) -> Dict[str, str]:
        """
        Get all teams for a given conference ID and season year.
        Returns a dictionary mapping team_id to team_name.
        
        Args:
            conference_id: 1 for Eastern, 2 for Western
            season_year: The season year to get teams for
        """
        from datetime import datetime, date
        
        current_year = datetime.now().year
        
        if season_year == current_year:
            # For current year, use the is_current flag for efficiency
            query = """
                SELECT t.team_id, t.team_name
                FROM team t
                JOIN team_affiliations ta ON t.team_id = ta.team_id
                WHERE ta.conference_id = :conference_id
                AND ta.is_current = true
                AND t.is_active = true
                ORDER BY t.team_name
            """
            values = {"conference_id": conference_id}
        else:
            # For historical years, check date ranges
            # Assume season runs from Jan 1 to Dec 31 of the given year
            season_start = date(season_year, 1, 1)
            season_end = date(season_year, 12, 31)
            
            query = """
                SELECT t.team_id, t.team_name
                FROM team t
                JOIN team_affiliations ta ON t.team_id = ta.team_id
                WHERE ta.conference_id = :conference_id
                AND t.is_active = true
                AND (
                    -- Team affiliation overlaps with the season
                    (ta.start_date IS NULL OR ta.start_date <= :season_end) AND
                    (ta.end_date IS NULL OR ta.end_date >= :season_start)
                )
                ORDER BY t.team_name
            """
            values = {
                "conference_id": conference_id,
                "season_start": season_start,
                "season_end": season_end
            }
    
        teams = await self.db.fetch_all(query, values=values)
        return {team['team_id']: team['team_name'] for team in teams}

    async def ensure_conferences_exist(self):
        """
        Ensures the Eastern and Western conferences exist in the database.
        This prevents errors in downstream processing that rely on conference data.
        """
        conferences = {1: 'Eastern Conference', 2: 'Western Conference'}
        for conf_id, conf_name in conferences.items():
            query = "SELECT * FROM conference WHERE conf_id = :conf_id"
            conference = await self.db.fetch_one(query, values={"conf_id": conf_id})
            if not conference:
                insert_query = """
                    INSERT INTO conference (conf_id, conf_name, league_id, created_at)
                    VALUES (:conf_id, :conf_name, 1, NOW())
                """
                await self.db.execute(insert_query, values={"conf_id": conf_id, "conf_name": conf_name})
                logger.info(f"Inserted {conf_name} into database.")

    # ==================== Game Data Management ====================
    
    async def get_games_for_season(self, season_year: int, conference: str = None,
                                   include_incomplete: bool = False) -> List[Dict]:
        """
        Retrieve all games for a season from the database, with optional filters.
        """
        query = """
            SELECT g.*, 
                   ht.team_name as home_team_name, at.team_name as away_team_name,
                   hta.conference_id as home_conference_id, ata.conference_id as away_conference_id
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            LEFT JOIN team_affiliations hta ON ht.team_id = hta.team_id AND hta.is_current = true
            LEFT JOIN team_affiliations ata ON at.team_id = ata.team_id AND ata.is_current = true
            WHERE g.season_year = :season_year
        """
        values = {"season_year": season_year}
        
        if conference:
            conf_id = 1 if conference.lower() == 'eastern' else 2
            # Correctly fetches games where at least one team is in the conference
            query += " AND (hta.conference_id = :conf_id OR ata.conference_id = :conf_id)"
            values["conf_id"] = conf_id
        
        if not include_incomplete:
            query += " AND g.is_completed = true"
        
        query += " ORDER BY g.date, g.game_id"
        
        games = await self.db.fetch_all(query, values=values)
        return [dict(game) for game in games]

    async def get_or_fetch_game(self, game_id: str) -> Optional[Dict]:
        """
        Get a single game from the database or fetch it from the API if missing.
        """
        query = "SELECT * FROM games WHERE game_id = :game_id"
        game = await self.db.fetch_one(query, values={"game_id": game_id})
        
        if game:
            return dict(game)
        
        logger.info(f"Game {game_id} not in database, fetching from ASA API...")
        try:
            games_data = self.asa_client.get_games(game_id=[game_id])
            if not games_data:
                logger.warning(f"Game {game_id} not found in ASA API.")
                return None
            
            # Store and return the new game data
            await self.store_game(games_data[0])
            return await self.db.fetch_one(query, values={"game_id": game_id})
        except Exception as e:
            logger.error(f"Error fetching game {game_id} from ASA: {e}")
            return None

    async def store_game(self, game_data: Dict) -> Dict:
        """
        Store a game in the database, handling data transformation and updates.
        Uses ON CONFLICT to efficiently update existing game records.
        Now also updates matchdays for existing games when ASA data is provided.
        """
        try:
            # First, try to update matchday for any existing games (if this is ASA data)
            if game_data.get('matchday') is not None:
                await self.update_matchday_from_asa_data(game_data)

            game_id = game_data.get('game_id')
            if not game_id:
                raise ValueError("game_id is required")
            
            # Handle date parsing more robustly
            date_str = game_data.get('date_time_utc', game_data.get('date'))
            game_date = None
            
            if date_str:
                if isinstance(date_str, datetime):
                    game_date = date_str
                elif isinstance(date_str, str):
                    # Robust date parsing - handle ASA's UTC format
                    date_str_clean = date_str.replace(' UTC', '+00:00').replace('Z', '+00:00')
                    for fmt in ('%Y-%m-%d %H:%M:%S%z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
                        try:
                            game_date = datetime.strptime(date_str_clean, fmt)
                            break
                        except ValueError:
                            continue
                            
            if not game_date:
                raise ValueError(f"Could not parse date: {date_str}")
            
            # Handle status and completion
            status = game_data.get('status', 'scheduled').lower()
            is_completed = game_data.get('is_completed', False)
            if isinstance(is_completed, str):
                is_completed = is_completed.lower() in ['true', '1', 'yes']
            
            # Handle scores - ensure they're integers or None
            def safe_int(value):
                if value is None or value == '':
                    return None
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None
            
            home_score = safe_int(game_data.get('home_score'))
            away_score = safe_int(game_data.get('away_score'))
            home_penalties = safe_int(game_data.get('home_penalties', 0))
            away_penalties = safe_int(game_data.get('away_penalties', 0))
            
            # Handle shootout logic
            went_to_shootout = game_data.get('went_to_shootout', False)
            if isinstance(went_to_shootout, str):
                went_to_shootout = went_to_shootout.lower() in ['true', '1', 'yes']
            
            # Handle integer fields safely
            matchday = safe_int(game_data.get('matchday', 0)) or 0
            attendance = safe_int(game_data.get('attendance', 0)) or 0
            expanded_minutes = safe_int(game_data.get('expanded_minutes', 90)) or 90
            
            # Handle season year
            season_year = game_data.get('season_year')
            if season_year is None:
                season_year = game_date.year
            season_year = safe_int(season_year) or game_date.year
            
            values = {
                "game_id": game_id,
                "date": game_date,
                "status": status,
                "home_team_id": game_data.get('home_team_id'),
                "away_team_id": game_data.get('away_team_id'),
                "matchday": matchday,
                "attendance": attendance,
                "is_completed": is_completed,
                "went_to_shootout": went_to_shootout,
                "season_year": season_year,
                "expanded_minutes": expanded_minutes,
                "home_score": home_score if is_completed else None,
                "away_score": away_score if is_completed else None,
                "home_penalties": home_penalties if went_to_shootout else None,
                "away_penalties": away_penalties if went_to_shootout else None
            }
            
            # Debug logging
            logger.debug(f"Storing game with values: {values}")
            
            query = """
                INSERT INTO games (
                    game_id, date, status, home_team_id, away_team_id, matchday,
                    attendance, is_completed, went_to_shootout, season_year, 
                    expanded_minutes, home_score, away_score, home_penalties, 
                    away_penalties, created_at, updated_at)
                VALUES (
                    :game_id, :date, :status, :home_team_id, :away_team_id, :matchday,
                    :attendance, :is_completed, :went_to_shootout, :season_year,
                    :expanded_minutes, :home_score, :away_score, :home_penalties,
                    :away_penalties, NOW(), NOW())
                ON CONFLICT (game_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    is_completed = EXCLUDED.is_completed,
                    went_to_shootout = EXCLUDED.went_to_shootout,
                    matchday = EXCLUDED.matchday,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    home_penalties = EXCLUDED.home_penalties,
                    away_penalties = EXCLUDED.away_penalties,
                    attendance = EXCLUDED.attendance,
                    expanded_minutes = EXCLUDED.expanded_minutes,
                    updated_at = NOW()
                RETURNING *
            """
            
            result = await self.db.fetch_one(query, values=values)
            logger.debug(f"Successfully stored game: {game_id}")
            return dict(result)
        
        except Exception as e:
            logger.error(f"Error in store_game for game_id {game_data.get('game_id', 'unknown')}: {e}", exc_info=True)
            raise

    async def update_incomplete_games(self, season_year: int):
        """
        Finds and updates all games that should have been played but are still
        marked as incomplete in the database.
        """
        # Create timezone-naive datetime to match database storage
        cutoff_date = datetime.now()
        
        query = """
            SELECT game_id FROM games 
            WHERE season_year = :season_year AND is_completed = false AND date < :cutoff_date
        """
        incomplete_games = await self.db.fetch_all(query, values={"season_year": season_year, "cutoff_date": cutoff_date})
        
        if not incomplete_games:
            logger.info(f"No incomplete games found to update for {season_year}.")
            return

        logger.info(f"Found {len(incomplete_games)} incomplete games to update for {season_year}.")
        for game in incomplete_games:
            await self.get_or_fetch_game(game['game_id'])
            await asyncio.sleep(0.1)  # Be nice to the API

    # ==================== Team Statistics Management ====================
    
    async def get_or_fetch_team_xg(self, team_id: str, season_year: int) -> Optional[Dict]:
        """
        Get team expected goals (xG) stats from the database or fetch from the API.
        Data is considered stale after 1 day and will be re-fetched.
        
        UPDATED: Removed x_shots_for/x_shots_against fields (not available in ASA MLSNP data)
        """
        query = """
            SELECT * FROM team_xg_history 
            WHERE team_id = :team_id AND season_year = :season_year
        """
        xg_data = await self.db.fetch_one(query, values={"team_id": team_id, "season_year": season_year})

        if xg_data and (datetime.now(timezone.utc) - xg_data['date_captured']) < timedelta(days=1):
            logger.debug(f"Returning fresh xG data for team {team_id} from database.")
            return dict(xg_data)

        logger.info(f"Fetching new xG data for team {team_id}, season {season_year} from ASA.")
        try:
            stats_df = self.asa_client.get_team_xgoals(
                leagues=['mlsnp'],
                team_ids=[team_id], 
                season_name=[str(season_year)]
            )
            
            if stats_df.empty:
                logger.warning(f"No xG data returned from ASA API for team {team_id}, season {season_year}.")
                return None

            stat = stats_df.iloc[0].to_dict()
            
            insert_values = {
                "team_id": team_id,
                "games_played": stat.get('count_games', 0),
                "shots_for": stat.get('shots_for', 0),
                "shots_against": stat.get('shots_against', 0),
                "x_goals_for": stat.get('xgoals_for', 0.0),
                "x_goals_against": stat.get('xgoals_against', 0.0),
                "season_year": season_year
            }
            
            insert_query = """
                INSERT INTO team_xg_history (
                    team_id, season_year, games_played, shots_for, shots_against, 
                    x_goals_for, x_goals_against, date_captured
                ) VALUES (
                    :team_id, :season_year, :games_played, :shots_for, :shots_against,
                    :x_goals_for, :x_goals_against, NOW()
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
            
            stored_data = await self.db.fetch_one(insert_query, values=insert_values)
            logger.info(f"Successfully stored xG data for team {team_id}, season {season_year}.")
            return dict(stored_data)
        except Exception as e:
            logger.error(f"Error fetching xG data for team {team_id}: {e}", exc_info=True)
            return None

    # ==================== Standings Management ====================
    
    async def calculate_and_store_standings(self, season_year: int, conference: str) -> List[Dict]:
        """
        Calculates current standings from completed games and stores them.
        Correctly implements the MLS Next Pro point system, including detailed goal stats.
        """
        conf_id = 1 if conference.lower() == 'eastern' else 2
        conf_teams = await self.get_conference_teams(conf_id, season_year)
        games = await self.get_games_for_season(season_year, conference, include_incomplete=False)

        standings = defaultdict(lambda: {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,  # Games that went to shootout
            "so_wins": 0,  # Shootout wins
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
            "home_goals_for": 0,
            "home_goals_against": 0,
            "away_goals_for": 0,
            "away_goals_against": 0
        })

        for team_id in conf_teams:
            standings[team_id] # Initialize every team in the conference

        for game in games:
            home_id, away_id = game['home_team_id'], game['away_team_id']
            home_score, away_score = game['home_score'], game['away_score']

            if home_id not in conf_teams or away_id not in conf_teams:
                continue # Skip games not between two conference teams

            # Increment games played for both teams
            standings[home_id]["games_played"] += 1
            standings[away_id]["games_played"] += 1

            # Update overall goals for and against
            standings[home_id]["goals_for"] += home_score
            standings[home_id]["goals_against"] += away_score
            standings[away_id]["goals_for"] += away_score
            standings[away_id]["goals_against"] += home_score

            # Update overall goal difference
            standings[home_id]["goal_difference"] += home_score - away_score
            standings[away_id]["goal_difference"] += away_score - home_score

            # Update home/away specific goals
            standings[home_id]["home_goals_for"] += home_score
            standings[home_id]["home_goals_against"] += away_score
            standings[away_id]["away_goals_for"] += away_score
            standings[away_id]["away_goals_against"] += home_score

            # Award points and update specific win/loss/draw stats
            if game['went_to_shootout']:
                standings[home_id]["draws"] += 1
                standings[away_id]["draws"] += 1
                if game['home_penalties'] > game['away_penalties']:
                    standings[home_id]["points"] += 2
                    standings[home_id]["so_wins"] += 1
                    standings[away_id]["points"] += 1
                    standings[away_id]["losses"] += 1 # A shootout loss is still a loss in this context
                else:
                    standings[away_id]["points"] += 2
                    standings[away_id]["so_wins"] += 1
                    standings[home_id]["points"] += 1
                    standings[home_id]["losses"] += 1 # A shootout loss is still a loss in this context
            else: # Regulation result
                if home_score > away_score:
                    standings[home_id]["points"] += 3
                    standings[home_id]["wins"] += 1
                    standings[away_id]["losses"] += 1
                elif away_score > home_score:
                    standings[away_id]["points"] += 3
                    standings[away_id]["wins"] += 1
                    standings[home_id]["losses"] += 1
                else:
                    # This case should ideally not happen for MLSNP games that go to shootout.
                    # If it does, it implies a regulation draw that didn't go to shootout,
                    # which would be 1 point for each, but MLSNP always goes to shootout.
                    # Keeping it for robustness, assuming it defaults to 1 point each like a regular draw.
                    standings[home_id]["draws"] += 1
                    standings[away_id]["draws"] += 1
                    standings[home_id]["points"] += 1
                    standings[away_id]["points"] += 1

        standings_list = [{"team_id": tid, **stats} for tid, stats in standings.items()]

        # Sort by MLS Next Pro tiebreakers
        # The tiebreakers are: Points, Wins, Goal Difference, Goals For, Shootout Wins
        # Assuming lower rank is better, so 'reverse=True' for sorting largest values first for points, wins, GD, GF, SO Wins.
        standings_list.sort(key=lambda x: (
            -x["points"],
            -x["wins"],
            -x["goal_difference"],
            -x["goals_for"],
            -x["so_wins"]
        )) # reverse=False by default, so sorting descending already

        return standings_list
    
    async def store_standings_record(self, team_id: str, season_year: int, matchday: int, stats: Dict) -> Dict:
        """
        Stores or updates a single team's standings record for a given season and matchday.
        Uses an UPSERT statement to handle conflicts.
        """
        insert_query = """
            INSERT INTO standings_history (
                team_id, season_year, matchday, games_played, wins, losses, 
                draws, so_wins, goals_for, goals_against, goal_difference, 
                points, home_goals_for, home_goals_against, away_goals_for, 
                away_goals_against, date_captured
            ) VALUES (
                :team_id, :season_year, :matchday, :games_played, :wins, :losses,
                :draws, :so_wins, :goals_for, :goals_against, :goal_difference,
                :points, :home_goals_for, :home_goals_against, :away_goals_for,
                :away_goals_against, NOW()
            )
            ON CONFLICT (team_id, season_year, matchday) DO UPDATE SET
                games_played = EXCLUDED.games_played,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                draws = EXCLUDED.draws,
                so_wins = EXCLUDED.so_wins,
                goals_for = EXCLUDED.goals_for,
                goals_against = EXCLUDED.goals_against,
                goal_difference = EXCLUDED.goal_difference,
                points = EXCLUDED.points,
                home_goals_for = EXCLUDED.home_goals_for,
                home_goals_against = EXCLUDED.home_goals_against,
                away_goals_for = EXCLUDED.away_goals_for,
                away_goals_against = EXCLUDED.away_goals_against,
                date_captured = NOW()
            RETURNING *
        """

        values = {
            "team_id": team_id,
            "season_year": season_year,
            "matchday": matchday,
            **stats # Unpack all the stats from the provided dictionary
        }
        
        try:
            result = await self.db.fetch_one(insert_query, values=values)
            logger.debug(f"Stored standings for team {team_id} (Season: {season_year}, Matchday: {matchday})")
            return dict(result)
        except Exception as e:
            logger.error(f"Error storing standings for team {team_id}: {e}", exc_info=True)
            raise

    
    # ==================== Fixture Management ====================
    
    async def store_fixtures(self, fixtures: List[Dict]):
        """
        Store scraped fixtures in the database as future games.
        """
        for fixture in fixtures:
            # Parse date
            fixture_date = None
            if fixture.get('date'):
                try:
                    fixture_date = datetime.strptime(fixture['date'], '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Could not parse fixture date: {fixture['date']}")
                    continue
            
            # Generate a unique game_id for fixtures
            game_id = f"{fixture['home_team_id']}_{fixture['away_team_id']}_{fixture['date']}"
            
            # Ensure both teams exist in the database
            if fixture.get('home_team_name'):
                await self.get_or_create_team(
                    fixture['home_team_id'], 
                    fixture['home_team_name']
                )
            if fixture.get('away_team_name'):
                await self.get_or_create_team(
                    fixture['away_team_id'], 
                    fixture['away_team_name']
                )
            
            # Create game record
            values = {
                "game_id": game_id,
                "date": fixture_date,
                "status": "scheduled",
                "home_team_id": fixture['home_team_id'],
                "away_team_id": fixture['away_team_id'],
                "matchday": fixture.get('matchday', 0),
                "attendance": 0,
                "is_completed": False,
                "went_to_shootout": False,
                "season_year": fixture_date.year if fixture_date else 2025,
                "expanded_minutes": 90,
                "home_score": None,
                "away_score": None,
                "home_penalties": None,
                "away_penalties": None
            }
            
            query = """
                INSERT INTO games (
                    game_id, date, status, home_team_id, away_team_id, matchday,
                    attendance, is_completed, went_to_shootout, season_year,
                     expanded_minutes, home_score, away_score, home_penalties,
                     away_penalties, created_at, updated_at
                ) VALUES (
                    :game_id, :date, :status, :home_team_id, :away_team_id, :matchday,
                    :attendance, :is_completed, :went_to_shootout, :season_year,
                    :expanded_minutes, :home_score, :away_score, :home_penalties,
                    :away_penalties, NOW(), NOW()
                )
                ON CONFLICT (game_id) DO NOTHING
            """
            
        try:
            await self.db.execute(query, values=values)
            logger.debug(f"Stored fixture: {game_id}")
        except Exception as e:
            logger.error(f"Error storing fixture {game_id}: {e}")

    # ==================== Simulation & Historical Data ====================

    async def get_data_for_simulation(self, conference: str, season_year: int) -> Dict[str, Any]:
        """
        Gathers all necessary data for a regular season simulation run.
        """
        logger.info(f"Gathering simulation data for {conference} conference, {season_year} season...")
        
        conf_id = 1 if conference.lower() == 'eastern' else 2
        conference_teams = await self.get_conference_teams(conf_id, season_year)
        team_ids = list(conference_teams.keys())
        
        await self.update_incomplete_games(season_year)
        
        all_games = await self.get_games_for_season(season_year, conference, include_incomplete=True)
        
        team_performance = {}
        for team_id in team_ids:
            xg_data = await self.get_or_fetch_team_xg(team_id, season_year)
            if xg_data and xg_data.get('games_played', 0) > 0:
                team_performance[team_id] = xg_data
            else: # Fallback for teams with no xG data
                team_performance[team_id] = {"team_id": team_id, "x_goals_for": 1.0, "x_goals_against": 1.0, "games_played": 0}

        logger.info(f"Successfully gathered data for {len(conference_teams)} teams.")
        return {
            "conference_teams": conference_teams,
            "games_data": all_games,
            "team_performance": team_performance
        }

    async def load_historical_season(self, season_year: int):
        """
        Loads all games and statistics for a historical season from the ASA API
        into the local database.
        """
        logger.info(f"Loading historical data for {season_year} season...")
        try:
            games = self.asa_client.get_games(leagues=['mlsnp'], season_name=[str(season_year)])
            logger.info(f"Found {len(games)} games for {season_year} season.")
            
            for game in games:
                await self.store_game(game)
                if len(games) > 100:
                    await asyncio.sleep(0.01)

            team_ids = await self.db.fetch_all("SELECT DISTINCT team_id FROM team")
            for team in team_ids:
                await self.get_or_fetch_team_xg(team['team_id'], season_year)
                await asyncio.sleep(0.1)
            
            logger.info(f"Successfully loaded and processed {season_year} season data.")
        except Exception as e:
            logger.error(f"Error loading historical season {season_year}: {e}", exc_info=True)
            raise

    async def update_matchday_from_asa_data(self, asa_game_data: Dict) -> bool:
        """
        Updates the matchday for existing games in the database using ASA game data.
        Matches games by home_team_id, away_team_id, and date since game_ids may differ.
        
        Args:
            asa_game_data: Game data from ASA API
            
        Returns:
            bool: True if a matching game was found and updated, False otherwise
        """
        try:
            # Extract the necessary data from ASA game
            home_team_id = asa_game_data.get('home_team_id')
            away_team_id = asa_game_data.get('away_team_id')
            matchday = asa_game_data.get('matchday')
            
            # Parse the date from ASA data
            date_str = asa_game_data.get('date_time_utc')
            if not date_str:
                logger.warning("No date found in ASA game data")
                return False
                
            # Parse the date to match our database format
            try:
                if isinstance(date_str, str):
                    # Handle the UTC format from ASA: "2025-05-24 23:00:00 UTC"
                    date_str = date_str.replace(' UTC', '+00:00')
                    for fmt in ('%Y-%m-%d %H:%M:%S%z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d'):
                        try:
                            game_date = datetime.strptime(date_str, fmt).date()
                            break
                        except ValueError:
                            continue
                    else:
                        logger.warning(f"Could not parse ASA date: {date_str}")
                        return False
                else:
                    game_date = date_str.date() if hasattr(date_str, 'date') else date_str
            except Exception as e:
                logger.warning(f"Error parsing date from ASA data: {e}")
                return False
            
            if not all([home_team_id, away_team_id, game_date]):
                logger.warning("Missing required data for game matching")
                return False
                
            # Find existing games that match by teams and date
            query = """
                SELECT game_id, matchday FROM games 
                WHERE home_team_id = :home_team_id 
                AND away_team_id = :away_team_id 
                AND DATE(date) = :game_date
            """
            
            existing_games = await self.db.fetch_all(
                query, 
                values={
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id, 
                    "game_date": game_date
                }
            )
            
            if not existing_games:
                logger.debug(f"No existing games found for {home_team_id} vs {away_team_id} on {game_date}")
                return False
                
            # Update matchday for all matching games (should typically be just one)
            updated_count = 0
            for existing_game in existing_games:
                # Only update if we have valid matchday data and it's different
                if matchday is not None and matchday != existing_game['matchday']:
                    update_query = """
                        UPDATE games 
                        SET matchday = :matchday, updated_at = NOW()
                        WHERE game_id = :game_id
                    """
                    
                    await self.db.execute(
                        update_query,
                        values={
                            "matchday": int(matchday),
                            "game_id": existing_game['game_id']
                        }
                    )
                    
                    logger.info(f"Updated matchday to {matchday} for game {existing_game['game_id']}")
                    updated_count += 1
                
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"Error updating matchday from ASA data: {e}", exc_info=True)
            return False
        
    async def get_team_goals_added_data(self, team_id: str, season_year: int) -> Dict:
        """
        Get Goals Added data for a team for a specific season.
        Aggregates across all action types to get total g+ for/against.
        """
        query = """
            SELECT 
                team_id,
                season_year,
                SUM(goals_added_for) as total_goals_added_for,
                SUM(goals_added_against) as total_goals_added_against,
                SUM(num_actions_for) as total_actions_for,
                SUM(num_actions_against) as total_actions_against,
                MAX(minutes_played) as minutes_played,
                COUNT(DISTINCT action_type) as action_types_count
            FROM team_ga_history 
            WHERE team_id = :team_id 
            AND season_year = :season_year
            GROUP BY team_id, season_year
        """
        
        result = await self.db.fetch_one(
            query, 
            values={"team_id": team_id, "season_year": season_year}
        )
        
        if result:
            # Estimate games played from minutes (assuming 90 min per game)
            estimated_games = max(1, result['minutes_played'] // 90) if result['minutes_played'] else 1
            
            return {
                'team_id': result['team_id'],
                'season_year': result['season_year'],
                'total_goals_added_for': result['total_goals_added_for'] or 0,
                'total_goals_added_against': result['total_goals_added_against'] or 0,
                'games_played': estimated_games,
                'action_types_available': result['action_types_count']
            }
        
        return None

    # ==================== Simulation Result Storage ====================

    async def store_simulation_run(self, conference: str, n_simulations: int, season_year: int) -> int:
        """
        Stores a new simulation run record and returns the run ID.
        """
        query = """
            INSERT INTO simulation_runs (conference, n_simulations, season_year, run_timestamp)
            VALUES (:conference, :n_simulations, :season_year, NOW())
            RETURNING run_id
        """
        values = {"conference": conference, "n_simulations": n_simulations, "season_year": season_year}
        run_id = await self.db.fetch_val(query, values=values)
        logger.info(f"Stored new simulation run with ID: {run_id}")
        return run_id

    async def store_simulation_results(self, run_id: int, summary_df: Any, final_rank_dist: Dict):
        """
        Stores detailed simulation results, including team summaries and rank distributions.
        """
        team_results_query = """
            INSERT INTO simulation_results (run_id, team_id, avg_points, avg_rank, 
                playoff_pct, final_rank_distribution, created_at)
            VALUES (:run_id, :team_id, :avg_points, :avg_rank, 
                :playoff_pct, :final_rank_distribution, NOW())
        """
        for _, row in summary_df.iterrows():
            team_id = row['_team_id']
            rank_dist = final_rank_dist.get(team_id, [])
            
            result_values = {
                "run_id": run_id,
                "team_id": team_id,
                "avg_points": row['Average Points'],
                "avg_rank": row['Average Final Rank'],
                "playoff_pct": row['Playoff Qualification %'],
                "final_rank_distribution": json.dumps(rank_dist)
            }
            await self.db.execute(team_results_query, values=result_values)
        logger.info(f"Stored {len(summary_df)} team simulation results for run_id: {run_id}")

    async def get_latest_simulation_run(self, conference: str, season_year: int) -> Optional[Dict]:
        """
        Retrieves the latest simulation run details for a given conference and season.
        """
        query = """
            SELECT * FROM simulation_runs
            WHERE conference = :conference AND season_year = :season_year
            ORDER BY run_timestamp DESC LIMIT 1
        """
        run = await self.db.fetch_one(query, values={"conference": conference, "season_year": season_year})
        return dict(run) if run else None

    async def get_simulation_results_for_run(self, run_id: int) -> List[Dict]:
        """
        Retrieves all stored simulation results for a given run ID.
        """
        query = """
            SELECT sr.*, t.team_name
            FROM simulation_results sr
            JOIN team t ON sr.team_id = t.team_id
            WHERE sr.run_id = :run_id
            ORDER BY sr.avg_rank ASC
        """
        results = await self.db.fetch_all(query, values={"run_id": run_id})
        # Parse the JSON string back into a list
        parsed_results = []
        for r in results:
            res_dict = dict(r)
            res_dict['final_rank_distribution'] = json.loads(res_dict['final_rank_distribution'])
            parsed_results.append(res_dict)
        return parsed_results