"""
Database Manager for MLS Next Pro Predictor
This module handles all database operations, providing a unified interface
for storing and retrieving game data, team statistics, and simulation results.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from databases import Database
import asyncio
import numpy as np
from collections import defaultdict
import json
from src.mlsnp_predictor.constants import EASTERN_CONFERENCE_TEAMS, WESTERN_CONFERENCE_TEAMS

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Central database manager that handles all database operations.
    
    This class follows a "database-first" approach:
    1. Always check database for existing data
    2. Only fetch from ASA API if data is missing or outdated
    3. Cache all fetched data for future use
    """
    
    def __init__(self, database: Database):
        """
        Initialize the database manager.
        
        Args:
            database: The database connection object from databases library
        """
        self.db = database
        self._asa_client = None  # Lazy loaded when needed
        
    async def initialize(self):
        """
        Initialize database manager and ensure all tables exist.
        This should be called after database connection is established.
        """
        logger.info("Initializing database manager...")
        # We could add table creation/migration logic here if needed
        
    @property
    def asa_client(self):
        """Lazy load ASA client only when needed"""
        if self._asa_client is None:
            from itscalledsoccer.client import AmericanSoccerAnalysis
            self._asa_client = AmericanSoccerAnalysis()
            logger.info("Initialized ASA client for data fetching")
        return self._asa_client
    
    # ==================== Team Management ====================
    
    async def get_or_create_team(self, team_id: str, team_name: str, 
                                 team_abbv: str = None, conference_id: int = None) -> Dict:
        """
        Get a team from database or create if it doesn't exist.
        
        This method ensures we have consistent team data across the system.
        """
        # First, try to get the team
        query = "SELECT * FROM team WHERE team_id = :team_id"
        team = await self.db.fetch_one(query, values={"team_id": team_id})
        
        if team:
            return dict(team)
        
        # Team doesn't exist, create it
        logger.info(f"Creating new team: {team_name} ({team_id})")
        
        insert_query = """
            INSERT INTO team (team_id, team_name, team_abbv, team_short_name, is_active, created_at)
            VALUES (:team_id, :team_name, :team_abbv, :short_name, true, NOW())
            RETURNING *
        """
        
        # Generate abbreviation if not provided
        if not team_abbv:
            team_abbv = ''.join(word[0].upper() for word in team_name.split()[:3])
        
        # Generate short name (first word of team name)
        short_name = team_name.split()[0]
        
        values = {
            "team_id": team_id,
            "team_name": team_name,
            "team_abbv": team_abbv,
            "short_name": short_name
        }
        
        team = await self.db.fetch_one(insert_query, values=values)
        
        # If conference_id provided, create affiliation
        if conference_id:
            await self.create_team_affiliation(team_id, conference_id)
        
        return dict(team)
    
    async def create_team_affiliation(self, team_id: str, conference_id: int, 
                                    league_id: int = 1, division_id: int = None):
        """
        Create or update team affiliation with conference/league.
        
        This is important for tracking which teams belong to Eastern/Western conferences.
        """
        # Check if affiliation exists
        check_query = """
            SELECT * FROM team_affiliations 
            WHERE team_id = :team_id AND conference_id = :conference_id 
            AND is_current = true
        """
        
        existing = await self.db.fetch_one(check_query, values={
            "team_id": team_id,
            "conference_id": conference_id
        })
        
        if existing:
            return dict(existing)
        
        # Mark any existing affiliations as not current
        update_query = """
            UPDATE team_affiliations 
            SET is_current = false, end_date = CURRENT_DATE
            WHERE team_id = :team_id AND is_current = true
        """
        await self.db.execute(update_query, values={"team_id": team_id})
        
        # Create new affiliation
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
    
    # ==================== Game Data Management ====================
    
    async def get_games_for_season(self, season_year: int, conference: str = None,
                                  include_incomplete: bool = False) -> List[Dict]:
        """
        Retrieve all games for a season from database.
        
        Args:
            season_year: The season year to retrieve
            conference: Optional - 'eastern' or 'western' to filter by conference
            include_incomplete: Whether to include games that haven't been played yet
            
        Returns:
            List of game dictionaries with all relevant information
        """
        # Build the base query
        query = """
            SELECT g.*, 
                   ht.team_name as home_team_name,
                   at.team_name as away_team_name,
                   hta.conference_id as home_conference_id,
                   ata.conference_id as away_conference_id
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            LEFT JOIN team_affiliations hta ON ht.team_id = hta.team_id AND hta.is_current = true
            LEFT JOIN team_affiliations ata ON at.team_id = ata.team_id AND ata.is_current = true
            WHERE g.season_year = :season_year
        """
        
        values = {"season_year": season_year}
        
        # Add conference filter if specified
        if conference:
            conf_id = 1 if conference.lower() == 'eastern' else 2
            query += " AND (hta.conference_id = :conf_id AND ata.conference_id = :conf_id)"
            values["conf_id"] = conf_id
        
        # Add completion filter
        if not include_incomplete:
            query += " AND g.is_completed = true"
        
        query += " ORDER BY g.date, g.game_id"
        
        games = await self.db.fetch_all(query, values=values)
        return [dict(game) for game in games]
    
    async def get_or_fetch_game(self, game_id: str, game_date: datetime = None) -> Optional[Dict]:
        """
        Get a game from database or fetch from ASA API if missing.
        
        This method implements the database-first approach:
        1. Check if game exists in database
        2. If not, fetch from ASA API
        3. Store in database for future use
        """
        # First, try to get from database
        query = "SELECT * FROM games WHERE game_id = :game_id"
        game = await self.db.fetch_one(query, values={"game_id": game_id})
        
        if game:
            return dict(game)
        
        # Game not in database, fetch from ASA
        logger.info(f"Game {game_id} not in database, fetching from ASA API...")
        
        try:
            # Use the ASA client to fetch game data
            games = self.asa_client.get_games(game_id=[game_id])
            
            if not games or len(games) == 0:
                logger.warning(f"Game {game_id} not found in ASA API")
                return None
            
            # Store the game in database
            game_data = games[0]
            await self.store_game(game_data)
            
            # Fetch and return the stored game
            game = await self.db.fetch_one(query, values={"game_id": game_id})
            return dict(game) if game else None
            
        except Exception as e:
            logger.error(f"Error fetching game {game_id} from ASA: {e}")
            return None
    
    async def store_game(self, game_data: Dict) -> Dict:
        """
        Store a game in the database.
        
        This method handles all the data transformation needed to store
        ASA game data in our database schema.
        """
        # Parse the game data
        game_id = game_data.get('game_id')
        date_str = game_data.get('date_time_utc', game_data.get('date'))
        
        # Parse date
        game_date = None
        if date_str:
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d %H:%M:%S UTC', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                    try:
                        game_date = datetime.strptime(date_str.replace(' UTC', ''), fmt)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.warning(f"Could not parse date '{date_str}': {e}")
        
        # Determine if game is completed
        status = game_data.get('status', '').lower()
        is_completed = status in ['fulltime', 'ft', 'finished', 'final']
        
        # Check if it was a shootout
        went_to_shootout = False
        home_penalties = game_data.get('home_penalties', 0)
        away_penalties = game_data.get('away_penalties', 0)
        
        if is_completed and game_data.get('home_score') == game_data.get('away_score'):
            if home_penalties > 0 or away_penalties > 0:
                went_to_shootout = True
        
        # Prepare values for insertion
        values = {
            "game_id": game_id,
            "date": game_date,
            "status": game_data.get('status', 'scheduled'),
            "home_team_id": game_data.get('home_team_id'),
            "away_team_id": game_data.get('away_team_id'),
            "matchday": game_data.get('matchday', 0),
            "attendance": game_data.get('attendance', 0),
            "is_completed": is_completed,
            "went_to_shootout": went_to_shootout,
            "season_year": game_data.get('season_year', game_date.year if game_date else 2025),
            "expanded_minutes": game_data.get('expanded_minutes', 90),
            "home_score": game_data.get('home_score', 0) if is_completed else None,
            "away_score": game_data.get('away_score', 0) if is_completed else None,
            "home_penalties": home_penalties if went_to_shootout else None,
            "away_penalties": away_penalties if went_to_shootout else None
        }
        
        # Use UPSERT to handle duplicates
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
            ON CONFLICT (game_id) DO UPDATE SET
                status = EXCLUDED.status,
                is_completed = EXCLUDED.is_completed,
                went_to_shootout = EXCLUDED.went_to_shootout,
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                home_penalties = EXCLUDED.home_penalties,
                away_penalties = EXCLUDED.away_penalties,
                updated_at = NOW()
            RETURNING *
        """
        
        result = await self.db.fetch_one(query, values=values)
        return dict(result)
    
    async def update_incomplete_games(self, season_year: int, cutoff_date: datetime = None):
        """
        Update all incomplete games that should have been played by now.
        
        This is called before running simulations to ensure we have the latest results.
        """
        if not cutoff_date:
            cutoff_date = datetime.now()
        
        # Get all incomplete games before cutoff date
        query = """
            SELECT game_id, date FROM games 
            WHERE season_year = :season_year 
            AND is_completed = false 
            AND date < :cutoff_date
        """
        
        values = {
            "season_year": season_year,
            "cutoff_date": cutoff_date
        }
        
        incomplete_games = await self.db.fetch_all(query, values=values)
        
        logger.info(f"Found {len(incomplete_games)} incomplete games to update")
        
        # Fetch updates from ASA API
        for game in incomplete_games:
            game_id = game['game_id']
            await self.get_or_fetch_game(game_id)  # This will update if found
            
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)
    
    # ==================== Team Statistics Management ====================
    
    async def get_or_fetch_team_xg(self, team_id: str, season_year: int) -> Dict:
        """
        Get team expected goals statistics from database or fetch from ASA.
        
        This data is crucial for the prediction model's accuracy.
        """
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
            data_age = datetime.now() - xg_data['date_captured']
            if data_age < timedelta(days=1):
                return dict(xg_data)
        
        # Fetch fresh data from ASA
        logger.info(f"Fetching xG data for team {team_id} from ASA API...")
        
        try:
            # Get team xG data from ASA
            xg_stats = self.asa_client.get_team_xgoals(
                team_id=[team_id],
                season_name=[str(season_year)]
            )
            
            if xg_stats and len(xg_stats) > 0:
                stat = xg_stats[0]
                
                # Store in database
                insert_query = """
                    INSERT INTO team_xg_history (
                        team_id, games_played, shots_for, shots_against,
                        x_goals_for, x_goals_against, date_captured, 
                        season_year, matchday
                    ) VALUES (
                        :team_id, :games_played, :shots_for, :shots_against,
                        :x_goals_for, :x_goals_against, NOW(), 
                        :season_year, :matchday
                    ) RETURNING *
                """
                
                values = {
                    "team_id": team_id,
                    "games_played": stat.get('count_games', 0),
                    "shots_for": stat.get('shots_for', 0),
                    "shots_against": stat.get('shots_against', 0),
                    "x_goals_for": stat.get('xgoals_for', 0.0),
                    "x_goals_against": stat.get('xgoals_against', 0.0),
                    "season_year": season_year,
                    "matchday": stat.get('matchday', 0)
                }
                
                result = await self.db.fetch_one(insert_query, values=values)
                return dict(result)
                
        except Exception as e:
            logger.error(f"Error fetching xG data for team {team_id}: {e}")
            
        # Return empty data if all else fails
        return {
            "team_id": team_id,
            "games_played": 0,
            "x_goals_for": 0.0,
            "x_goals_against": 0.0
        }
    
    # ==================== Standings Management ====================
    
    async def calculate_and_store_standings(self, season_year: int, conference: str = None,
                                          matchday: int = None) -> List[Dict]:
        """
        Calculate current standings from games and store in standings_history.
        
        This method properly handles the MLS Next Pro point system:
        - 3 points for regulation win
        - 2 points for shootout win
        - 1 point for shootout loss
        - 0 points for regulation loss
        """
        # Get all completed games
        games = await self.get_games_for_season(season_year, conference, include_incomplete=False)
        
        # Initialize standings dictionary
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
        
        # Process each game
        for game in games:
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            home_score = game['home_score'] or 0
            away_score = game['away_score'] or 0
            went_to_shootout = game['went_to_shootout']
            
            # Update games played
            standings[home_id]["games_played"] += 1
            standings[away_id]["games_played"] += 1
            
            # Update goals (excluding shootout goals)
            standings[home_id]["goals_for"] += home_score
            standings[home_id]["goals_against"] += away_score
            standings[away_id]["goals_for"] += away_score
            standings[away_id]["goals_against"] += home_score
            
            # Update home/away specific stats
            standings[home_id]["home_goals_for"] += home_score
            standings[home_id]["home_goals_against"] += away_score
            standings[away_id]["away_goals_for"] += away_score
            standings[away_id]["away_goals_against"] += home_score
            
            # Determine winner and update records
            if went_to_shootout:
                # Game went to shootout (regulation draw)
                standings[home_id]["draws"] += 1
                standings[away_id]["draws"] += 1
                
                # Determine shootout winner
                home_pens = game['home_penalties'] or 0
                away_pens = game['away_penalties'] or 0
                
                if home_pens > away_pens:
                    standings[home_id]["so_wins"] += 1
                    standings[home_id]["points"] += 2
                    standings[away_id]["points"] += 1
                elif away_pens > home_pens:
                    standings[away_id]["so_wins"] += 1
                    standings[away_id]["points"] += 2
                    standings[home_id]["points"] += 1
                else:
                    # Shouldn't happen, but handle gracefully
                    standings[home_id]["points"] += 1
                    standings[away_id]["points"] += 1
            else:
                # Regulation result
                if home_score > away_score:
                    standings[home_id]["wins"] += 1
                    standings[home_id]["points"] += 3
                    standings[away_id]["losses"] += 1
                elif away_score > home_score:
                    standings[away_id]["wins"] += 1
                    standings[away_id]["points"] += 3
                    standings[home_id]["losses"] += 1
                else:
                    # Regulation draw without shootout (shouldn't happen in MLS Next Pro)
                    standings[home_id]["draws"] += 1
                    standings[away_id]["draws"] += 1
                    standings[home_id]["points"] += 1
                    standings[away_id]["points"] += 1
        
        # Calculate goal difference
        for team_id, stats in standings.items():
            stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]
        
        # Store in database
        for team_id, stats in standings.items():
            await self.store_standings_record(
                team_id=team_id,
                season_year=season_year,
                matchday=matchday or 0,
                stats=stats
            )
        
        # Convert to list and sort by points, then tiebreakers
        standings_list = []
        for team_id, stats in standings.items():
            stats["team_id"] = team_id
            standings_list.append(stats)
        
        # Sort by MLS Next Pro tiebreakers
        standings_list.sort(key=lambda x: (
            -x["points"],
            -x["wins"],
            -x["goal_difference"],
            -x["goals_for"],
            -x["so_wins"]
        ))
        
        return standings_list
    
    async def store_standings_record(self, team_id: str, season_year: int, 
                                   matchday: int, stats: Dict):
        """
        Store a standings record in the database.
        """
        query = """
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
        """
        
        values = {
            "team_id": team_id,
            "season_year": season_year,
            "matchday": matchday,
            **stats
        }
        
        await self.db.execute(query, values=values)
    
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
            
            # Create game record
            values = {
                "game_id": f"{fixture['home_team_id']}_{fixture['away_team_id']}_{fixture['date']}",
                "date": fixture_date,
                "status": "scheduled",
                "home_team_id": fixture['home_team_id'],
                "away_team_id": fixture['away_team_id'],
                "matchday": 0,  # Will be updated later
                "attendance": 0,
                "is_completed": False,
                "went_to_shootout": False,
                "season_year": fixture_date.year if fixture_date else 2025,
                "expanded_minutes": 90
            }
            
            query = """
                INSERT INTO games (
                    game_id, date, status, home_team_id, away_team_id, matchday,
                    attendance, is_completed, went_to_shootout, season_year,
                    expanded_minutes, created_at
                ) VALUES (
                    :game_id, :date, :status, :home_team_id, :away_team_id, :matchday,
                    :attendance, :is_completed, :went_to_shootout, :season_year,
                    :expanded_minutes, NOW()
                )
                ON CONFLICT (game_id) DO NOTHING
            """
            
            await self.db.execute(query, values=values)
    
    # ==================== Historical Data Loading ====================
    
    async def load_historical_season(self, season_year: int):
        """
        Load all games and statistics for a historical season.
        
        This is used to populate the database with past seasons.
        """
        logger.info(f"Loading historical data for {season_year} season...")
        
        try:
            # Fetch all games for the season from ASA
            games = self.asa_client.get_games(
                leagues=['mlsnp'],
                season_name=[str(season_year)]
            )
            
            logger.info(f"Found {len(games)} games for {season_year} season")
            
            # Store each game
            for game in games:
                await self.store_game(game)
                
                # Small delay to be nice to the database
                if len(games) > 100:
                    await asyncio.sleep(0.01)
            
            # Calculate and store final standings
            await self.calculate_and_store_standings(season_year)
            
            # Fetch and store team statistics
            teams = await self.db.fetch_all("SELECT DISTINCT team_id FROM team")
            
            for team in teams:
                team_id = team['team_id']
                await self.get_or_fetch_team_xg(team_id, season_year)
                
                # Small delay
                await asyncio.sleep(0.1)
            
            logger.info(f"Successfully loaded {season_year} season data")
            
        except Exception as e:
            logger.error(f"Error loading historical season {season_year}: {e}")
            raise
    
    # ==================== Simulation Management =======================
    
    async def get_data_for_simulation(self, conference: str, season_year: int) -> Dict[str, Any]:
        """
        Gathers all necessary data for a regular season simulation run.
        This method prepares the data required by the MLSNPRegSeasonPredictor.

        Args:
            conference (str): The conference to get data for ('eastern' or 'western').
            season_year (int): The season year.

        Returns:
            A dictionary containing all the data needed for the simulation.
        """
        logger.info(f"Gathering all simulation data for {conference} conference, {season_year} season...")
        
        # 1. Get all teams for the conference
        conference_teams = await self.get_conference_teams(conference, season_year)
        team_ids = list(conference_teams.keys())
        
        # 2. Get all games for the season (can be filtered client-side by predictor)
        all_games = await self.get_games_for_season(season_year, include_incomplete=True)
        
        # 3. Get team performance data (xG or goals) for each team
        team_performance = {}
        for team_id in team_ids:
            # First, try to get xG data
            xg_data = await self.get_or_fetch_team_xg(team_id, season_year)
            if xg_data and xg_data.get('games_played', 0) > 0:
                team_performance[team_id] = {
                    "team_id": team_id,
                    "games_played": xg_data['games_played'],
                    "x_goals_for": xg_data.get('x_goals_for', 0.0),
                    "x_goals_against": xg_data.get('x_goals_against', 0.0)
                }
            else:
                # Fallback to goal-based stats if no xG
                # This requires fetching standings or game data. We can calculate this from all_games.
                team_performance[team_id] = {"games_played": 0, "goals_for": 0, "goals_against": 0}

        # Calculate goal-based stats for fallback
        for game in all_games:
            if not game.get('is_completed'):
                continue
            home_id, away_id = game['home_team_id'], game['away_team_id']
            if home_id in team_performance:
                team_performance[home_id]['games_played'] += 1
                team_performance[home_id]['goals_for'] = team_performance[home_id].get('goals_for', 0) + game['home_score']
                team_performance[home_id]['goals_against'] = team_performance[home_id].get('goals_against', 0) + game['away_score']
            if away_id in team_performance:
                team_performance[away_id]['games_played'] += 1
                team_performance[away_id]['goals_for'] = team_performance[away_id].get('goals_for', 0) + game['away_score']
                team_performance[away_id]['goals_against'] = team_performance[away_id].get('goals_against', 0) + game['home_score']

        logger.info(f"Successfully gathered data for {len(conference_teams)} teams.")
        
        return {
            "conference_teams": conference_teams,
            "games_data": all_games,
            "team_performance": team_performance
        }
    
    async def store_simulation_run(self, user_id: int, conference: str, 
                                 n_simulations: int, season_year: int,
                                 matchday: int = None) -> int:
        """
        Store a simulation run record and return the run_id.
        """
        # Determine conference_id
        conference_id = 1 if conference.lower() == 'eastern' else 2
        
        query = """
            INSERT INTO prediction_runs (
                run_date, season_year, n_simulations, league_id, 
                conference_id, matchday, is_stored, created_at
            ) VALUES (
                NOW(), :season_year, :n_simulations, 1,
                :conference_id, :matchday, false, NOW()
            ) RETURNING run_id
        """
        
        values = {
            "season_year": season_year,
            "n_simulations": n_simulations,
            "conference_id": conference_id,
            "matchday": matchday or 0
        }
        
        result = await self.db.fetch_one(query, values=values)
        return result['run_id']
    
    async def store_simulation_results(self, run_id: int, summary_df: Any, 
                                     simulation_results: Dict[str, List],
                                     qualification_data: Dict[str, Dict]):
        """
        Store complete simulation results in the database.
        
        This includes:
        - Team summaries with playoff probabilities
        - Shootout analysis
        - Rank distributions
        """
        # Store prediction summary for each team
        for _, row in summary_df.iterrows():
            team_id = row['_team_id']
            
            # Get rank distribution
            ranks = simulation_results.get(team_id, [])
            if not ranks:
                continue
            
            values = {
                "run_id": run_id,
                "team_id": team_id,
                "games_remaining": qualification_data.get(team_id, {}).get('games_remaining', 0),
                "current_points": row['Current Points'],
                "current_rank": 0,  # Will calculate from standings
                "avg_final_rank": np.mean(ranks),
                "median_final_rank": np.median(ranks),
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "rank_25": np.percentile(ranks, 25),
                "rank_75": np.percentile(ranks, 75),
                "playoff_prob_pct": row['Playoff Qualification %'],
                "clinched": qualification_data.get(team_id, {}).get('status', '').startswith('x-'),
                "eliminated": qualification_data.get(team_id, {}).get('status', '').startswith('e-')
            }
            
            query = """
                INSERT INTO prediction_summary (
                    run_id, team_id, games_remaining, current_points, current_rank,
                    avg_final_rank, median_final_rank, best_rank, worst_rank,
                    rank_25, rank_75, playoff_prob_pct, clinched, eliminated, created_at
                ) VALUES (
                    :run_id, :team_id, :games_remaining, :current_points, :current_rank,
                    :avg_final_rank, :median_final_rank, :best_rank, :worst_rank,
                    :rank_25, :rank_75, :playoff_prob_pct, :clinched, :eliminated, NOW()
                )
            """
            
            await self.db.execute(query, values=values)
            
            # Store shootout analysis
            shootout_impact = qualification_data.get(team_id, {}).get('shootout_win_impact', {})
            
            # Find wins needed for 50% and 75% odds
            wins_for_50 = None
            wins_for_75 = None
            
            for wins, prob in shootout_impact.items():
                if prob >= 50 and wins_for_50 is None:
                    wins_for_50 = wins
                if prob >= 75 and wins_for_75 is None:
                    wins_for_75 = wins
            
            if shootout_impact:
                shootout_values = {
                    "run_id": run_id,
                    "team_id": team_id,
                    "games_remaining": qualification_data.get(team_id, {}).get('games_remaining', 0),
                    "wins_for_50_odds": wins_for_50,
                    "wins_for_75_odds": wins_for_75,
                    "current_odds": row['Playoff Qualification %']
                }
                
                shootout_query = """
                    INSERT INTO shootout_analysis (
                        run_id, team_id, games_remaining, wins_for_50_odds,
                        wins_for_75_odds, current_odds
                    ) VALUES (
                        :run_id, :team_id, :games_remaining, :wins_for_50_odds,
                        :wins_for_75_odds, :current_odds
                    )
                """
                
                await self.db.execute(shootout_query, values=shootout_values)
        
        # Mark run as stored
        await self.db.execute(
            "UPDATE prediction_runs SET is_stored = true WHERE run_id = :run_id",
            values={"run_id": run_id}
        )
    
    # ==================== Conference Helper Methods ====================
    
    async def get_conference_teams(self, conference: str, season_year: int) -> Dict[str, str]:
        """
        Get all teams in a conference for a specific season.
        
        Returns a dictionary mapping team_id to team_name.
        """
        conference_id = 1 if conference.lower() == 'eastern' else 2
        
        query = """
            SELECT t.team_id, t.team_name
            FROM team t
            JOIN team_affiliations ta ON t.team_id = ta.team_id
            WHERE ta.conference_id = :conference_id
            AND ta.is_current = true
            AND t.is_active = true
            ORDER BY t.team_name
        """
        
        teams = await self.db.fetch_all(query, values={"conference_id": conference_id})
        
        return {team['team_id']: team['team_name'] for team in teams}
    
    async def ensure_conferences_exist(self):
        """
        Ensure Eastern and Western conferences exist in the database.
        """
        # Check for Eastern Conference
        eastern = await self.db.fetch_one(
            "SELECT * FROM conference WHERE conf_id = 1"
        )
        
        if not eastern:
            await self.db.execute("""
                INSERT INTO conference (conf_id, conf_name, league_id, created_at)
                VALUES (1, 'Eastern Conference', 1, NOW())
            """)
        
        # Check for Western Conference
        western = await self.db.fetch_one(
            "SELECT * FROM conference WHERE conf_id = 2"
        )
        
        if not western:
            await self.db.execute("""
                INSERT INTO conference (conf_id, conf_name, league_id, created_at)
                VALUES (2, 'Western Conference', 1, NOW())
            """)
        
        logger.info("Conferences verified/created")
