import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from itscalledsoccer.client import AmericanSoccerAnalysis as AsaClient
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import json
from src.common.utils import safe_api_call, parse_game_date
from src.mlsnp_predictor import constants

logger = logging.getLogger(__name__)

class MLSNPRegSeasonPredictor:
    def __init__(self):
        """Initialize the predictor with API client and team mappings."""
        try:
            self.client = AsaClient()
            logger.info("Successfully initialized API client")
        except Exception as e:
            logger.error(f"Failed to initialize AsaClient: {e}")
            raise
        
        # Initialize team-related attributes
        self.eastern_teams = set(constants.EASTERN_CONFERENCE_TEAMS.keys())
        self.team_names = constants.EASTERN_CONFERENCE_TEAMS.copy()
        
        # League statistics for regression to mean
        self.league_avg_goals_for = None
        self.league_avg_goals_against = None
        self.league_avg_xgf = None
        self.league_avg_xga = None
        
        # Track home/away statistics for tiebreakers
        self.home_away_stats = defaultdict(lambda: {
            "home_goals_for": 0,
            "home_goals_against": 0,
            "away_goals_for": 0,
            "away_goals_against": 0,
            "home_games": 0,
            "away_games": 0
        })

    def get_games_data(self) -> List[Dict]:
        """
        Retrieve games data for MLS Next Pro.
        Updated to handle the actual API response structure.
        """
        games_data = []
        
        # Method 1: Try the client's get_games method
        logger.info("Attempting to get games data via client...")
        client_params_to_try = [
            {"leagues": [constants.LEAGUE]},  # Preferred
            {"leagues": constants.LEAGUE}     # Fallback if list is not accepted by client
        ]

        for params in client_params_to_try:
            try:
                result = safe_api_call(self.client.get_games, **params)
                if result and len(result) > 0:
                    games_data = result
                    logger.info(f"Retrieved {len(games_data)} games from client with params {params}")
                    break
            except Exception as e:
                logger.warning(f"Client method with params {params} failed: {e}")
            if games_data: # If data found, no need to try other params
                break
        
        # Method 2: Direct API call if client method fails or returns no data
        if not games_data:
            logger.info("Attempting direct API call...")
            try:
                base_url = getattr(self.client, 'BASE_URL', 'https://app.americansocceranalysis.com/api/v1/')
                url = f"{base_url}{constants.LEAGUE}/games"
                
                response = self.client.session.get(url)  # Use client's session
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                
                # Check if response is actually JSON
                if 'application/json' in response.headers.get('Content-Type', ''):
                    games_data_json = response.json()
                    if games_data_json and len(games_data_json) > 0:
                        games_data = games_data_json
                        logger.info(f"Retrieved {len(games_data)} games from direct API")
                    else:
                        logger.warning("Direct API call returned empty or non-list data.")
                else:
                    logger.warning(f"Direct API call did not return JSON. Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                    
            except Exception as e: # Catches requests.exceptions.HTTPError too
                logger.error(f"Direct API call failed: {e}")
        
        if not games_data:
            logger.error("All methods to retrieve games data failed.")
            
        return games_data

    def filter_eastern_conference_games(self, all_games: List[Dict]) -> List[Dict]:
        """
        Filter games to Eastern Conference
        """
        eastern_games = []
        skipped_games = 0
        
        for game in all_games:
            # Extract team IDs using the actual API field names
            home_id = game.get("home_team_id")
            away_id = game.get("away_team_id")
            
            # Only include games where both teams are Eastern Conference
            if home_id in self.eastern_teams and away_id in self.eastern_teams:
                # Filter by season dates using the actual date field name
                game_date = None
                date_field = game.get("date_time_utc")
                
                if date_field:
                    game_date = parse_game_date(str(date_field))
                
                # Include game if date is within season or if we can't parse the date (to be safe)
                if game_date is None or (constants.SEASON_START <= game_date <= constants.SEASON_END):
                    eastern_games.append(game)
                else:
                    skipped_games += 1
            else:
                skipped_games += 1
        
        logger.info(f"Filtered to {len(eastern_games)} Eastern Conference games (skipped {skipped_games} games)")
        return eastern_games

    def calculate_current_standings(self, games_data: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate current standings from completed games.
        """
        standings = defaultdict(lambda: {
            "name": "",
            "points": 0,
            "goal_difference": 0,
            "games_played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "shootout_wins": 0,
            "team_id": None
        })
        
        completed_games_processed = 0 
        
        for game in games_data:
            status = game.get("status", "").lower()
            has_scores = (game.get("home_score") is not None and 
                         game.get("away_score") is not None)
            
            is_completed = has_scores and (status in ["fulltime", "ft", "finished", "final"] or 
                                         (status == "" and has_scores)) 
            
            if not is_completed:
                continue
            
            home_id = game.get("home_team_id")
            away_id = game.get("away_team_id")
            
            if not home_id or not away_id:
                continue
                
            try:
                home_goals = int(game.get("home_score", 0))
                away_goals = int(game.get("away_score", 0))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse scores for game: {game.get('id', 'N/A')}. Skipping game.")
                continue
            
            for team_id_init in [home_id, away_id]:
                if standings[team_id_init]["team_id"] is None: 
                    standings[team_id_init]["team_id"] = team_id_init
                    standings[team_id_init]["name"] = self.team_names.get(team_id_init, f"Team {team_id_init}")
            
            completed_games_processed += 1
            
            self._update_home_away_stats(home_id, away_id, home_goals, away_goals)

            if home_goals == away_goals:
                home_pen, away_pen = 0, 0
                try:
                    home_pen = int(game.get("home_penalties", 0))
                    away_pen = int(game.get("away_penalties", 0))
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse penalty scores for game: {game.get('id', 'N/A')}")

                if home_pen > away_pen:
                    self._update_shootout_winner(standings[home_id], home_goals, away_goals)
                    self._update_shootout_loser(standings[away_id], away_goals, home_goals)
                elif away_pen > home_pen:
                    self._update_shootout_winner(standings[away_id], away_goals, home_goals)
                    self._update_shootout_loser(standings[home_id], home_goals, away_goals)
                else:
                    for team_stat_update, g_for, g_against in [
                        (standings[home_id], home_goals, away_goals), 
                        (standings[away_id], away_goals, home_goals)
                    ]:
                        team_stat_update["points"] += 1
                        team_stat_update["draws"] += 1
                        team_stat_update["games_played"] += 1
                        team_stat_update["goals_for"] += g_for
                        team_stat_update["goals_against"] += g_against
                        team_stat_update["goal_difference"] += g_for - g_against
                    
                    logger.warning(
                        f"Game ID {game.get('id', 'N/A')} between {self.team_names.get(home_id, home_id)} and "
                        f"{self.team_names.get(away_id, away_id)} was a draw ({home_goals}-{away_goals}) "
                        f"but shootout scores were inconclusive ({home_pen}-{away_pen}). "
                        f"Awarded 1 pt to each. 'shootout_wins' not incremented."
                    )
            else:
                self._update_team_standings(standings[home_id], home_goals, away_goals)
                self._update_team_standings(standings[away_id], away_goals, home_goals)
        
        final_standings = {}
        for team_id, stats in standings.items():
            if stats["games_played"] > 0: 
                final_standings[team_id] = dict(stats)
        
        logger.info(f"Processed {completed_games_processed} completed games for {len(final_standings)} teams")
        return final_standings

    def _update_home_away_stats(self, home_id: str, away_id: str, home_goals: int, away_goals: int) -> None:
        self.home_away_stats[home_id]["home_goals_for"] += home_goals
        self.home_away_stats[home_id]["home_goals_against"] += away_goals
        self.home_away_stats[home_id]["home_games"] += 1
        
        self.home_away_stats[away_id]["away_goals_for"] += away_goals
        self.home_away_stats[away_id]["away_goals_against"] += home_goals
        self.home_away_stats[away_id]["away_games"] += 1

    def _update_shootout_winner(self, team_stats: Dict, goals_for: int, goals_against: int) -> None:
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] += goals_for - goals_against
        team_stats["points"] += 2 
        team_stats["draws"] += 1 
        team_stats["shootout_wins"] += 1

    def _update_shootout_loser(self, team_stats: Dict, goals_for: int, goals_against: int) -> None:
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] += goals_for - goals_against
        team_stats["points"] += 1 
        team_stats["draws"] += 1 

    def _update_team_standings(self, team_stats: Dict, goals_for: int, goals_against: int) -> None:
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] += goals_for - goals_against
        
        if goals_for > goals_against: 
            team_stats["wins"] += 1
            team_stats["points"] += 3
        elif goals_for < goals_against: 
            team_stats["losses"] += 1

    def get_team_performance_data(self, games_data: List[Dict]) -> Dict[str, Dict]:
        team_performance = {}
        
        logger.info("Attempting to get xG data from API...")
        xg_data = safe_api_call(self.client.get_team_xgoals, leagues=constants.LEAGUE)
        
        if xg_data:
            logger.info(f"Processing xG data for {len(xg_data)} team records")
            for stat in xg_data:
                team_id = stat.get("team_id")
                if team_id in self.eastern_teams:
                    games = stat.get("count_games", 0)
                    xgoals_for = stat.get("xgoals_for", 0.0)
                    xgoals_against = stat.get("xgoals_against", 0.0)
                    
                    if games > 0:
                        team_performance[team_id] = {
                            "xgf_per_game": float(xgoals_for) / games,
                            "xga_per_game": float(xgoals_against) / games,
                            "games": games,
                            "data_type": "xg"
                        }
                        logger.debug(f"Added xG data for {self.team_names.get(team_id, team_id)}: "
                                    f"{xgoals_for:.2f} xGF in {games} games")
        
        goal_stats = defaultdict(lambda: {"goals_for": 0, "goals_against": 0, "games": 0})
        
        for game in games_data:
            status = game.get("status", "").lower()
            has_scores = (game.get("home_score") is not None and 
                         game.get("away_score") is not None)
            
            is_completed = has_scores and (status in ["fulltime", "ft", "finished", "final"] or 
                                         (status == "" and has_scores))
            
            if not is_completed:
                continue
                
            home_id = game.get("home_team_id")
            away_id = game.get("away_team_id")
            
            if not home_id or not away_id or home_id not in self.eastern_teams or away_id not in self.eastern_teams:
                continue
                
            try:
                home_goals = int(game.get("home_score", 0))
                away_goals = int(game.get("away_score", 0))
                
                goal_stats[home_id]["goals_for"] += home_goals
                goal_stats[home_id]["goals_against"] += away_goals
                goal_stats[home_id]["games"] += 1
                
                goal_stats[away_id]["goals_for"] += away_goals
                goal_stats[away_id]["goals_against"] += home_goals
                goal_stats[away_id]["games"] += 1
                
            except (ValueError, TypeError):
                continue
        
        for team_id in self.eastern_teams:
            if team_id not in team_performance and goal_stats[team_id]["games"] > 0:
                stats = goal_stats[team_id]
                team_performance[team_id] = {
                    "xgf_per_game": stats["goals_for"] / stats["games"],
                    "xga_per_game": stats["goals_against"] / stats["games"],
                    "games": stats["games"],
                    "data_type": "goals"
                }
                logger.debug(f"Added goal-based data for {self.team_names.get(team_id, team_id)}: "
                            f"{stats['goals_for']} goals in {stats['games']} games")
        
        logger.info(f"Performance data available for {len(team_performance)} teams")
        
        xg_teams = sum(1 for stats in team_performance.values() if stats["data_type"] == "xg")
        goal_teams = sum(1 for stats in team_performance.values() if stats["data_type"] == "goals")
        logger.info(f"Using xG data for {xg_teams} teams, goal data for {goal_teams} teams")
        
        return team_performance

    def calculate_league_averages(self, team_performance: Dict[str, Dict]) -> None:
        if not team_performance:
            self.league_avg_xgf = 1.2
            self.league_avg_xga = 1.2
            logger.warning("Using fallback league averages due to no team performance data.")
            return
        
        total_xgf = sum(stats["xgf_per_game"] * stats["games"] for stats in team_performance.values() if stats.get("games", 0) > 0)
        total_xga = sum(stats["xga_per_game"] * stats["games"] for stats in team_performance.values() if stats.get("games", 0) > 0)
        total_games = sum(stats["games"] for stats in team_performance.values() if stats.get("games", 0) > 0)
        
        if total_games > 0:
            self.league_avg_xgf = total_xgf / total_games
            self.league_avg_xga = total_xga / total_games
        else:
            self.league_avg_xgf = 1.2
            self.league_avg_xga = 1.2
            logger.warning("Using fallback league averages as total_games in performance data is zero.")
            
        logger.info(f"League averages - Attack: {self.league_avg_xgf:.3f}, Defense: {self.league_avg_xga:.3f}")

    def load_fixtures_from_file(self) -> List[Dict]:
        try:
            # Ensure the path is relative to the project root, data directory
            fixture_file_path = f"data/{constants.FIXTURES_FILE}"
            with open(fixture_file_path, 'r') as f:
                fixtures_data = json.load(f)
            
            eastern_fixtures = []
            for fixture in fixtures_data:
                home_id = fixture.get("home_team_id")
                away_id = fixture.get("away_team_id")
                
                if home_id in self.eastern_teams and away_id in self.eastern_teams:
                    fixture_date = parse_game_date(fixture.get("date"))
                    if fixture_date and constants.SEASON_START <= fixture_date <= constants.SEASON_END:
                        eastern_fixtures.append(fixture)
            
            logger.info(f"Loaded {len(eastern_fixtures)} Eastern Conference fixtures from file: {fixture_file_path}")
            return eastern_fixtures
            
        except FileNotFoundError:
            logger.error(f"Fixture file {fixture_file_path} not found")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing fixture file: {e}")
            return []

    def calculate_expected_goals(self, home_id: str, away_id: str, team_performance: Dict[str, Dict]) -> Tuple[float, float]:
        home_stats = team_performance.get(home_id, {})
        away_stats = team_performance.get(away_id, {})

        if self.league_avg_xgf is None or self.league_avg_xga is None:
            logger.warning("League averages not pre-calculated. Using defaults for this match.")
            league_avg_xgf_eff = 1.2
            league_avg_xga_eff = 1.2
        else:
            league_avg_xgf_eff = self.league_avg_xgf
            league_avg_xga_eff = self.league_avg_xga
        
        def get_regressed_rates(stats: Dict, lg_avg_att: float, lg_avg_def: float) -> Tuple[float, float]:
            if not stats or stats.get("games", 0) == 0:
                return lg_avg_att, lg_avg_def
            
            games = stats["games"]
            observed_attack = stats["xgf_per_game"]
            observed_defense = stats["xga_per_game"]
            
            weight_for_league_avg = constants.REGRESSION_WEIGHT / (1 + max(0, games - constants.MIN_GAMES_FOR_RELIABILITY) / constants.MIN_GAMES_FOR_RELIABILITY) 
            weight_for_observed = 1 - weight_for_league_avg
            
            regressed_attack = weight_for_observed * observed_attack + weight_for_league_avg * lg_avg_att
            regressed_defense = weight_for_observed * observed_defense + weight_for_league_avg * lg_avg_def
            
            return regressed_attack, regressed_defense
        
        home_attack, home_defense = get_regressed_rates(home_stats, league_avg_xgf_eff, league_avg_xga_eff)
        away_attack, away_defense = get_regressed_rates(away_stats, league_avg_xgf_eff, league_avg_xga_eff)

        # Default expected goals based on league average for attack
        # These are used if league averages are too low for the detailed formula.
        home_expected_fallback = league_avg_xgf_eff
        away_expected_fallback = league_avg_xgf_eff

        if league_avg_xgf_eff > 0.05 and league_avg_xga_eff > 0.05:
            # Standard calculation using team strengths relative to league averages
            home_expected_calculated = (home_attack / league_avg_xgf_eff) * \
                                     (away_defense / league_avg_xga_eff) * \
                                     league_avg_xgf_eff

            away_expected_calculated = (away_attack / league_avg_xgf_eff) * \
                                     (home_defense / league_avg_xga_eff) * \
                                     league_avg_xgf_eff

            # Apply home advantage to the calculated home expectation
            home_expected = home_expected_calculated + constants.HOME_ADVANTAGE_GOALS
            away_expected = away_expected_calculated
        else:
            # Fallback if league averages are too low for the main formula.
            # Ensure home advantage is still applied to the home team's fallback expectation.
            home_expected = home_expected_fallback + constants.HOME_ADVANTAGE_GOALS
            away_expected = away_expected_fallback
        
        # Ensure a minimum floor for expected goals.
        return max(0.1, home_expected), max(0.1, away_expected)

    def simulate_match_with_shootout(self, home_xg: float, away_xg: float) -> Tuple[int, int, bool, bool]:
        home_goals = np.random.poisson(home_xg)
        away_goals = np.random.poisson(away_xg)
        
        went_to_shootout = False
        home_won_shootout = False
        
        if home_goals == away_goals:
            went_to_shootout = True
            home_won_shootout = np.random.random() < 0.55  # 55% chance home team wins shootout
        
        return home_goals, away_goals, went_to_shootout, home_won_shootout

    def simulate_remaining_season(self, current_standings: Dict[str, Dict], 
                                remaining_fixtures: List[Dict], 
                                team_performance: Dict[str, Dict]) -> Dict[str, Dict]:
        projected_standings = {}
        for team_id, stats in current_standings.items():
            projected_standings[team_id] = stats.copy()
            projected_standings[team_id].setdefault('wins', 0)
            projected_standings[team_id].setdefault('losses', 0)
            projected_standings[team_id].setdefault('shootout_wins', 0)
            projected_standings[team_id].setdefault('draws', 0) 

        simulated_home_away = defaultdict(lambda: {
            "home_goals_for": 0, "home_goals_against": 0,
            "away_goals_for": 0, "away_goals_against": 0
        })

        for fixture in remaining_fixtures:
            home_id = fixture.get("home_team_id")
            away_id = fixture.get("away_team_id")

            if not home_id or not away_id or home_id not in projected_standings or away_id not in projected_standings:
                logger.warning(f"Skipping fixture due to missing team(s) in standings: {home_id} vs {away_id}")
                continue

            home_xg, away_xg = self.calculate_expected_goals(home_id, away_id, team_performance)
            home_goals, away_goals, went_to_shootout, home_won_shootout = self.simulate_match_with_shootout(home_xg, away_xg)

            simulated_home_away[home_id]["home_goals_for"] += home_goals
            simulated_home_away[home_id]["home_goals_against"] += away_goals
            simulated_home_away[away_id]["away_goals_for"] += away_goals
            simulated_home_away[away_id]["away_goals_against"] += home_goals

            if went_to_shootout:
                if home_won_shootout:
                    self._update_simulated_shootout_winner(projected_standings[home_id], home_goals, away_goals)
                    self._update_simulated_shootout_loser(projected_standings[away_id], away_goals, home_goals)
                else:
                    self._update_simulated_shootout_winner(projected_standings[away_id], away_goals, home_goals)
                    self._update_simulated_shootout_loser(projected_standings[home_id], home_goals, away_goals)
            else:
                self._update_simulated_standings(projected_standings[home_id], home_goals, away_goals)
                self._update_simulated_standings(projected_standings[away_id], away_goals, home_goals)
        
        for team_id in projected_standings:
            h_a_stats = simulated_home_away[team_id]
            actual_stats = self.home_away_stats.get(team_id, {"home_goals_for":0, "home_goals_against":0, "away_goals_for":0, "away_goals_against":0}) 
            
            projected_standings[team_id]["_home_goals_for"] = actual_stats["home_goals_for"] + h_a_stats["home_goals_for"]
            projected_standings[team_id]["_home_goals_against"] = actual_stats["home_goals_against"] + h_a_stats["home_goals_against"]
            projected_standings[team_id]["_away_goals_for"] = actual_stats["away_goals_for"] + h_a_stats["away_goals_for"]
            projected_standings[team_id]["_away_goals_against"] = actual_stats["away_goals_against"] + h_a_stats["away_goals_against"]
                
        return projected_standings

    def _update_simulated_shootout_winner(self, team_stats: Dict, goals_for: int, goals_against: int) -> None:
        team_stats["points"] += 2
        team_stats["goal_difference"] += goals_for - goals_against
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["games_played"] += 1
        team_stats["shootout_wins"] += 1
        team_stats["draws"] += 1 

    def _update_simulated_shootout_loser(self, team_stats: Dict, goals_for: int, goals_against: int) -> None:
        team_stats["points"] += 1
        team_stats["goal_difference"] += goals_for - goals_against
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["games_played"] += 1
        team_stats["draws"] += 1 


    def _update_simulated_standings(self, team_stats: Dict, goals_for: int, goals_against: int) -> None:
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] += goals_for - goals_against

        if goals_for > goals_against: 
            team_stats["points"] += 3
            team_stats["wins"] += 1
        elif goals_for < goals_against: 
            team_stats["losses"] += 1

    def apply_tiebreakers(self, standings: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        tiebreaker_data = []
        
        for team_id, stats in standings.items():
            points = stats.get("points", 0)
            wins = stats.get("wins", 0) 
            goal_difference = stats.get("goal_difference", 0)
            goals_for = stats.get("goals_for", 0)
            shootout_wins = stats.get("shootout_wins", 0) 
            
            away_gf = stats.get("_away_goals_for", 0)
            away_ga = stats.get("_away_goals_against", 0)
            away_goal_diff = away_gf - away_ga

            home_gf = stats.get("_home_goals_for", 0)
            home_ga = stats.get("_home_goals_against", 0)
            home_goal_diff = home_gf - home_ga
            
            tiebreaker_tuple = (
                -points,                        
                -wins,                          
                -goal_difference,               
                -goals_for,                     
                -shootout_wins,                 
                -away_goal_diff,                
                -away_gf,                       
                -home_goal_diff,                
                -home_gf,                       
                np.random.random()              
            )
            
            tiebreaker_data.append((tiebreaker_tuple, team_id, stats))
        
        tiebreaker_data.sort(key=lambda x: x[0])
        
        return [(team_id, stats) for _, team_id, stats in tiebreaker_data]

    def calculate_playoff_qualification(self, simulation_results: Dict[str, List], 
                                      current_standings: Dict[str, Dict], 
                                      remaining_fixtures: List[Dict]) -> Dict[str, Dict]:
        qualification_data = {}
        if not simulation_results or not any(simulation_results.values()):
            logger.warning("No simulation results to calculate playoff qualification.")
            return qualification_data 

        first_team_id = next(iter(simulation_results))
        n_simulations = len(simulation_results[first_team_id])
        if n_simulations == 0:
            logger.warning("Zero simulations found for playoff qualification.")
            return qualification_data

        remaining_games = defaultdict(int)
        for fixture in remaining_fixtures:
            home_id = fixture.get("home_team_id")
            away_id = fixture.get("away_team_id")
            if home_id in self.eastern_teams: 
                remaining_games[home_id] += 1
            if away_id in self.eastern_teams:
                remaining_games[away_id] += 1
        
        for team_id, ranks in simulation_results.items():
            if not ranks : 
                playoff_prob = 0.0
                best_possible_rank = float('inf')
                worst_possible_rank = float('-inf')
            else:
                made_playoffs = sum(1 for rank in ranks if rank <= 8) 
                playoff_prob = (made_playoffs / n_simulations) * 100 if n_simulations > 0 else 0
                best_possible_rank = min(ranks) if ranks else float('inf')
                worst_possible_rank = max(ranks) if ranks else float('-inf')

            status = ""
            if worst_possible_rank <= 8 and worst_possible_rank > 0 : 
                status = "x-"  
            elif best_possible_rank > 8:
                status = "e-"  
            
            shootout_impact = {}
            games_left = remaining_games.get(team_id, 0) 
            
            for extra_wins in range(min(6, games_left + 1)):
                extra_points = extra_wins * 1 
                estimated_prob = min(100, playoff_prob + (extra_points * 5)) 
                shootout_impact[extra_wins] = estimated_prob
            
            qualification_data[team_id] = {
                "status": status,
                "playoff_probability": playoff_prob,
                "games_remaining": games_left,
                "shootout_win_impact": shootout_impact, 
                "team_name_debug": self.team_names.get(team_id, team_id) 
            }
        
        return qualification_data

    def run_simulations(self, n_simulations: int = constants.N_SIMULATIONS) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
        logger.info("Retrieving games data...")
        all_games = self.get_games_data()
        if not all_games:
            logger.error("No games data available, cannot run simulations.")
            return pd.DataFrame(), {}, self.team_names, {}
        
        eastern_games = self.filter_eastern_conference_games(all_games)
        if not eastern_games:
            logger.error("No Eastern Conference games found, cannot run simulations.")
            return pd.DataFrame(), {}, self.team_names, {}

        logger.info("Calculating current standings...")
        current_standings = self.calculate_current_standings(eastern_games)
        if not current_standings:
            logger.error("Could not calculate current standings, cannot run simulations.")
            current_standings = {
                team_id: {
                    "name": name, "points": 0, "goal_difference": 0, "games_played": 0,
                    "wins": 0, "draws": 0, "losses": 0, "goals_for": 0, "goals_against": 0,
                    "shootout_wins": 0, "team_id": team_id
                } for team_id, name in self.team_names.items() if team_id in self.eastern_teams
            }
            if not current_standings:
                return pd.DataFrame(), {}, self.team_names, {}
            logger.warning("Proceeding with empty or default initial standings for all teams.")

        standings_df_current = pd.DataFrame(current_standings.values())
        if not standings_df_current.empty:
            standings_df_current = standings_df_current.sort_values(by=['points', 'goal_difference', 'wins'], ascending=[False, False, False])
            print("\n" + "="*80)
            print("CURRENT EASTERN CONFERENCE STANDINGS")
            print("="*80)
            display_cols = ['name', 'games_played', 'wins', 'draws', 'losses', 
                           'shootout_wins', 'points', 'goal_difference']
            display_cols = [col for col in display_cols if col in standings_df_current.columns]
            print(standings_df_current[display_cols].to_string(index=False))
            print("="*80)
        else:
            print("Current standings are empty.")

        logger.info("Analyzing team performance...")
        team_performance = self.get_team_performance_data(eastern_games)
        self.calculate_league_averages(team_performance)
        
        remaining_fixtures = self.load_fixtures_from_file()
        if not remaining_fixtures:
            logger.warning("No remaining fixtures found. Season simulation will be based on current standings only.")
        logger.info(f"Will simulate {len(remaining_fixtures)} remaining fixtures")
        
        teams_for_simulation = list(current_standings.keys())
        if not teams_for_simulation:
            teams_for_simulation = list(self.eastern_teams)

        simulation_results = {team_id: [] for team_id in teams_for_simulation}
        playoff_qualification_count = {team_id: 0 for team_id in teams_for_simulation}
        rank_distributions = {team_id: np.zeros(len(teams_for_simulation) if teams_for_simulation else 1) for team_id in teams_for_simulation}
        
        simulation_points = {team_id: [] for team_id in teams_for_simulation}

        logger.info(f"Running {n_simulations} simulations...")
        for _ in tqdm(range(n_simulations), desc="Simulating seasons"):
            final_standings = self.simulate_remaining_season(current_standings, remaining_fixtures, team_performance)
            if not final_standings:
                logger.warning("A simulation run produced empty final standings.")
                continue

            ranked_teams = self.apply_tiebreakers(final_standings)
            
            for rank, (team_id, team_stats) in enumerate(ranked_teams):
                final_rank = rank + 1
                if team_id in simulation_results:
                    simulation_results[team_id].append(final_rank)
                    simulation_points[team_id].append(team_stats.get("points", 0))
                    
                    if rank < len(rank_distributions[team_id]):
                        rank_distributions[team_id][rank] += 1
                    else:
                        logger.warning(f"Rank {rank} out of bounds for team {team_id} rank_distributions size {len(rank_distributions[team_id])}")

                    if final_rank <= 8:
                        playoff_qualification_count[team_id] += 1
                else:
                    logger.warning(f"Team ID {team_id} from ranked_teams not found in simulation_results keys.")

        qualification_data = self.calculate_playoff_qualification(
            simulation_results, current_standings, remaining_fixtures
        )
        
        summary_data = []
        for team_id in teams_for_simulation:
            ranks = simulation_results.get(team_id, [])
            points_list = simulation_points.get(team_id, [])  
            current_team_stats = current_standings.get(team_id, {})

            if not current_team_stats:
                current_team_stats = {
                    "name": self.team_names.get(team_id, f"Team {team_id}"), "points": 0, "games_played": 0,
                    "shootout_wins": 0
                }

            if ranks:
                avg_rank = np.mean(ranks)
                playoff_pct = (playoff_qualification_count.get(team_id, 0) / n_simulations) * 100 if n_simulations > 0 else 0.0
                rank_25 = np.percentile(ranks, 25)
                rank_75 = np.percentile(ranks, 75)
                rank_min = min(ranks)
                rank_max = max(ranks)
                rank_median = np.median(ranks)
            else:
                avg_rank, rank_median, rank_min, rank_max, rank_25, rank_75 = [float('nan')] * 6
                playoff_pct = 0.0
            
            if points_list:
                points_min = min(points_list)
                points_max = max(points_list)
                points_avg = np.mean(points_list)
            else:
                points_min = points_max = points_avg = current_team_stats.get("points", 0)
            
            qual_status_info = qualification_data.get(team_id, {})
            qual_status = qual_status_info.get("status", "")
            team_name = self.team_names.get(team_id, f"Team {team_id}")
            display_name = f"{qual_status}{team_name}" if qual_status else team_name
            
            summary_data.append({
                "Team": display_name,
                "Current Points": current_team_stats.get("points", 0),
                "Games Played": current_team_stats.get("games_played", 0),
                "Current Shootout Wins": current_team_stats.get("shootout_wins", 0),
                "Worst Points": points_min,
                "Average Points": round(points_avg, 1),
                "Best Points": points_max,
                "Average Final Rank": round(avg_rank, 2) if not np.isnan(avg_rank) else 'N/A',
                "Median Final Rank": round(rank_median, 1) if not np.isnan(rank_median) else 'N/A',
                "Best Rank": rank_min if not np.isnan(rank_min) else 'N/A',
                "Worst Rank": rank_max if not np.isnan(rank_max) else 'N/A',
                "Playoff Qualification %": round(playoff_pct, 1),
                "_team_id": team_id,
                "_rank_25": rank_25 if not np.isnan(rank_25) else 'N/A',
                "_rank_75": rank_75 if not np.isnan(rank_75) else 'N/A'
            })

        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            if 'Average Final Rank' in summary_df.columns and not summary_df['Average Final Rank'].eq('N/A').all():
                summary_df['Average Final Rank Numeric'] = pd.to_numeric(summary_df['Average Final Rank'], errors='coerce')
                summary_df = summary_df.sort_values(by="Average Final Rank Numeric").drop(columns=['Average Final Rank Numeric'])
            
            ordered_rank_distributions = []
            if "_team_id" in summary_df.columns:
                for team_id_in_df in summary_df["_team_id"]:
                    dist = rank_distributions.get(team_id_in_df, np.array([]))
                    ordered_rank_distributions.append(dist)
                summary_df["_rank_distribution"] = ordered_rank_distributions
            else:
                logger.error("'_team_id' column missing from summary_df, cannot add rank distributions.")
                summary_df["_rank_distribution"] = [np.array([])] * len(summary_df)
        
        return summary_df, simulation_results, self.team_names, qualification_data


    def plot_results(self, summary_df: pd.DataFrame, qualification_data: Dict[str, Dict]) -> None:
        if summary_df.empty:
            logger.warning("Summary DataFrame is empty. Skipping plot generation.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        teams = summary_df["Team"].tolist()
        playoff_probs = summary_df["Playoff Qualification %"].tolist()
        
        ax1.barh(teams, playoff_probs, color="navy")
        ax1.set_title("Playoff Qualification Probability", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Playoff Qualification %")
        ax1.grid(axis='x', alpha=0.7)
        ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5)
        
        n_teams = len(summary_df)
        ax2.clear() 
        
        candlestick_y_labels = []
        candlestick_y_ticks = []

        for i, row_idx in enumerate(summary_df.index): 
            row = summary_df.loc[row_idx]
            candlestick_y_labels.append(row["Team"])
            candlestick_y_ticks.append(i)

            rank_min = pd.to_numeric(row["Best Rank"], errors='coerce')
            rank_max = pd.to_numeric(row["Worst Rank"], errors='coerce')
            rank_25 = pd.to_numeric(row["_rank_25"], errors='coerce')
            rank_75 = pd.to_numeric(row["_rank_75"], errors='coerce')
            rank_median = pd.to_numeric(row["Median Final Rank"], errors='coerce')

            if any(pd.isna([rank_min, rank_max, rank_25, rank_75, rank_median])):
                logger.warning(f"Skipping candlestick for team {row['Team']} due to missing rank data.")
                continue
            
            ax2.plot([i, i], [rank_min, rank_max], 'k-', linewidth=1)
            box_height = rank_75 - rank_25
            if box_height < 0: box_height = 0 

            rect = Rectangle((i - 0.3, rank_25), 0.6, box_height,
                           facecolor='lightblue', edgecolor='black', linewidth=1)
            ax2.add_patch(rect)
            ax2.plot([i - 0.3, i + 0.3], [rank_median, rank_median], 'r-', linewidth=2)
            ax2.plot([i - 0.1, i + 0.1], [rank_min, rank_min], 'k-', linewidth=1)
            ax2.plot([i - 0.1, i + 0.1], [rank_max, rank_max], 'k-', linewidth=1)
        
        ax2.set_xlim(-0.5, n_teams - 0.5)
        ax2.set_ylim(n_teams + 0.5, 0.5)
        ax2.set_xticks(candlestick_y_ticks) 
        ax2.set_xticklabels(candlestick_y_labels, rotation=45, ha='right') 
        ax2.set_ylabel("Final Rank")
        ax2.set_title("Final Rank Distribution (Candlestick Chart)", fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=8.5, color='red', linestyle='--', alpha=0.5, label='Playoff cutoff')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='r', linewidth=2, label='Median rank'),
            Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='25th-75th percentile'),
            Line2D([0], [0], color='k', linewidth=1, label='Best/Worst rank')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        current_points_numeric = pd.to_numeric(summary_df["Current Points"], errors='coerce')
        avg_final_rank_numeric = pd.to_numeric(summary_df["Average Final Rank"], errors='coerce')
        
        valid_plot_data = summary_df.dropna(subset=['Current Points', 'Average Final Rank', 'Playoff Qualification %'])
        current_points_numeric = pd.to_numeric(valid_plot_data["Current Points"], errors='coerce')
        avg_final_rank_numeric = pd.to_numeric(valid_plot_data["Average Final Rank"], errors='coerce')
        playoff_qual_pct_numeric = pd.to_numeric(valid_plot_data["Playoff Qualification %"], errors='coerce')

        if not current_points_numeric.empty and not avg_final_rank_numeric.empty:
            scatter = ax3.scatter(current_points_numeric, avg_final_rank_numeric, 
                       s=100, alpha=0.6, c=playoff_qual_pct_numeric, 
                       cmap='RdYlGn', edgecolors='black')
            ax3.set_xlabel("Current Points")
            ax3.set_ylabel("Average Final Rank")
            ax3.set_title("Current Standing vs Projected Finish", fontsize=14, fontweight='bold')
            ax3.invert_yaxis()
            ax3.grid(alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label("Playoff Probability %")
        else:
            ax3.text(0.5, 0.5, "Not enough data for this plot", ha='center', va='center', transform=ax3.transAxes)

        ax4.clear()
        if "Games Played" in summary_df.columns and "_team_id" in summary_df.columns:
            teams_for_shootout_analysis_df = summary_df[pd.to_numeric(summary_df["Games Played"], errors='coerce').notna()]
            teams_for_shootout_analysis_df["Games Played Numeric"] = pd.to_numeric(teams_for_shootout_analysis_df["Games Played"], errors='coerce')
            
            if not teams_for_shootout_analysis_df.empty:
                teams_to_analyze_ids = teams_for_shootout_analysis_df.nlargest(5, "Games Played Numeric")["_team_id"].tolist()
            
                for team_id in teams_to_analyze_ids:
                    qual_info = qualification_data.get(team_id, {})
                    shootout_impact = qual_info.get("shootout_win_impact", {})
                    team_name_qual = qual_info.get("team_name_debug", self.team_names.get(team_id, team_id)) 
                    
                    if shootout_impact:
                        wins = sorted(list(shootout_impact.keys())) 
                        probs = [shootout_impact[w] for w in wins]
                        ax4.plot(wins, probs, marker='o', label=team_name_qual, linewidth=2)
            
            ax4.set_xlabel("Additional Shootout Wins (Hypothetical)")
            ax4.set_ylabel("Estimated Playoff Probability %")
            ax4.set_title("Impact of Add. Shootout Wins", fontsize=14, fontweight='bold') 
            ax4.grid(alpha=0.3)
            ax4.legend(loc='best', fontsize='small')
            ax4.set_xlim(-0.5, 5.5) 
            ax4.set_ylim(0, 105)
        else:
            ax4.text(0.5, 0.5, "Not enough data for shootout impact plot", ha='center', va='center', transform=ax4.transAxes)

        
        plt.tight_layout()
        plt.show()

    def create_shootout_analysis_table(self, summary_df: pd.DataFrame, 
                                     qualification_data: Dict[str, Dict]) -> pd.DataFrame:
        analysis_data = []
        if summary_df.empty:
            return pd.DataFrame(analysis_data)

        for _, row in summary_df.iterrows():
            team_id = row["_team_id"]
            team_name = row["Team"] 
            current_prob = row["Playoff Qualification %"]
            
            qual_info = qualification_data.get(team_id, {})
            shootout_impact = qual_info.get("shootout_win_impact", {})
            games_left = qual_info.get("games_remaining", 0)
            
            wins_for_50 = "N/A"
            wins_for_75 = "N/A"
            
            for wins in sorted(shootout_impact.keys()):
                prob = shootout_impact[wins]
                if prob >= 50 and wins_for_50 == "N/A":
                    wins_for_50 = wins
                if prob >= 75 and wins_for_75 == "N/A":
                    wins_for_75 = wins
            
            analysis_data.append({
                "Team": team_name,
                "Current Playoff %": f"{current_prob:.1f}%" if isinstance(current_prob, (int, float)) else current_prob,
                "Games Remaining": games_left,
                "Add. SO Wins for ~50% Chance": wins_for_50, 
                "Add. SO Wins for ~75% Chance": wins_for_75
            })
        
        return pd.DataFrame(analysis_data)
