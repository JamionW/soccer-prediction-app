import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import pickle
from pathlib import Path

# AutoML imports
try:
    from pycaret.regression import *
    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False
    logging.warning("PyCaret not installed. Install with: pip install pycaret")

logger = logging.getLogger(__name__)

class MLSNPRegSeasonPredictor:
    """
    Enhanced MLS Next Pro regular season predictor with AutoML capabilities.
    """

    def __init__(self, conference: str, conference_teams: Dict[str, str], 
                 games_data: List[Dict], team_performance: Dict[str, Dict], 
                 league_averages: Dict[str, float], use_automl: bool = True):
        """
        Initialize the predictor with all necessary data.
        
        Args:
            conference (str): The conference to simulate ('eastern' or 'western').
            conference_teams (Dict[str, str]): A dictionary of team_id -> team_name for the conference.
            games_data (List[Dict]): A list of all game data for the season.
            team_performance (Dict[str, Dict]): Pre-calculated team performance metrics (xG, goals per game).
            league_averages (Dict[str, float]): League-wide average goals and xG.
            use_automl (bool): Whether to use AutoML model if available.
        """
        self.conference = conference
        self.conference_teams = set(conference_teams.keys())
        self.team_names = conference_teams
        self.games_data = games_data
        self.team_performance = team_performance
        self.league_avg_xgf = league_averages.get('league_avg_xgf', 1.2)
        self.league_avg_xga = league_averages.get('league_avg_xga', 1.2)
        self.use_automl = use_automl and AUTOML_AVAILABLE

        self.current_standings = self._calculate_current_standings()
        self.remaining_games = self._filter_remaining_games()
        
        # Initialize ML model
        self.ml_model = None
        self.model_path = Path(f"models/xg_predictor_{conference}_{datetime.now().strftime('%Y%m')}.pkl")
        
        if self.use_automl:
            self._initialize_ml_model()

    def _initialize_ml_model(self):
        """Initialize or load the AutoML model for xG prediction."""
        # Try to load existing model first
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                logger.info(f"Loaded existing ML model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
        # Build new model if needed
        logger.info("Building new AutoML model for xG prediction...")
        training_data = self._prepare_training_data()
        
        if len(training_data) < 50:
            logger.warning("Insufficient training data for ML model. Falling back to traditional method.")
            self.use_automl = False
            return
        
        try:
            # Initialize PyCaret environment
            reg_setup = setup(
                data=training_data,
                target='goals',  # We'll predict actual goals
                train_size=0.8,
                session_id=42,
                verbose=False,
                normalize=True,
                transformation=True,
                remove_multicollinearity=True,
                feature_interaction=True,
                polynomial_features=True,
                ignore_low_variance=True
            )
            
            # Create and train the model
            # Using blend of best models for better performance
            best_model = compare_models(
                include=['lr', 'ridge', 'rf', 'gbr', 'xgboost', 'lightgbm'],
                fold=5,
                n_select=3
            )
            
            self.ml_model = blend_models(best_model)
            
            # Save the model
            self.model_path.parent.mkdir(exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.ml_model, f)
            
            logger.info("AutoML model training completed successfully")

        except Exception as e:
            logger.error(f"AutoML model training failed: {e}")
            self.use_automl = False

    def _prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare training data for the ML model with engineered features.
        
        Returns:
            DataFrame with features and target variable (goals scored)
        """
        training_records = []
        
        # Get completed games for training
        completed_games = [g for g in self.games_data if g.get('is_completed')]
        
        for game in completed_games:
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            
            # Skip if teams not in our conference (for conference-specific models)
            if home_id not in self.conference_teams or away_id not in self.conference_teams:
                continue
            
            # Create two records per game (one for each team)
            # Home team record
            home_features = self._extract_features(
                team_id=home_id,
                opponent_id=away_id,
                is_home=True,
                game_date=game.get('date'),
                games_before=completed_games
            )
            home_features['goals'] = game.get('home_score', 0)
            training_records.append(home_features)
            
            # Away team record
            away_features = self._extract_features(
                team_id=away_id,
                opponent_id=home_id,
                is_home=False,
                game_date=game.get('date'),
                games_before=completed_games
            )
            away_features['goals'] = game.get('away_score', 0)
            training_records.append(away_features)
        
        return pd.DataFrame(training_records)

    def _extract_features(self, team_id: str, opponent_id: str, is_home: bool, 
                         game_date: str, games_before: List[Dict]) -> Dict[str, float]:
        """
        Extract features for ML model prediction.
        
        This is where the magic happens - good feature engineering is crucial!
        """
        features = {}
        
        # Basic features
        features['is_home'] = 1.0 if is_home else 0.0
        
        # Team strength features (from xG data)
        team_stats = self.team_performance.get(team_id, {})
        opp_stats = self.team_performance.get(opponent_id, {})
        
        # Offensive and defensive strength
        team_games = max(team_stats.get('games_played', 1), 1)
        opp_games = max(opp_stats.get('games_played', 1), 1)
        
        features['team_xgf_per_game'] = team_stats.get('x_goals_for', 0) / team_games
        features['team_xga_per_game'] = team_stats.get('x_goals_against', 0) / team_games
        features['opp_xgf_per_game'] = opp_stats.get('x_goals_for', 0) / opp_games
        features['opp_xga_per_game'] = opp_stats.get('x_goals_against', 0) / opp_games
        
        # Relative strength features
        features['xg_diff'] = features['team_xgf_per_game'] - features['team_xga_per_game']
        features['opp_xg_diff'] = features['opp_xgf_per_game'] - features['opp_xga_per_game']
        
        # Form features (last 5 games)
        team_form = self._calculate_form(team_id, game_date, games_before, n_games=5)
        opp_form = self._calculate_form(opponent_id, game_date, games_before, n_games=5)
        
        features['team_form_points'] = team_form['points_per_game']
        features['team_form_gf'] = team_form['goals_for_per_game']
        features['team_form_ga'] = team_form['goals_against_per_game']
        features['opp_form_points'] = opp_form['points_per_game']
        features['opp_form_gf'] = opp_form['goals_for_per_game']
        features['opp_form_ga'] = opp_form['goals_against_per_game']
        
        # Head-to-head features
        h2h = self._calculate_h2h_features(team_id, opponent_id, game_date, games_before)
        features.update(h2h)
        
        # Rest days (if we can calculate it)
        features['team_rest_days'] = self._calculate_rest_days(team_id, game_date, games_before)
        features['opp_rest_days'] = self._calculate_rest_days(opponent_id, game_date, games_before)
        
        # Temporal features
        if game_date:
            try:
                dt = pd.to_datetime(game_date)
                features['month'] = dt.month
                features['day_of_week'] = dt.dayofweek
                features['is_weekend'] = 1.0 if dt.dayofweek >= 5 else 0.0
            except:
                features['month'] = 6  # Default to June
                features['day_of_week'] = 3  # Default to Wednesday
                features['is_weekend'] = 0.0
        
        # League average features for context
        features['league_avg_xgf'] = self.league_avg_xgf
        features['league_avg_xga'] = self.league_avg_xga
        
        return features

    def _calculate_current_standings(self) -> Dict[str, Dict]:
        """
        Calculates current standings based on completed games from the provided data.
        This mirrors the logic from DatabaseManager but operates on the local data copy.
        """
        standings = defaultdict(lambda: {
            "team_id": None, "name": "", "points": 0, "goal_difference": 0,
            "games_played": 0, "wins": 0, "draws": 0, "losses": 0,
            "goals_for": 0, "goals_against": 0, "shootout_wins": 0
        })

        for game in self.games_data:
            if not game.get("is_completed"):
                continue

            home_id, away_id = game["home_team_id"], game["away_team_id"]
            if home_id not in self.conference_teams or away_id not in self.conference_teams:
                continue

            for team_id in [home_id, away_id]:
                if standings[team_id]["team_id"] is None:
                    standings[team_id]["team_id"] = team_id
                    standings[team_id]["name"] = self.team_names.get(team_id, f"Team {team_id}")

            home_score, away_score = game.get("home_score", 0), game.get("away_score", 0)

            if game.get("went_to_shootout"):
                home_pens, away_pens = game.get("home_penalties", 0), game.get("away_penalties", 0)
                if home_pens > away_pens:
                    self._update_shootout_winner(standings[home_id], home_score, away_score)
                    self._update_shootout_loser(standings[away_id], away_score, home_score)
                else:
                    self._update_shootout_winner(standings[away_id], away_score, home_score)
                    self._update_shootout_loser(standings[home_id], home_score, away_score)
            else:
                if home_score > away_score:
                    self._update_team_standings(standings[home_id], home_score, away_score, "win")
                    self._update_team_standings(standings[away_id], away_score, home_score, "loss")
                elif away_score > home_score:
                    self._update_team_standings(standings[away_id], away_score, home_score, "win")
                    self._update_team_standings(standings[home_id], home_score, away_score, "loss")

        for team_id, stats in standings.items():
            stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]

        return {team_id: dict(stats) for team_id, stats in standings.items()}

    def _filter_remaining_games(self) -> List[Dict]:
        """Filters for future games to be simulated."""
        return [
            game for game in self.games_data
            if not game.get("is_completed") and
               game.get("home_team_id") in self.conference_teams and
               game.get("away_team_id") in self.conference_teams
        ]
    
    def _get_team_strength(self, team_id: str) -> Tuple[float, float]:
        """Gets a team's offensive and defensive strength, falling back to league average."""
        stats = self.team_performance.get(team_id)
        if stats and stats.get('games_played', 0) > 0:
            # Use xG if available, otherwise fall back to goals
            attack_metric = stats.get('x_goals_for', stats.get('goals_for', 0))
            defend_metric = stats.get('x_goals_against', stats.get('goals_against', 0))
            games_played = stats['games_played']
            
            attack_strength = (attack_metric / games_played) / self.league_avg_xgf
            defend_strength = (defend_metric / games_played) / self.league_avg_xga
            return attack_strength, defend_strength
        
        return 1.0, 1.0 # Fallback to league average strength

    def _calculate_form(self, team_id: str, before_date: str, games: List[Dict], 
                       n_games: int = 5) -> Dict[str, float]:
        """Calculate recent form statistics for a team."""
        recent_games = []
        
        for game in reversed(games):  # Most recent first
            if not game.get('is_completed'):
                continue
                
            game_date = game.get('date')
            if game_date and before_date and game_date >= before_date:
                continue
                
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                recent_games.append(game)
                
            if len(recent_games) >= n_games:
                break
        
        if not recent_games:
            return {
                'points_per_game': 1.0,  # League average assumption
                'goals_for_per_game': self.league_avg_xgf,
                'goals_against_per_game': self.league_avg_xga
            }
        
        total_points = 0
        total_gf = 0
        total_ga = 0
        
        for game in recent_games:
            if game['home_team_id'] == team_id:
                gf = game.get('home_score', 0)
                ga = game.get('away_score', 0)
            else:
                gf = game.get('away_score', 0)
                ga = game.get('home_score', 0)
            
            total_gf += gf
            total_ga += ga
            
            # Calculate points (including shootout logic)
            if game.get('went_to_shootout'):
                if game['home_team_id'] == team_id:
                    won_shootout = (game.get('home_penalties', 0) > game.get('away_penalties', 0))
                else:
                    won_shootout = (game.get('away_penalties', 0) > game.get('home_penalties', 0))
                total_points += 2 if won_shootout else 1
            else:
                if gf > ga:
                    total_points += 3
                elif gf == ga:
                    total_points += 1
        
        n = len(recent_games)
        return {
            'points_per_game': total_points / n,
            'goals_for_per_game': total_gf / n,
            'goals_against_per_game': total_ga / n
        }

    def _calculate_h2h_features(self, team_id: str, opponent_id: str, 
                               before_date: str, games: List[Dict]) -> Dict[str, float]:
        """Calculate head-to-head features between two teams."""
        h2h_games = []
        
        for game in games:
            if not game.get('is_completed'):
                continue
                
            game_date = game.get('date')
            if game_date and before_date and game_date >= before_date:
                continue
                
            if ((game['home_team_id'] == team_id and game['away_team_id'] == opponent_id) or
                (game['home_team_id'] == opponent_id and game['away_team_id'] == team_id)):
                h2h_games.append(game)
        
        if not h2h_games:
            return {
                'h2h_games_played': 0.0,
                'h2h_win_rate': 0.5,
                'h2h_goals_for_avg': self.league_avg_xgf,
                'h2h_goals_against_avg': self.league_avg_xga
            }
        
        wins = 0
        total_gf = 0
        total_ga = 0
        
        for game in h2h_games:
            if game['home_team_id'] == team_id:
                gf = game.get('home_score', 0)
                ga = game.get('away_score', 0)
            else:
                gf = game.get('away_score', 0)
                ga = game.get('home_score', 0)
            
            total_gf += gf
            total_ga += ga
            
            if gf > ga:
                wins += 1
            elif game.get('went_to_shootout'):
                # Check shootout winner
                if game['home_team_id'] == team_id:
                    if game.get('home_penalties', 0) > game.get('away_penalties', 0):
                        wins += 0.5  # Count shootout wins as half
                else:
                    if game.get('away_penalties', 0) > game.get('home_penalties', 0):
                        wins += 0.5
        
        n = len(h2h_games)
        return {
            'h2h_games_played': float(n),
            'h2h_win_rate': wins / n,
            'h2h_goals_for_avg': total_gf / n,
            'h2h_goals_against_avg': total_ga / n
        }

    def _calculate_rest_days(self, team_id: str, game_date: str, games: List[Dict]) -> float:
        """Calculate days of rest since last game."""
        if not game_date:
            return 7.0  # Default assumption
        
        try:
            current_date = pd.to_datetime(game_date)
        except:
            return 7.0
        
        last_game_date = None
        
        for game in reversed(games):  # Most recent first
            if not game.get('is_completed'):
                continue
                
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                try:
                    gd = pd.to_datetime(game.get('date'))
                    if gd < current_date:
                        last_game_date = gd
                        break
                except:
                    continue
        
        if last_game_date:
            return (current_date - last_game_date).days
        
        return 7.0  # Default

    def _simulate_game(self, game: Dict) -> Tuple[int, int, bool]:
        """Simulates a single game using either ML model or traditional method."""
        home_id, away_id = game["home_team_id"], game["away_team_id"]
        
        if self.use_automl and self.ml_model:
            # Use ML model for prediction
            home_features = self._extract_features(
                team_id=home_id,
                opponent_id=away_id,
                is_home=True,
                game_date=game.get('date'),
                games_before=self.games_data
            )
            
            away_features = self._extract_features(
                team_id=away_id,
                opponent_id=home_id,
                is_home=False,
                game_date=game.get('date'),
                games_before=self.games_data
            )
            
            # Convert to DataFrame for prediction
            home_df = pd.DataFrame([home_features])
            away_df = pd.DataFrame([away_features])
            
            try:
                # Get predictions (expected goals)
                home_exp_goals = max(0.1, self.ml_model.predict(home_df)[0])
                away_exp_goals = max(0.1, self.ml_model.predict(away_df)[0])
            except Exception as e:
                logger.warning(f"ML prediction failed, falling back to traditional method: {e}")
                return self._simulate_game_traditional(game)
        else:
            # Use traditional method
            return self._simulate_game_traditional(game)
        
        # Sample from Poisson distribution
        home_goals = np.random.poisson(home_exp_goals)
        away_goals = np.random.poisson(away_exp_goals)
        
        went_to_shootout = False
        if home_goals == away_goals:
            went_to_shootout = True
            # Shootout probability could also be learned, but keeping simple for now
            if np.random.rand() > 0.5:
                home_goals += 1
            else:
                away_goals += 1
                
        return home_goals, away_goals, went_to_shootout

    def _simulate_game_traditional(self, game: Dict) -> Tuple[int, int, bool]:
        """Traditional simulation method (fallback)."""
        home_id, away_id = game["home_team_id"], game["away_team_id"]
        
        home_attack, home_defense = self._get_team_strength(home_id)
        away_attack, away_defense = self._get_team_strength(away_id)
        
        # Calculate expected goals for this matchup
        home_exp_goals = home_attack * away_defense * self.league_avg_xgf
        away_exp_goals = away_attack * home_defense * self.league_avg_xga
        
        # Get result from Poisson distribution
        home_goals = np.random.poisson(home_exp_goals)
        away_goals = np.random.poisson(away_exp_goals)
        
        went_to_shootout = False
        if home_goals == away_goals:
            went_to_shootout = True
            if np.random.rand() > 0.5:
                home_goals += 1
            else:
                away_goals += 1
                
        return home_goals, away_goals, went_to_shootout

    def _get_team_strength(self, team_id: str) -> Tuple[float, float]:
        """Gets a team's offensive and defensive strength, falling back to league average."""
        stats = self.team_performance.get(team_id)
        if stats and stats.get('games_played', 0) > 0:
            # Use xG if available, otherwise fall back to goals
            attack_metric = stats.get('x_goals_for', stats.get('goals_for', 0))
            defend_metric = stats.get('x_goals_against', stats.get('goals_against', 0))
            games_played = stats['games_played']
            
            attack_strength = (attack_metric / games_played) / self.league_avg_xgf
            defend_strength = (defend_metric / games_played) / self.league_avg_xga
            return attack_strength, defend_strength
        
        return 1.0, 1.0 # Fallback to league average strength

    def run_simulations(self, n_simulations: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """
        Runs the Monte Carlo simulation for n_simulations.
        """
        final_ranks = defaultdict(list)
        final_points = defaultdict(list)

        for _ in range(n_simulations): # Removed tqdm
            sim_standings = {team_id: stats.copy() for team_id, stats in self.current_standings.items()}

            for game in self.remaining_games:
                home_id, away_id = game["home_team_id"], game["away_team_id"]
                h_goals, a_goals, shootout = self._simulate_game(game)

                if shootout:
                    if h_goals > a_goals: # Home won shootout
                        self._update_shootout_winner(sim_standings[home_id], h_goals - 1, a_goals)
                        self._update_shootout_loser(sim_standings[away_id], a_goals, h_goals - 1)
                    else: # Away won shootout
                        self._update_shootout_winner(sim_standings[away_id], a_goals - 1, h_goals)
                        self._update_shootout_loser(sim_standings[home_id], h_goals, a_goals - 1)
                else: # Regulation result
                    if h_goals > a_goals:
                        self._update_team_standings(sim_standings[home_id], h_goals, a_goals, "win")
                        self._update_team_standings(sim_standings[away_id], a_goals, h_goals, "loss")
                    else:
                        self._update_team_standings(sim_standings[away_id], a_goals, h_goals, "win")
                        self._update_team_standings(sim_standings[home_id], h_goals, a_goals, "loss")

            # Sort standings and record ranks
            sorted_teams = sorted(sim_standings.values(), key=lambda x: (-x['points'], -x['wins'], -x['goal_difference'], -x['goals_for'], -x['shootout_wins']))
            for rank, stats in enumerate(sorted_teams, 1):
                team_id = stats['team_id']
                final_ranks[team_id].append(rank)
                final_points[team_id].append(stats['points'])
        
        summary_df, qualification_data = self._create_summary_df(final_ranks, final_points)
        
        # This function should return the same signature as the original
        # The third element (rank_dist_df) can be an empty DataFrame if not used.
        return summary_df, final_ranks, pd.DataFrame(), qualification_data

    def _create_summary_df(self, final_ranks: Dict, final_points: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Creates the final summary DataFrame and qualification data dictionary."""
        summary_data = []
        qualification_data = {}
        
        for team_id, ranks in final_ranks.items():
            current_stats = self.current_standings.get(team_id, {})
            playoff_prob = (np.array(ranks) <= 8).mean() * 100
            
            summary_data.append({
                'Team': self.team_names.get(team_id, team_id),
                '_team_id': team_id,
                'Current Points': current_stats.get('points', 0),
                'Games Played': current_stats.get('games_played', 0),
                'Playoff Qualification %': playoff_prob,
                'Average Final Rank': np.mean(ranks),
                'Average Points': np.mean(final_points[team_id]),
            })

            qualification_data[team_id] = {
                'games_remaining': len(self.remaining_games),
                'status': '', # This can be enhanced later
                'shootout_win_impact': {} # Placeholder for compatibility
            }

        summary_df = pd.DataFrame(summary_data).sort_values(by='Playoff Qualification %', ascending=False).reset_index(drop=True)
        return summary_df, qualification_data

    # Helper methods to update standings state
    def _update_team_standings(self, team_stats: Dict, goals_for: int, goals_against: int, result: str):
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] += (goals_for - goals_against)
        if result == "win":
            team_stats["wins"] += 1
            team_stats["points"] += 3
        elif result == "loss":
            team_stats["losses"] += 1

    def _update_shootout_winner(self, team_stats: Dict, goals_for: int, goals_against: int):
        self._update_team_standings(team_stats, goals_for, goals_against, "loss") # No points for loss part
        team_stats["draws"] += 1
        team_stats["shootout_wins"] += 1
        team_stats["points"] += 2 # 2 points for SO win

    def _update_shootout_loser(self, team_stats: Dict, goals_for: int, goals_against: int):
        self._update_team_standings(team_stats, goals_for, goals_against, "loss") # No points for loss part
        team_stats["draws"] += 1
        team_stats["points"] += 1 # 1 point for SO loss
