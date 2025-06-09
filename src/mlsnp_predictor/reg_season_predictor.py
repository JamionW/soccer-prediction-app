import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class MLSNPRegSeasonPredictor:
    """
    A pure computational engine for running MLS Next Pro regular season simulations.
    
    This class takes pre-fetched game data and team statistics to run Monte Carlo
    simulations. It does not perform any I/O operations (database or API calls),
    which is handled in the database manager.
    """

    def __init__(self, conference: str, conference_teams: Dict[str, str], games_data: List[Dict], team_performance: Dict[str, Dict], league_averages: Dict[str, float]):
        """
        Initialize the predictor with all necessary data.
        
        Args:
            conference (str): The conference to simulate ('eastern' or 'western').
            conference_teams (Dict[str, str]): A dictionary of team_id -> team_name for the conference.
            games_data (List[Dict]): A list of all game data for the season.
            team_performance (Dict[str, Dict]): Pre-calculated team performance metrics (xG, goals per game).
            league_averages (Dict[str, float]): League-wide average goals and xG.
        """
        self.conference = conference
        self.conference_teams = set(conference_teams.keys())
        self.team_names = conference_teams
        self.games_data = games_data
        self.team_performance = team_performance
        self.league_avg_xgf = league_averages.get('league_avg_xgf', 1.2)
        self.league_avg_xga = league_averages.get('league_avg_xga', 1.2)

        self.current_standings = self._calculate_current_standings()
        self.remaining_games = self._filter_remaining_games()

        logger.info(f"=== GAME FILTERING DEBUG ===")
        logger.info(f"Total games passed to predictor: {len(self.games_data)}")
        logger.info(f"Conference teams: {len(self.conference_teams)}")
        logger.info(f"Remaining games after filtering: {len(self.remaining_games)}")
        
        # Let's see what's in the first few games for good measure
        for i, game in enumerate(self.games_data[:3]):
            logger.info(f"Sample game {i+1}: completed={game.get('is_completed')}, home={game.get('home_team_id')}, away={game.get('away_team_id')}")


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
            
            # SAFETY GUARDS: Ensure we never divide by zero
            safe_league_avg_xgf = max(self.league_avg_xgf, 0.1)  # Minimum 0.1
            safe_league_avg_xga = max(self.league_avg_xga, 0.1)  # Minimum 0.1
            
            # Calculate per-game metrics
            attack_per_game = attack_metric / games_played if games_played > 0 else safe_league_avg_xgf
            defend_per_game = defend_metric / games_played if games_played > 0 else safe_league_avg_xga
            
            # Calculate strength ratios
            attack_strength = attack_per_game / safe_league_avg_xgf
            defend_strength = defend_per_game / safe_league_avg_xga
            
            # Ensure reasonable bounds (between 0.1 and 5.0)
            attack_strength = max(min(attack_strength, 5.0), 0.1)
            defend_strength = max(min(defend_strength, 5.0), 0.1)
            
            return attack_strength, defend_strength
        
        return 1.0, 1.0 # Fallback to league average strength

    def _simulate_game(self, game: Dict) -> Tuple[int, int, bool]:
        """Simulates a single game and returns score and shootout status."""
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
            # Simple coin flip for shootout winner
            if np.random.rand() > 0.5:
                home_goals += 1 # Representing a shootout win
            else:
                away_goals += 1
                
        return home_goals, away_goals, went_to_shootout

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