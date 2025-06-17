import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MLSNPPlayoffPredictor:
    def __init__(self, games_data: List[Dict], team_performance: Dict[str, Dict],
                 league_avg_xgf: float = 1.2, league_avg_xga: float = 1.2,
                 regular_season_records: Dict[str, Dict] = None):
        """
        Initialize playoff predictor with season data.
        
        Args:
            games_data: List of completed games from the season.
            team_performance: Dictionary of team performance metrics.
            league_avg_xgf: League average expected goals for.
            league_avg_xga: League average expected goals against.
            regular_season_records: Regular season records for determining home field advantage
        """
        logger.info("Initializing MLSNPPlayoffPredictor.")
        self.games_data = games_data
        self.team_performance = team_performance
        self.league_avg_xgf = league_avg_xgf if league_avg_xgf > 0.05 else 1.2
        self.league_avg_xga = league_avg_xga if league_avg_xga > 0.05 else 1.2
        self.regular_season_records = regular_season_records or {}
        logger.info(f"Playoff Predictor initialized with {len(games_data)} games, {len(team_performance)} teams' performance data.")
        logger.info(f"League averages for Playoff predictor: xGF={self.league_avg_xgf:.2f}, xGA={self.league_avg_xga:.2f}")
        
        # Cache for head-to-head records
        self.h2h_cache = {}
        
        # Cache for recent form
        self.form_cache = {}
        
        # Playoff selection weights
        self.SELECTION_WEIGHTS = {
            'best_h2h': 0.5,     # 50% - Pick opponent with best Head-to-Head rating
            'worst_form': 0.3,   # 30% - Pick opponent in worst form
            'lowest_seed': 0.2   # 20% - Pick lowest seed
        }
        # Simulation constants
        self.HOME_ADVANTAGE_XG_MULTIPLIER = 1.10  # Home team gets a 10% xG boost
        self.HOME_TEAM_SHOOTOUT_WIN_PROB = 0.55 # Home team has 55% chance of winning shootout
        self.NEUTRAL_SITE_SHOOTOUT_WIN_PROB = 0.50 # Neutral site shootout is 50/50
        logger.info(f"Playoff sim constants: Home Adv xG Mult={self.HOME_ADVANTAGE_XG_MULTIPLIER}, Home SO Prob={self.HOME_TEAM_SHOOTOUT_WIN_PROB}, Neutral SO Prob={self.NEUTRAL_SITE_SHOOTOUT_WIN_PROB}")
    
    def determine_championship_home_team(self, east_champ_id: str, west_champ_id: str) -> Tuple[str, str]:
        """
        Determine home team for championship based on regular season records.
        
        Uses the following tiebreakers in order:
        1. Total points
        2. Total wins
        3. Goal difference
        4. Goals for
        5. Coin flip (random)
        
        Returns:
            Tuple of (home_team_id, away_team_id)
        """
        # Get regular season records from both conferences
        east_record = self.regular_season_records.get('eastern', {}).get(east_champ_id, {})
        west_record = self.regular_season_records.get('western', {}).get(west_champ_id, {})
        
        # If we don't have records, fall back to coin flip
        if not east_record or not west_record:
            logger.warning(f"Missing regular season records for championship between {east_champ_id} and {west_champ_id}. Using coin flip.")
            reason_for_missing = "missing records (coin flip)"
            home_team, away_team = (east_champ_id, west_champ_id) if np.random.random() < 0.5 else (west_champ_id, east_champ_id)
            logger.info(f"Championship home team determined by {reason_for_missing}: {home_team} (home) vs {away_team} (away)")
            return home_team, away_team

        reason = "N/A" # Will be updated by the tiebreakers
        # Default to East champ as home; will be overridden if West wins tiebreakers or by coin flip
        home_team, away_team = east_champ_id, west_champ_id 
        
        # Compare records using tiebreakers
        # 1. Total points (use average final points for better accuracy)
        east_points = east_record.get('average_final_points', east_record.get('points', 0))
        west_points = west_record.get('average_final_points', west_record.get('points', 0))
        logger.debug(f"Championship HFA check: {east_champ_id} pts={east_points} vs {west_champ_id} pts={west_points}")

        if east_points > west_points:
            reason = "points"
            # home_team, away_team already set to east_champ_id, west_champ_id
        elif west_points > east_points:
            reason = "points"
            home_team, away_team = west_champ_id, east_champ_id
        else:
            # 2. Total wins
            east_wins = east_record.get('wins', 0)
            west_wins = west_record.get('wins', 0)
            logger.debug(f"Championship HFA check (tie on pts): {east_champ_id} wins={east_wins} vs {west_champ_id} wins={west_wins}")
            if east_wins > west_wins:
                reason = "wins"
                # home_team, away_team are correct as points were tied (still default East home)
            elif west_wins > east_wins:
                reason = "wins"
                home_team, away_team = west_champ_id, east_champ_id
            else:
                # 3. Goal difference
                east_gd = east_record.get('goal_difference', 0)
                west_gd = west_record.get('goal_difference', 0)
                logger.debug(f"Championship HFA check (tie on wins): {east_champ_id} GD={east_gd} vs {west_champ_id} GD={west_gd}")
                if east_gd > west_gd:
                    reason = "goal difference"
                    # home_team, away_team are correct
                elif west_gd > east_gd:
                    reason = "goal difference"
                    home_team, away_team = west_champ_id, east_champ_id
                else:
                    # 4. Goals for
                    east_gf = east_record.get('goals_for', 0)
                    west_gf = west_record.get('goals_for', 0)
                    logger.debug(f"Championship HFA check (tie on GD): {east_champ_id} GF={east_gf} vs {west_champ_id} GF={west_gf}")
                    if east_gf > west_gf:
                        reason = "goals for"
                        # home_team, away_team are correct
                    elif west_gf > east_gf:
                        reason = "goals for"
                        home_team, away_team = west_champ_id, east_champ_id
                    else:
                        # 5. Coin flip
                        logger.debug(f"Championship HFA check (tie on GF): Using coin flip.")
                        reason = "coin flip"
                        if np.random.random() < 0.5:
                            # home_team, away_team are already east_champ_id, west_champ_id (East wins flip)
                            pass
                        else:
                            home_team, away_team = west_champ_id, east_champ_id # West wins flip

        logger.info(f"Championship home team determined: {home_team} (home) vs {away_team} (away) based on {reason}")
        return home_team, away_team
    
    def calculate_head_to_head_rating(self, team_id: str, opponent_id: str) -> float:
        """
        Calculate a head-to-head rating for a team against a specific opponent.
        Based on goal differential in past H2H games, or difference of season xG differentials if no H2H games.
        Positive value means team_id has performed better against opponent_id.
        
        Args:
            team_id: ID of the selecting team
            opponent_id: ID of the potential opponent
            
        Returns:
            Head-to-head rating value.
        """
        cache_key = f"{team_id}-{opponent_id}"
        if cache_key in self.h2h_cache:
            return self.h2h_cache[cache_key]
        
        goals_for_team = 0
        goals_against_team = 0
        games_played = 0
        
        for game in self.games_data:
            is_completed = game.get('status', '').lower() in ['fulltime', 'ft', 'finished', 'final']
            if not is_completed:
                continue

            home_id = game.get("home_team_id")
            away_id = game.get("away_team_id")

            try:
                home_goals = int(game.get("home_score", 0))
                away_goals = int(game.get("away_score", 0))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse scores for H2H game: {game.get('id', 'N/A')}. Skipping game in H2H.")
                continue

            if home_id == team_id and away_id == opponent_id:
                goals_for_team += home_goals
                goals_against_team += away_goals
                games_played += 1
            elif home_id == opponent_id and away_id == team_id:
                goals_for_team += away_goals
                goals_against_team += home_goals
                games_played += 1
        
        if games_played > 0:
            # Use actual goal difference from H2H games
            h2h_goal_differential = (goals_for_team - goals_against_team) / games_played
        else:
            # Fallback: difference of season xG differentials
            # (team's season xG_for - team's season xG_against) - (opponent's season xG_for - opponent's season xG_against)
            team_xgf = self.team_performance.get(team_id, {}).get('xgf_per_game', 1.2)
            team_xga = self.team_performance.get(team_id, {}).get('xga_per_game', 1.2)
            opp_xgf = self.team_performance.get(opponent_id, {}).get('xgf_per_game', 1.2)
            opp_xga = self.team_performance.get(opponent_id, {}).get('xga_per_game', 1.2)
            
            h2h_goal_differential = (team_xgf - team_xga) - (opp_xgf - opp_xga)
            logger.debug(f"No H2H games for {team_id} vs {opponent_id}. Using season xG diff: {h2h_goal_differential:.2f}")

        self.h2h_cache[cache_key] = h2h_goal_differential
        return h2h_goal_differential
    
    def calculate_recent_form(self, team_id: str, n_games: int = 5) -> float:
        """
        Calculate a team's form over the last N games.
        Lower score = worse form (what we want for opponent selection).
        
        Args:
            team_id: Team ID
            n_games: Number of recent games to consider
            
        Returns:
            Form score (0-1, where 0 is worst form)
        """
        cache_key = f"{team_id}-{n_games}"
        if cache_key in self.form_cache:
            return self.form_cache[cache_key]
        
        # Get team's recent games
        team_games = []
        for game in reversed(self.games_data):  # Most recent first
            if game['status'] not in ['fulltime', 'ft', 'finished', 'final']:
                continue
                
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                team_games.append(game)
                
            if len(team_games) >= n_games:
                break
        
        if not team_games:
            # No recent games, assume average form
            self.form_cache[cache_key] = 0.5
            return 0.5
        
        # Calculate form based on results and goal difference
        form_points = 0
        total_goal_diff_in_recent_games = 0 # Using actual goal diff from these games
        
        for game in team_games:
            home_id = game.get("home_team_id")
            away_id = game.get("away_team_id")

            try:
                game_home_goals = int(game.get("home_score", 0))
                game_away_goals = int(game.get("away_score", 0))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse scores for form calculation game: {game.get('id', 'N/A')}. Skipping game in form calc.")
                continue

            if home_id == team_id: # Team was home
                goals_for = game_home_goals
                goals_against = game_away_goals
            else: # Team was away
                goals_for = game_away_goals
                goals_against = game_home_goals
            
            # Points (3 for win, 1 for draw)
            if goals_for > goals_against:
                form_points += 3
            elif goals_for == goals_against:
                form_points += 1
                
            # Actual goal differential for the team in this game
            total_goal_diff_in_recent_games += (goals_for - goals_against)
        
        # Normalize form score (0-1) based on points
        max_points = len(team_games) * 3
        form_score_from_points = form_points / max_points if max_points > 0 else 0.5
        
        # Adjust by actual recent game performance (goal difference)
        # Avoid division by zero if team_games is empty (though checked earlier)
        avg_goal_diff_in_recent_games = total_goal_diff_in_recent_games / len(team_games) if team_games else 0
        
        # Scale adjustment: e.g., an average GD of +1 adds 0.1 to form, -1 subtracts 0.1
        # Capped at +/- 0.2 to prevent GD from dominating points too much.
        goal_diff_adjustment = min(max(avg_goal_diff_in_recent_games * 0.1, -0.2), 0.2)

        final_form = min(max(form_score_from_points + goal_diff_adjustment, 0), 1)
        self.form_cache[cache_key] = final_form
        return final_form
    
    def select_opponent(self, selecting_team: str, available_opponents: List[str], 
                       seeding: Dict[str, int]) -> str:
        """
        Select an opponent based on the weighted criteria.
        
        Args:
            selecting_team: Team ID doing the selecting
            available_opponents: List of available opponent team IDs
            seeding: Dictionary mapping team_id to seed number
            
        Returns:
            Selected opponent team ID
        """
        if len(available_opponents) == 1:
            return available_opponents[0]
        
        # Calculate scores for each potential opponent
        opponent_scores = {}
        
        for opponent in available_opponents:
            # Best Head-to-Head rating (higher is better for selecting team)
            h2h_rating = self.calculate_head_to_head_rating(selecting_team, opponent)
            
            # Worst form (lower form score is what we want)
            form_score = self.calculate_recent_form(opponent)
            
            # Lowest seed (higher seed number is what we want)
            seed_score = seeding[opponent] / 8.0  # Normalize to 0-1
            
            opponent_scores[opponent] = {
                'h2h_score': h2h_rating,
                'form_score': -form_score,  # Negative because we want worst form (higher is better after negation)
                'seed_score': seed_score,
            }
        
        # Determine selection method using weighted random choice
        selection_method = np.random.choice(
            list(self.SELECTION_WEIGHTS.keys()), # ['best_h2h', 'worst_form', 'lowest_seed']
            p=list(self.SELECTION_WEIGHTS.values())
        )
        
        # Select opponent based on chosen method
        if selection_method == 'best_h2h':
            # Pick opponent with best H2H rating for selecting team
            selected = max(opponent_scores.keys(), 
                          key=lambda x: opponent_scores[x]['h2h_score'])
        elif selection_method == 'worst_form':
            # Pick opponent with worst recent form
            selected = max(opponent_scores.keys(), 
                          key=lambda x: opponent_scores[x]['form_score'])
        else:  # lowest_seed
            # Pick opponent with lowest seed (highest seed number)
            selected = max(opponent_scores.keys(), 
                          key=lambda x: opponent_scores[x]['seed_score'])
        
        logger.debug(f"{selecting_team} selected {selected} using {selection_method} method")
        return selected, selection_method
    
    def simulate_match(self, home_team_id: str, away_team_id: str,
                      is_neutral_site: bool = False) -> Tuple[str, int, int]:
        """
        Simulate a playoff match between two teams.
        
        Args:
            home_team_id: ID of the designated home team.
            away_team_id: ID of the designated away team.
            is_neutral_site: Whether match is at neutral site (no home advantage effects).
            
        Returns:
            Tuple of (winner_id, home_score, away_score)
        """

        # Get team performance metrics
        logger.debug(f"Simulating match: Home={home_team_id}, Away={away_team_id}, Neutral={is_neutral_site}")
        home_perf = self.team_performance.get(home_team_id, {})
        away_perf = self.team_performance.get(away_team_id, {})

        # Calculate expected goals
        home_attack_strength = home_perf.get('xgf_per_game', self.league_avg_xgf)
        away_attack_strength = away_perf.get('xgf_per_game', self.league_avg_xgf)

        # Adjust for defensive strength using league averages passed in __init__
        # Defensive ratio: opponent_xga / league_avg_xga. If ratio < 1, opponent is good defensively.
        home_exp_goals_adj_vs_away_def = away_perf.get('xga_per_game', self.league_avg_xga) / self.league_avg_xga
        away_exp_goals_adj_vs_home_def = home_perf.get('xga_per_game', self.league_avg_xga) / self.league_avg_xga

        home_exp_goals = home_attack_strength * home_exp_goals_adj_vs_away_def
        away_exp_goals = away_attack_strength * away_exp_goals_adj_vs_home_def
        
        # Apply home advantage if not a neutral site
        if not is_neutral_site:
            # Using a multiplier for simplicity right now.
            home_exp_goals *= self.HOME_ADVANTAGE_XG_MULTIPLIER

        # Ensure xG is not negative and has a minimum floor
        logger.debug(f"Match xG before floor: HomeID={home_team_id}, HomeExp={home_exp_goals:.2f}, AwayID={away_team_id}, AwayExp={away_exp_goals:.2f}")
        home_exp_goals = max(0.05, home_exp_goals)
        away_exp_goals = max(0.05, away_exp_goals)
        logger.debug(f"Match xG after floor: HomeID={home_team_id}, HomeExp={home_exp_goals:.2f}, AwayID={away_team_id}, AwayExp={away_exp_goals:.2f}")

        # Simulate goals
        home_goals = np.random.poisson(home_exp_goals)
        away_goals = np.random.poisson(away_exp_goals)

        # Handle draws with penalty shootout
        shootout_occurred = False
        if home_goals == away_goals:
            shootout_occurred = True
            if is_neutral_site:
                shootout_home_win_prob = self.NEUTRAL_SITE_SHOOTOUT_WIN_PROB
                logger.debug(f"Shootout on neutral site. Home ({home_team_id}) win prob: {shootout_home_win_prob:.2f}")
            else:
                # Give home team a slight advantage in shootout
                shootout_home_win_prob = self.HOME_TEAM_SHOOTOUT_WIN_PROB
                logger.debug(f"Shootout at home team's ({home_team_id}) venue. Win prob: {shootout_home_win_prob:.2f}")

            winner = home_team_id if np.random.random() < shootout_home_win_prob else away_team_id
        else:
            winner = home_team_id if home_goals > away_goals else away_team_id
        
        logger.debug(f"Match result: {home_team_id} {home_goals} - {away_goals} {away_team_id}. Winner: {winner}. Shootout: {shootout_occurred}")
        return winner, home_goals, away_goals
    
    def simulate_single_playoff(self, eastern_seeds: Dict[str, int], 
                               western_seeds: Dict[str, int]) -> Dict:
        """
        Simulate a single playoff bracket
        """
        results = {
            'eastern': {'round1': [], 'round2': [], 'final': None},
            'western': {'round1': [], 'round2': [], 'final': None},
            'championship': None
        }
        
        # Simulate each conference
        for conf, seeds in [('eastern', eastern_seeds), ('western', western_seeds)]:
            logger.info(f"Simulating {conf} conference playoffs...")
            # Create reverse mapping (seed -> team_id)
            seed_to_team = {v: k for k, v in seeds.items()}
            
            results[conf]['round1_selection_details'] = []
            logger.debug(f"{conf} Round 1 opponent selection starting.")
            # Round 1: Seeds 1-3 choose opponents
            available_opponents = [seed_to_team[i] for i in range(5, 9)] # Seeds 5,6,7,8
            round1_matchups = []
            
            # Seeds 1-3 choose in order
            for seed_num_selector in range(1, 4): # Seeds 1, 2, 3
                selecting_team_id = seed_to_team[seed_num_selector]
                chosen_opponent_id, selection_method = self.select_opponent(selecting_team_id, available_opponents, seeds)
                results[conf]['round1_selection_details'].append({
                    'selector_seed': seed_num_selector,
                    'selector_id': selecting_team_id,
                    'selected_opponent_id': chosen_opponent_id,
                    'method': selection_method
                })
                available_opponents.remove(chosen_opponent_id)
                round1_matchups.append((selecting_team_id, chosen_opponent_id))
            
            # Seed 4 gets remaining opponent
            round1_matchups.append((seed_to_team[4], available_opponents[0]))
            logger.debug(f"{conf} Round 1 matchups: {round1_matchups}")
            
            # Simulate Round 1
            logger.info(f"Simulating {conf} Round 1 matches...")
            round1_winners = []
            for i, (selected_matchup_team1, selected_matchup_team2) in enumerate(round1_matchups):
                logger.debug(f"R1 Match {i+1} ({conf}): {selected_matchup_team1} vs {selected_matchup_team2}")
                if seeds[selected_matchup_team1] > seeds[selected_matchup_team2]:
                    # Team2 is higher seed, so they are home.
                    actual_home, actual_away = selected_matchup_team2, selected_matchup_team1
                else:
                    # Team1 is higher seed (or equal, default to team1 as home - selection order implies this)
                    actual_home, actual_away = selected_matchup_team1, selected_matchup_team2

                winner, home_score, away_score = self.simulate_match(actual_home, actual_away, is_neutral_site=False)

                # Store scores based on the actual home/away for clarity if needed elsewhere
                # For the 'matchup' tuple, it's just for reference of who played whom.
                results[conf]['round1'].append({
                    'matchup': tuple(sorted((selected_matchup_team1, selected_matchup_team2))),
                    'sim_details': {'home': actual_home, 'away': actual_away, 'home_score': home_score, 'away_score': away_score},
                    'winner': winner,
                })
                round1_winners.append(winner)
            
            # Round 2: Highest seed chooses opponent
            # Note: highest seed means lowest seed number
            round1_winner_seeds = sorted([(seeds[w], w) for w in round1_winners]) # List of (seed_num, team_id)
            logger.debug(f"{conf} Round 1 Winners (seed, team_id): {round1_winner_seeds}")
            
            highest_seed_selector_id = round1_winner_seeds[0][1] # Team ID of the highest seed
            highest_seed_selector_seed_num = round1_winner_seeds[0][0]
            logger.debug(f"{conf} Round 2 highest seed selector: {highest_seed_selector_id} (Seed {highest_seed_selector_seed_num})")

            opponents_for_round2_choice = [details[1] for details in round1_winner_seeds[1:]] # List of team_ids
            logger.debug(f"{conf} Round 2 available opponents for selection: {opponents_for_round2_choice}")
            
            results[conf]['round2_selection_details'] = []
            if len(round1_winners) == 4 and len(opponents_for_round2_choice) == 3 : # Standard scenario: 4 winners, 3 choices for top seed
                logger.debug(f"{conf} Round 2 opponent selection starting for {highest_seed_selector_id}.")
                chosen_opponent_r2_id, selection_method_r2 = self.select_opponent(
                    highest_seed_selector_id, opponents_for_round2_choice, seeds
                )
                results[conf]['round2_selection_details'].append({
                    'selector_seed': highest_seed_selector_seed_num,
                    'selector_id': highest_seed_selector_id,
                    'selected_opponent_id': chosen_opponent_r2_id,
                    'method': selection_method_r2
                })

                opponents_for_round2_choice.remove(chosen_opponent_r2_id) # Remove chosen
                round2_matchups = [
                    (highest_seed_selector_id, chosen_opponent_r2_id),
                    (opponents_for_round2_choice[0], opponents_for_round2_choice[1]) # The remaining two play each other
                ]
            elif len(round1_winners) < 4 and len(round1_winners) >=2 : # Handle scenarios with fewer than 4 winners if possible (e.g. straight to final)
                 logger.warning(f"Fewer than 4 winners in {conf} Round 1 ({len(round1_winners)}), proceeding with available teams for Round 2 if possible.")
                 if len(round1_winners) >= 2: # Need at least two for a match
                    round2_matchups = [(round1_winner_seeds[0][1], round1_winner_seeds[1][1])] # Highest two play
                 else: # Not enough for even one match
                    round2_matchups = []
            else:
                logger.error(f"Unexpected number of R1 winners ({len(round1_winners)}) or available opponents ({len(opponents_for_round2_choice)}) for {conf} Round 2. Cannot form matchups.")
                round2_matchups = []
            logger.debug(f"{conf} Round 2 matchups: {round2_matchups}")

            # Simulate Round 2
            logger.info(f"Simulating {conf} Round 2 matches...")
            round2_winners = []
            for i, (selected_matchup_team1, selected_matchup_team2) in enumerate(round2_matchups):
                # Determine actual home team by seed
                logger.debug(f"R2 Match {i+1} ({conf}): {selected_matchup_team1} vs {selected_matchup_team2}")
                if seeds[selected_matchup_team1] > seeds[selected_matchup_team2]:
                    actual_home, actual_away = selected_matchup_team2, selected_matchup_team1
                else:
                    actual_home, actual_away = selected_matchup_team1, selected_matchup_team2
                logger.debug(f"R2 Match {i+1} ({conf}) Home: {actual_home}, Away: {actual_away}")
                winner, home_score, away_score = self.simulate_match(actual_home, actual_away, is_neutral_site=False)
                results[conf]['round2'].append({
                    'matchup': tuple(sorted((selected_matchup_team1, selected_matchup_team2))),
                    'sim_details': {'home': actual_home, 'away': actual_away, 'home_score': home_score, 'away_score': away_score},
                    'winner': winner,
                })
                round2_winners.append(winner)
            logger.debug(f"{conf} Round 2 Winners: {round2_winners}")
            
            # Conference Final
            logger.info(f"Simulating {conf} Conference Final...")
            if len(round2_winners) < 2:
                logger.error(f"Not enough winners for {conf} Conference Final. Winners: {round2_winners}")
                results[conf]['final'] = {'winner': None, 'matchup': (None,None), 'sim_details': None}
            else:
                team_a_cf, team_b_cf = round2_winners[0], round2_winners[1] # Should be the only two if logic is correct
                logger.debug(f"{conf} Conf Final: {team_a_cf} vs {team_b_cf}")
                if seeds[team_a_cf] > seeds[team_b_cf]:
                    actual_home_cf, actual_away_cf = team_b_cf, team_a_cf
                else:
                    actual_home_cf, actual_away_cf = team_a_cf, team_b_cf
                logger.debug(f"{conf} Conf Final Home: {actual_home_cf}, Away: {actual_away_cf}")
                winner_cf, home_score_cf, away_score_cf = self.simulate_match(actual_home_cf, actual_away_cf, is_neutral_site=False)
                results[conf]['final'] = {
                    'matchup': tuple(sorted((team_a_cf, team_b_cf))),
                    'sim_details': {'home': actual_home_cf, 'away': actual_away_cf, 'home_score': home_score_cf, 'away_score': away_score_cf},
                    'winner': winner_cf,
                }
            if results[conf]['final'] and results[conf]['final']['winner']:
                 logger.info(f"{conf} Conference Champion: {results[conf]['final']['winner']}")
            else:
                 logger.warning(f"{conf} Conference Final could not be determined due to insufficient winners.")

        # Championship - Hosted by higher seed from original regular season seeding
        logger.info("Simulating MLSNP Championship game...")
        east_champ_id = results.get('eastern', {}).get('final', {}).get('winner')
        west_champ_id = results.get('western', {}).get('final', {}).get('winner')

        if not east_champ_id or not west_champ_id:
            logger.error(f"Cannot simulate championship. Eastern Champ: {east_champ_id}, Western Champ: {west_champ_id}")
            results['championship'] = {'winner': None, 'matchup': (None,None), 'sim_details': None}
        else:
            # Determine home team based on regular season performance
            logger.debug(f"Championship Game: {east_champ_id} (East) vs {west_champ_id} (West)")
            actual_home_cup, actual_away_cup = self.determine_championship_home_team(
                east_champ_id,
                west_champ_id
            )

            winner_cup, home_score_cup, away_score_cup = self.simulate_match(
                actual_home_cup,
                actual_away_cup,
                is_neutral_site=False
            )
            results['championship'] = {
                'matchup': tuple(sorted((east_champ_id, west_champ_id))),
                'sim_details': {
                    'home': actual_home_cup,
                    'away': actual_away_cup,
                    'home_score': home_score_cup,
                    'away_score': away_score_cup,
                    'home_field_reason': 'regular_season_record'
                },
                'winner': winner_cup,
            }
            if winner_cup:
                logger.info(f"MLSNP Champion: {winner_cup}")
            else:
                logger.error("MLSNP Championship could not be determined.")
        
        return results
    
    def run_playoff_simulations(self, eastern_seeds: Dict[str, int], 
                               western_seeds: Dict[str, int],
                               n_simulations: int = 10000) -> Dict:
        """
        Run multiple playoff simulations and aggregate results.
        
        Args:
            eastern_seeds: Dictionary mapping team_id to seed (1-8) for Eastern Conference
            western_seeds: Dictionary mapping team_id to seed (1-8) for Western Conference
            n_simulations: Number of simulations to run
            
        Returns:
            Dictionary with aggregated results and probabilities
        """
        # Initialize tracking
        all_teams = list(eastern_seeds.keys()) + list(western_seeds.keys())
        
        results = {
            'team_probabilities': {
                team: {
                    'round1_win': 0,
                    'round2_win': 0,
                    'conf_final_win': 0,
                    'championship_win': 0
                } for team in all_teams
            },
            'matchup_frequency': defaultdict(int),
            'opponent_selection_frequency': defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        }
        
        # Run simulations
        logger.info(f"Starting {n_simulations} full playoff bracket simulations...")

        for i in tqdm(range(n_simulations), desc="Simulating playoffs", disable=None):
            # Log progress sparsely for very large N to avoid flooding logs
            log_interval = n_simulations // 10 if n_simulations >= 1000 else (n_simulations // 5 if n_simulations >=100 else 1)
            if log_interval > 0 and (i + 1) % log_interval == 0:
                 logger.debug(f"Running playoff simulation {i + 1}/{n_simulations}...")
            elif n_simulations < 20 and (i+1) % 1 == 0: # Log every sim if very few
                 logger.debug(f"Running playoff simulation {i + 1}/{n_simulations}...")

            sim_result = self.simulate_single_playoff(eastern_seeds, western_seeds)
            
            # Track results for each conference
            for conf in ['eastern', 'western']:
                # Round 1
                for match in sim_result[conf]['round1']:
                    winner = match['winner']
                    results['team_probabilities'][winner]['round1_win'] += 1
                    
                    # Track matchup frequency
                    matchup = tuple(sorted(match['matchup']))
                    results['matchup_frequency'][matchup] += 1
                
                # Round 2
                for match in sim_result[conf]['round2']:
                    winner = match['winner']
                    results['team_probabilities'][winner]['round2_win'] += 1
                
                # Conference Final
                conf_winner = sim_result[conf]['final']['winner']
                results['team_probabilities'][conf_winner]['conf_final_win'] += 1
            
            # Championship
            champion = sim_result.get('championship', {}).get('winner')
            if champion:
                results['team_probabilities'][champion]['championship_win'] += 1

            # Aggregate opponent selection frequency
            for conf in ['eastern', 'western']:
                current_conf_results = sim_result.get(conf, {})
                if current_conf_results and 'round1_selection_details' in current_conf_results:
                    for selection_detail in sim_result[conf]['round1_selection_details']:
                        selector = selection_detail['selector_id']
                        selected = selection_detail['selected_opponent_id']
                        method = selection_detail['method']
                        results['opponent_selection_frequency'][selector][selected][method] = \
                            results['opponent_selection_frequency'][selector][selected].get(method, 0) + 1

                if current_conf_results and 'round2_selection_details' in current_conf_results:
                    for selection_detail in current_conf_results['round2_selection_details']:
                        selector = selection_detail['selector_id']
                        selected = selection_detail['selected_opponent_id']
                        method = selection_detail['method']
                        results['opponent_selection_frequency'][selector][selected][method] = \
                            results['opponent_selection_frequency'][selector][selected].get(method, 0) + 1
        
        # Convert counts to probabilities
        if n_simulations > 0:
            for team in all_teams:
                for stage in results['team_probabilities'][team]:
                    results['team_probabilities'][team][stage] /= n_simulations
                    results['team_probabilities'][team][stage] *= 100  # Convert to percentage
        else:
            logger.warning("n_simulations was 0 for run_playoff_simulations, probabilities will be zero.")

        logger.info(f"Finished all {n_simulations} playoff simulations.")
        
        # Add summary statistics
        results['summary'] = {
            'n_simulations': n_simulations,
            'eastern_seeds': eastern_seeds,
            'western_seeds': western_seeds,
            'timestamp': datetime.now().isoformat()
        }
        
        return results