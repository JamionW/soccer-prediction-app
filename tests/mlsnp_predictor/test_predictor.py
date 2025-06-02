import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import pandas as pd
import numpy as np
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor
from src.mlsnp_predictor import PlayoffPredictor
from src.mlsnp_predictor import constants
from src.common.utils import parse_game_date
import json
from collections import defaultdict
import logging
from datetime import datetime

# Disable most logging for tests to keep output clean, allow CRITICAL for actual test failures
logging.disable(logging.CRITICAL)

# Updated to 10 teams
SAMPLE_EASTERN_TEAM_IDS_REG = {f"ID{i}" for i in range(1, 11)} # ID1 to ID10
SAMPLE_TEAM_NAMES_REG = {f"ID{i}": f"Team {chr(64+i)}" for i in range(1, 11)} # Team A to Team J
SAMPLE_TEAM_NAMES_REG["IDW1"] = "Western Team One" # For filtering tests
SAMPLE_TEAM_NAMES_REG["IDW2"] = "Western Team Two" # For other tests if needed


TEST_SEASON_START = datetime(2025, 3, 1)
TEST_SEASON_END = datetime(2025, 10, 5)

class TestMLSNPRegSeasonPredictor(unittest.TestCase):
    @patch('itscalledsoccer.client.AmericanSoccerAnalysis')
    def setUp(self, MockAsaClient):
        self.mock_asa_client_instance = MockAsaClient.return_value
        self.predictor = MLSNPRegSeasonPredictor()
        self.predictor.eastern_teams = SAMPLE_EASTERN_TEAM_IDS_REG.copy()
        self.predictor.team_names = SAMPLE_TEAM_NAMES_REG.copy()
        self.predictor.league_avg_xgf = 1.2
        self.predictor.league_avg_xga = 1.2
        self.predictor.home_away_stats = defaultdict(lambda: {
            "home_goals_for": 0, "home_goals_against": 0,
            "away_goals_for": 0, "away_goals_against": 0,
            "home_games": 0, "away_games": 0
        })
        
        self.original_season_start = constants.SEASON_START
        self.original_season_end = constants.SEASON_END
        self.original_home_advantage = constants.HOME_ADVANTAGE_GOALS
        self.original_min_games = constants.MIN_GAMES_FOR_RELIABILITY
        self.original_regression_weight = constants.REGRESSION_WEIGHT
        self.original_n_simulations = constants.N_SIMULATIONS

        constants.SEASON_START = TEST_SEASON_START
        constants.SEASON_END = TEST_SEASON_END
        constants.HOME_ADVANTAGE_GOALS = 0.20
        constants.MIN_GAMES_FOR_RELIABILITY = 1
        constants.REGRESSION_WEIGHT = 0.5
        constants.N_SIMULATIONS = 2 # Default for tests unless overridden

    def tearDown(self):
        constants.SEASON_START = self.original_season_start
        constants.SEASON_END = self.original_season_end
        constants.HOME_ADVANTAGE_GOALS = self.original_home_advantage
        constants.MIN_GAMES_FOR_RELIABILITY = self.original_min_games
        constants.REGRESSION_WEIGHT = self.original_regression_weight
        constants.N_SIMULATIONS = self.original_n_simulations

    @patch('src.mlsnp_predictor.reg_season_predictor.safe_api_call')
    def test_get_games_data_client_success(self, mock_safe_api_call):
        mock_games = [{"id": "game1"}]
        mock_safe_api_call.return_value = mock_games
        games = self.predictor.get_games_data()
        self.assertEqual(games, mock_games)

    @patch('src.mlsnp_predictor.reg_season_predictor.safe_api_call', side_effect=[None, None])
    @patch('itscalledsoccer.client.AmericanSoccerAnalysis')
    def test_get_games_data_direct_api_success(self, MockAsaClientAgain, mock_safe_api_call_attempts):
        mock_client_fresh = MockAsaClientAgain.return_value
        mock_response = MagicMock(); mock_response.json.return_value = [{"id": "direct_game1"}]; mock_response.headers = {'Content-Type': 'application/json'}
        mock_client_fresh.session.get.return_value = mock_response
        self.predictor.client = mock_client_fresh
        games = self.predictor.get_games_data()
        self.assertEqual(games[0]['id'], "direct_game1")

    def test_filter_eastern_conference_games(self):
        self.predictor.eastern_teams = {"ID1", "ID2", "ID3"}
        all_games = [{"home_team_id": "ID1", "away_team_id": "ID2", "date_time_utc": "2025-07-01T12:00:00Z"}, {"home_team_id": "ID1", "away_team_id": "ID7", "date_time_utc": "2025-07-02T12:00:00Z"}, {"home_team_id": "ID1", "away_team_id": "ID3", "date_time_utc": "invalid-date"}]
        filtered = self.predictor.filter_eastern_conference_games(all_games)
        self.assertEqual(len(filtered), 2)

    @patch('src.mlsnp_predictor.reg_season_predictor.safe_api_call')
    def test_get_team_performance_data_xg_priority(self, mock_safe_api_call_xg):
        self.predictor.eastern_teams = {"ID1", "ID2", "ID3"}
        mock_xg_data = [{"team_id": "ID1", "count_games": 1, "xgoals_for": 1.5, "xgoals_against": 0.5}]
        mock_safe_api_call_xg.return_value = mock_xg_data
        games_data_for_goals = [{"home_team_id": "ID2", "away_team_id": "ID3", "home_score": 2, "away_score": 1, "status": "FT", "date_time_utc": "2025-07-01T00:00:00Z"}]
        performance = self.predictor.get_team_performance_data(games_data_for_goals)
        self.assertEqual(performance["ID1"]["data_type"], "xg"); self.assertAlmostEqual(performance["ID1"]["xgf_per_game"], 1.5)
        self.assertEqual(performance["ID2"]["data_type"], "goals"); self.assertAlmostEqual(performance["ID2"]["xgf_per_game"], 2.0)

    def test_calculate_league_averages(self):
        team_performance = {"ID1": {"xgf_per_game": 1.6, "xga_per_game": 0.8, "games": 10}, "ID2": {"xgf_per_game": 1.4, "xga_per_game": 1.2, "games": 10}}
        self.predictor.calculate_league_averages(team_performance)
        self.assertAlmostEqual(self.predictor.league_avg_xgf, 1.5)

    @patch.object(MLSNPRegSeasonPredictor, 'get_games_data')
    @patch.object(MLSNPRegSeasonPredictor, 'load_fixtures_from_file')
    @patch('src.mlsnp_predictor.reg_season_predictor.safe_api_call')
    def test_run_simulations_flow(self, mock_safe_api_xg, mock_load_fixtures, mock_get_games):
        mock_get_games.return_value = [{"home_team_id": "ID1", "away_team_id": "ID2", "home_score": 2, "away_score": 1, "status": "FT", "date_time_utc": "2025-07-01T00:00:00Z"}]
        mock_load_fixtures.return_value = [{"home_team_id": "ID1", "away_team_id": "ID3", "date": "2025-08-01"}]
        mock_safe_api_xg.return_value = [{"team_id": tid, "count_games": 1, "xgoals_for": 1.5, "xgoals_against": 1.0} for tid in self.predictor.eastern_teams]
        summary_df, _, _, qual_data = self.predictor.run_simulations(n_simulations=1)
        self.assertIsInstance(summary_df, pd.DataFrame); self.assertEqual(len(summary_df), len(self.predictor.eastern_teams))
        self.assertIn("Playoff Qualification %", summary_df.columns); self.assertEqual(len(qual_data), len(self.predictor.eastern_teams))

    @patch.object(MLSNPRegSeasonPredictor, 'calculate_expected_goals')
    @patch.object(MLSNPRegSeasonPredictor, 'simulate_match_with_shootout')
    def test_simulate_remaining_season(self, mock_simulate_match, mock_calculate_xg):
        self.predictor.eastern_teams = {"ID1", "ID2"}; self.predictor.team_names = {"ID1": "Team 1", "ID2": "Team 2"}
        self.predictor.home_away_stats = defaultdict(lambda: dict.fromkeys(self.predictor.home_away_stats.default_factory().keys(), 0))
        current_standings = {"ID1": {"points": 3, "wins": 1, "losses": 0, "draws":0, "shootout_wins":0, "goal_difference": 2, "goals_for":3, "goals_against":1, "games_played":1, "name": "ID1", "team_id": "ID1"},
                             "ID2": {"points": 0, "wins": 0, "losses": 1, "draws":0, "shootout_wins":0, "goal_difference": -2, "goals_for":1, "goals_against":3, "games_played":1, "name": "ID2","team_id": "ID2"}}
        self.predictor.home_away_stats["ID1"]["home_goals_for"] = 3; self.predictor.home_away_stats["ID1"]["home_goals_against"] = 1
        self.predictor.home_away_stats["ID2"]["away_goals_for"] = 1; self.predictor.home_away_stats["ID2"]["away_goals_against"] = 3
        remaining_fixtures = [{"home_team_id": "ID1", "away_team_id": "ID2"}]
        team_performance = { "ID1": {"xgf_per_game": 1.5, "xga_per_game": 1.0, "games": 1}, "ID2": {"xgf_per_game": 1.0, "xga_per_game": 1.5, "games": 1}}
        mock_calculate_xg.return_value = (1.6, 1.1); mock_simulate_match.return_value = (2, 1, False, False)
        final_standings = self.predictor.simulate_remaining_season(current_standings, remaining_fixtures, team_performance)
        self.assertEqual(final_standings["ID1"]["points"], 6)

    def test_calculate_playoff_qualification(self):
        self.predictor.eastern_teams = {"ID1", "ID2", "ID3"}
        simulation_results = { "ID1": [1, 1, 2], "ID2": [8, 9, 10], "ID3": [9, 10, 11] }
        current_standings = {tid: {} for tid in simulation_results.keys()}
        remaining_fixtures = [{"home_team_id": "ID1", "away_team_id": "ID2"}]
        qualification_data = self.predictor.calculate_playoff_qualification(simulation_results, current_standings, remaining_fixtures)
        self.assertAlmostEqual(qualification_data["ID1"]["playoff_probability"], 100.0)
        self.assertEqual(qualification_data["ID1"]["status"], "x-")

    def test_calculate_expected_goals_basic(self):
        self.predictor.league_avg_xgf = 1.25; self.predictor.league_avg_xga = 1.25
        team_performance = {"home_id": {"xgf_per_game": 1.5, "xga_per_game": 1.0, "games": 2}, "away_id": {"xgf_per_game": 1.1, "xga_per_game": 1.3, "games": 2}}
        home_xg, away_xg = self.predictor.calculate_expected_goals("home_id", "away_id", team_performance)
        self.assertAlmostEqual(home_xg, 1.680375, places=5)
        self.assertAlmostEqual(away_xg, 0.966875, places=5)

    def test_apply_tiebreakers_all_factors(self):
        def get_stats(pts, w, gd, gf, sow, agf=0, aga=0, hgf=0, hga=0): return {"points": pts, "wins": w, "goal_difference": gd, "goals_for": gf, "shootout_wins": sow, "_away_goals_for": agf, "_away_goals_against": aga, "_home_goals_for": hgf, "_home_goals_against": hga}
        standings = {"T1": get_stats(10,3,5,10,0,agf=5,aga=2,hgf=5,hga=3), "T2": get_stats(10,3,5,10,0,agf=6,aga=2,hgf=4,hga=3), "T3": get_stats(12,4,6,12,0), "T4": get_stats(10,3,5,10,1), "T5": get_stats(10,3,5,10,0,agf=5,aga=2,hgf=6,hga=2)}
        ranked = self.predictor.apply_tiebreakers(standings)
        self.assertEqual([r[0] for r in ranked], ["T3", "T4", "T2", "T5", "T1"])

class TestPlayoffPredictor(unittest.TestCase):
    def setUp(self):
        self.sample_games_data = [{"home_team_id": "E1", "away_team_id": "E2", "home_score": 2, "away_score": 1, "status": "FT", "date_time_utc": "2025-07-01T00:00:00Z"}, {"home_team_id": "E3", "away_team_id": "E4", "home_score": 1, "away_score": 1, "status": "FT", "date_time_utc": "2025-07-01T00:00:00Z"}, {"home_team_id": "E1", "away_team_id": "E3", "home_score": 3, "away_score": 0, "status": "FT", "date_time_utc": "2025-07-05T00:00:00Z"}, {"home_team_id": "E2", "away_team_id": "E4", "home_score": 2, "away_score": 2, "status": "FT", "date_time_utc": "2025-07-05T00:00:00Z"}]
        self.sample_team_performance = { "E1": {"xgf_per_game": 1.8, "xga_per_game": 0.8, "games": 28}, "E2": {"xgf_per_game": 1.5, "xga_per_game": 1.2, "games": 28}, "E3": {"xgf_per_game": 1.2, "xga_per_game": 1.5, "games": 28}, "E4": {"xgf_per_game": 1.0, "xga_per_game": 1.8, "games": 28}, "E5": {"xgf_per_game": 1.4, "xga_per_game": 1.4, "games": 28}, "E6": {"xgf_per_game": 1.3, "xga_per_game": 1.3, "games": 28}, "E7": {"xgf_per_game": 1.1, "xga_per_game": 1.1, "games": 28}, "E8": {"xgf_per_game": 0.9, "xga_per_game": 0.9, "games": 28}, "W1": {"xgf_per_game": 1.7, "xga_per_game": 0.9, "games": 28}, "W2": {"xgf_per_game": 1.4, "xga_per_game": 1.1, "games": 28}}
        self.league_avg_xgf = 1.35; self.league_avg_xga = 1.35
        self.playoff_predictor = PlayoffPredictor(games_data=self.sample_games_data, team_performance=self.sample_team_performance, league_avg_xgf=self.league_avg_xgf, league_avg_xga=self.league_avg_xga)

    def test_calculate_head_to_head_rating(self):
        self.assertAlmostEqual(self.playoff_predictor.calculate_head_to_head_rating("E1", "E2"), 1.0)
        self.assertAlmostEqual(self.playoff_predictor.calculate_head_to_head_rating("E1", "E4"), 1.8)

    def test_calculate_recent_form(self):
        self.assertAlmostEqual(self.playoff_predictor.calculate_recent_form("E1", n_games=2), 1.0)
        self.assertAlmostEqual(self.playoff_predictor.calculate_recent_form("E4", n_games=2), 1/3, places=5)

    @patch('numpy.random.choice')
    def test_select_opponent(self, mock_np_choice):
        mock_np_choice.return_value = 'best_h2h'
        seeding = {"E1": 1, "E2": 5, "E4": 7}
        selected_opp, method = self.playoff_predictor.select_opponent("E1", ["E2", "E4"], seeding)
        self.assertEqual(selected_opp, "E4"); self.assertEqual(method, "best_h2h")

    @patch('numpy.random.poisson')
    def test_simulate_match_home_win_and_hfa(self, mock_poisson):
        mock_poisson.side_effect = [2, 1]
        # Expected xG for E1 (home, vs E2): 1.8 * (1.2/1.35) * 1.1 (HFA) = 1.6 * 1.1 = 1.76
        # Expected xG for E2 (away, vs E1): 1.5 * (0.8/1.35) = 0.888...
        # Mock poisson based on these adjusted expectations if needed, or just outcome.
        with patch.object(self.playoff_predictor, 'league_avg_xga', 1.35): # Ensure it uses the intended value
             winner, score1, score2 = self.playoff_predictor.simulate_match("E1", "E2", is_neutral_site=False)
        self.assertEqual(winner, "E1"); self.assertEqual(score1, 2); self.assertEqual(score2, 1)
        # Check if xG values for poisson were scaled due to HFA (hard to check directly without more mocking)

    @patch('numpy.random.poisson', side_effect=[2,2])
    @patch('numpy.random.random', return_value=0.4)
    def test_simulate_match_shootout_win(self, mock_random, mock_poisson):
        winner, score1, score2 = self.playoff_predictor.simulate_match("E1", "E2", is_neutral_site=True)
        self.assertEqual(winner, "E1"); self.assertEqual(score1, 2); self.assertEqual(score2, 2)

    def test_simulate_single_playoff_structure(self):
        eastern_seeds = {f"E{i+1}":i+1 for i in range(8)}
        western_seeds = {f"W{i+1}":i+1 for i in range(8)}
        for i in range(8):
            self.sample_team_performance.setdefault(f"E{i+1}", {"xgf_per_game":1.2, "xga_per_game":1.2, "games":10})
            self.sample_team_performance.setdefault(f"W{i+1}", {"xgf_per_game":1.2, "xga_per_game":1.2, "games":10})
        self.playoff_predictor.team_performance = self.sample_team_performance

        results = self.playoff_predictor.simulate_single_playoff(eastern_seeds, western_seeds)
        self.assertIn('championship', results); self.assertTrue(len(results['eastern']['round1']) == 4)

    def test_run_playoff_simulations_aggregation(self):
        eastern_seeds = {f"E{i+1}":i+1 for i in range(8)}
        western_seeds = {f"W{i+1}":i+1 for i in range(8)}
        for i in range(8):
            self.sample_team_performance.setdefault(f"E{i+1}", {"xgf_per_game":1.2, "xga_per_game":1.2, "games":10})
            self.sample_team_performance.setdefault(f"W{i+1}", {"xgf_per_game":1.2, "xga_per_game":1.2, "games":10})
        self.playoff_predictor.team_performance = self.sample_team_performance

        mock_single_run = {'eastern': {'round1': [{'winner': 'E1', 'matchup':('E1','E8'), 'sim_details':{}}, {'winner': 'E2', 'matchup':('E2','E7'), 'sim_details':{}}, {'winner': 'E3', 'matchup':('E3','E6'), 'sim_details':{}}, {'winner': 'E4', 'matchup':('E4','E5'), 'sim_details':{}}], 'round2': [{'winner': 'E1', 'matchup':('E1','E4'), 'sim_details':{}}, {'winner': 'E2', 'matchup':('E2','E3'), 'sim_details':{}}], 'final': {'winner': 'E1', 'matchup':('E1','E2'), 'sim_details':{}}, 'round1_selection_details': [{'selector_id':'E1', 'selected_opponent_id':'E8', 'method':'best_h2h'}], 'round2_selection_details': []}, 'western': {'round1': [{'winner': 'W1', 'matchup':('W1','W8'), 'sim_details':{}}, {'winner': 'W2', 'matchup':('W2','W7'), 'sim_details':{}}, {'winner': 'W3', 'matchup':('W3','W6'), 'sim_details':{}}, {'winner': 'W4', 'matchup':('W4','W5'), 'sim_details':{}}], 'round2': [{'winner': 'W1', 'matchup':('W1','W4'), 'sim_details':{}}, {'winner': 'W2', 'matchup':('W2','W3'), 'sim_details':{}}], 'final': {'winner': 'W1', 'matchup':('W1','W2'), 'sim_details':{}}, 'round1_selection_details': [], 'round2_selection_details': []}, 'championship': {'winner': 'E1', 'matchup':('E1','W1'), 'sim_details':{}}}
        with patch.object(self.playoff_predictor, 'simulate_single_playoff', return_value=mock_single_run):
            results = self.playoff_predictor.run_playoff_simulations(eastern_seeds, western_seeds, n_simulations=2)
        self.assertAlmostEqual(results['team_probabilities']['E1']['championship_win'], 100.0)
        self.assertEqual(results['matchup_frequency'][tuple(sorted(('E1','E8')))], 2)
        self.assertEqual(results['opponent_selection_frequency']['E1']['E8']['best_h2h'], 2)

if __name__ == '__main__':
    unittest.main()
