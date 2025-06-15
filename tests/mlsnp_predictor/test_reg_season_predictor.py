import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor
from collections import defaultdict
import logging
from datetime import datetime

# Disable most logging for tests to keep output clean
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

# --- Pytest Fixtures ---

@pytest.fixture
def basic_conference_teams():
    return {f"T{i}": f"Team {i}" for i in range(1, 5)} # T1, T2, T3, T4

@pytest.fixture
def sample_league_averages():
    return {"league_avg_xgf": 1.3, "league_avg_xga": 1.3}

@pytest.fixture
def sample_team_performance_data():
    return {
        "T1": {"x_goals_for": 1.5 * 10, "x_goals_against": 1.0 * 10, "games_played": 10}, # xGF 1.5, xGA 1.0
        "T2": {"x_goals_for": 1.2 * 10, "x_goals_against": 1.4 * 10, "games_played": 10}, # xGF 1.2, xGA 1.4
        "T3": {"goals_for": 1.0 * 5, "goals_against": 1.2 * 5, "games_played": 5},      # No xG, GF 1.0, GA 1.2
        "T4": {"games_played": 0}, # No games played, should use league avg strength
        "T5": {"x_goals_for": 2.0 * 2, "x_goals_against": 0.5 * 2, "games_played": 2},   # For _get_team_strength specific tests
    }

@pytest.fixture
def predictor_instance(basic_conference_teams, sample_team_performance_data, sample_league_averages):
    # Minimal games_data for initialization, specific tests will provide their own
    games_data = []
    return MLSNPRegSeasonPredictor(
        conference="eastern",
        conference_teams=basic_conference_teams,
        games_data=games_data,
        team_performance=sample_team_performance_data,
        league_averages=sample_league_averages
    )

# --- Test Cases ---

class TestMLSNPRegSeasonPredictor:

    def test_get_team_strength(self, basic_conference_teams, sample_team_performance_data, sample_league_averages):
        # Test with a team that has xG data
        predictor = MLSNPRegSeasonPredictor("test", {"T1": "Team 1"}, [], {"T1": sample_team_performance_data["T1"]}, sample_league_averages)
        att_T1, def_T1 = predictor._get_team_strength("T1")
        assert att_T1 == pytest.approx(1.5 / 1.3)
        assert def_T1 == pytest.approx(1.0 / 1.3)

        # Test with a team that only has goals data (no xG)
        predictor_goals_only = MLSNPRegSeasonPredictor("test", {"T3": "Team 3"}, [], {"T3": sample_team_performance_data["T3"]}, sample_league_averages)
        att_T3, def_T3 = predictor_goals_only._get_team_strength("T3")
        assert att_T3 == pytest.approx(1.0 / 1.3) # Uses goals_for / league_avg_xgf
        assert def_T3 == pytest.approx(1.2 / 1.3) # Uses goals_against / league_avg_xga

        # Test with a team that has no games played
        predictor_no_games = MLSNPRegSeasonPredictor("test", {"T4": "Team 4"}, [], {"T4": sample_team_performance_data["T4"]}, sample_league_averages)
        att_T4, def_T4 = predictor_no_games._get_team_strength("T4")
        assert att_T4 == 1.0  # Fallback to league average strength
        assert def_T4 == 1.0

        # Test with a team not in performance data
        att_TX, def_TX = predictor_no_games._get_team_strength("TX") # TX not in sample_team_performance_data
        assert att_TX == 1.0
        assert def_TX == 1.0

        # Test strength bounds (using T5 and modified league averages)
        extreme_league_avg = {"league_avg_xgf": 0.2, "league_avg_xga": 0.2} # Low league avg to make T5 appear very strong
        predictor_extreme = MLSNPRegSeasonPredictor("test", {"T5":"Team 5"}, [], {"T5": sample_team_performance_data["T5"]}, extreme_league_avg)
        att_T5, def_T5 = predictor_extreme._get_team_strength("T5")
        # T5: xGF 2.0, xGA 0.5. Relative strength: Attack 2.0/0.2 = 10. Defense 0.5/0.2 = 2.5
        assert att_T5 == 5.0 # Capped at 5.0
        assert def_T5 == pytest.approx(2.5)

        extreme_league_avg_high = {"league_avg_xgf": 10.0, "league_avg_xga": 10.0} # High league avg to make T5 appear very weak
        predictor_extreme_low = MLSNPRegSeasonPredictor("test", {"T5":"Team 5"}, [], {"T5": sample_team_performance_data["T5"]}, extreme_league_avg_high)
        att_T5_low, def_T5_low = predictor_extreme_low._get_team_strength("T5")
        # T5: xGF 2.0, xGA 0.5. Relative strength: Attack 2.0/10.0 = 0.2. Defense 0.5/10.0 = 0.05
        assert att_T5_low == pytest.approx(0.2)
        assert def_T5_low == 0.1 # Floored at 0.1

    def test_calculate_current_standings_no_games(self, basic_conference_teams, sample_team_performance_data, sample_league_averages):
        predictor = MLSNPRegSeasonPredictor(
            conference="eastern",
            conference_teams=basic_conference_teams,
            games_data=[], # No games
            team_performance=sample_team_performance_data,
            league_averages=sample_league_averages
        )
        standings = predictor._calculate_current_standings()
        for team_id in basic_conference_teams:
            assert standings[team_id]["points"] == 0
            assert standings[team_id]["games_played"] == 0
            assert standings[team_id]["name"] == basic_conference_teams[team_id]

    def test_calculate_current_standings_completed_games(self, basic_conference_teams, sample_team_performance_data, sample_league_averages):
        games_data = [
            {"game_id": "g1", "home_team_id": "T1", "away_team_id": "T2", "home_score": 2, "away_score": 1, "is_completed": True, "went_to_shootout": False},
            {"game_id": "g2", "home_team_id": "T3", "away_team_id": "T4", "home_score": 1, "away_score": 1, "is_completed": True, "went_to_shootout": True, "home_penalties": 3, "away_penalties": 2}, # T3 wins SO
            {"game_id": "g3", "home_team_id": "T1", "away_team_id": "T3", "home_score": 0, "away_score": 0, "is_completed": True, "went_to_shootout": True, "home_penalties": 1, "away_penalties": 3}, # T3 wins SO
            {"game_id": "g4", "home_team_id": "T2", "away_team_id": "T4", "home_score": 3, "away_score": 2, "is_completed": True, "went_to_shootout": False},
            {"game_id": "g5", "home_team_id": "T1", "away_team_id": "T4", "home_score": 1, "away_score": 1, "is_completed": False, "went_to_shootout": False}, # Incomplete
             # Game involving a non-conference team (should be ignored if teams are filtered before this method)
            {"game_id": "g6", "home_team_id": "T1", "away_team_id": "TX", "home_score": 5, "away_score": 0, "is_completed": True, "went_to_shootout": False},
        ]
        predictor = MLSNPRegSeasonPredictor(
            conference="eastern",
            conference_teams=basic_conference_teams, # T1, T2, T3, T4
            games_data=games_data,
            team_performance=sample_team_performance_data,
            league_averages=sample_league_averages
        )
        standings = predictor.current_standings # _calculate_current_standings is called in __init__

        # T1: Win vs T2 (3pts), SO Loss vs T3 (1pt) = 4 pts
        assert standings["T1"]["points"] == 4
        assert standings["T1"]["games_played"] == 2
        assert standings["T1"]["wins"] == 1
        assert standings["T1"]["losses"] == 0 # SO loss is a draw in W/L record
        assert standings["T1"]["draws"] == 1
        assert standings["T1"]["shootout_wins"] == 0
        assert standings["T1"]["goals_for"] == 2 # 2-1, 0-0
        assert standings["T1"]["goals_against"] == 1
        assert standings["T1"]["goal_difference"] == 1

        # T2: Loss vs T1 (0pts), Win vs T4 (3pts) = 3 pts
        assert standings["T2"]["points"] == 3
        assert standings["T2"]["games_played"] == 2
        assert standings["T2"]["wins"] == 1
        assert standings["T2"]["losses"] == 1
        assert standings["T2"]["draws"] == 0
        assert standings["T2"]["shootout_wins"] == 0
        assert standings["T2"]["goals_for"] == 4 # 1-2, 3-2
        assert standings["T2"]["goals_against"] == 4
        assert standings["T2"]["goal_difference"] == 0

        # T3: SO Win vs T4 (2pts), SO Win vs T1 (2pts) = 4 pts
        assert standings["T3"]["points"] == 4
        assert standings["T3"]["games_played"] == 2
        assert standings["T3"]["wins"] == 0 # SO win is a draw in W/L record
        assert standings["T3"]["losses"] == 0
        assert standings["T3"]["draws"] == 2
        assert standings["T3"]["shootout_wins"] == 2
        assert standings["T3"]["goals_for"] == 1 # 1-1, 0-0
        assert standings["T3"]["goals_against"] == 1
        assert standings["T3"]["goal_difference"] == 0

        # T4: SO Loss vs T3 (1pt), Loss vs T2 (0pts) = 1 pt
        assert standings["T4"]["points"] == 1
        assert standings["T4"]["games_played"] == 2
        assert standings["T4"]["wins"] == 0
        assert standings["T4"]["losses"] == 1 # SO loss is a draw in W/L record
        assert standings["T4"]["draws"] == 1
        assert standings["T4"]["shootout_wins"] == 0
        assert standings["T4"]["goals_for"] == 3 # 1-1, 2-3
        assert standings["T4"]["goals_against"] == 4
        assert standings["T4"]["goal_difference"] == -1

    @patch('numpy.random.poisson')
    def test_simulate_game_scores_and_shootout(self, mock_poisson, predictor_instance):
        # Mock team strengths to be identical and equal to league average for simplicity
        with patch.object(predictor_instance, '_get_team_strength', return_value=(1.0, 1.0)):
            # Scenario 1: Home win
            mock_poisson.side_effect = [2, 1] # Home goals, Away goals
            game_details = {"home_team_id": "T1", "away_team_id": "T2"}
            h_goals, a_goals, went_to_shootout, home_wins_shootout = predictor_instance._simulate_game(game_details)
            assert h_goals == 2
            assert a_goals == 1
            assert not went_to_shootout
            assert not home_wins_shootout

            # Scenario 2: Draw, home wins shootout
            mock_poisson.side_effect = [2, 2] # Regulation draw
            with patch('numpy.random.rand', return_value=predictor_instance.HOME_SHOOTOUT_WIN_PROB - 0.1): # Ensure home wins SO
                h_goals, a_goals, went_to_shootout, home_wins_shootout = predictor_instance._simulate_game(game_details)
                assert h_goals == 2
                assert a_goals == 2
                assert went_to_shootout
                assert home_wins_shootout

            # Scenario 3: Draw, away wins shootout
            mock_poisson.side_effect = [1, 1] # Regulation draw
            with patch('numpy.random.rand', return_value=predictor_instance.HOME_SHOOTOUT_WIN_PROB + 0.1): # Ensure away wins SO
                h_goals, a_goals, went_to_shootout, home_wins_shootout = predictor_instance._simulate_game(game_details)
                assert h_goals == 1
                assert a_goals == 1
                assert went_to_shootout
                assert not home_wins_shootout

    def test_run_simulations_output_structure(self, basic_conference_teams, sample_team_performance_data, sample_league_averages):
        # Use a very small set of remaining games and simulations for speed
        remaining_games = [
            {"game_id": "g_rem1", "home_team_id": "T1", "away_team_id": "T2", "is_completed": False},
            {"game_id": "g_rem2", "home_team_id": "T3", "away_team_id": "T4", "is_completed": False},
        ]
        # Initial standings (empty, all games are future)
        initial_games = []

        predictor = MLSNPRegSeasonPredictor(
            conference="eastern",
            conference_teams=basic_conference_teams,
            games_data=initial_games + remaining_games, # All games are remaining
            team_performance=sample_team_performance_data,
            league_averages=sample_league_averages
        )

        n_sims = 5 # Small number of simulations
        summary_df, final_ranks_dist, _, qualification_data = predictor.run_simulations(n_sims)

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == len(basic_conference_teams)
        expected_cols = ['Team', '_team_id', 'Current Points', 'Games Played',
                           'Playoff Qualification %', 'Average Final Rank', 'Average Points']
        for col in expected_cols:
            assert col in summary_df.columns

        assert isinstance(final_ranks_dist, dict)
        assert len(final_ranks_dist) == len(basic_conference_teams)
        for team_id in basic_conference_teams:
            assert team_id in final_ranks_dist
            assert len(final_ranks_dist[team_id]) == n_sims # Each team has a list of ranks from N sims
            assert all(isinstance(rank, int) for rank in final_ranks_dist[team_id])

        assert isinstance(qualification_data, dict)
        assert len(qualification_data) == len(basic_conference_teams)
        for team_id in basic_conference_teams:
            assert team_id in qualification_data
            assert 'games_remaining' in qualification_data[team_id]
            # This assertion needs to be more robust if games can be filtered out.
            # For this test, all remaining_games are between conference teams.
            # games_in_conf_remaining = len([g for g in remaining_games if g["home_team_id"] in basic_conference_teams and g["away_team_id"] in basic_conference_teams])
            # assert qualification_data[team_id]['games_remaining'] == games_in_conf_remaining


        # Check playoff probabilities are within bounds
        assert summary_df['Playoff Qualification %'].between(0, 100).all()
        # Check current points and GP are 0 as no initial games were completed
        assert (summary_df['Current Points'] == 0).all()
        assert (summary_df['Games Played'] == 0).all()
