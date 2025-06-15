import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.mlsnp_predictor.playoff_predictor import MLSNPPlayoffPredictor
import logging

# Disable most logging for tests to keep output clean
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

# --- Pytest Fixtures ---

@pytest.fixture
def sample_games_data_playoffs():
    games = [
        {"game_id": "g1", "home_team_id": "E1", "away_team_id": "E2", "home_score": 2, "away_score": 1, "status": "final", "date_time_utc": "2025-07-01T00:00:00Z", "went_to_shootout": False},
        {"game_id": "g2", "home_team_id": "E3", "away_team_id": "E4", "home_score": 1, "away_score": 1, "status": "final", "date_time_utc": "2025-07-01T00:00:00Z", "went_to_shootout": True, "home_penalties": 3, "away_penalties": 2}, # E3 wins SO
        {"game_id": "g3", "home_team_id": "E1", "away_team_id": "E3", "home_score": 3, "away_score": 0, "status": "final", "date_time_utc": "2025-07-05T00:00:00Z", "went_to_shootout": False},
        {"game_id": "g4", "home_team_id": "E2", "away_team_id": "E4", "home_score": 2, "away_score": 2, "status": "final", "date_time_utc": "2025-07-05T00:00:00Z", "went_to_shootout": True, "home_penalties": 4, "away_penalties": 5}, # E4 wins SO
    ]
    # Ensure 'status' is one of the expected values for H2H/Form calcs
    for game in games:
        game['status'] = 'final'
    return games

@pytest.fixture
def sample_team_performance_playoffs():
    return {
        "E1": {"xgf_per_game": 1.8, "xga_per_game": 0.8, "games_played": 28},
        "E2": {"xgf_per_game": 1.5, "xga_per_game": 1.2, "games_played": 28},
        "E3": {"xgf_per_game": 1.2, "xga_per_game": 1.5, "games_played": 28},
        "E4": {"xgf_per_game": 1.0, "xga_per_game": 1.8, "games_played": 28},
        "E5": {"xgf_per_game": 1.4, "xga_per_game": 1.4, "games_played": 28},
        "E6": {"xgf_per_game": 1.3, "xga_per_game": 1.3, "games_played": 28},
        "E7": {"xgf_per_game": 1.1, "xga_per_game": 1.1, "games_played": 28},
        "E8": {"xgf_per_game": 0.9, "xga_per_game": 0.9, "games_played": 28},
        "W1": {"xgf_per_game": 1.7, "xga_per_game": 0.9, "games_played": 28},
        "W2": {"xgf_per_game": 1.4, "xga_per_game": 1.1, "games_played": 28},
    }

@pytest.fixture
def sample_regular_season_records_playoffs():
    return {
        "eastern": {
            "E1": {"average_final_points": 60, "wins": 18, "goal_difference": 30, "goals_for": 50, "team_name": "Team E1"},
            "E2": {"average_final_points": 50, "wins": 15, "goal_difference": 10, "goals_for": 40, "team_name": "Team E2"},
             "E3": {"average_final_points": 45, "wins": 12, "goal_difference": 5, "goals_for": 35, "team_name": "Team E3"},
            "E4": {"average_final_points": 40, "wins": 10, "goal_difference": 0, "goals_for": 30, "team_name": "Team E4"},
        },
        "western": {
            "W1": {"average_final_points": 62, "wins": 19, "goal_difference": 35, "goals_for": 55, "team_name": "Team W1"},
            "W2": {"average_final_points": 50, "wins": 15, "goal_difference": 10, "goals_for": 40, "team_name": "Team W2"},
        }
    }

@pytest.fixture
def playoff_predictor_instance(sample_games_data_playoffs, sample_team_performance_playoffs, sample_regular_season_records_playoffs):
    return MLSNPPlayoffPredictor(
        games_data=sample_games_data_playoffs,
        team_performance=sample_team_performance_playoffs,
        league_avg_xgf=1.35,
        league_avg_xga=1.35,
        regular_season_records=sample_regular_season_records_playoffs
    )

# --- Test Cases ---

def test_determine_championship_home_team_points(playoff_predictor_instance):
    # W1 has more points than E1
    home, away = playoff_predictor_instance.determine_championship_home_team("E1", "W1")
    assert home == "W1"
    assert away == "E1"

def test_determine_championship_home_team_wins_tiebreak(sample_games_data_playoffs, sample_team_performance_playoffs):
    records = {
        "eastern": {"E1": {"average_final_points": 60, "wins": 18, "goal_difference": 30, "goals_for": 50}},
        "western": {"W1": {"average_final_points": 60, "wins": 19, "goal_difference": 20, "goals_for": 40}}, # W1 more wins
    }
    predictor = MLSNPPlayoffPredictor(sample_games_data_playoffs, sample_team_performance_playoffs, 1.35, 1.35, records)
    home, away = predictor.determine_championship_home_team("E1", "W1")
    assert home == "W1"

def test_determine_championship_home_team_gd_tiebreak(sample_games_data_playoffs, sample_team_performance_playoffs):
    records = {
        "eastern": {"E1": {"average_final_points": 60, "wins": 18, "goal_difference": 30, "goals_for": 50}},
        "western": {"W1": {"average_final_points": 60, "wins": 18, "goal_difference": 35, "goals_for": 40}}, # W1 better GD
    }
    predictor = MLSNPPlayoffPredictor(sample_games_data_playoffs, sample_team_performance_playoffs, 1.35, 1.35, records)
    home, away = predictor.determine_championship_home_team("E1", "W1")
    assert home == "W1"

def test_determine_championship_home_team_gf_tiebreak(sample_games_data_playoffs, sample_team_performance_playoffs):
    records = {
        "eastern": {"E1": {"average_final_points": 60, "wins": 18, "goal_difference": 30, "goals_for": 50}},
        "western": {"W1": {"average_final_points": 60, "wins": 18, "goal_difference": 30, "goals_for": 55}}, # W1 better GF
    }
    predictor = MLSNPPlayoffPredictor(sample_games_data_playoffs, sample_team_performance_playoffs, 1.35, 1.35, records)
    home, away = predictor.determine_championship_home_team("E1", "W1")
    assert home == "W1"

@patch('numpy.random.random')
def test_determine_championship_home_team_coin_flip(mock_random, sample_games_data_playoffs, sample_team_performance_playoffs):
    records = {
        "eastern": {"E1": {"average_final_points": 60, "wins": 18, "goal_difference": 30, "goals_for": 50}},
        "western": {"W1": {"average_final_points": 60, "wins": 18, "goal_difference": 30, "goals_for": 50}}, # Identical
    }
    predictor = MLSNPPlayoffPredictor(sample_games_data_playoffs, sample_team_performance_playoffs, 1.35, 1.35, records)
    mock_random.return_value = 0.4 # E1 wins coin flip
    home, away = predictor.determine_championship_home_team("E1", "W1")
    assert home == "E1"
    mock_random.return_value = 0.6 # W1 wins coin flip
    home, away = predictor.determine_championship_home_team("E1", "W1")
    assert home == "W1"

def test_calculate_head_to_head_rating(playoff_predictor_instance):
    # E1 vs E2: E1 won 2-1 (GD +1 for E1)
    assert playoff_predictor_instance.calculate_head_to_head_rating("E1", "E2") == pytest.approx(1.0)
    # E1 vs E4: No direct games, fallback to xG diff
    # E1 xG_diff = 1.8 - 0.8 = 1.0
    # E4 xG_diff = 1.0 - 1.8 = -0.8
    # Rating for E1 vs E4 = 1.0 - (-0.8) = 1.8
    assert playoff_predictor_instance.calculate_head_to_head_rating("E1", "E4") == pytest.approx(1.8)

def test_calculate_recent_form(playoff_predictor_instance):
    # E1: Game g3 (3-0 vs E3), Game g1 (2-1 vs E2). Points: 3+3=6. Max points = 6. Score=1.0
    # GD: (3-0)=3, (2-1)=1. Total GD = 4. Avg GD = 2. Adjustment = min(max(2 * 0.1, -0.2), 0.2) = 0.2
    # Final Form E1 = min(1.0 + 0.2, 1.0) = 1.0 (capped at 1.0)
    assert playoff_predictor_instance.calculate_recent_form("E1", n_games=2) == pytest.approx(1.0)

    # E4: Game g4 (2-2 vs E2, SO loss), Game g2 (1-1 vs E3, SO loss)
    # Points for E4: 1 (vs E2) + 1 (vs E3) = 2. Max points = 6. Score = 2/6 = 0.333...
    # GD for E4: (2-2)=0, (1-1)=0. Total GD = 0. Avg GD = 0. Adjustment = 0.
    # Final Form E4 = 0.333...
    assert playoff_predictor_instance.calculate_recent_form("E4", n_games=2) == pytest.approx(1/3, abs=1e-5)

@patch('numpy.random.choice')
def test_select_opponent_h2h(mock_np_choice, playoff_predictor_instance):
    mock_np_choice.return_value = 'best_h2h'
    seeding = {"E1": 1, "E2": 5, "E4": 7}
    # Mock underlying calculation methods to control their outputs for this test
    with patch.object(playoff_predictor_instance, 'calculate_head_to_head_rating', side_effect=lambda t1, t2: {"E1-E2": 1.0, "E1-E4": 1.8}[f"{t1}-{t2}"]):
        selected_opp, method = playoff_predictor_instance.select_opponent("E1", ["E2", "E4"], seeding)
        assert selected_opp == "E4" # E1 vs E4 H2H is 1.8, E1 vs E2 is 1.0
        assert method == "best_h2h"

@patch('numpy.random.choice')
def test_select_opponent_worst_form(mock_np_choice, playoff_predictor_instance):
    mock_np_choice.return_value = 'worst_form'
    seeding = {"E1": 1, "E2": 5, "E4": 7}
    # Mock form: E2 form = 0.8 (better), E4 form = 0.2 (worse)
    with patch.object(playoff_predictor_instance, 'calculate_recent_form', side_effect=lambda team_id: {"E2": 0.8, "E4": 0.2}[team_id]):
        selected_opp, method = playoff_predictor_instance.select_opponent("E1", ["E2", "E4"], seeding)
        assert selected_opp == "E4" # E4 has worse form
        assert method == "worst_form"

@patch('numpy.random.choice')
def test_select_opponent_lowest_seed(mock_np_choice, playoff_predictor_instance):
    mock_np_choice.return_value = 'lowest_seed'
    seeding = {"E1": 1, "E2": 5, "E4": 7} # E2 is seed 5, E4 is seed 7 (lower rank)
    # We want the opponent with the highest seed number (which means lowest rank)
    selected_opp, method = playoff_predictor_instance.select_opponent("E1", ["E2", "E4"], seeding)
    assert selected_opp == "E4" # E4 is seed 7, numerically higher (lower rank)
    assert method == "lowest_seed"


@patch('numpy.random.poisson')
def test_simulate_match_home_advantage(mock_poisson, playoff_predictor_instance):
    mock_poisson.side_effect = [2, 1] # Home scores 2, Away scores 1
    winner, home_score, away_score = playoff_predictor_instance.simulate_match("E1", "E2", is_neutral_site=False)
    assert winner == "E1"
    assert home_score == 2
    assert away_score == 1
    # To verify HFA was applied, we'd need to capture args to np.random.poisson
    # This requires more complex mocking or refactoring simulate_match to return expected goals.

@patch('numpy.random.poisson', side_effect=[2,2]) # Draw
@patch('numpy.random.random')
def test_simulate_match_shootout_probabilities(mock_random, mock_poisson, playoff_predictor_instance):
    # Home win in shootout
    mock_random.return_value = playoff_predictor_instance.HOME_TEAM_SHOOTOUT_WIN_PROB - 0.01
    winner, _, _ = playoff_predictor_instance.simulate_match("E1", "E2", is_neutral_site=False)
    assert winner == "E1"

    # Away win in shootout (Home loses)
    mock_random.return_value = playoff_predictor_instance.HOME_TEAM_SHOOTOUT_WIN_PROB + 0.01
    winner, _, _ = playoff_predictor_instance.simulate_match("E1", "E2", is_neutral_site=False)
    assert winner == "E2"

    # Neutral site - 50/50
    mock_random.return_value = playoff_predictor_instance.NEUTRAL_SITE_SHOOTOUT_WIN_PROB - 0.01 # Team1 (home_team_id) wins
    winner, _, _ = playoff_predictor_instance.simulate_match("E1", "E2", is_neutral_site=True)
    assert winner == "E1"

    mock_random.return_value = playoff_predictor_instance.NEUTRAL_SITE_SHOOTOUT_WIN_PROB + 0.01 # Team2 (away_team_id) wins
    winner, _, _ = playoff_predictor_instance.simulate_match("E1", "E2", is_neutral_site=True)
    assert winner == "E2"


def test_run_playoff_simulations_structure(playoff_predictor_instance, sample_team_performance_playoffs):
    eastern_seeds = {f"E{i}": i for i in range(1, 9)}
    western_seeds = {f"W{i}": i for i in range(1, 3)} # Smaller West for quicker test

    # Ensure all seeded teams exist in performance data for this specific test run
    current_team_perf = sample_team_performance_playoffs.copy()
    for i in range(1,9):
        if f"E{i}" not in current_team_perf: current_team_perf[f"E{i}"] = {"xgf_per_game":1.1, "xga_per_game":1.1, "games_played":10}
    for i in range(1,3):
        if f"W{i}" not in current_team_perf: current_team_perf[f"W{i}"] = {"xgf_per_game":1.1, "xga_per_game":1.1, "games_played":10}

    # Update predictor instance for this test
    playoff_predictor_instance.team_performance = current_team_perf

    num_sims = 3 # Very small number for structure testing
    results = playoff_predictor_instance.run_playoff_simulations(eastern_seeds, western_seeds, n_simulations=num_sims)

    assert 'team_probabilities' in results
    assert 'matchup_frequency' in results
    assert 'opponent_selection_frequency' in results
    assert 'summary' in results
    assert results['summary']['n_simulations'] == num_sims

    all_teams_in_seeds = list(eastern_seeds.keys()) + list(western_seeds.keys())
    for team_id in all_teams_in_seeds:
        assert team_id in results['team_probabilities']
        for stage_prob in results['team_probabilities'][team_id].values():
            assert 0 <= stage_prob <= 100
