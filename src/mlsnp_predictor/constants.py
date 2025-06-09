from datetime import datetime

N_SIMULATIONS = 10000
SEASON = 2025
LEAGUE = "mlsnp"
HOME_ADVANTAGE_GOALS = 0.2
REGRESSION_WEIGHT = 0.3
MIN_GAMES_FOR_RELIABILITY = 5
FIXTURES_FILE = "output/fox_sports_mlsnp_fixtures.json"

# Season boundaries - games between these dates count for 2025 regular season
SEASON_START = datetime(2025, 3, 1)  # March 2025
SEASON_END = datetime(2025, 10, 5)   # October 5, 2025