from datetime import datetime

N_SIMULATIONS = 10000
SEASON = 2025
LEAGUE = "mlsnp"
HOME_ADVANTAGE_GOALS = 0.2
REGRESSION_WEIGHT = 0.3
MIN_GAMES_FOR_RELIABILITY = 5
FIXTURES_FILE = "fox_sports_mls_fixtures_20250526_170453.json"

# Season boundaries - games between these dates count for 2025 regular season
SEASON_START = datetime(2025, 3, 1)  # March 2025
SEASON_END = datetime(2025, 10, 5)   # October 5, 2025

# Eastern Conference teams mapping - using exact data from ASA API
EASTERN_CONFERENCE_TEAMS = {
    "0Oq6Yad56D": "Columbus Crew 2",
    "0x5gb3bM7O": "Chicago Fire FC II", 
    "4JMAkpDqKg": "Toronto FC II",
    "9Yqdwg85vJ": "New York Red Bulls II",
    "a35reDLML6": "Crown Legacy FC",
    "gpMO1Pyqzy": "New England Revolution II",
    "gpMOXy25zy": "FC Cincinnati 2",
    "jYQJXkP5GR": "New York City FC II",
    "KPqjwN4Q6v": "Philadelphia Union II",
    "kRQaW3L5KZ": "Orlando City B",
    "NWMWJezQlz": "Carolina Core",
    "OlMlKDEQLz": "Huntsville City FC", 
    "raMyeZAMd2": "Chattanooga FC",
    "vzqowoZqap": "Inter Miami CF II",
    "wvq9jx1QWn": "Atlanta United 2"
}

# Western Conference teams mapping - using exact data from ASA API
WESTERN_CONFERENCE_TEAMS = {
    "2lqRX1AMr0": "Minnesota United FC 2",
    "2vQ14GKqrA": "Sporting Kansas City II",
    "2vQ1XzlqrA": "Los Angeles FC 2",
    "4wM4E4d5jB": "Ventura County FC",
    "7VqG1oWMvW": "Colorado Rapids 2",
    "BLMv6m3Mxe": "Real Monarchs",
    "eV5Dw4EMKn": "Austin FC II",
    "eVq3Z0D5WO": "St. Louis City SC 2",
    "gOMnJnOMwN": "Houston Dynamo FC 2",
    "KXMe8Z2Q64": "Tacoma Defiance",
    "ljqE94Vqx0": "North Texas SC",
    "N6MmWV0qEG": "Vancouver Whitecaps FC 2",
    "Oa5wDy8q14": "The Town FC",
    "zeQZe4DqKw": "Portland Timbers 2"
}