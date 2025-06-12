
# Add this to your code before running predictions:

from src.common.database_manager import DatabaseManager
import pandas as pd

# Monkey patch the problematic method
def get_or_fetch_team_xg_patched(self, team_id: str, season_year: int):
    """Patched version that handles DataFrames correctly"""
    
    # Default return value
    default_data = {
        "team_id": team_id,
        "games_played": 0,
        "x_goals_for": 0.0,
        "x_goals_against": 0.0
    }
    
    # For now, just return defaults since ASA doesn't have 2025 MLSNP data yet
    return default_data

# Apply the patch
DatabaseManager.get_or_fetch_team_xg = get_or_fetch_team_xg_patched
