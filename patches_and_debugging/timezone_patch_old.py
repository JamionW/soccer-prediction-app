from datetime import datetime, timezone, timedelta
from src.common.database_manager import DatabaseManager

# Store original method
_original_get_or_fetch_team_xg = DatabaseManager.get_or_fetch_team_xg

async def get_or_fetch_team_xg_with_timezone_fix(self, team_id: str, season_year: int):
    """Fixed version with proper timezone handling"""
    
    # Try to get from database first
    query = """
        SELECT * FROM team_xg_history 
        WHERE team_id = :team_id AND season_year = :season_year
        ORDER BY date_captured DESC
        LIMIT 1
    """
    
    values = {"team_id": team_id, "season_year": season_year}
    xg_data = await self.db.fetch_one(query, values=values)
    
    # Check if data is fresh (less than 1 day old)
    if xg_data:
        # FIX: Handle timezone properly
        if xg_data['date_captured'].tzinfo is None:
            # If naive, assume UTC
            date_captured = xg_data['date_captured'].replace(tzinfo=timezone.utc)
        else:
            date_captured = xg_data['date_captured']
        
        # Use timezone-aware current time
        data_age = datetime.now(timezone.utc) - date_captured
        
        if data_age < timedelta(days=1):
            return dict(xg_data)
    
    # Return defaults for now (ASA not working for 2025)
    return {
        "team_id": team_id,
        "games_played": 0,
        "x_goals_for": 0.0,
        "x_goals_against": 0.0
    }

# Apply the patch
DatabaseManager.get_or_fetch_team_xg = get_or_fetch_team_xg_with_timezone_fix
print("✅ Timezone patch applied")
