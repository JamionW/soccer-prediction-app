from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SimulationRequest(BaseModel):
    conference: str  # "eastern" or "western"
    n_simulations: int = 10000
    include_playoffs: bool = False
    
class PlayoffSeedingRequest(BaseModel):
    """Request model for custom playoff seeding"""
    eastern_seeds: Dict[int, str]  # {1: "team_id", 2: "team_id", ...}
    western_seeds: Dict[int, str]
    n_simulations: int = 10000

class TeamPerformance(BaseModel):
    """Model for team performance data"""
    team_id: str
    team_name: str
    current_points: int
    games_played: int
    playoff_probability: float
    average_final_rank: float
    
class SimulationResponse(BaseModel):
    """Response model for simulation results"""
    simulation_id: str
    conference: str
    status: str  # "running", "completed", "failed"
    regular_season_complete: bool
    playoff_simulation_available: bool
    results: Optional[List[TeamPerformance]]
    
class PlayoffBracket(BaseModel):
    """Model for playoff bracket structure"""
    round_1: List[Dict[str, Any]]
    round_2: List[Dict[str, Any]]
    conference_final: Dict[str, Any]
    championship: Optional[Dict[str, Any]]

class LoginCredentials(BaseModel):
    username_or_email: str
    password: str
