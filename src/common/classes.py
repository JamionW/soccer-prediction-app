from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from typing import ClassVar

class SimulationRequest(BaseModel):
    conference: str  # "eastern" or "western"
    n_simulations: int = 25000
    include_playoffs: bool = False
    simulation_preset: Optional[str] = "standard"
    SIMULATION_PRESETS: ClassVar[Dict[str, Dict[str, any]]] = {
        "quick": {
            "count": 1000,
            "description": "Quick estimate (~3 seconds)",
            "accuracy": "±3% margin of error"
        },
        "standard": {
            "count": 25000,
            "description": "Professional accuracy (~1 minute)",
            "accuracy": "±0.6% margin of error"
        },
        "detailed": {
            "count": 50000,
            "description": "High precision (~2 minutes)",
            "accuracy": "±0.4% margin of error"
        },
        "research": {
            "count": 100000,
            "description": "Maximum precision (~4 minutes)",
            "accuracy": "±0.3% margin of error"
        }
    }
    
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
