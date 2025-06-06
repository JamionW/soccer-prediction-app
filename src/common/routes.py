from fastapi import APIRouter, Body, HTTPException, Request, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional, List, Dict, Any
import asyncio
import os
from ..auth_system import AuthManager
from .classes import SimulationRequest, PlayoffSeedingRequest, TeamPerformance, SimulationResponse, LoginCredentials
from .database import database
from .database_manager import DatabaseManager
from .utils import logger
from src.mlsnp_predictor import MLSNPRegSeasonPredictor, MLSNPPlayoffPredictor

router = APIRouter()

auth_manager = AuthManager(database)
db_manager = DatabaseManager(database)

# ==================== Public Routes (No Auth Required) ====================

@router.get("/")
async def root():
    return {
        "message": "Soccer Table and Playoff Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "Authentication": {
                "POST /auth/register": "Register new user",
                "POST /auth/login": "Login with credentials",
                "POST /auth/oauth/{provider}": "OAuth login (google/github)",
                "POST /auth/logout": "Logout current user"
            },
            "Simulations": {
                "POST /simulations/regular-season": "Run regular season simulation",
                "GET /simulations/{simulation_id}": "Get simulation results",
                "POST /simulations/playoffs": "Run playoff simulation",
                "GET /simulations/all": "View all public simulations",
                "GET /users/me/simulations": "View your simulations"
            },
            "Data": {
                "GET /teams/{conference}": "Get teams in a conference",
                "GET /data/seasons": "Get available seasons",
                "POST /data/load-historical/{year}": "Load historical data (admin)"
            }
        }
    }

@router.get("/teams/{conference}")
async def get_conference_teams(conference: str):
    """Get all teams in a conference with their current standings"""
    if conference not in ["eastern", "western"]:
        raise HTTPException(status_code=400, detail="Conference must be 'eastern' or 'western'")
    
    teams = await db_manager.get_conference_teams(conference, 2025)
    standings = await db_manager.calculate_and_store_standings(2025, conference)
    
    return {
        "conference": conference,
        "teams": teams,
        "standings": standings
    }

@router.get("/health")
async def health_check():
    """
    Returns a simple OK status without checking the database.
    This allows the health check to pass even if DATABASE_URL is not set
    in the health check environment, or if the database is temporarily unavailable.
    The actual database connection is checked during application startup in main.py.
    """
    return {
        "status": "healthy",
        "message": "Application is running.",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "railway_environment": os.getenv("RAILWAY_ENVIRONMENT")
    }


# ==================== Authentication Routes ====================

@router.post("/auth/register")
async def register(
    username: str = Body(...),
    email: str = Body(...),
    password: str = Body(...)
):
    """
    Register a new user with username and password.
    The password will be hashed using bcrypt before storage.
    Returns a JWT token for immediate login.
    """
    return await auth_manager.register_user(username, email, password)

@router.post("/auth/login")
async def login(credentials: LoginCredentials):
    """
    Login with username/email and password.
    Returns a JWT token that should be included in subsequent requests.
    """
    return await auth_manager.login_user(
        credentials.username_or_email,
        credentials.password
    )

@router.post("/auth/oauth/{provider}")
async def oauth_login(provider: str, code: str = Body(...)):
    """
    Handle OAuth login callback.
    The code is the authorization code from the OAuth provider.
    """
    return await auth_manager.oauth_login(provider, code)

@router.post("/auth/logout")
async def logout(current_user: Dict = Depends(auth_manager.validate_token)):
    """
    Logout the current user by revoking their token.
    """
    await auth_manager.logout(current_user.get('token'))
    return {"message": "Successfully logged out"}

# ==================== User Profile Routes ====================

@router.get("/users/me")
async def get_current_user(current_user: Dict = Depends(auth_manager.validate_token)):
    """Get the current authenticated user's profile."""
    return current_user

@router.get("/users/me/simulations")
async def get_user_simulations(
    current_user: Dict = Depends(auth_manager.validate_token),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get all simulations run by the current user.
    Supports pagination with skip and limit parameters.
    """
    query = """
        SELECT 
            pr.run_id,
            pr.run_date,
            pr.season_year,
            pr.n_simulations,
            pr.matchday,
            c.conf_name as conference,
            COUNT(ps.team_id) as teams_simulated,
            pr.is_stored
        FROM prediction_runs pr
        LEFT JOIN conference c ON pr.conference_id = c.conf_id
        LEFT JOIN prediction_summary ps ON pr.run_id = ps.run_id
        WHERE pr.user_id = :user_id
        GROUP BY pr.run_id, pr.run_date, pr.season_year, pr.n_simulations, 
                 pr.matchday, c.conf_name, pr.is_stored
        ORDER BY pr.run_date DESC
        LIMIT :limit OFFSET :skip
    """
    
    simulations = await database.fetch_all(
        query,
        values={
            "user_id": current_user['user_id'],
            "limit": limit,
            "skip": skip
        }
    )
    
    # Get total count for pagination
    count_query = """
        SELECT COUNT(*) as total 
        FROM prediction_runs 
        WHERE user_id = :user_id
    """
    total_result = await database.fetch_one(
        count_query,
        values={"user_id": current_user['user_id']}
    )
    
    return {
        "simulations": [dict(sim) for sim in simulations],
        "total": total_result['total'],
        "skip": skip,
        "limit": limit
    }

# ==================== Public Simulation Routes ====================

@router.get("/simulations/all")
async def get_all_simulations(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    conference: Optional[str] = Query(None, regex="^(eastern|western|both)$")
):
    """
    Get all simulations from all users (public view).    
    Can filter by conference and supports pagination.
    """
    where_clause = ""
    values = {"limit": limit, "skip": skip}
    
    if conference and conference != "both":
        conf_id = 1 if conference == "eastern" else 2
        where_clause = "WHERE pr.conference_id = :conf_id"
        values["conf_id"] = conf_id
    
    query = f"""
        SELECT 
            pr.run_id,
            pr.run_date,
            pr.season_year,
            pr.n_simulations,
            pr.matchday,
            c.conf_name as conference,
            u.username,
            COUNT(ps.team_id) as teams_simulated
        FROM prediction_runs pr
        LEFT JOIN conference c ON pr.conference_id = c.conf_id
        LEFT JOIN users u ON pr.user_id = u.user_id
        LEFT JOIN prediction_summary ps ON pr.run_id = ps.run_id
        {where_clause}
        GROUP BY pr.run_id, pr.run_date, pr.season_year, pr.n_simulations, 
                 pr.matchday, c.conf_name, u.username
        ORDER BY pr.run_date DESC
        LIMIT :limit OFFSET :skip
    """
    
    simulations = await database.fetch_all(query, values=values)
    
    # Get total count
    count_query = f"""
        SELECT COUNT(*) as total 
        FROM prediction_runs pr
        {where_clause}
    """
    total_result = await database.fetch_one(count_query, values=values)
    
    return {
        "simulations": [dict(sim) for sim in simulations],
        "total": total_result['total'],
        "skip": skip,
        "limit": limit
    }

# ==================== Simulation Execution Routes ====================

@router.post("/simulations/regular-season")
async def run_regular_season_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(auth_manager.validate_token)
):
    """
    Run a regular season simulation for the specified conference(s).
    
    This endpoint starts the simulation in the background and returns immediately.
    The simulation will:
    1. Check database for existing game data
    2. Fetch any missing data from ASA API
    3. Run Monte Carlo simulations
    4. Store results in the database
    """
    simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.conference}_{current_user['user_id']}"
    
    # Initialize the simulation status
    simulation_cache[simulation_id] = {
        "status": "running",
        "conference": request.conference,
        "regular_season_complete": False,
        "playoff_simulation_available": False,
        "results": None,
        "user_id": current_user['user_id']
    }
    
    # Run simulation in background
    background_tasks.add_task(
        run_simulation_task,
        simulation_id,
        request.conference,
        request.n_simulations,
        request.include_playoffs,
        current_user['user_id']
    )
    
    return SimulationResponse(
        simulation_id=simulation_id,
        conference=request.conference,
        status="running",
        regular_season_complete=False,
        playoff_simulation_available=False,
        results=None
    )

async def run_simulation_task(
    simulation_id: str,
    conference: str,
    n_simulations: int,
    include_playoffs: bool,
    user_id: int
):
    """
    Background task to run the actual simulation.
    """
    try:
        # First, ensure we have historical data loaded
        await ensure_historical_data_loaded()
        
        # Update any incomplete games that should be complete by now
        await db_manager.update_incomplete_games(2025)
        
        if conference == "both":
            # Calculate league-wide averages ONCE for both conferences to share
            league_averages = await calculate_league_averages()
            
            eastern_task = run_conference_simulation(
                "eastern", 
                n_simulations, 
                user_id,
                league_averages
            )
            western_task = run_conference_simulation(
                "western", 
                n_simulations, 
                user_id,
                league_averages
            )
            
            # Run both simulations in parallel
            eastern_results, western_results = await asyncio.gather(
                eastern_task, 
                western_task
            )
            
            # Store regular season records for championship home field advantage
            regular_season_records = {
                "eastern": extract_regular_season_records(eastern_results),
                "western": extract_regular_season_records(western_results)
            }
            
            # Combine results for display
            combined_results = {
                "eastern": eastern_results,
                "western": western_results,
                "regular_season_records": regular_season_records
            }
            
            simulation_cache[simulation_id].update({
                "status": "completed",
                "regular_season_complete": True,
                "playoff_simulation_available": include_playoffs,
                "results": combined_results
            })
            
            if include_playoffs:
                parent_run_id = eastern_results.get('run_id')  # Doesn't matter which conference we use to get the run_id
                
                eastern_seeds = {
                    team_id: rank 
                    for team_id, rank in eastern_results['top_8_teams'].items()
                }
                western_seeds = {
                    team_id: rank 
                    for team_id, rank in western_results['top_8_teams'].items()
                }
                
                playoff_results = await run_playoff_simulation_with_records(
                    eastern_seeds=eastern_seeds,
                    western_seeds=western_seeds,
                    regular_season_records=regular_season_records,
                    n_simulations=n_simulations,
                    user_id=user_id,
                    parent_run_id=parent_run_id
                )
                
                simulation_cache[simulation_id]["playoff_results"] = playoff_results
        
        else:
            # Run single conference (existing code)
            league_averages = await calculate_league_averages()
            results = await run_conference_simulation(
                conference, 
                n_simulations, 
                user_id,
                league_averages
            )
            
            simulation_cache[simulation_id].update({
                "status": "completed",
                "regular_season_complete": True,
                "playoff_simulation_available": False,
                "results": results
            })
        
    except Exception as e:
        logger.error(f"Simulation {simulation_id} failed: {str(e)}")
        simulation_cache[simulation_id]["status"] = "failed"
        simulation_cache[simulation_id]["error"] = str(e)


async def calculate_league_averages() -> Dict[str, float]:
    """
    Calculate league-wide averages for all teams.
    These will be shared between conference simulations.
    """
    # Get all teams' performance data
    all_teams_xg = await db_manager.db.fetch_all("""
        SELECT 
            team_id,
            x_goals_for,
            x_goals_against,
            games_played
        FROM team_xg_history
        WHERE season_year = 2025
        AND games_played > 0
        ORDER BY team_id, date_captured DESC
    """)
    
    # Group by team and take most recent
    team_latest = {}
    for row in all_teams_xg:
        if row['team_id'] not in team_latest:
            team_latest[row['team_id']] = row
    
    # Calculate weighted averages
    total_xgf = sum(t['x_goals_for'] for t in team_latest.values())
    total_xga = sum(t['x_goals_against'] for t in team_latest.values())
    total_games = sum(t['games_played'] for t in team_latest.values())
    
    if total_games > 0:
        league_avg_xgf = total_xgf / total_games
        league_avg_xga = total_xga / total_games
    else:
        # Fallback values
        league_avg_xgf = 1.2
        league_avg_xga = 1.2
    
    return {
        "league_avg_xgf": league_avg_xgf,
        "league_avg_xga": league_avg_xga,
        "total_teams": len(team_latest),
        "total_games": total_games
    }


async def run_conference_simulation(
    conference: str, 
    n_simulations: int, 
    user_id: int,
    league_averages: Dict[str, float]
) -> Dict:
    # Run simulation for a single conference with provided league averages.

    run_id = await db_manager.store_simulation_run(
        user_id=user_id,
        conference=conference,
        n_simulations=n_simulations,
        season_year=2025
    )
    
    # 1. FETCH: Get all data from DatabaseManager
    sim_data = await db_manager.get_data_for_simulation(conference, 2025)
    
    # 2. COMPUTE: Run simulation with fetched data
    predictor = MLSNPRegSeasonPredictor(
        conference=conference,
        conference_teams=sim_data["conference_teams"],
        games_data=sim_data["games_data"],
        team_performance=sim_data["team_performance"],
        league_averages=league_averages
    )
    
    summary_df, simulation_results, _, qualification_data = predictor.run_simulations(n_simulations)
    
    # 3. STORE: Save results to database
    await db_manager.store_simulation_results(
        run_id, summary_df, simulation_results, qualification_data
    )
    
    # 4. FORMAT: Prepare API response with ALL required fields
    team_performances = []
    top_8_teams = {}
    team_records = {}
    
    for _, row in summary_df.iterrows():
        team_id = row['_team_id']
        
        # Build team performance for API
        performance = TeamPerformance(
            team_id=team_id,
            team_name=row['Team'],
            current_points=row['Current Points'],
            games_played=row['Games Played'],
            playoff_probability=row['Playoff Qualification %'],
            average_final_rank=row['Average Final Rank']
        )
        team_performances.append(performance)
        
        # Build top_8_teams for playoffs
        if row['Average Final Rank'] <= 8:
            top_8_teams[team_id] = int(row['Average Final Rank'])
        
        # Build team_records for playoff home field advantage
        current_standing = predictor.current_standings.get(team_id, {})
        team_records[team_id] = {
            "team_name": row['Team'],
            "points": row['Current Points'],
            "average_final_points": row['Average Points'],
            "wins": current_standing.get('wins', 0),
            "goal_difference": current_standing.get('goal_difference', 0),
            "goals_for": current_standing.get('goals_for', 0)
        }
    
    return {
        "run_id": run_id,
        "conference": conference,
        "teams": team_performances,
        "top_8_teams": top_8_teams,
        "team_records": team_records,
        "league_averages_used": league_averages
    }


def extract_regular_season_records(conference_results: Dict) -> Dict[str, Dict]:
    """
    Extract regular season records for all teams in a conference.
    Used for determining home field advantage in playoffs.
    """
    return conference_results.get("team_records", {})

async def run_playoff_simulation_with_records(
    eastern_seeds: Dict[str, int],
    western_seeds: Dict[str, int],
    regular_season_records: Dict[str, Dict],
    n_simulations: int,
    user_id: int,
    parent_run_id: int = None  # Link to regular season simulation
) -> Dict:
    """
    Run playoff simulations with regular season records for home field advantage.
    This is an internal function, not a route handler.
    """
    try:
        # Get game data for team performance metrics
        games = await db_manager.get_games_for_season(2025)
        
        # Get team performance data
        team_performance = {}
        league_totals = {"xgf": 0, "xga": 0, "games": 0}
        
        for conference in ['eastern', 'western']:
            conference_teams = await db_manager.get_conference_teams(conference, 2025)
            for team_id in conference_teams:
                xg_data = await db_manager.get_or_fetch_team_xg(team_id, 2025)
                if xg_data['games_played'] > 0:
                    team_performance[team_id] = {
                        'xgf_per_game': xg_data['x_goals_for'] / xg_data['games_played'],
                        'xga_per_game': xg_data['x_goals_against'] / xg_data['games_played']
                    }
                    # Track for league averages
                    league_totals["xgf"] += xg_data['x_goals_for']
                    league_totals["xga"] += xg_data['x_goals_against']
                    league_totals["games"] += xg_data['games_played']
        
        # Calculate league averages
        league_avg_xgf = league_totals["xgf"] / league_totals["games"] if league_totals["games"] > 0 else 1.2
        league_avg_xga = league_totals["xga"] / league_totals["games"] if league_totals["games"] > 0 else 1.2
        
        # Initialize playoff predictor with regular season records
        playoff_predictor = MLSNPPlayoffPredictor(
            games_data=games,
            team_performance=team_performance,
            league_avg_xgf=league_avg_xgf,
            league_avg_xga=league_avg_xga,
            regular_season_records=regular_season_records
        )
        
        # Run simulations
        results = playoff_predictor.run_playoff_simulations(
            eastern_seeds=eastern_seeds,
            western_seeds=western_seeds,
            n_simulations=n_simulations
        )
        
        # Store playoff results in database
        playoff_run_id = await store_playoff_results(
            results=results,
            user_id=user_id,
            parent_run_id=parent_run_id,
            eastern_seeds=eastern_seeds,
            western_seeds=western_seeds,
            n_simulations=n_simulations
        )
        
        return {
            "playoff_run_id": playoff_run_id,
            "message": "Playoff simulation completed",
            "results": results,
            "championship_home_field": "Based on regular season record"
        }
        
    except Exception as e:
        logger.error(f"Playoff simulation failed: {str(e)}")
        raise

# ==================== Data Management Routes ====================

@router.post("/data/load-historical/{year}")
async def load_historical_data(
    year: int,
    current_user: Dict = Depends(auth_manager.validate_token)
):
    """
    Load historical season data from ASA API.
    
    This endpoint is admin-only and loads all games and statistics
    for the specified season.
    """
    if not current_user.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        await db_manager.load_historical_season(year)
        return {"message": f"Successfully loaded {year} season data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/seasons")
async def get_available_seasons():
    """Get list of seasons with data in the database."""
    query = """
        SELECT DISTINCT season_year, COUNT(*) as game_count
        FROM games
        GROUP BY season_year
        ORDER BY season_year DESC
    """
    
    seasons = await database.fetch_all(query)
    return [{"year": s['season_year'], "games": s['game_count']} for s in seasons]

async def store_playoff_results(
    results: Dict,
    user_id: int,
    parent_run_id: int,
    eastern_seeds: Dict[str, int],
    western_seeds: Dict[str, int],
    n_simulations: int
) -> int:
    """
    Store playoff simulation results in the database.
    Returns the playoff projection ID.
    """
    proj_query = """
        INSERT INTO playoff_projection (
            proj_date, season_year, parent_projection, proj_type,
            proj_name, n_simulations, user_id, created_at
        ) VALUES (
            NOW(), 2025, :parent_projection, 'simulation',
            :proj_name, :n_simulations, :user_id, NOW()
        ) RETURNING proj_id
    """
    
    proj_values = {
        "parent_projection": parent_run_id,
        "proj_name": f"Playoff Simulation - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "n_simulations": n_simulations,
        "user_id": user_id
    }
    
    proj_result = await db_manager.db.fetch_one(proj_query, values=proj_values)
    proj_id = proj_result['proj_id']
    
    # Store team-specific playoff results
    for team_id, probs in results['team_probabilities'].items():
        # Determine seed and conference
        seed = eastern_seeds.get(team_id) or western_seeds.get(team_id)
        
        summary_query = """
            INSERT INTO team_playoff_summary (
                proj_id, team_id, seed, r1_win_pct, r2_win_pct,
                conf_finals_win_pct, champion_pct
            ) VALUES (
                :proj_id, :team_id, :seed, :r1_win_pct, :r2_win_pct,
                :conf_finals_win_pct, :champion_pct
            )
        """
        
        summary_values = {
            "proj_id": proj_id,
            "team_id": team_id,
            "seed": seed,
            "r1_win_pct": probs['round1_win'],
            "r2_win_pct": probs['round2_win'],
            "conf_finals_win_pct": probs['conf_final_win'],
            "champion_pct": probs['championship_win']
        }
        
        await db_manager.db.execute(summary_query, values=summary_values)
    
    # Store projected field (the seeding used)
    for conf_name, seeds in [("Eastern", eastern_seeds), ("Western", western_seeds)]:
        for team_id, seed in seeds.items():
            field_query = """
                INSERT INTO projected_field (
                    proj_id, team_id, team_name, seed, avg_final_rank,
                    playoff_prob_pct, reg_season_points
                ) VALUES (
                    :proj_id, :team_id, :team_name, :seed, :seed,
                    100.0, :points
                )
            """
            
            # Get team name and points from regular season records
            team_record = results.get('regular_season_records', {}).get(conf_name.lower(), {}).get(team_id, {})
            
            field_values = {
                "proj_id": proj_id,
                "team_id": team_id,
                "team_name": team_record.get('team_name', f'Team {team_id}'),
                "seed": seed,
                "points": team_record.get('average_final_points', 0)
            }
            
            await db_manager.db.execute(field_query, values=field_values)
    
    logger.info(f"Stored playoff results with projection ID: {proj_id}")
    return proj_id

# Manual playoff simulations
@router.post("/simulations/playoffs")
async def run_playoff_simulation(
    request: PlayoffSeedingRequest,
    current_user: Dict = Depends(auth_manager.validate_token)
):
    """
    Run playoff simulations with manually specified seeding.
    This is the actual HTTP endpoint for users who want to run custom playoff brackets.
    """

    regular_season_records = {}
    
    # Fetch current standings for both conferences
    for conference in ['eastern', 'western']:
        standings = await db_manager.calculate_and_store_standings(2025, conference)
        conference_records = {}
        
        for team_standing in standings:
            team_id = team_standing['team_id']
            conference_records[team_id] = {
                "team_name": team_standing.get('name', ''),
                "points": team_standing['points'],
                "average_final_points": team_standing['points'],  # Use current for manual
                "wins": team_standing['wins'],
                "goal_difference": team_standing['goal_difference'],
                "goals_for": team_standing['goals_for']
            }
        
        regular_season_records[conference] = conference_records
    
    result = await run_playoff_simulation_with_records(
        eastern_seeds=request.eastern_seeds,
        western_seeds=request.western_seeds,
        regular_season_records=regular_season_records,
        n_simulations=request.n_simulations,
        user_id=current_user['user_id']
    )
    
    return result

# ==================== Helper Functions ====================

async def ensure_historical_data_loaded():
    """
    Ensure we have historical data loaded.    
    Checks if 2025 season is loaded and loads if not.
    """
    for year in [2025]:
        # Check if we have games for this year
        count_result = await database.fetch_one(
            "SELECT COUNT(*) as count FROM games WHERE season_year = :year",
            values={"year": year}
        )
        
        if count_result['count'] == 0:
            logger.info(f"Loading {year} season data...")
            await db_manager.load_historical_season(year)

# In-memory cache for simulation status (replace with Redis in production)
simulation_cache = {}