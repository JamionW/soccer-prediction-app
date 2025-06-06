# Soccer Table and Playoff Predictor API (Updated: 2025-06-06)

This API provides services for simulating soccer league seasons (specifically only MLS Next Pro at the moment), predicting standings, and running playoff simulations.

## Overview

The application is built using FastAPI and involves several key components:

*   **API Endpoints**: Defined in `src/common/routes.py`, providing access to various functionalities.
*   **Authentication**: Uses JWT and OAuth (Google/GitHub) for securing user-specific endpoints.
*   **Data Management**: Data is sourced from an external service (ASA API) for historical game and team statistics, and stored in a primary database. A `DatabaseManager` class handles database interactions.
*   **Simulation Engine**:
    *   Regular season simulations are handled by the `MLSNPRegSeasonPredictor` class.
    *   Playoff simulations are handled by the `MLSNPPlayoffPredictor` class.
    *   Simulations are computationally intensive and are run as background tasks to ensure API responsiveness.
*   **In-Memory Cache**: A temporary in-memory cache (`simulation_cache` in `routes.py`) is used to track the status of ongoing simulations.

## Data Retrieval and Storage

### Data Retrieval

1. **FOX SPORTS SCRAPER**
    *   The primary source for getting the season schedule. It's limited to MLSNP at the moment.
    *   This creates a shell for the ASA API to put data into. Empty shells will be used as the remaining games in the simulator.
    *   Currently this does not have an API endpoint. It can be run locally by an admin, and then the matches may be uploaded to the database.

2.  **ASA API**:
    *   Thank you to American Soccer Analysis for the public data, allowing this to be possible at all. It can be found at https://www.americansocceranalysis.com/.
    *   The primary source for historical game data, team statistics (including expected goals - xG).
    *   Data is fetched when a new season's data is explicitly loaded (via an admin endpoint) or when simulations require data not yet in the local database.
    *   Relevant functions: `DatabaseManager.load_historical_season()`, `DatabaseManager.get_or_fetch_team_xg()`.

3.  **Database**:
    *   Stores user accounts, fetched historical data, team information, simulation configurations, and results.
    *   The `DatabaseManager` class serves as an abstraction layer for all database operations.
    *   Used to fetch data required for simulations, serve API requests for team/season info, and retrieve stored simulation outcomes.

### Data Storage

1.  **Database**:
    *   **Users**: Credentials (hashed passwords), OAuth details.
    *   **Leagues**: Used to store leagues, their conferences, and any divisions.
    *   **Teams**: Team identifiers, org affiliations, xG history (`team_xg_history`), standings history, and other statistics.
    *   **Games**: Historical game details (scores, dates, teams, season).
    *   **Simulation Runs**: Metadata for each simulation executed (ID, date, parameters, user, conference). Stored in `prediction_runs`.
    *   **Simulation Results**:
        *   Regular Season: Aggregated results like average points, final rank, playoff probabilities (`prediction_summary`).
        *   Playoffs: Round-by-round advancement probabilities, championship win probability (`playoff_projection`, `team_playoff_summary`, `projected_field`).
    *   **Standings**: Calculated league standings.

2.  **In-Memory Cache (`simulation_cache`)**:
    *   Temporarily holds the status (`running`, `completed`, `failed`) and potentially intermediate results of simulations initiated via the API.

## Simulation Execution

This allows users to predict regular season outcomes and playoff brackets.

1.  **Initiation**:
    *   Users trigger simulations via specific API endpoints (`/simulations/regular-season`, `/simulations/playoffs`).
    *   Regular season simulations can automatically trigger subsequent playoff simulations.

2.  **Background Processing**:
    *   To handle potentially long computation times, regular season simulations are offloaded to background tasks using FastAPI's `BackgroundTasks`. The API returns an initial response quickly with a simulation ID.

3.  **Data Preparation**:
    *   Ensures required historical data is loaded from ASA API if not already in the database.
    *   Updates any game statuses (e.g., from incomplete to final).
    *   Calculates league-wide averages (e.g., xGf, xGa) as a baseline for predictions.

4.  **Regular Season Simulation (`MLSNPRegSeasonPredictor`)**:
    *   Users can choose to run a single conference or both.
    *   Takes conference details, team data, game schedules, and league averages as input.
    *   Runs Monte Carlo simulations for the remainder of the season.
    *   Outputs predicted standings, playoff qualification odds, and other metrics.
    *   Results are stored in the database.

5.  **Playoff Simulation (`MLSNPPlayoffPredictor`)**:
    *   Can be run with seeding derived from a regular season simulation or with manually provided seeds.
    *   Uses team performance metrics (like xG) and regular season records (for home-field advantage) to simulate playoff rounds.
    *   Outputs probabilities for teams advancing through each round and winning the championship.
    *   Results are stored in the database.

6.  **Status Tracking**:
    *   The `simulation_cache` provides real-time status updates for ongoing simulations.

## API Endpoints

All endpoints are relative to the base URL of the API.

### Public Routes (No Authentication Required)

*   **`GET /`**
    *   **Purpose**: Root endpoint. Provides API information and a summary of other endpoints.
*   **`GET /health`**
    *   **Purpose**: Health check for monitoring system status and database connectivity.
*   **`GET /teams/{conference}`**
    *   **Purpose**: Retrieves teams and current standings for the specified conference (`eastern` or `western`) for the 2025 season.
*   **`GET /simulations/all`**
    *   **Purpose**: Lists all publicly viewable simulation runs. Supports pagination and conference filtering.
*   **`GET /data/seasons`**
    *   **Purpose**: Returns a list of seasons for which the database contains game data.

### Authentication Routes

*   **`POST /auth/register`**
    *   **Purpose**: Allows new users to register with username, email, and password. Returns a JWT.
*   **`POST /auth/login`**
    *   **Purpose**: Allows existing users to log in. Returns a JWT.
*   **`POST /auth/oauth/{provider}`**
    *   **Purpose**: Handles OAuth 2.0 login flow with supported providers (e.g., `google`, `github`). Returns a JWT.
*   **`POST /auth/logout`**
    *   **Purpose**: Logs out the currently authenticated user by invalidating their token.
    *   **Authentication**: Required.

### User Profile Routes

*   **`GET /users/me`**
    *   **Purpose**: Fetches the profile of the currently authenticated user.
    *   **Authentication**: Required.
*   **`GET /users/me/simulations`**
    *   **Purpose**: Lists all simulations run by the currently authenticated user. Supports pagination.
    *   **Authentication**: Required.

### Simulation Execution Routes

*   **`POST /simulations/regular-season`**
    *   **Purpose**: Initiates a regular season simulation for a conference (or both). Runs in the background.
    *   **Authentication**: Required.
*   **`POST /simulations/playoffs`**
    *   **Purpose**: Initiates a playoff simulation using manually provided team seedings.
    *   **Authentication**: Required.

### Simulation Result Routes

*   **`GET /simulations/{simulation_id}`**
    *   **Purpose**: Retrieves the status and results of a specific simulation.
    *   **Authentication**: Public (assumed, for accessing results via a known ID).

### Data Management Routes

*   **`POST /data/load-historical/{year}`**
    *   **Purpose**: Loads historical data for a specified season from the ASA API.
    *   **Authentication**: Required (Admin only).