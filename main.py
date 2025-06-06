import os
import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
from src.common.utils import logger
from src.common import database
from src.common.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    This can be used to initialize resources or connections.
    """
    app.state.db_connected = False  # Initialize db_connected state
    db_url = os.getenv('DATABASE_URL')
    if db_url is None:
        logger.error("DATABASE_URL environment variable not set. Please configure it before running the application.")
        # We might not want to raise HTTPException here anymore if we want the app to start for health checks
        # For now, let's keep it to ensure config is present, but this could be revisited.
        raise HTTPException(status_code=500, detail="Application is not configured correctly. DATABASE_URL is missing.")

    logger.info("Attempting to connect to the database...")
    try:
        await database.connect()
        logger.info("Database connected successfully.")
        app.state.db_connected = True
    except asyncpg.exceptions.PostgresConnectionError as e: # Catches InvalidPasswordError and other connection issues
        logger.error(f"Database connection failed during startup: {e}")
        # app.state.db_connected remains False
    except Exception as e: # Catch any other potential errors during connection
        logger.error(f"An unexpected error occurred during database connection: {e}")
        # app.state.db_connected remains False
    
    yield

    if app.state.db_connected:
        logger.info("Shutting down database connection...")
        await database.disconnect()
        logger.info("Database disconnected.")
    else:
        logger.info("Skipping database disconnection as it was not connected.")

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://pkbipcas.com",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)