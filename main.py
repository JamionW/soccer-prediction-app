from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
from src.common.utils import logger
from src.common import connect, disconnect
from src.common.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    This can be used to initialize resources or connections.
    """
    logger.info("Starting FastAPI application...")
    await connect()
    logger.info("Database connected.")
    
    yield

    logger.info("Shutting down connection...")
    await disconnect()
    logger.info("Database disconnected.")

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://*.vercel.app",  # Allow all Vercel subdomains
    "https://pkbipcas.com",
    "https://*.railway.app"
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