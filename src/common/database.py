import os
import logging
from databases import Database
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (for local testing, ignored on Railway)
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable not set. Please configure it before running the application.")

# Log the database URL (masking password for security in production logs)
# For debugging, you can temporarily print the full URL, but be cautious in public logs.
# For now, let's just confirm it's being read.
# If DATABASE_URL contains '@', split and rejoin to mask password.
# Otherwise, just log the URL.
if '@' in DATABASE_URL:
    parts = DATABASE_URL.split('@')
    # Assuming password is part of the first segment after '://' and before '@'
    user_and_host = parts[0].split('://')
    masked_url = f"{user_and_host[0]}://{user_and_host[1].split(':')[0]}:********@{parts[1]}"
else:
    masked_url = DATABASE_URL # Fallback if format is unexpected

logger.info(f"App attempting to connect to database using URL: {masked_url}")

database = Database(DATABASE_URL)

async def connect():
    """Connect to the database."""
    try:
        await database.connect()
        logger.info("Successfully connected to the database.")
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise # Re-raise to ensure the application still fails if connection fails

async def disconnect():
    """Disconnect from the database."""
    try:
        await database.disconnect()
        logger.info("Successfully disconnected from the database.")
    except Exception as e:
        logger.error(f"Error disconnecting from database: {e}")

__all__ = ['database', 'connect', 'disconnect']