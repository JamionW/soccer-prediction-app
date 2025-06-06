from databases import Database
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Database connection URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable not set. Please configure it before running the application.")

# Initialize the database connection
database = Database(DATABASE_URL)

# Export connect and disconnect functions
async def connect():
    """Connect to the database."""
    await database.connect()

async def disconnect():
    """Disconnect from the database."""
    await database.disconnect()

__all__ = ['database', 'connect', 'disconnect']