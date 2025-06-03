from databases import Database
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Database connection URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://postgres:huzKZXKRlfXQeWveALbXWcnyPKHypaRr@nozomi.proxy.rlwy.net:15606/railway")

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