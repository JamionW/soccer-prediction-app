from databases import Database
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Database connection URL from environment variable
DATABASE_URL="postgresql://postgres:huzKZXKRlfXQeWveALbXWcnyPKHypaRr@nozomi.proxy.rlwy.net:15606/railway"
# Initialize the database connection
database = Database(DATABASE_URL)