from databases import Database
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Database connection URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "spostgresql://neondb_owner:npg_4RlL2ucAHBTq@ep-purple-mode-a47dmgec-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require") # UPDATE with flyio later
# Initialize the database connection
database = Database(DATABASE_URL)