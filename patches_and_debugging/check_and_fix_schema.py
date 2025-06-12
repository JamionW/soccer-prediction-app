#!/usr/bin/env python3
"""
Check and fix the database schema issue with game_id
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.common.database import database, connect, disconnect

async def check_and_fix_schema():
    """Check the current schema and fix if needed"""
    
    print("🔧 Checking and Fixing Database Schema")
    print("=" * 50)
    
    try:
        await connect()
        
        # 1. Check current schema
        print("\n1. Checking current games table schema...")
        
        schema_query = """
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = 'games'
        AND column_name = 'game_id'
        """
        
        result = await database.fetch_one(schema_query)
        
        if result:
            print(f"   Current game_id column:")
            print(f"   - Type: {result['data_type']}")
            print(f"   - Max length: {result['character_maximum_length']}")
            
            if result['data_type'] in ['integer', 'bigint', 'serial', 'bigserial']:
                print("\n❌ Problem found: game_id is an integer type!")
                print("   It should be VARCHAR to store IDs like 'teamA_teamB_date'")
                
                # 2. Ask for confirmation
                print("\n⚠️  WARNING: This will modify the database schema!")
                print("   We need to:")
                print("   1. Create a new temporary column")
                print("   2. Drop the old game_id column")
                print("   3. Recreate it as VARCHAR")
                print("   4. Any existing games will be lost")
                
                response = input("\nProceed with fix? (yes/no): ")
                
                if response.lower() == 'yes':
                    await fix_schema()
                else:
                    print("\n❌ Schema fix cancelled")
                    print("\n💡 Alternative: Ask your admin to run this SQL:")
                    print(get_fix_sql())
                    
            else:
                print("\n✅ game_id is already VARCHAR - schema is correct!")
                print("\n🤔 The error might be from a different table or query")
                
                # Check for other potential issues
                await check_other_issues()
        else:
            print("❌ games table or game_id column not found!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()


async def fix_schema():
    """Fix the schema by recreating game_id as VARCHAR"""
    
    print("\n🔧 Fixing schema...")
    
    try:
        # Check if there are any games
        game_count = await database.fetch_one("SELECT COUNT(*) as count FROM games")
        if game_count['count'] > 0:
            print(f"   ⚠️  Warning: {game_count['count']} games will be deleted!")
            
        # Start transaction
        async with database.transaction():
            # 1. Drop foreign key constraints if any
            print("   1. Checking for foreign key constraints...")
            constraints = await database.fetch_all("""
                SELECT constraint_name 
                FROM information_schema.table_constraints 
                WHERE table_name = 'games' 
                AND constraint_type = 'FOREIGN KEY'
            """)
            
            for constraint in constraints:
                print(f"      Dropping constraint: {constraint['constraint_name']}")
                await database.execute(f"ALTER TABLE games DROP CONSTRAINT {constraint['constraint_name']}")
            
            # 2. Create new table with correct schema
            print("   2. Creating new games table with correct schema...")
            
            await database.execute("ALTER TABLE games RENAME TO games_old")
            
            await database.execute("""
                CREATE TABLE games (
                    game_id VARCHAR(100) PRIMARY KEY,
                    date TIMESTAMP,
                    status VARCHAR(20),
                    home_team_id VARCHAR(20) REFERENCES team(team_id),
                    away_team_id VARCHAR(20) REFERENCES team(team_id),
                    home_score INTEGER,
                    away_score INTEGER,
                    home_penalties INTEGER,
                    away_penalties INTEGER,
                    matchday INTEGER DEFAULT 0,
                    attendance INTEGER DEFAULT 0,
                    is_completed BOOLEAN DEFAULT false,
                    went_to_shootout BOOLEAN DEFAULT false,
                    season_year INTEGER,
                    expanded_minutes INTEGER DEFAULT 90,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # 3. Create indexes
            print("   3. Creating indexes...")
            await database.execute("CREATE INDEX idx_games_season ON games(season_year)")
            await database.execute("CREATE INDEX idx_games_teams ON games(home_team_id, away_team_id)")
            await database.execute("CREATE INDEX idx_games_date ON games(date)")
            
            # 4. Drop old table
            print("   4. Dropping old table...")
            await database.execute("DROP TABLE games_old")
            
            print("\n✅ Schema fixed successfully!")
            
    except Exception as e:
        print(f"\n❌ Error fixing schema: {e}")
        raise


async def check_other_issues():
    """Check for other potential issues"""
    
    print("\n2. Checking for other potential issues...")
    
    # Check team table
    team_schema = await database.fetch_all("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'team'
        ORDER BY ordinal_position
    """)
    
    print("\n   Team table schema:")
    for col in team_schema[:3]:  # Show first 3 columns
        print(f"   - {col['column_name']}: {col['data_type']}")
    
    # Check if team_id is VARCHAR
    team_id_col = next((col for col in team_schema if col['column_name'] == 'team_id'), None)
    if team_id_col and team_id_col['data_type'] not in ['character varying', 'varchar', 'text']:
        print(f"\n❌ Another issue: team_id is {team_id_col['data_type']}, should be VARCHAR!")


def get_fix_sql():
    """Get the SQL to fix the schema"""
    return """
-- Fix game_id column type
BEGIN;

-- Rename old table
ALTER TABLE games RENAME TO games_old;

-- Create new table with correct schema
CREATE TABLE games (
    game_id VARCHAR(100) PRIMARY KEY,
    date TIMESTAMP,
    status VARCHAR(20),
    home_team_id VARCHAR(20) REFERENCES team(team_id),
    away_team_id VARCHAR(20) REFERENCES team(team_id),
    home_score INTEGER,
    away_score INTEGER,
    home_penalties INTEGER,
    away_penalties INTEGER,
    matchday INTEGER DEFAULT 0,
    attendance INTEGER DEFAULT 0,
    is_completed BOOLEAN DEFAULT false,
    went_to_shootout BOOLEAN DEFAULT false,
    season_year INTEGER,
    expanded_minutes INTEGER DEFAULT 90,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_games_season ON games(season_year);
CREATE INDEX idx_games_teams ON games(home_team_id, away_team_id);
CREATE INDEX idx_games_date ON games(date);

-- Drop old table
DROP TABLE games_old;

COMMIT;
"""


if __name__ == "__main__":
    print("🔍 Database Schema Checker & Fixer")
    print("=" * 50)
    print("This will check if the game_id column has the wrong type")
    print("and offer to fix it if needed.")
    print()
    
    asyncio.run(check_and_fix_schema())