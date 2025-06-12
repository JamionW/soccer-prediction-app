#!/bin/bash

echo "🚀 Setting up MLS Next Pro Predictor with Railway Database"
echo "========================================================="

# 1. Create .env file with Railway connection
echo "📝 Creating .env file with Railway database connection..."

cat > .env << EOL
# Railway Database Configuration
# Get the full connection string from your admin
DATABASE_URL=postgresql://postgres:aApGVbbXwaMVqWvHeeUfljDIWtxJIbBr@trolley.proxy.rlwy.net:13360/railway

# Authentication Keys (generate your own for development)
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Environment
ENVIRONMENT=development

# Optional OAuth (leave blank for now)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
EOL

# 2. Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# 3. Install AutoML dependencies
echo "🤖 Installing AutoML libraries..."
pip install pycaret[full]
# Alternative: pip install autogluon

# 4. Create necessary directories
echo "📁 Creating required directories..."
mkdir -p output/archive
mkdir -p models
mkdir -p data

# 5. Test database connection
echo "🔍 Testing Railway database connection..."
python << EOF
import os
import asyncpg
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_connection():
    try:
        # Get connection string
        db_url = os.getenv('DATABASE_URL')
        if not db_url or '[USERNAME]' in db_url:
            print("❌ Please update .env with actual Railway database credentials!")
            return
        
        # Test connection
        conn = await asyncpg.connect(db_url)
        version = await conn.fetchval('SELECT version()')
        await conn.close()
        
        print("✅ Successfully connected to Railway PostgreSQL!")
        print(f"   Database version: {version.split(',')[0]}")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   Check your DATABASE_URL in .env file")

asyncio.run(test_connection())
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env with your Railway database credentials"
echo "2. Start the API server: uvicorn main:app --reload"
echo "3. The database schema should already exist on Railway"
echo "4. Load any missing data if needed"
