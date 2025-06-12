#!/usr/bin/env python3
"""
Test that everything works after timezone fix
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Apply timezone fix
import patches_and_debugging.timezone_patch_old as timezone_patch_old

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager

async def test_everything_fixed():
    """Quick test to ensure everything works"""
    
    print("🧪 Testing System After Timezone Fix")
    print("=" * 50)
    
    try:
        await connect()
        db_manager = DatabaseManager(database)
        
        # This should work now
        print("\n✅ Testing data fetch (this previously failed)...")
        sim_data = await db_manager.get_data_for_simulation('eastern', 2025)
        
        print(f"   Success! Loaded {len(sim_data['conference_teams'])} teams")
        print(f"   Found {len(sim_data['games_data'])} total games")
        
        completed = [g for g in sim_data['games_data'] if g.get('is_completed')]
        print(f"   Completed games: {len(completed)}")
        
        print("\n✅ All systems operational!")
        print("\nYou can now:")
        print("1. Run: python train_with_real_data.py")
        print("2. Run: python test_railway_automl.py (should work now)")
        
    except Exception as e:
        print(f"❌ Still having issues: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await disconnect()

if __name__ == "__main__":
    asyncio.run(test_everything_fixed())