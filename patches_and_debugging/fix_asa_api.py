#!/usr/bin/env python3
"""
Test and fix ASA API calls
"""

from itscalledsoccer.client import AmericanSoccerAnalysis

def test_asa_api():
    """Test different ASA API call formats"""
    
    print("🔍 Testing ASA API Calls")
    print("=" * 50)
    
    asa = AmericanSoccerAnalysis()
    
    # Test 1: Get available leagues
    print("\n1. Testing get_leagues()...")
    try:
        leagues = asa.get_leagues()
        print(f"   Available leagues: {[l.get('league_name', l.get('league_id')) for l in leagues[:5]]}")
        
        # Check if mlsnp is there
        mlsnp = [l for l in leagues if 'mlsnp' in str(l).lower() or 'next pro' in str(l).lower()]
        if mlsnp:
            print(f"   ✅ Found MLS Next Pro: {mlsnp}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Try different parameter names for team xG
    print("\n2. Testing get_team_xgoals() with different parameters...")
    
    test_params = [
        {'team_ids': ['raMyeZAMd2']},  # Original
        {'team_id': ['raMyeZAMd2']},   # Singular
        {'teams': ['raMyeZAMd2']},     # Alternative
    ]
    
    for params in test_params:
        try:
            print(f"\n   Trying: {params}")
            result = asa.get_team_xgoals(**params)
            if result:
                print(f"   ✅ Success! Got {len(result)} records")
                print(f"   Sample: {result[0] if result else 'No data'}")
                break
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Test 3: Get some games to verify team IDs
    print("\n3. Testing get_games() to verify team IDs...")
    try:
        # Try to get any recent games
        games = asa.get_games(leagues=['mls', 'mlsnp', 'uslc'], seasons=['2024', '2025'])
        
        if games:
            print(f"   Found {len(games)} games")
            
            # Look for Chattanooga FC games
            cfc_games = [g for g in games if 'raMyeZAMd2' in str(g.get('home_team_id', '')) + str(g.get('away_team_id', ''))]
            if cfc_games:
                print(f"   ✅ Found {len(cfc_games)} Chattanooga FC games")
                print(f"   Sample game: {cfc_games[0]}")
        else:
            print("   ❌ No games found")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Try the correct method signature
    print("\n4. Checking correct ASA client methods...")
    print("   Available methods:")
    methods = [m for m in dir(asa) if not m.startswith('_') and callable(getattr(asa, m))]
    relevant = [m for m in methods if 'team' in m.lower() or 'xg' in m.lower()]
    for method in relevant:
        print(f"   - {method}")


if __name__ == "__main__":
    test_asa_api()