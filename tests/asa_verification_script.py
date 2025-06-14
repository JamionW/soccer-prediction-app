#!/usr/bin/env python3
"""
ASA Data Verification Script
Fetches specific games from ASA API to verify against database data.
"""

import asyncio
import sys
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.common.database import database
from src.common.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASADataVerifier:
    def __init__(self):
        self.db_manager = None
        self.asa_client = None
        
    async def initialize(self):
        """Initialize database and ASA client."""
        await database.connect()
        self.db_manager = DatabaseManager(database)
        await self.db_manager.initialize()
        
        # Initialize ASA client
        from itscalledsoccer.client import AmericanSoccerAnalysis
        self.asa_client = AmericanSoccerAnalysis()
        logger.info("Initialized ASA client and database connection")

    async def get_problematic_games_from_db(self) -> List[Dict]:
        """Get the problematic games identified in the SQL queries."""
        
        # Get games marked as shootout but with different scores
        query = """
            SELECT 
                g.game_id,
                g.date,
                g.home_team_id,
                g.away_team_id,
                ht.team_name as home_team,
                at.team_name as away_team,
                g.home_score,
                g.away_score,
                g.home_penalties,
                g.away_penalties,
                g.went_to_shootout,
                g.asa_loaded,
                'INVALID_SHOOTOUT' as issue_type
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE g.went_to_shootout = true 
              AND g.home_score != g.away_score
              AND g.is_completed = true
              AND g.season_year = 2025
            
            UNION ALL
            
            SELECT 
                g.game_id,
                g.date,
                g.home_team_id,
                g.away_team_id,
                ht.team_name as home_team,
                at.team_name as away_team,
                g.home_score,
                g.away_score,
                g.home_penalties,
                g.away_penalties,
                g.went_to_shootout,
                g.asa_loaded,
                'MISSING_PENALTY_DATA' as issue_type
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE g.went_to_shootout = true 
              AND (g.home_penalties IS NULL OR g.away_penalties IS NULL 
                   OR (g.home_penalties = 0 AND g.away_penalties = 0))
              AND g.is_completed = true
              AND g.season_year = 2025
            ORDER BY date DESC
        """
        
        games = await self.db_manager.db.fetch_all(query)
        return [dict(game) for game in games]

    async def get_specific_teams_games_from_db(self, team_ids: List[str]) -> List[Dict]:
        """Get all games for specific teams from database."""
        
        team_ids_str = "', '".join(team_ids)
        query = f"""
            SELECT 
                g.game_id,
                g.date,
                g.home_team_id,
                g.away_team_id,
                ht.team_name as home_team,
                at.team_name as away_team,
                g.home_score,
                g.away_score,
                g.home_penalties,
                g.away_penalties,
                g.went_to_shootout,
                g.asa_loaded,
                g.status
            FROM games g
            JOIN team ht ON g.home_team_id = ht.team_id
            JOIN team at ON g.away_team_id = at.team_id
            WHERE (g.home_team_id IN ('{team_ids_str}') 
                   OR g.away_team_id IN ('{team_ids_str}'))
              AND g.is_completed = true
              AND g.season_year = 2025
            ORDER BY g.date ASC
        """
        
        games = await self.db_manager.db.fetch_all(query)
        return [dict(game) for game in games]

    def fetch_asa_games_for_teams(self, team_ids: List[str], season_year: int = 2025) -> pd.DataFrame:
        """Fetch games from ASA for specific teams."""
        
        try:
            logger.info(f"Fetching ASA games for teams: {team_ids}")
            
            # Fetch games from ASA
            asa_games = self.asa_client.get_games(
                leagues=['mlsnp'],
                team_ids=team_ids,
                seasons=[str(season_year)]
            )
            
            logger.info(f"Found {len(asa_games)} games in ASA for specified teams")
            return asa_games
            
        except Exception as e:
            logger.error(f"Error fetching games from ASA: {e}")
            return pd.DataFrame()

    def compare_game_data(self, db_game: Dict, asa_games: pd.DataFrame) -> Optional[Dict]:
        """Compare a database game with ASA data."""
        
        # Find matching ASA game by teams and date
        game_date = db_game['date'].date() if hasattr(db_game['date'], 'date') else db_game['date']
        
        # Look for matching game in ASA data
        matching_asa_games = asa_games[
            (asa_games['home_team_id'] == db_game['home_team_id']) &
            (asa_games['away_team_id'] == db_game['away_team_id'])
        ].copy()  # ADD .copy() to avoid pandas warnings
        
        # Filter by date and RETURN ONLY ONE MATCH
        best_match = None
        for idx, asa_game in matching_asa_games.iterrows():
            try:
                asa_date_str = asa_game.get('date_time_utc', asa_game.get('date'))
                if asa_date_str:
                    asa_datetime = pd.to_datetime(asa_date_str, utc=True)
                    asa_date = asa_datetime.date()
                    
                    date_diff = abs((asa_date - game_date).days)
                    if date_diff <= 1:  # Found matching game
                        best_match = asa_game.to_dict()
                        break  # Take the first match and stop
            except Exception as e:
                logger.warning(f"Error parsing ASA date: {e}")
                continue
        
        if best_match:
            return self._create_comparison_result(db_game, best_match)
        
        return None

    def _create_comparison_result(self, db_game: Dict, asa_game: Dict) -> Dict:
        """Create a comparison result between database and ASA game."""
        
        # Extract ASA game data
        asa_home_score = asa_game.get('home_score', 0)
        asa_away_score = asa_game.get('away_score', 0)
        asa_went_to_penalties = asa_game.get('penalties', False)
        asa_home_pens = asa_game.get('home_penalties', 0) if asa_went_to_penalties else None
        asa_away_pens = asa_game.get('away_penalties', 0) if asa_went_to_penalties else None
        
        # Handle boolean conversion for penalties field
        if isinstance(asa_went_to_penalties, str):
            asa_went_to_penalties = asa_went_to_penalties.lower() in ['true', '1', 'yes']
        
        comparison = {
            'game_id': db_game['game_id'],
            'date': db_game['date'],
            'matchup': f"{db_game['home_team']} vs {db_game['away_team']}",
            
            # Database data
            'db_home_score': db_game['home_score'],
            'db_away_score': db_game['away_score'],
            'db_went_to_shootout': db_game['went_to_shootout'],
            'db_home_penalties': db_game.get('home_penalties'),
            'db_away_penalties': db_game.get('away_penalties'),
            
            # ASA data
            'asa_home_score': asa_home_score,
            'asa_away_score': asa_away_score,
            'asa_went_to_penalties': asa_went_to_penalties,
            'asa_home_penalties': asa_home_pens,
            'asa_away_penalties': asa_away_pens,
            
            # Comparison flags
            'score_matches': (db_game['home_score'] == asa_home_score and 
                            db_game['away_score'] == asa_away_score),
            'shootout_flag_matches': db_game['went_to_shootout'] == asa_went_to_penalties,
            'penalty_scores_match': (db_game.get('home_penalties') == asa_home_pens and 
                                   db_game.get('away_penalties') == asa_away_pens),
            
            # Issue detection
            'has_issues': False
        }
        
        # Detect issues
        issues = []
        if not comparison['score_matches']:
            issues.append('SCORE_MISMATCH')
        if not comparison['shootout_flag_matches']:
            issues.append('SHOOTOUT_FLAG_MISMATCH')
        if not comparison['penalty_scores_match'] and asa_went_to_penalties:
            issues.append('PENALTY_SCORE_MISMATCH')
        if db_game['went_to_shootout'] and db_game['home_score'] != db_game['away_score']:
            issues.append('INVALID_SHOOTOUT_IN_DB')
        
        comparison['issues'] = issues
        comparison['has_issues'] = len(issues) > 0
        
        return comparison

    def print_comparison_results(self, comparisons: List[Dict]):
        """Print detailed comparison results."""
        
        print("\n" + "="*100)
        print("ASA DATA VERIFICATION RESULTS")
        print("="*100)
        
        if not comparisons:
            print("No games found to compare.")
            return
        
        issues_found = [c for c in comparisons if c['has_issues']]
        
        print(f"\nTotal games compared: {len(comparisons)}")
        print(f"Games with issues: {len(issues_found)}")
        print(f"Games matching perfectly: {len(comparisons) - len(issues_found)}")
        
        if issues_found:
            print(f"\n🚨 GAMES WITH ISSUES:")
            print("-" * 100)
            
            for i, comp in enumerate(issues_found, 1):
                print(f"\n{i}. {comp['matchup']} - {comp['date'].strftime('%Y-%m-%d')}")
                print(f"   Game ID: {comp['game_id']}")
                print(f"   Issues: {', '.join(comp['issues'])}")
                
                print(f"\n   DATABASE DATA:")
                print(f"     Score: {comp['db_home_score']}-{comp['db_away_score']}")
                print(f"     Went to shootout: {comp['db_went_to_shootout']}")
                if comp['db_went_to_shootout']:
                    print(f"     Penalty score: {comp['db_home_penalties']}-{comp['db_away_penalties']}")
                
                print(f"\n   ASA DATA:")
                print(f"     Score: {comp['asa_home_score']}-{comp['asa_away_score']}")
                print(f"     Went to penalties: {comp['asa_went_to_penalties']}")
                if comp['asa_went_to_penalties']:
                    print(f"     Penalty score: {comp['asa_home_penalties']}-{comp['asa_away_penalties']}")
                
                print(f"\n   RECOMMENDED FIX:")
                if 'INVALID_SHOOTOUT_IN_DB' in comp['issues']:
                    print(f"     UPDATE games SET went_to_shootout = false, home_penalties = NULL, away_penalties = NULL")
                    print(f"     WHERE game_id = '{comp['game_id']}';")
                elif 'SHOOTOUT_FLAG_MISMATCH' in comp['issues']:
                    print(f"     UPDATE games SET went_to_shootout = {comp['asa_went_to_penalties']}")
                    print(f"     WHERE game_id = '{comp['game_id']}';")
                elif 'SCORE_MISMATCH' in comp['issues']:
                    print(f"     UPDATE games SET home_score = {comp['asa_home_score']}, away_score = {comp['asa_away_score']}")
                    print(f"     WHERE game_id = '{comp['game_id']}';")
                
                print("-" * 50)
        
        else:
            print(f"\n✅ All games match ASA data perfectly!")

    async def verify_problematic_games(self):
        """Main function to verify problematic games against ASA."""
        
        print("Fetching problematic games from database...")
        problematic_games = await self.get_problematic_games_from_db()
        
        if not problematic_games:
            print("No problematic games found in database!")
            return
        
        print(f"Found {len(problematic_games)} problematic games")
        
        # Get unique team IDs from problematic games
        team_ids = set()
        for game in problematic_games:
            team_ids.add(game['home_team_id'])
            team_ids.add(game['away_team_id'])
        
        print(f"Fetching ASA data for {len(team_ids)} teams...")
        asa_games = self.fetch_asa_games_for_teams(list(team_ids))
        
        if asa_games.empty:
            print("No ASA games found for these teams!")
            return
        
        # Compare each problematic game
        comparisons = []
        for db_game in problematic_games:
            comparison = self.compare_game_data(db_game, asa_games)
            if comparison:
                comparisons.append(comparison)
        
        self.print_comparison_results(comparisons)

    async def verify_specific_teams(self, team_ids: List[str]):
        """Verify all games for specific teams."""
        
        print(f"Fetching all games for teams: {team_ids}")
        db_games = await self.get_specific_teams_games_from_db(team_ids)
        
        print(f"Found {len(db_games)} games in database")
        
        asa_games = self.fetch_asa_games_for_teams(team_ids)
        
        if asa_games.empty:
            print("No ASA games found!")
            return
        
        # Compare all games
        comparisons = []
        for db_game in db_games:
            comparison = self.compare_game_data(db_game, asa_games)
            if comparison:
                comparisons.append(comparison)
        
        self.print_comparison_results(comparisons)

    async def cleanup(self):
        """Cleanup database connection."""
        await database.disconnect()

async def main():
    """Main function."""
    
    verifier = ASADataVerifier()
    
    try:
        await verifier.initialize()
        
        print("ASA Data Verification Tool")
        print("=" * 50)
        print("1. Verify problematic games (from SQL queries)")
        print("2. Verify specific teams (RBNY II and NYCFC II)")
        print("3. Verify custom team IDs")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            await verifier.verify_problematic_games()
        
        elif choice == "2":
            # RBNY II and NYCFC II team IDs
            team_ids = ['9Yqdwg85vJ', 'jYQJXkP5GR']
            await verifier.verify_specific_teams(team_ids)
        
        elif choice == "3":
            team_ids_input = input("Enter team IDs separated by commas: ").strip()
            team_ids = [tid.strip() for tid in team_ids_input.split(',') if tid.strip()]
            if team_ids:
                await verifier.verify_specific_teams(team_ids)
            else:
                print("No valid team IDs provided!")
        
        else:
            print("Invalid choice!")
    
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    
    finally:
        await verifier.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
