"""
Fox Sports MLS Next Pro Scraper
This version is specifically designed for Fox Sports' date-based schedule system
and handles their unique team abbreviation mappings.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Optional, Tuple, Set
import logging
from pathlib import Path
import time
import sys
import io
import os

# Define output directory and log file path
output_dir = "output"
log_file_path = os.path.join(output_dir, "fox_sports_scraper.log")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class FoxSportsMLSNextProScraper:
    """
    Specialized scraper for Fox Sports MLS Next Pro schedule.
    
    This scraper understands Fox Sports' specific structure:
    1. Date-based URL navigation
    2. Custom team abbreviations
    3. Logo-based team disambiguation
    """
    
    # Fox Sports team abbreviation mappings
    FOX_ABBREVIATIONS = {
        'CHT': 'Chattanooga FC',
        'HUN': 'Huntsville City FC', 
        'NII': 'New York Red Bulls II',
        'BES': 'Philadelphia Union II',
        'OCB': 'Orlando City B',
        'CHI': 'Chicago Fire FC II',
        'CRO': 'Crown Legacy FC',
        'TII': 'Toronto FC II',
        'ATL': 'Atlanta United 2',
        'INT': 'Inter Miami CF II',
        'CAR': 'Carolina Core',
        'CIN': 'FC Cincinnati 2',
        'SAI': 'Saint Louis City SC 2',
        'SPR': 'Sporting Kansas City II',
        'NOR': 'North Texas SC',
        'LOS': 'Los Angeles FC 2',
        'RMO': 'Real Monarchs',
        'MIN': 'Minnesota United FC 2',
        'PII': 'Portland Timbers 2',
        'SII': 'Tacoma Defiance',
        'SAN': 'The Town FC',
        'VAN': 'Vancouver Whitecaps FC 2',
        'VEN': 'Ventura County FC',
        'AUS': 'Austin FC II',
        'HOU': 'Houston Dynamo FC 2',
        'VII': 'Vancouver Whitecaps FC 2',
        'COL': 'AMBIGUOUS',  # Will be resolved by logo
        'NEW': 'AMBIGUOUS'  # Will be resolved by logo
    }
    
    # Logo patterns to distinguish between the two "NEW" teams
    LOGO_PATTERNS = {
        'new-york-city-fc-2': 'New York City FC II',
        'new-england-revolution-2': 'New England Revolution II',
        'columbus-crew-2': 'Columbus Crew 2',
        'colorado-rapids-2': 'Colorado Rapids 2'

    }
    
    def __init__(self, asa_teams_file_path: str):
        """
        Initialize the scraper with ASA team data and Fox Sports mappings.
        
        Args:
            asa_teams_file_path: Path to the ASA teams JSON file
        """
        self.asa_teams = self._load_asa_teams(asa_teams_file_path)
        
        if not self.asa_teams:
            raise ValueError("Could not load ASA team data.")
        
        # Create session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Create team lookup dictionary for ASA matching
        self.team_lookup = self._create_team_lookup()
        
        # Track scraping statistics
        self.stats = {
            'dates_scraped': 0,
            'fixtures_found': 0,
            'teams_not_matched': set(),
            'dates_with_games': [],
            'ambiguous_resolutions': [],
            'resolution_failures': [],
            'same_team_errors': []
        }
        
        logger.info(f"Initialized Fox Sports scraper with {len(self.asa_teams)} ASA teams")
    
    def _load_asa_teams(self, file_path: str) -> List[Dict]:
        """Load and validate ASA team data."""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as file:
                teams_data = json.load(file)
            
            # Validate that we have the required fields
            validated_teams = []
            for team in teams_data:
                if all(field in team for field in ['team_id', 'team_name']):
                    validated_teams.append(team)
            
            logger.info(f"Loaded {len(validated_teams)} valid ASA teams")
            return validated_teams
            
        except Exception as e:
            logger.error(f"Error loading ASA teams: {e}")
            return []
    
    def _create_team_lookup(self) -> Dict[str, str]:
        """
        Create a lookup dictionary to match team names to ASA IDs.
        This includes variations to handle different naming conventions.
        """
        lookup = {}
        
        for team in self.asa_teams:
            team_id = team['team_id']
            team_name = team['team_name']
            
            # Add exact name
            lookup[team_name.lower()] = team_id
            
            # Add variations without numbers/suffixes
            base_name = re.sub(r'\s+(2|II|B)$', '', team_name, flags=re.I)
            lookup[base_name.lower()] = team_id
            
            # Add specific patterns for disambiguation - AVOID CONFLICTS
            if 'New York City' in team_name:
                lookup['new york city fc ii'] = team_id
                lookup['nyc fc ii'] = team_id
                lookup['nycfc ii'] = team_id
                # Don't use generic 'new york' to avoid conflicts
            elif 'New England' in team_name:
                lookup['new england revolution ii'] = team_id
                lookup['new england'] = team_id
                lookup['revolution ii'] = team_id
            elif 'Red Bulls' in team_name:
                lookup['new york red bulls ii'] = team_id
                lookup['red bulls ii'] = team_id
                lookup['nyrb ii'] = team_id
            elif 'Columbus' in team_name:
                lookup['columbus crew 2'] = team_id
                lookup['columbus'] = team_id
                lookup['crew 2'] = team_id
            elif 'Colorado' in team_name:
                lookup['colorado rapids 2'] = team_id
                lookup['colorado'] = team_id
                lookup['rapids 2'] = team_id
            
            # Add city name only for non-conflicting cases
            city_parts = team_name.split()
            if city_parts and city_parts[0].lower() not in ['new']:  # Avoid 'New' conflicts
                city = city_parts[0]
                city_key = city.lower()
                if city_key not in lookup:
                    lookup[city_key] = team_id
        
        logging.info(f"Created {len(lookup)} team name patterns for ASA matching")
        
        # Log key lookups for debugging
        debug_keys = ['new york city fc ii', 'columbus crew 2', 'columbus', 'colorado rapids 2', 'colorado']
        for key in debug_keys:
            if key in lookup:
                team_name = self._get_team_name_by_id(lookup[key])
                logging.info(f"Lookup '{key}' → {team_name} ({lookup[key]})")
        
        return lookup
    
    def _get_team_name_by_id(self, team_id: str) -> str:
        """Helper to get team name by ID."""
        for team in self.asa_teams:
            if team['team_id'] == team_id:
                return team['team_name']
        return f"Unknown team (ID: {team_id})"
    
    def _extract_available_dates(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract available dates from the Fox Sports page by looking for date headers.        
        Args:
            soup: BeautifulSoup object of the page            
        Returns:
            List of date strings in YYYY-MM-DD format
        """
        dates = []
        
        # Fox Sports uses divs with class 'table-title' for date headers
        date_titles = soup.find_all('div', class_='table-title')
        
        current_year = datetime.now().year # Assume current year if not specified
        
        for title_div in date_titles:
            date_text = title_div.get_text(strip=True)
            
            # Example format: "SAT, AUG 23"
            date_match = re.search(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*(\w+)\s*(\d{1,2})', date_text, re.IGNORECASE)
            if date_match:
                month_str = date_match.group(2)
                day_str = date_match.group(3)
                
                try:
                    # Parse the month and day, assume current year
                    date_obj = datetime.strptime(f"{month_str} {day_str} {current_year}", "%b %d %Y")
                    dates.append(date_obj.strftime("%Y-%m-%d"))
                except ValueError:
                    logger.warning(f"Could not parse date from '{date_text}'")
                    continue
        
        # Remove duplicates and sort
        dates = sorted(list(set(dates)))
        
        if dates:
            logger.info(f"Found {len(dates)} dates with games on the current page: {dates}")
        else:
            logger.warning("Could not extract any date headers from the page.")
        
        return dates
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """
        Generate a range of dates to check for games.        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format            
        Returns:
            List of date strings in YYYY-MM-DD format
        """
        dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return dates
    
    def _resolve_team_from_abbreviation(self, abbrev: str, row_element, cell_index: int = None) -> Optional[str]:
        """
        Resolve a Fox Sports abbreviation to a team name.        
        For the ambiguous "NEW" abbreviation, this checks the team logo
        to determine which team it represents.        
        Args:
            abbrev: The Fox Sports abbreviation
            row_element: The BeautifulSoup element containing the match info            
        Returns:
            The full team name, or None if not found
        """
        logging.info(f"🔍 Resolving abbreviation: '{abbrev}' (cell {cell_index})")

        # First check if it's a known non-ambiguous abbreviation
        if abbrev in self.FOX_ABBREVIATIONS and self.FOX_ABBREVIATIONS[abbrev] != 'AMBIGUOUS':
            team_name = self.FOX_ABBREVIATIONS[abbrev]
            logging.info(f"✅ Resolved '{abbrev}' to '{team_name}' via direct mapping")
            return team_name
        
        # Handle ambiguous cases (NEW, COL)
        if abbrev in ['NEW', 'COL'] or self.FOX_ABBREVIATIONS.get(abbrev) == 'AMBIGUOUS':
            logging.info(f"🚨 Attempting to resolve ambiguous abbreviation: '{abbrev}'")
            
            # Strategy 1: Enhanced logo detection
            resolved_team = self._resolve_by_enhanced_logo(abbrev, row_element, cell_index)
            if resolved_team:
                logging.info(f"✅ Resolved '{abbrev}' to '{resolved_team}' via logo")
                self.stats['ambiguous_resolutions'].append({
                    'abbrev': abbrev,
                    'resolved_to': resolved_team,
                    'method': 'logo',
                    'cell_index': cell_index
                })
                return resolved_team
            
            # Strategy 2: Alt text analysis
            resolved_team = self._resolve_by_alt_text(abbrev, row_element)
            if resolved_team:
                logging.info(f"✅ Resolved '{abbrev}' to '{resolved_team}' via alt text")
                self.stats['ambiguous_resolutions'].append({
                    'abbrev': abbrev,
                    'resolved_to': resolved_team,
                    'method': 'alt_text',
                    'cell_index': cell_index
                })
                return resolved_team
            
            # If we get here, resolution failed
            logging.error(f"❌ Could not resolve ambiguous abbreviation '{abbrev}'")
            logging.error(f"Row HTML snippet: {str(row_element)[:500]}")
            
            # Emergency fallback - extract all images for debugging
            images = row_element.find_all('img')
            for i, img in enumerate(images):
                logging.error(f"Image {i}: src='{img.get('src', '')}', alt='{img.get('alt', '')}'")
            
            self.stats['resolution_failures'].append({
                'abbrev': abbrev,
                'html_snippet': str(row_element)[:500],
                'cell_index': cell_index
            })
            return None
        
        # Unknown abbreviation
        logging.warning(f"❓ Unknown Fox Sports abbreviation: '{abbrev}'")
        self.stats['teams_not_matched'].add(abbrev)
        return None
    
    def _resolve_by_enhanced_logo(self, abbrev: str, row_element, cell_index: int = None) -> Optional[str]:
        """Enhanced logo resolution with specific patterns for NEW/COL."""
        target_search_area = None
        if cell_index is not None:
            # When called from _extract_fixtures_from_page, row_element is <tr>
            # and cell_index is the index of the <td> we care about.
            all_tds_in_row = row_element.find_all('td')
            if cell_index < len(all_tds_in_row):
                target_search_area = all_tds_in_row[cell_index]
            else:
                logging.warning(f"Cell index {cell_index} out of bounds for row for '{abbrev}'. HTML: {str(row_element)[:200]}")
                # Fallback: try to find any 'cell-entity' if specific index fails badly
                # This might occur if the table structure is unexpected.
                # However, for the test case, cell_index should be valid if row structure is as expected.
                # If all_tds_in_row is empty, it means row_element wasn't a <tr> or had no <td>s.
                # If cell_index is too large, it means the row didn't have enough <td>s.
                # In these critical error cases, returning None is appropriate.
                return None
        else:
            # This path is taken by direct unit tests for _resolve_team_from_abbreviation.
            # The row_element in those tests is typically a <tr><td><img ...></td></tr>.
            # We need to find the <td> that contains the image.
            if row_element.name == 'td': # If row_element itself is the td
                 target_search_area = row_element
            else: # Assume row_element is a <tr> or similar wrapper
                 target_search_area = row_element.find('td') # Find the first td

        if not target_search_area:
            logging.debug(f"No valid TD cell found for logo resolution for '{abbrev}'. Searched in: {str(row_element)[:200]}")
            return None
        
        images = target_search_area.find_all('img')
        logging.info(f"Found {len(images)} images in current cell for '{abbrev}'")
        
        for img in images:
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            logging.info(f"Checking image: src='{src}', alt='{alt}'")
            
            # Enhanced logo patterns based on actual Fox Sports URLs
            logo_patterns = {
                # New York teams
                'new-york-city-fc-2': 'New York City FC II',
                'new-england-revolution-2': 'New England Revolution II', 
                'new-york-red-bulls-2': 'New York Red Bulls II',
                # Columbus vs Colorado
                'columbus-crew-2': 'Columbus Crew 2',
                'colorado-rapids-2': 'Colorado Rapids 2',
            }
            
            # Check patterns against src
            for pattern, team_name in logo_patterns.items():
                if pattern in src.lower():
                    logging.info(f"🎯 Logo match: '{pattern}' in '{src}' → '{team_name}'")
                    return team_name
            
            # Also check alt text for team names
            alt_lower = alt.lower()
            if 'new york city' in alt_lower:
                logging.info(f"🎯 Alt text match: 'new york city' in '{alt}' → 'New York City FC II'")
                return 'New York City FC II'
            elif 'columbus' in alt_lower:
                logging.info(f"🎯 Alt text match: 'columbus' in '{alt}' → 'Columbus Crew 2'")
                return 'Columbus Crew 2'
            elif 'colorado' in alt_lower:
                logging.info(f"🎯 Alt text match: 'colorado' in '{alt}' → 'Colorado Rapids 2'")
                return 'Colorado Rapids 2'
            elif 'new england' in alt_lower:
                logging.info(f"🎯 Alt text match: 'new england' in '{alt}' → 'New England Revolution II'")
                return 'New England Revolution II'
        
        logging.warning(f"No logo patterns matched for '{abbrev}'")
        return None

    def _resolve_by_alt_text(self, abbrev: str, row_element) -> Optional[str]:
        """Fallback resolution using alt text and location data."""
        # Get the current cell for this abbreviation
        current_cell = row_element.find('td', class_='cell-entity')
        if not current_cell:
            return None
        
        # Look for any text that might help identify the team
        cell_text = current_cell.get_text().lower()
        
        # Check if we can find city/location clues
        if abbrev == 'NEW':
            if 'new york city' in cell_text or 'nyc' in cell_text:
                return 'New York City FC II'
            elif 'new england' in cell_text or 'foxborough' in cell_text:
                return 'New England Revolution II'
            elif 'red bulls' in cell_text:
                return 'New York Red Bulls II'
        elif abbrev == 'COL':
            if 'columbus' in cell_text or 'crew' in cell_text:
                return 'Columbus Crew 2'
            elif 'colorado' in cell_text or 'rapids' in cell_text:
                return 'Colorado Rapids 2'
        
        return None
    
    def _extract_fixtures_from_page(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract fixtures from a Fox Sports page.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            List of fixture dictionaries
        """
        fixtures = []
        
        # Find all table segments, which contain a date header and a table
        table_segments = soup.find_all('div', class_='table-segment')
        
        for segment in table_segments:
            # Extract the date from the segment's title
            date_div = segment.find('div', class_='table-title')
            current_date = None
            if date_div:
                date_text = date_div.get_text(strip=True)
                # Example: "SAT, AUG 23"
                date_match = re.search(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*(\w+)\s*(\d{1,2})', date_text, re.IGNORECASE)
                if date_match:
                    month_str = date_match.group(2)
                    day_str = date_match.group(3)
                    try:
                        # Assume current year for parsing
                        current_date = datetime.strptime(f"{month_str} {day_str} {datetime.now().year}", "%b %d %Y").strftime("%Y-%m-%d")
                    except ValueError:
                        logger.warning(f"Could not parse date from segment title: '{date_text}'")
            
            if not current_date:
                logger.warning("Skipping table segment due to missing or unparsable date.")
                continue

            table = segment.find('table', class_='data-table')
            if not table:
                continue

            rows = table.find_all('tr', id=re.compile(r'tbl-row-\d+')) # Match rows with fixture data
            
            for row in rows:
                cells = row.find_all('td')
                
                if len(cells) < 5: # Expecting at least 5 cells for matchup, status, location
                    continue
                
                # Team abbreviations are usually in the first and third data cells
                # These cells have class 'cell-entity'
                home_abbrev_elem = cells[0].find('a', class_='table-entity-name')
                away_abbrev_elem = cells[2].find('a', class_='table-entity-name')
                
                home_abbrev = home_abbrev_elem.get_text(strip=True) if home_abbrev_elem else None
                away_abbrev = away_abbrev_elem.get_text(strip=True) if away_abbrev_elem else None
                
                if home_abbrev and away_abbrev:
                    home_team = self._resolve_team_from_abbreviation(home_abbrev, row, cell_index=0)
                    # The away team's info (name and logo) is in cells[2]
                    away_team = self._resolve_team_from_abbreviation(away_abbrev, row, cell_index=2)
                    
                    if home_team and away_team:
                        if home_team == away_team:
                            print(f"\n🚨 SAME TEAM ERROR DETECTED:")
                            print(f"Date: {current_date}")
                            print(f"Abbreviations: {home_abbrev} vs {away_abbrev}")
                            print(f"Both resolved to: {home_team}")
                            print(f"Row HTML snippet:")
                            print(str(row)[:500] + "...")
                            
                            # Get the specific cells for analysis
                            cells = row.find_all('td', class_='cell-entity')
                            if len(cells) >= 2:
                                # Analyze home cell (index 0)
                                home_images = cells[0].find_all('img')
                                print(f"\nHome cell images ({len(home_images)}):")
                                for i, img in enumerate(home_images):
                                    print(f"  {i}: src='{img.get('src', '')}', alt='{img.get('alt', '')}'")
                                
                                # Analyze away cell (index 1)  
                                away_images = cells[1].find_all('img')
                                print(f"\nAway cell images ({len(away_images)}):")
                                for i, img in enumerate(away_images):
                                    print(f"  {i}: src='{img.get('src', '')}', alt='{img.get('alt', '')}'")
                            
                            # Add to your stats tracking
                            if not hasattr(self.stats, 'same_team_errors'):
                                self.stats['same_team_errors'] = []
                            
                            self.stats['same_team_errors'].append({
                                'date': current_date,
                                'home_abbrev': home_abbrev,
                                'away_abbrev': away_abbrev,
                                'resolved_name': home_team,
                                'row_snippet': str(row)[:200]
                            })
                            
                            print("=" * 50)
                            continue  # Skip this fixture

                        # Match to ASA IDs
                        home_id = self._match_to_asa_id(home_team)
                        away_id = self._match_to_asa_id(away_team)
                        
                        if home_id and away_id:
                            fixture = {
                                'date': current_date,
                                'home_team': home_team,
                                'home_team_id': home_id,
                                'away_team': away_team,
                                'away_team_id': away_id,
                                'location': self._extract_location(row),
                                'score_or_status': self._extract_score_or_status(row),
                                'source': 'fox_sports'
                            }
                            fixtures.append(fixture)
                            logger.info(f"Found fixture: {home_team} vs {away_team} on {current_date} at {fixture['score_or_status']}")
                        else:
                            logger.warning(f"Could not match teams to ASA IDs: {home_team} vs {away_team} for date {current_date}")
        
        return fixtures
    
    def _match_to_asa_id(self, team_name: str) -> Optional[str]:
        """
        Match a team name to an ASA team ID.
        
        Args:
            team_name: The team name to match
            
        Returns:
            The ASA team ID, or None if not found
        """
        # Try exact match first
        clean_name = team_name.lower().strip()
        if clean_name in self.team_lookup:
            return self.team_lookup[clean_name]
        
        # Try without common suffixes
        for suffix in [' 2', ' ii', ' b']:
            if clean_name.endswith(suffix):
                base = clean_name[:-len(suffix)].strip()
                if base in self.team_lookup:
                    return self.team_lookup[base]
        
        # Try fuzzy matching as last resort
        from rapidfuzz import fuzz, process
        
        best_match = process.extractOne(
            clean_name,
            list(self.team_lookup.keys()),
            scorer=fuzz.ratio,
            score_cutoff=80
        )
        
        if best_match:
            matched_key = best_match[0]
            logger.debug(f"Fuzzy matched '{team_name}' to '{matched_key}' (score: {best_match[1]})")
            return self.team_lookup[matched_key]
        
        logger.warning(f"Could not match team '{team_name}' to ASA database")
        self.stats['teams_not_matched'].add(team_name) # Add full name for unmatched teams
        return None
    
    def _extract_location(self, row) -> str:
        """Extract venue/location information from a row."""
        # Location is typically in the 5th cell (index 4)
        cells = row.find_all('td')
        if len(cells) > 4:
            location_cell = cells[4]
            # The actual location text might be within an 'alt' attribute of an image or a span in the 'table-subtext'
            
            # Check for img alt attribute first
            img_tag = cells[0].find('img') # Home team logo
            if img_tag and img_tag.get('alt'):
                alt_text = img_tag['alt'].strip()
                # Alt text often contains "City, State Team Name" e.g., "Chattanooga, TN Chattanooga FC"
                location_parts = alt_text.split(',')
                if len(location_parts) > 1:
                    city_state = f"{location_parts[0].strip()}, {location_parts[1].strip().split(' ')[0]}" # Extract city, state
                    return city_state
            
            # Fallback to the span with 'table-subtext' or direct text in location cell if available
            location_span = location_cell.find('span', class_='table-subtext') # Location is often in a span with class 'table-subtext'
            if location_span and location_span.get_text(strip=True) != '-': # Check if it's not the default empty placeholder
                return location_span.get_text(strip=True)
            
            # If nothing specific found, check the cell's direct text, excluding the '-'
            cell_text = location_cell.get_text(strip=True)
            if cell_text and cell_text != '-':
                return cell_text
                
        return "TBD"
    
    def _extract_score_or_status(self, row) -> str:
        """Extract game time from a row."""
        # Time is typically in the 4th cell (index 3), which has class 'cell-text broadcast'
        cells = row.find_all('td')
        if len(cells) > 3:
            time_cell = cells[3]
            time_link = time_cell.find('a') # Time is within an <a> tag
            if time_link:
                return time_link.get_text(strip=True)
        
        return "TBD"
    
    def scrape_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Scrape fixtures for a date range, iteratively checking dates.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of fixture dictionaries
        """
        all_fixtures = []
        base_url = "https://www.foxsports.com/soccer/mls-next-pro/schedule"
        
        # Initialize with the start date and a set to keep track of visited dates
        current_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates_to_visit = [current_date_obj.strftime('%Y-%m-%d')]
        visited_dates = set()
        
        logger.info(f"Starting scraping from {start_date} to {end_date}")
        
        while dates_to_visit:
            date_str = dates_to_visit.pop(0)
            
            if date_str in visited_dates:
                continue
            
            visited_dates.add(date_str)
            self.stats['dates_scraped'] += 1
            
            # Stop if we've gone past the end date
            if datetime.strptime(date_str, '%Y-%m-%d') > end_date_obj:
                logger.info(f"Reached end date {end_date}. Stopping.")
                break

            try:
                url = f"{base_url}?date={date_str}"
                logger.info(f"Scraping date: {date_str} (URL: {url})")
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract fixtures for this date (and potentially surrounding dates)
                fixtures_on_page = self._extract_fixtures_from_page(soup)
                
                if fixtures_on_page:
                    all_fixtures.extend(fixtures_on_page)
                    # Add distinct dates from the current page to dates_with_games
                    for f in fixtures_on_page:
                        if f['date'] not in self.stats['dates_with_games']:
                            self.stats['dates_with_games'].append(f['date'])
                    
                    logger.info(f"Found {len(fixtures_on_page)} fixtures on {date_str} (and possibly surrounding dates).")
                else:
                    logger.info(f"No fixtures found directly associated with date {date_str} on this page.")
                
                # Extract all available dates from the current page (including past/future)
                # and add them to dates_to_visit if they haven't been visited and are within range
                available_dates_from_page = self._extract_available_dates(soup)
                
                for next_date_str in available_dates_from_page:
                    next_date_obj = datetime.strptime(next_date_str, '%Y-%m-%d')
                    if next_date_str not in visited_dates and next_date_obj <= end_date_obj:
                        dates_to_visit.append(next_date_str)
                
                # Sort dates_to_visit to process chronologically
                dates_to_visit = sorted(list(set(dates_to_visit)))
                
                # Slow down to not make FOX mad.
                time.sleep(1)
                
            except requests.exceptions.RequestException as req_e:
                logger.error(f"Network error scraping date {date_str}: {req_e}")
                continue
            except Exception as e:
                logger.error(f"General error scraping date {date_str}: {e}")
                continue
        
        # Deduplicate fixtures
        unique_fixtures = self._deduplicate_fixtures(all_fixtures)
        self.stats['fixtures_found'] = len(unique_fixtures)
        
        return unique_fixtures
    
    def _deduplicate_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Remove duplicate fixtures."""
        seen = set()
        unique = []
        
        for fixture in fixtures:
            # Create a unique key based on teams and date
            key = f"{fixture['date']}-{fixture['home_team_id']}-{fixture['away_team_id']}"
            
            if key not in seen:
                seen.add(key)
                unique.append(fixture)
        
        return unique
    
    def _save_json_results(self, fixtures: List[Dict], filename: str):
        """
        Save the scraped fixtures to a JSON file.

        Args:
            fixtures: List of fixture dictionaries
            filename: Name of the JSON file to save
        """
        if not fixtures:
            logger.info("No fixtures to save to JSON.")
            return

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(fixtures, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved {len(fixtures)} fixtures to {filename}")
        except Exception as e:
            logger.error(f"Error saving fixtures to JSON: {e}")
    
    def print_summary(self) -> None:
        """Print a summary of the scraping results."""
        print("\n" + "="*70)
        print("FOX SPORTS SCRAPING SUMMARY")
        print("="*70)
        
        print(f"\nDates scraped (unique pages visited): {self.stats['dates_scraped']}")
        print(f"Total unique fixtures found: {self.stats['fixtures_found']}")
        
        if self.stats['dates_with_games']:
            # Ensure dates_with_games are sorted for min/max
            sorted_game_dates = sorted(list(set(self.stats['dates_with_games'])))
            print(f"Number of dates with games found: {len(sorted_game_dates)}")
            print(f"First game date: {sorted_game_dates[0]}")
            print(f"Last game date: {sorted_game_dates[-1]}")
        else:
            print("No specific game dates identified.")
        
        if self.stats['teams_not_matched']:
            print(f"\nUnrecognized team abbreviations (or full names not matched to ASA):")
            for abbrev in sorted(self.stats['teams_not_matched']):
                print(f"  - {abbrev}")
        else:
            print("\nAll teams successfully matched to ASA IDs.")
