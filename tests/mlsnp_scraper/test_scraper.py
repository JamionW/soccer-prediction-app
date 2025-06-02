import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from datetime import datetime # Added for dynamic year
import pandas as pd
from bs4 import BeautifulSoup
from src.mlsnp_scraper.fox_schedule_scraper import FoxSportsMLSNextProScraper
import json
import os
import logging

# Disable logging for tests to keep output clean
logging.disable(logging.CRITICAL)

class TestFoxSportsMLSNextProScraper(unittest.TestCase):
    def setUp(self):
        """Set up for test methods."""
        self.mock_asa_teams_data = [
            {"team_id": "ID1", "team_name": "Real Team Name"},
            {"team_id": "ID2", "team_name": "Another Team FC"},
            {"team_id": "ID3", "team_name": "Team III"},
            {"team_id": "IDCHT", "team_name": "Chattanooga FC"},
            {"team_id": "IDNYC", "team_name": "New York City FC II"},
            {"team_id": "IDNER", "team_name": "New England Revolution II"}
        ]

        # Mock _load_asa_teams directly on the class to affect the instance creation
        self.patcher = patch.object(FoxSportsMLSNextProScraper, '_load_asa_teams', return_value=self.mock_asa_teams_data)
        self.mock_load_asa_teams = self.patcher.start()

        self.scraper = FoxSportsMLSNextProScraper(asa_teams_file_path="dummy_path.json")

    def tearDown(self):
        """Clean up after test methods."""
        self.patcher.stop()

    def test_load_asa_teams_success(self):
        # This test now tests the setup indirectly, or we can test it by calling it on a new instance
        # For this exercise, we assume setUp's patch handles the direct call during __init__
        # To test _load_asa_teams isolated, we would unpatch and patch open
        
        # To directly test _load_asa_teams, we need a separate instance or unpatch
        scraper_for_load_test = FoxSportsMLSNextProScraper(asa_teams_file_path="dummy_path.json") # _load_asa_teams is already patched
        
        sample_data = '[{"team_id": "T1", "team_name": "Team One"}]'
        expected_data = [{"team_id": "T1", "team_name": "Team One"}]

        # Unpatch _load_asa_teams for this specific test scope if you want to test its internal logic
        self.patcher.stop() # Stop the class-level patch

        with patch("builtins.open", mock_open(read_data=sample_data)) as mock_file:
            with patch("pathlib.Path.exists", return_value=True): # Mock path.exists()
                # Need to re-initialize a scraper or call method directly if it's static/class method
                # For instance method, it's tricky without re-init or making it static
                # Let's assume we are testing a new instance's initialization path for _load_asa_teams
                temp_scraper = FoxSportsMLSNextProScraper(asa_teams_file_path="another_dummy.json")
                self.assertEqual(temp_scraper.asa_teams, expected_data)
                mock_file.assert_called_once_with(Path("another_dummy.json"), 'r', encoding='utf-8')
        
        self.mock_load_asa_teams = self.patcher.start() # Restart the class-level patch for other tests

    def test_load_asa_teams_file_not_found(self):
        self.patcher.stop() # Stop the class-level patch to test the real _load_asa_teams
        # Path.exists will be mocked by the FoxSportsMLSNextProScraper's _load_asa_teams
        # The important thing is that open raises FileNotFoundError
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("pathlib.Path.exists", return_value=False): # Ensure path.exists returns False
                with self.assertRaises(ValueError) as context:
                    FoxSportsMLSNextProScraper(asa_teams_file_path="non_existent.json")
                self.assertIn("Could not load ASA team data", str(context.exception))
        self.mock_load_asa_teams = self.patcher.start() # Restart the patch for other tests

    def test_load_asa_teams_invalid_json(self):
        self.patcher.stop() # Stop the class-level patch
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with patch("pathlib.Path.exists", return_value=True): # Ensure path.exists returns True
                with self.assertRaises(ValueError) as context:
                    FoxSportsMLSNextProScraper(asa_teams_file_path="invalid_data.json")
                self.assertIn("Could not load ASA team data", str(context.exception))
        self.mock_load_asa_teams = self.patcher.start() # Restart the patch

    def test_resolve_team_from_abbreviation_known(self):
        self.assertEqual(self.scraper._resolve_team_from_abbreviation("CHT", BeautifulSoup("", "html.parser")), "Chattanooga FC")

    def test_resolve_team_from_abbreviation_ambiguous_nyc(self):
        html_row = '<tr><td><img src="https://b.fssta.com/uploads/application/soccer/team-logos/new-york-city-fc-2.vresize.72.72.medium.0.png"/></td></tr>'
        soup_row = BeautifulSoup(html_row, "html.parser")
        self.assertEqual(self.scraper._resolve_team_from_abbreviation("NEW", soup_row), "New York City FC II")

    def test_resolve_team_from_abbreviation_ambiguous_ner(self):
        html_row = '<tr><td><img src="https://b.fssta.com/uploads/application/soccer/team-logos/new-england-revolution-2.vresize.72.72.medium.0.png"/></td></tr>'
        soup_row = BeautifulSoup(html_row, "html.parser")
        self.assertEqual(self.scraper._resolve_team_from_abbreviation("NEW", soup_row), "New England Revolution II")

    def test_resolve_team_from_abbreviation_unknown(self):
        self.assertIsNone(self.scraper._resolve_team_from_abbreviation("UNKNOWN", BeautifulSoup("", "html.parser")))

    def test_match_to_asa_id_exact(self):
        self.assertEqual(self.scraper._match_to_asa_id("Real Team Name"), "ID1")

    def test_match_to_asa_id_case_insensitive(self):
        self.assertEqual(self.scraper._match_to_asa_id("real team name"), "ID1")

    def test_match_to_asa_id_suffix_stripping(self):
        # Based on the mock_asa_teams_data, "Team III" -> "Team" should match "ID3"
        # The _create_team_lookup logic adds variations like "Team" from "Team III"
        self.assertEqual(self.scraper._match_to_asa_id("Team"), "ID3")
        self.assertEqual(self.scraper._match_to_asa_id("Another Team"), "ID2") # From "Another Team FC"

    def test_match_to_asa_id_not_found(self):
        self.assertIsNone(self.scraper._match_to_asa_id("Unknown Team Name"))

    def test_extract_fixtures_from_page_basic(self):
        # Simplified HTML structure
        html_page = """
        <html><body>
            <div class="table-segment">
                <div class="table-title">SAT, AUG 23</div>
                <table class="data-table">
                    <tr id="tbl-row-1">
                        <td><a class="table-entity-name">CHT</a><img src="..."/></td>
                        <td>vs</td>
                        <td><a class="table-entity-name">NYC</a><img src="...new-york-city-fc-2..."/></td>
                        <td><a href="#">7:00 PM ET</a></td>
                        <td><span>Some Location, ST</span></td>
                    </tr>
                </table>
            </div>
        </body></html>
        """
        # Corrected to use NYC for NEW abbreviation for New York City FC II
        # For the purpose of this test, let's assume 'NYC' is a valid abbreviation for 'New York City FC II'
        # or that _resolve_team_from_abbreviation correctly handles it.
        # To make it more robust, we should ensure 'NYC' abbreviation is handled or use 'NEW'
        # Let's adjust the test to use 'NEW' for NYCFC II as per scraper's actual logic
        
        html_page_corrected = """
        <html><body>
            <div class="table-segment">
                <div class="table-title">SAT, AUG 23</div>
                <table class="data-table">
                    <tr id="tbl-row-1">
                        <td><a class="table-entity-name">CHT</a><img src="..."/></td>
                        <td>vs</td>
                        <td><a class="table-entity-name">NEW</a><img src="https://b.fssta.com/uploads/application/soccer/team-logos/new-york-city-fc-2.vresize.72.72.medium.0.png"/></td>
                        <td><a href="#">7:00 PM ET</a></td>
                        <td><span>Some Location, ST</span></td>
                    </tr>
                </table>
            </div>
        </body></html>
        """
        soup = BeautifulSoup(html_page_corrected, "html.parser")
        
        # Mock _resolve_team_from_abbreviation and _match_to_asa_id for focused testing if needed,
        # but for this "basic structure" test, we can rely on the ones set up in setUp.
        # self.scraper.FOX_ABBREVIATIONS['NYC'] = 'New York City FC II' # if we used NYC

        fixtures = self.scraper._extract_fixtures_from_page(soup)
        self.assertIsInstance(fixtures, list)
        self.assertEqual(len(fixtures), 1)
        
        if fixtures:
            fixture = fixtures[0]
            self.assertEqual(fixture['home_team'], "Chattanooga FC")
            self.assertEqual(fixture['away_team'], "New York City FC II")
            self.assertIn(f"{datetime.now().year}-08-23", fixture['date']) # Dynamic year
            self.assertEqual(fixture['time'], "7:00 PM ET")
            # Location extraction is more complex, for basic test just check it's there
            self.assertIsNotNone(fixture['location'])


if __name__ == '__main__':
    unittest.main()
