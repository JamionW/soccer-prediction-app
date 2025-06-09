import logging
import sys
import io
from datetime import datetime
from src.mlsnp_scraper import FoxSportsMLSNextProScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/run_scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Fox Sports MLS Next Pro Scraper execution...")
    
    # Configuration
    asa_file = "data/asa_mls_next_pro_teams.json" 
    
    # Date range to scrape
    start_date = "2025-03-07" 
    end_date = "2025-10-31"   
    
    try:
        # Initialize scraper
        scraper = FoxSportsMLSNextProScraper(asa_file)
        
        logger.info(f"Scraping fixtures from {start_date} to {end_date}")
        
        # Scrape fixtures
        fixtures = scraper.scrape_date_range(start_date, end_date)
        
        # Print summary
        scraper.print_summary()
        
        # Save results
        if fixtures:
            json_filename = "output/fox_sports_mlsnp_fixtures.json"
            scraper._save_json_results(fixtures, json_filename)
            logger.info(f"JSON results saved to: {json_filename}")
            
            # A backup with timestamp just in case
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"output/archive/fox_sports_mlsnp_fixtures_{timestamp}.json"
            scraper._save_json_results(fixtures, backup_filename)
            logger.info(f"Backup saved to: {backup_filename}")
            
            logger.info("Sample fixtures found:")
            for fixture in fixtures[:5]:
                logger.info(f"  {fixture['date']}: {fixture['home_team']} vs {fixture['away_team']} at ({fixture['location']})")
            
            if len(fixtures) > 5:
                logger.info(f"  ... and {len(fixtures) - 5} more")
        else:
            logger.warning("No fixtures found. Possible reasons:")
            logger.warning("- The specified date range may not have games.")
            logger.warning("- Fox Sports may have changed their HTML structure significantly.")
            logger.warning(f"- The ASA team data file ('{asa_file}') might be missing or malformed.")
            logger.warning("- Check the log files for detailed error information.")
    
    except FileNotFoundError as fnf_error:
        logger.error(f"Configuration file not found: {fnf_error}")
        logger.error("Please ensure 'data/asa_mls_next_pro_teams.json' exists.")
    except ValueError as val_error:
        logger.error(f"Initialization error: {val_error}")
        logger.error("This might be due to issues with loading ASA team data.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during scraper execution: {e}", exc_info=True)
        logger.error("Check the log files for detailed error information.")
    
    logger.info("Fox Sports MLS Next Pro Scraper execution finished.")

if __name__ == "__main__":
    main()