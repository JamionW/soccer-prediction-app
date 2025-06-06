import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command: list, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command and arguments as a list
        description: Description of what's being run
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(command)}")
    
    # Correctly determine the project root (the parent directory of 'src')
    # This ensures that 'src' is a top-level package Python can find.
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Set the environment to include the project root in Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(project_root) # Set the working directory to the project root
        )
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False


def main():
    """Run the complete fixture update workflow."""
    
    print("\n" + "="*70)
    print("MLS NEXT PRO FIXTURE UPDATE WORKFLOW")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Run the Fox Sports scraper
    logger.info("\nStep 1: Scraping fixtures from Fox Sports...")
    # Execute the script as a module to ensure correct package resolution
    if not run_command(
        [sys.executable, "-m", "src.mlsnp_scraper.run_fox_scraper"],
        "Fox Sports scraper"
    ):
        logger.error("Failed to scrape fixtures. Exiting.")
        return 1
    
    # Check if the output file was created (Corrected filename)
    fixture_file = Path("output/fox_sports_mlsnp_fixtures.json")
    if not fixture_file.exists():
        logger.error(f"Expected output file not found: {fixture_file}")
        logger.error("The scraper might have run but produced no output. Check logs.")
        return 1
    
    logger.info(f"✓ Fixture file created: {fixture_file}")
    
    # Step 2: Load fixtures into database
    logger.info("\nStep 2: Loading fixtures into database...")
    # Execute the loader as a module
    if not run_command(
        [sys.executable, "-m", "src.mlsnp_scraper.game_loader"],
        "Database fixture loader"
    ):
        logger.error("Failed to load fixtures into database.")
        return 1
    
    # Success!
    print("\n" + "="*70)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("- The fixtures are now in your database")
    print("- ASA API will fill in match details as games are played")
    print("- Your simulators can use the future games for predictions")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)