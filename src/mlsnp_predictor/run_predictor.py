import logging
import sys
import io
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor
from src.mlsnp_predictor import constants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/run_predictor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    logger.info("Starting MLS Next Pro Predictor execution...")

    try:
        predictor = MLSNPRegSeasonPredictor()
    
        # Adjust file paths in constants for predictor if necessary (e.g., for FIXTURES_FILE)
        # constants.FIXTURES_FILE = "data/fox_sports_mls_fixtures_20250526_170453.json" # Example if it needs to be overridden
        # For now, we assume constants.py has the correct default paths or they are handled within the class.

        summary_df, simulation_results, _, qualification_data = predictor.run_simulations() 
    
        if not summary_df.empty:
            logger.info("="*100)
            logger.info("  MLS NEXT PRO EASTERN CONFERENCE - SEASON PREDICTIONS")
            logger.info("="*100)
            
            display_columns = ["Team", "Current Points", "Games Played", "Current Shootout Wins",
                          "Worst Points", "Average Points", "Best Points",
                          "Average Final Rank", "Median Final Rank", "Best Rank", 
                          "Worst Rank", "Playoff Qualification %"]
            actual_display_columns = [col for col in display_columns if col in summary_df.columns]
            
            logger.info("\n" + summary_df[actual_display_columns].to_string(index=False))
            logger.info("="*100)
            
            shootout_analysis_df = predictor.create_shootout_analysis_table(summary_df, qualification_data)
            logger.info("\n" + "="*80)
            logger.info("  SHOOTOUT WINS IMPACT ANALYSIS (ESTIMATED)")
            logger.info("="*80)
            logger.info("\n" + shootout_analysis_df.to_string(index=False))
            logger.info("="*80)
            
            # Ensure plots are saved to output directory (handled within plot_results if updated, or save fig here)
            predictor.plot_results(summary_df, qualification_data) 
            
        else:
            logger.error("Unable to generate predictions - summary DataFrame is empty.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during predictor execution: {e}", exc_info=True)
        logger.error("Check the log files for detailed error information.")

    logger.info("MLS Next Pro Predictor execution finished.")

if __name__ == "__main__":
    main()
