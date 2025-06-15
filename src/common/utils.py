import logging
import pandas as pd
import requests # requests is used by AsaClient, but not directly in safe_api_call. Keeping for now.
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

def safe_api_call(func: callable, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """
    Wrapper for API calls with error handling.
    This handles both exceptions and DataFrame ambiguity issues.
    """
    try:
        result = func(*args, **kwargs)
        
        # Handle DataFrame responses - convert to list of dictionaries
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return []
            return result.to_dict('records')
        
        # Handle None or empty responses
        if result is None:
            return []
            
        # If it's already a list, return as is
        if isinstance(result, list):
            return result
            
        # If it's a single item (and not None, which is handled above), wrap in list.
        # This ensures that an empty dict {} or empty list [] (if it somehow bypasses earlier checks)
        # gets wrapped, e.g. {} -> [{}]
        return [result]
        
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return []

def parse_game_date(date_string: str) -> Optional[datetime]:
    """
    Parse game date string into datetime object.
    Updated to handle both API format and fixture format dates.
    """
    if not date_string:
        return None
    
    # Remove common timezone indicators that might interfere with parsing
    clean_date = date_string.replace(' UTC', '').replace('UTC', '').replace(' Z', '').replace('Z', '')
    
    # Common date formats used in soccer APIs - updated based on actual API responses
    date_formats = [
        "%Y-%m-%d %H:%M:%S",      # 2025-05-24 23:00:00 (after removing UTC)
        "%Y-%m-%dT%H:%M:%S",      # 2025-05-24T23:00:00
        "%Y-%m-%d",               # 2025-05-24 (fixture format)
        "%Y-%m-%dT%H:%M:%S.%f",   # 2025-05-24T23:00:00.000000
    ]
    
    for fmt in date_formats:
        try:
            # Split on decimal point to handle microseconds if present
            date_part = clean_date.split('.')[0]
            return datetime.strptime(date_part, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_string}")
    return None
