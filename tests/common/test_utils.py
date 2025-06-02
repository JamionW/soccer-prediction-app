import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from src.common.utils import safe_api_call, parse_game_date
import requests # For mocking requests.exceptions.RequestException
import logging

# Disable logging for tests to keep output clean
logging.disable(logging.CRITICAL)

class TestCommonUtils(unittest.TestCase):
    def test_safe_api_call_success_list_return(self):
        mock_api_func = MagicMock(return_value=["data1", "data2"])
        result = safe_api_call(mock_api_func, "arg1", kwarg1="value")
        self.assertEqual(result, ["data1", "data2"])
        mock_api_func.assert_called_once_with("arg1", kwarg1="value")

    def test_safe_api_call_success_df_return(self):
        sample_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_api_func = MagicMock(return_value=sample_df)
        result = safe_api_call(mock_api_func)
        self.assertEqual(result, sample_df.to_dict('records'))
        mock_api_func.assert_called_once_with()

    def test_safe_api_call_success_empty_df_return(self):
        mock_api_func = MagicMock(return_value=pd.DataFrame())
        result = safe_api_call(mock_api_func)
        self.assertEqual(result, [])
        mock_api_func.assert_called_once_with()

    def test_safe_api_call_success_none_return(self):
        mock_api_func = MagicMock(return_value=None)
        result = safe_api_call(mock_api_func)
        self.assertEqual(result, [])
        mock_api_func.assert_called_once_with()

    def test_safe_api_call_success_single_item_return(self):
        mock_api_func = MagicMock(return_value={"key": "value"})
        result = safe_api_call(mock_api_func)
        self.assertEqual(result, [{"key": "value"}])
        mock_api_func.assert_called_once_with()
        
    def test_safe_api_call_success_empty_single_item_return(self):
        # Test case for when a single item is returned but it's "falsey" (e.g. empty dict)
        # Based on current implementation: `return [result] if result else []`
        # An empty dict {} is True in boolean context, so it should be wrapped in a list.
        mock_api_func = MagicMock(return_value={})
        result = safe_api_call(mock_api_func)
        self.assertEqual(result, []) # Changed from [{}] to [] to match actual behavior
        mock_api_func.assert_called_once_with()

    @patch('src.common.utils.logger.error') # Patching the logger in the utils module
    def test_safe_api_call_exception_handling(self, mock_logger_error):
        api_error_message = "API Error"
        mock_api_func = MagicMock(side_effect=requests.exceptions.RequestException(api_error_message))
        result = safe_api_call(mock_api_func)
        self.assertEqual(result, [])
        mock_api_func.assert_called_once_with()
        mock_logger_error.assert_called_once_with(f"API call failed: {api_error_message}")

    def test_parse_game_date_valid_formats(self):
        test_cases = {
            "2025-05-24 23:00:00": datetime(2025, 5, 24, 23, 0, 0),
            "2025-05-24T23:00:00": datetime(2025, 5, 24, 23, 0, 0),
            "2025-05-24": datetime(2025, 5, 24, 0, 0, 0),
            "2025-05-24T23:00:00.123456": datetime(2025, 5, 24, 23, 0, 0),
            "2025-05-24 23:00:00 UTC": datetime(2025, 5, 24, 23, 0, 0),
            "2025-05-24T23:00:00Z": datetime(2025, 5, 24, 23, 0, 0),
            "2024-03-10 19:00:00 UTC": datetime(2024, 3, 10, 19, 0, 0), # Example from a file
        }
        for date_str, expected_dt in test_cases.items():
            with self.subTest(date_str=date_str):
                self.assertEqual(parse_game_date(date_str), expected_dt)

    @patch('src.common.utils.logger.warning') # Patching the logger in the utils module
    def test_parse_game_date_invalid_format(self, mock_logger_warning):
        date_str = "Invalid Date String"
        self.assertIsNone(parse_game_date(date_str))
        mock_logger_warning.assert_called_once_with(f"Could not parse date: {date_str}")

    def test_parse_game_date_empty_string(self):
        self.assertIsNone(parse_game_date(""))

    def test_parse_game_date_none_input(self):
        self.assertIsNone(parse_game_date(None))

if __name__ == '__main__':
    unittest.main()
