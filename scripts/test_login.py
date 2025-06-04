import requests
import json
import urllib3
import logging
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings()

base_url = "https://soccerapi.pkbipcas.com"

def check_server_status():
    """Check if the server is responding"""
    try:
        response = requests.get(f"{base_url}/health", verify=False)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Server health check failed: {e}")
        return False

def test_login_request(test_name, payload, content_type=None):
    """Helper function to test login requests with different formats"""
    print(f"Test: {test_name}")

    headers = {
        'Content-Type': content_type,
        'Accept': 'application/json'
    } if content_type else {'Accept': 'application/json'}

    try:
        if content_type == "application/x-www-form-urlencoded":
            # data = urlencode(payload) # This line is removed or commented out
            response = requests.post(
                f"{base_url}/auth/login",
                data=payload,  # Pass the dictionary directly
                headers=headers,
                verify=False
            )
        else:
            response = requests.post(
                f"{base_url}/auth/login",
                json=payload,  # Let requests handle JSON serialization
                headers=headers,
                verify=False
            )

        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        print(f"Request Headers: {headers}")
        print(f"Request Payload: {payload}")

        if test_name == "Form data request":
            if response.status_code == 422:
                logger.info("Form data request correctly received a 422 error (as expected, endpoint is JSON-only).")
            else:
                logger.error(f"Form data request received status {response.status_code}, but expected 422.")
                logger.debug(f"Status Code: {response.status_code}")
                logger.debug(f"Response: {response.text}")
                logger.debug(f"Request payload: {payload}")
                logger.debug(f"Request headers: {headers}")
        elif response.status_code >= 400: # For JSON request or other unexpected errors
            logger.error(f"Error occurred during {test_name}")
            logger.debug(f"Status Code: {response.status_code}")
            logger.debug(f"Response: {response.text}")
            logger.debug(f"Request payload: {payload}")
            logger.debug(f"Request headers: {headers}")

        return response

    except Exception as e:
        logger.error(f"Request failed during {test_name}: {str(e)}")
        return None

def main():
    # Check server status first
    if not check_server_status():
        logger.error("Server is not responding. Please check if it's running.")
        return

    print("Testing different login request formats...")

    credentials = {
        "username_or_email": "test_user2",
        "password": "secure_password_123"
    }

    # Test 1: JSON request
    test_login_request(
        "JSON request",
        credentials,
        "application/json"
    )

    # Test 2: Form data request
    test_login_request(
        "Form data request",
        credentials,
        "application/x-www-form-urlencoded"
    )

    # Test 3: Register test
    print("Test: Register endpoint")
    register_data = {
        "username": "test_user3",
        "email": "test3@example.com",
        "password": "secure_password_123"
    }

    try:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.post(
            f"{base_url}/auth/register",
            json=register_data,
            headers=headers,
            verify=False
        )
        print(f"Register Status: {response.status_code}")
        print(f"Register Response: {response.text[:200]}")
        print(f"Request Headers: {headers}")
        print(f"Request Payload: {register_data}")

        if response.status_code >= 400:
            logger.error("Registration failed")
            logger.debug(f"Status Code: {response.status_code}")
            logger.debug(f"Response: {response.text}")
            logger.debug(f"Request payload: {register_data}")
            logger.debug(f"Request headers: {headers}")
    except Exception as e:
        logger.error(f"Register request failed: {str(e)}")

if __name__ == "__main__":
    main()
