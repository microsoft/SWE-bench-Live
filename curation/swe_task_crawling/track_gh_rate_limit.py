import time
import logging
import requests
from http.client import RemoteDisconnected
from ghapi.core import GhApi
from fastcore.net import HTTP403ForbiddenError
from urllib.error import URLError
import urllib3
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)

IDLE_THRESHOLD = 5  # Number of consecutive checks to consider a token IDLE

# Track the idle status count for each token
idle_count = {}

def check_rate_limit(token: str, token_id: int, retries: int = 5, delay: int = 10):
    """
    Check and log the current rate limit status for the token with retry logic.

    Args:
        token (str): GitHub token
        token_id (int): Token ID to associate with the PID file
        retries (int): Number of retry attempts
        delay (int): Delay between retries in seconds
    """
    api = GhApi(token=token)
    attempt = 0
    while attempt < retries:
        try:
            rl = api.rate_limit.get()
            # Get the GraphQL rate limit
            graphql_rl = rl.resources.graphql
            logger.info(
                f"Token {token[:20]}*:\n\tRESTful API remaining calls: {rl.resources.core.remaining}, reset time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rl.resources.core.reset))}\n\tGraphQL API remaining calls: {graphql_rl.remaining}, reset time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(graphql_rl.reset))}"
            )

            # Initialize idle_count for the token if not already present
            if token_id not in idle_count:
                idle_count[token_id] = 0

            if rl.resources.core.remaining == 5000 and graphql_rl.remaining == 5000:
                idle_count[token_id] += 1
            else:
                idle_count[token_id] = 0

            status = 'IDLE' if idle_count[token_id] >= IDLE_THRESHOLD else 'BUSY'

            pid_file = f"job_status/pid_{token_id}.txt"
            if os.path.exists(pid_file):
                with open(pid_file, 'r') as f:
                    pid = f.read().strip()
                with open(f'job_status/rate_limit_status_{token_id}.txt', 'w') as f:
                    f.write(f'{status} {pid}\n')
            else:
                logger.warning(f'PID file for token {token[:20]}* not found.')

            break  # Exit the loop if the API call is successful
        except (requests.exceptions.RequestException, URLError, RemoteDisconnected, \
                urllib3.exceptions.MaxRetryError, requests.exceptions.ConnectTimeout, \
                HTTP403ForbiddenError) as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed: Remote end closed connection without response. Retrying in {delay} seconds...")
            if attempt < retries:
                time.sleep(delay)
            else:
                logger.error("Max retry attempts reached.")
                logger.error(e)

if __name__ == "__main__":
    # List of GitHub API tokens
    with open("GH_token_sub.txt", 'r') as f:
        tokens = f.read().splitlines()
    print(f"Found {len(tokens)} tokens")

    # Keep the main thread running to allow background threads to work
    while True:
        for token_id, token in enumerate(tokens):
            check_rate_limit(token, token_id)
        print("\n*******\n")
        time.sleep(180)
