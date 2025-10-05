import logging
import os
from datetime import datetime

LOG_PATH = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)
FILE_PATH = os.path.join(log_path, LOG_PATH)

logging.basicConfig(
    filename=FILE_PATH,
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


logging.info(f"Logging setup complete. Log file: {FILE_PATH}")