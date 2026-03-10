import logging
import os

LOG_DIR = "../logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

import logging
import sys

class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg.encode(stream.encoding, errors='ignore').decode(stream.encoding) + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
            
logging.basicConfig(
    level=logging.ERROR,  # Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler(os.path.join(LOG_DIR, os.environ["LOG_FILE"])),  # Log to file
        SafeStreamHandler()         # Log to console
    ],
    encoding='utf-8',
)

# Create a logger for modules to use
# By setting propagate to False, we prevent the handlers from sending messages to the root logger
# This allows us to control the log output more precisely
logger = logging.getLogger("HiGRaAgent")
logger.propagate = True
