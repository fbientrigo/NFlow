import json
import logging
import os

logger = logging.getLogger(__name__)

def get_next_attempt_number(base_dir="outputs"):
# Creating incremental running directories

    os.makedirs(base_dir, exist_ok=True)
    metadata_file = os.path.join(base_dir, "run_metadata.json")
    attempt_number = 1
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            attempt_number = metadata.get("last_attempt", 0) + 1
    with open(metadata_file, "w") as f:
        json.dump({"last_attempt": attempt_number}, f)
    run_dir = os.path.join(base_dir, f"run_{attempt_number}")
    os.makedirs(run_dir, exist_ok=True)
    
    return attempt_number, run_dir
