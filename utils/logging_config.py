import logging
import logging.config
import os


def setup_logging(log_config, run_dir):
    # Determine the log file path
    log_file = os.path.join(run_dir, log_config.get("file", "logs/project.log"))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": log_config.get("level", "INFO"),
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "standard",
                "level": log_config.get("level", "INFO"),
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": log_config.get("level", "INFO"),
        },
    }
    logging.config.dictConfig(logging_config)
