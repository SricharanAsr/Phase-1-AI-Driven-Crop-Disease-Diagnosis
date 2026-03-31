import logging
import sys
from pathlib import Path
from colorama import init, Fore, Style

from logging.handlers import TimedRotatingFileHandler
import os

# Initialize colorama
init(autoreset=True)

class CustomFormatter(logging.Formatter):
    """Custom formatter for colorized output."""
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: Fore.CYAN + format_str + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + format_str + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + format_str + Style.RESET_ALL,
        logging.ERROR: Fore.RED + format_str + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format_str + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name: str = "CropDiagnosis", level: int = logging.INFO) -> logging.Logger:
    """Sets up a professional logger with console and rotating file handlers."""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers if already configured
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)

        # Rotating File handler (daily rotation, keeps 30 days)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = TimedRotatingFileHandler(
            log_dir / "app.log",
            when="D",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger

# Global logger instance
logger = setup_logger()
