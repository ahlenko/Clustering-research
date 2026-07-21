"""
logger.py – налаштування системи логування.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from config import Config

def setup_logger():
    cfg = Config()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Вивід у консоль
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Вивід у файл з ротацією
    file_handler = RotatingFileHandler(cfg.log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)