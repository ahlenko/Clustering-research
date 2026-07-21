"""
main.py – головна точка входу.
"""
from config import Config
from core.logger import setup_logger
from core.experiment import ExperimentRunner
import logging

def main():
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Старт дослідження алгоритмів кластеризації текстів")
    cfg = Config()
    runner = ExperimentRunner(cfg)
    runner.run_all()
    logger.info("Експерименти завершено. Результати в папці reports/")

if __name__ == "__main__":
    main()