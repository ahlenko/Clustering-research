"""Application entry point.  Starts the desktop interface by default."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="run the legacy batch configuration")
    args = parser.parse_args()
    if args.batch:
        from config import Config
        from core.logger import setup_logger
        from core.experiment import ExperimentRunner
        import logging
        setup_logger()
        logger = logging.getLogger(__name__)
        runner = ExperimentRunner(Config())
        runner.run_all()
        logger.info("Експерименти завершено. Результати в папці reports/")
    else:
        from gui import launch
        launch()

if __name__ == "__main__":
    main()
