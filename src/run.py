"""Master file to run the pipeline"""

from pipelines import SnarXiv_training_pipeline

from zenml.logger import get_logger

logger = get_logger(__name__)

def main():
    """Primary entry point for pipeline execution"""

    logger.info("Beginning training pipeline...")
    SnarXiv_training_pipeline()
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    main()