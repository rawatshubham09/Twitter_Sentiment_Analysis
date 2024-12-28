import sys
from twitter_sentiment.exception import TwetterException
from twitter_sentiment.logger import logging
from twitter_sentiment.components.data_ingestion import DataIngestion
#from twitter_sentiment.components.data_validation import DataValidation
#from twitter_sentiment.components.data_transformation import DataTransformation
#from twitter_sentiment.components.model_trainer import ModelTrainer
#from twitter_sentiment.components.model_evaluation import ModelEvaluation
#from twitter_sentiment.components.model_pusher import ModelPusher


from twitter_sentiment.entity.config_entity import (DataIngestionConfig
                                         )
"""

DataValidationConfig,
                                         DataTransformationConfig,
                                         ModelTrainerConfig,
                                         ModelEvaluationConfig,
                                         ModelPusherConfig
"""


from twitter_sentiment.entity.artifact_entity import (DataIngestionArtifact
                                            )

"""DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact
                                            """

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise TwetterException(e, sys) from e  
        

    
    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
    
        except Exception as e:
            raise TwetterException(e, sys)
        