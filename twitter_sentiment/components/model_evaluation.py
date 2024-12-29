from twitter_sentiment.entity.config_entity import ModelEvaluationConfig, DataTransformationConfig
from twitter_sentiment.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact,DataTransformationArtifact
from sklearn.metrics import f1_score
from twitter_sentiment.exception import TwetterException
from twitter_sentiment.constants import TARGET_COLUMN, CURRENT_YEAR
from twitter_sentiment.utils.main_utils import get_stop_word, clean_text, load_object
from twitter_sentiment.logger import logging
from nltk.stem.porter import PorterStemmer
import sys
import pandas as pd
from typing import Optional
from twitter_sentiment.entity.s3_estimator import TweetsEstimator
from dataclasses import dataclass
from twitter_sentiment.entity.estimator import TweetsModel

stop = get_stop_word()
port_stem = PorterStemmer()

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise TwetterException(e, sys) from e

    def get_best_model(self) -> Optional[TweetsEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            tweets_estimator = TweetsEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if tweets_estimator.is_model_present(model_path=model_path):
                return tweets_estimator
            return None
        except Exception as e:
            raise  TwetterException(e,sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Inside Model Evalulation")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            stop = get_stop_word()

            test_df["tweet"] = test_df["tweet"].apply(clean_text)

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info(f" X shape {x.shape}, y shape {y.shape}")
            #preprocessing X data
            

            logging.info("Text cleaning completed")

            transform_tfidf = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            x = transform_tfidf.transform(x)
            x = x.toarray()

            logging.info(f"tfidf object created and x array shape is : {x.shape}")

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"f1 score {best_model_f1_score}")
            
            
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise TwetterException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                #is_model_accepted=evaluate_model_response.is_model_accepted,
                is_model_accepted=False
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise TwetterException(e, sys) from e