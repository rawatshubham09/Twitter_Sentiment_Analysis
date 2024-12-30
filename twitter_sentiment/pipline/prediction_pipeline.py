import os
import sys

import numpy as np
import pandas as pd
from twitter_sentiment.entity.config_entity import TweetsPredictorConfig, tfidfConfig
from twitter_sentiment.entity.s3_estimator import TweetsEstimator
from twitter_sentiment.exception import TwetterException
from twitter_sentiment.logger import logging
from twitter_sentiment.utils.main_utils import read_yaml_file, clean_text, load_object
from pandas import DataFrame

tfidf = load_object(tfidfConfig.data_transformation_dir)
pred_model = load_object(tfidfConfig.predict_model_dir)

class TweetsData:
    def __init__(self, tweet):
        """
        Tweet Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.tweet = tweet

        except Exception as e:
            raise TwetterException(e, sys) from e

    def get_tweets_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from TweetsData class input
        """
        try:
            
            tweets_input_dict = self.get_tweets_data_as_dict()
            return DataFrame(tweets_input_dict)
        
        except Exception as e:
            raise TwetterException(e, sys) from e


    def get_tweets_data_as_dict(self):
        """
        This function returns a dictionary from Tweet class input 
        """
        logging.info("Entered  prediction pipeline get_tweets_data_as_dict method as TweetsData class")

        try:
            input_data = {
                "tweet": [self.tweet],
            }

            logging.info("Created Tweets data dict")

            logging.info("Exited get_tweets_data_as_dict method as TweetsData class")

            return input_data

        except Exception as e:
            raise TwetterException(e, sys) from e

class TweetsClassifier:
    def __init__(self,prediction_pipeline_config: TweetsPredictorConfig = TweetsPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise TwetterException(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of TweetsClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of TweetsClassifier class")
            
            model = TweetsEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            

            print(dataframe["tweet"])
            # preprocessing the data
            try:
                result =  model.predict(dataframe)
            except:
                x = clean_text(dataframe["tweet"][0])

                x = tfidf.transform([x])
                x = x.toarray()

                result = pred_model.predict(x)
             
            print(result)
            return result
        
        except Exception as e:
            raise TwetterException(e, sys)