import sys
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


from twitter_sentiment.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from twitter_sentiment.entity.config_entity import DataTransformationConfig
from twitter_sentiment.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from twitter_sentiment.exception import TwetterException
from twitter_sentiment.logger import logging
from twitter_sentiment.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

port_stem = PorterStemmer()

stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
 "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

tfidf = TfidfVectorizer()

def clean_text(text):  
    pat1 = r'@[^ ]+'                   #@signs and value
    pat2 = r'https?://[A-Za-z0-9./]+'  #links
    pat3 = r'\'s'                      #floating s's
    pat4 = r'\#\w+'                     # hashtags and value
    pat5 = r'&amp '
    pat6 = r'[^A-Za-z\s]'         #remove non-alphabet
    combined_pat = r'|'.join((pat1, pat2,pat3,pat4,pat5, pat6))
    text = re.sub(combined_pat,"",text).lower()
    text = text.split()
    text = [word for word in text if not word in stop ]
    text = [port_stem.stem(word) for word in text]
    text = " ".join(text)
    return text.strip()

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise TwetterException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise TwetterException(e, sys)


    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
                
                train_df = train_df.dropna(subset=["tweet", "polarity"])
                test_df = test_df.dropna(subset=["tweet", "polarity"])

                transform_columns = "tweet"
                num_features = "polarity"

                # playing with train data

                input_feature_train_final = train_df[transform_columns].apply(clean_text)
                train_data_tweet = tfidf.fit_transform(train_df[transform_columns])
                target_feature_train_df = train_df[num_features].values

                target_feature_train_df = sp.csr_matrix(target_feature_train_df.reshape(-1, 1))

                logging.info("Got train features and test features of Training dataset")

                input_feature_test_final = test_df[transform_columns].apply(clean_text)
                test_data_tweet = tfidf.transform(test_df[transform_columns])
                target_feature_test_df = test_df[num_features].values

                target_feature_test_df = sp.csr_matrix(target_feature_test_df.reshape(-1, 1))

                
                # transforming train data using tfidf function
                
                
                # Transforming using

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                logging.info("Created train array and test array")
                
                # Assuming input_feature_train_final and 

                combined_data_sparse_train = sp.hstack([train_data_tweet, target_feature_train_df])

                combined_data_sparse_test = sp.hstack([test_data_tweet, target_feature_test_df])


                save_object(self.data_transformation_config.transformed_object_file_path, tfidf)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=combined_data_sparse_train)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=combined_data_sparse_test)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise TwetterException(e, sys) from e