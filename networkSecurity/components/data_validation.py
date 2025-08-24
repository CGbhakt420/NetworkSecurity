from networkSecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networkSecurity.entity.config_entity import DataValidationConfig
from networkSecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networkSecurity.logging.logger import logging
from networkSecurity.exceptionHandling.exception import NetworkSecurityException
from networkSecurity.utils.main_utils.utils import read_yaml, write_yaml
from scipy.stats import ks_2samp
import pandas as pd
import os,sys

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            schema_columns = self._schema_config['data_schema']['columns']
            dataframe_columns = dataframe.columns
        
            missing_columns = [col for col in schema_columns if col not in dataframe_columns]

            if missing_columns:
                logging.info(f"Missing columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05)->bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist=ks_2samp(d1, d2)
                if threshold<=is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column:{
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml(file_path=drift_report_file_path, content=report)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.training_file_path
            test_file_path = self.data_ingestion_artifact.testing_file_path
            
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            status = self.validate_columns(train_dataframe)
            if not status:
                error_message = f"Train dataframe doesnt contain all columns \n"
            status = self.validate_columns(test_dataframe)
            if not status:
                error_message = f"Test dataframe doesnt contain all columns \n"
                
            # Validate if numerical columns exist
            numerical_columns = self._schema_config['data_schema'].get('numerical_columns', [])
            missing_numerical_cols_train = [col for col in numerical_columns if col not in train_dataframe.columns]
            missing_numerical_cols_test = [col for col in numerical_columns if col not in test_dataframe.columns]

            if missing_numerical_cols_train:
                error_message = f"Train dataframe is missing numerical columns: {missing_numerical_cols_train}\n"
                logging.info(error_message)
                raise NetworkSecurityException(error_message, sys)
            if missing_numerical_cols_test:
                error_message = f"Test dataframe is missing numerical columns: {missing_numerical_cols_test}\n"
                logging.info(error_message)
                raise NetworkSecurityException(error_message, sys)
            
            #check datadrift
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.training_file_path,
                valid_test_file_path=self.data_ingestion_artifact.testing_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            
            return data_validation_artifact
        
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
 