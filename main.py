from networkSecurity.components.data_ingestion import DataIngestion
from networkSecurity.exceptionHandling.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
from networkSecurity.entity.config_entity import DataIngestionConfig
from networkSecurity.entity.config_entity import TrainingPipelineConfig
import os
import sys

if __name__ == "__main__":
    try:
        logging.info("Data Ingestion started")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        
        # df = data_ingestion.export_collection_as_dataframe()
        # logging.info("Exported collection as dataframe")
        
        # feature_store_df = data_ingestion.export_data_into_feature_store(dataframe=df)
        # logging.info("Exported data into feature store")
        
        # train_df, test_df = data_ingestion.split_data_as_train_test(dataframe=feature_store_df)
        # logging.info("Data ingestion completed successfully")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)