from networkSecurity.components.data_ingestion import DataIngestion
from networkSecurity.components.data_validation import DataValidation
from networkSecurity.components.data_transformation import DataTransformation
from networkSecurity.components.model_trainer import ModelTrainer
from networkSecurity.exceptionHandling.exception import NetworkSecurityException

from networkSecurity.logging.logger import logging
from networkSecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainingConfig
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
        
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Data validation started")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)
        
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Data transformation started")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifact)
        
        logging.info("Model training started")
        model_trainer_config = ModelTrainingConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training artifact created")
        print(model_trainer_artifact)

        # df = data_ingestion.export_collection_as_dataframe()
        # logging.info("Exported collection as dataframe")
        
        # feature_store_df = data_ingestion.export_data_into_feature_store(dataframe=df)
        # logging.info("Exported data into feature store")
        
        # train_df, test_df = data_ingestion.split_data_as_train_test(dataframe=feature_store_df)
        # logging.info("Data ingestion completed successfully")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)