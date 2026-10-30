import os
import sys
from networkSecurity.exceptionHandling.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
from networkSecurity.entity.artifact_entity import DataTrasformationArtifact, ModelTrainerArtifact
from networkSecurity.entity.config_entity import ModelTrainingConfig

from networkSecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networkSecurity.utils.ml_utils.model.estimator import NetworkModel
from networkSecurity.utils.ml_utils.metric.classification_metric import get_classification_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow
import dagshub

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainingConfig, data_transformation_artifact: DataTrasformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classification_metric):
        try:
            with mlflow.start_run():
                f1_score = classification_metric.f1_score
                precision_score = classification_metric.precision_score
                recall_score = classification_metric.recall_score
                
                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)
                
                mlflow.sklearn.log_model(best_model, artifact_path="model")
        except Exception as e:
            logging.warning(f"MLflow tracking skipped or failed: {e}")
                    
    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Logistic Regression": LogisticRegression()
            }
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [None, 'sqrt', 'log2']
                },
                "Random Forest": {
                    'n_estimators': [32, 64, 100],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 15, 25],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [32, 64, 100],
                    'subsample': [0.7, 0.85, 1.0],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5]
                },
                "AdaBoost": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                    'n_estimators': [32, 64, 100]
                },
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'max_iter': [300]
                }
            }
            
            model_report: dict = evaluate_models(x_train, y_train, x_test, y_test, models, params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            logging.info(f"Selected Best Model: {best_model_name} with Weighted F1 Score: {best_model_score:.4f}")
            logging.info(f"Best Model Hyperparameters: {best_model.get_params()}")
            
            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_metrics(y_true=y_train, y_pred=y_train_pred)
            
            # Track the experiments with mlflow (safe)
            self.track_mlflow(best_model, classification_train_metric)
            
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_metrics(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model, classification_test_metric)
            
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            Network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_model)
            
            os.makedirs("final_models", exist_ok=True)
            save_object("final_models/model.pkl", best_model)
            
            # Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
                best_model_name=best_model_name
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            model = self.train_model(x_train, y_train, x_test, y_test)
            return model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
