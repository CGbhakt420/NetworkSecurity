import yaml
from networkSecurity.exceptionHandling.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
import numpy as np
import os, sys
import dill
import pickle 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.metrics import r2_score, f1_score

def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def write_yaml(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object)->None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of Mainutils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def evaluate_models(x_train, y_train, x_test, y_test, models, params, n_iter=6):
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params.get(model_name, {})
            
            total_combinations = len(ParameterGrid(param)) if param else 0
            
            # If search space is small, do exhaustive GridSearchCV.
            # If search space is large/rich, do RandomizedSearchCV with n_iter to maintain high speed.
            if total_combinations <= n_iter or total_combinations <= 1:
                search = GridSearchCV(model, param, cv=3, scoring="f1_weighted", n_jobs=1)
            else:
                search = RandomizedSearchCV(
                    model, 
                    param_distributions=param, 
                    n_iter=n_iter, 
                    cv=3, 
                    scoring="f1_weighted", 
                    random_state=42, 
                    n_jobs=1
                )
            
            search.fit(x_train, y_train)
            
            best_model = search.best_estimator_
            models[model_name] = best_model
            
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)
            
            test_model_score = f1_score(y_test, y_test_pred, average="weighted")
            report[model_name] = test_model_score
            
            logging.info(f"Model: {model_name} | Best Params: {search.best_params_} | Weighted F1: {test_model_score:.4f}")
            
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)
