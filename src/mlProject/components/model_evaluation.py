import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        recall = recall_score(actual, pred)
        precision = precision_score(actual, pred)
        class_report = classification_report(actual, pred)
        return accuracy, f1, recall, precision, classification_report
    


    def log_into_mlflow(self):

        test_x = pd.read_csv(self.config.test_data_path)
        test_y = pd.read_csv(self.config.test_target_data_path).iloc[:,0]
        model = joblib.load(self.config.model_path)


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            pred = model.predict(test_x)

            (accuracy, f1, recall, precision, class_report) = self.eval_metrics(test_y, pred)
            
            # Saving metrics as local
            scores = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            #mlflow.log_metric("classification_report", class_report)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")