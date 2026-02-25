import pandas as pd
import os
from mlProject import logger
from sklearn.ensemble import RandomForestClassifier
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        
        train_x = pd.read_csv(self.config.train_data_path)
        test_x = pd.read_csv(self.config.test_data_path)
        # The target column is read as a dataframe on converted back to a series using the iloc[:,0]
        train_y = pd.read_csv(self.config.train_target_data_path).iloc[:,0]
        test_y = pd.read_csv(self.config.test_target_data_path).iloc[:,0]

        #logger.info(type(train_y), type(test_y))

        rf = RandomForestClassifier()
        rf.fit(train_x, train_y)

        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))
