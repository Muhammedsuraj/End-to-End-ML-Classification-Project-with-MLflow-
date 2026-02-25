import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, FunctionTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    
    # Converting objects into numeric
    def coerce_numeric_func(self, X):
        return X.apply(pd.to_numeric, errors="coerce")




    def get_data_transformer_obj(self):

        """"
        This function transforms the the dataset.
        
        """

        numeric_columns, categorical_columns, mixed_columns = [], [], []

        for col, dtype in self.config.all_schema.items():
            if dtype=="number": 
                numeric_columns.append(col)
            elif dtype=="object":
                categorical_columns.append(col)
            else:
                mixed_columns.append(col)

        

        num_pipeline= SkPipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

        cat_pipeline=SkPipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )


        numeric_coercer = FunctionTransformer(
                self.coerce_numeric_func,
                feature_names_out="one-to-one"
            )

        mixed_pipeline = SkPipeline(
                steps=[
                ("coerce_numeric", numeric_coercer),
                ("imputer",SimpleImputer(strategy="mean")),
                #("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
                #("target_encoder",TargetEncoder(smoothing=10, min_samples_leaf=5))
                ]
            )

        # ColumnTransformer
        preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numeric_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns),
                ("mixed_pipeline", mixed_pipeline, mixed_columns)
                ]
            )
        
#        # Smote Pipeline
#        train_data_preprocessor = Pipeline(
#            [
#                ("preprocessor", test_data_preprocessor),
#                ("smote", SMOTE())
#            ]
#        )

#        joblib.dump(train_data_preprocessor, os.path.join(self.config.root_dir, self.config.train_preprocessor_name))

        return preprocessor



    def initiate_data_transformation(self):
        data = pd.read_excel(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data, test_size=0.25, random_state=1)


        logger.info("Splitted data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        train_x = train.drop([self.config.target_column], axis=1)
        test_x = test.drop([self.config.target_column], axis=1)
        train_y = train[self.config.target_column]
        test_y = test[self.config.target_column]

        logger.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

        preprocessor = self.get_data_transformer_obj()

        train_x = preprocessor.fit_transform(train_x)
        test_x = preprocessor.transform(test_x)

        feature_names = preprocessor.get_feature_names_out()

        train_x = pd.DataFrame(train_x, columns=feature_names)
        test_x = pd.DataFrame(test_x, columns=feature_names)

        train_x.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test_x.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        train_y.to_csv(os.path.join(self.config.root_dir, "train_target.csv"),index = False)
        test_y.to_csv(os.path.join(self.config.root_dir, "test_target.csv"),index = False)

        print(train_x.shape)
        print(test_x.shape)
        print(train_y.shape)
        print(test_y.shape)
        print(type(train_y), type(test_y))

        joblib.dump(preprocessor, os.path.join(self.config.root_dir, self.config.preprocessor))

        return train_x, test_x, train_y, test_y