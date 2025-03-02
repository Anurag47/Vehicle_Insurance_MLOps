import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, save_json_data, load_json_data


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            ss_columns = self._schema_config['ss_columns']
            mm_columns = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, ss_columns),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def _map_columns(self, df):
        """
        Map Gender column to 0 for Female and 1 for Male.
        Map Vehicle_Damage col to 0 for No and 1 for Yes
        Map Vehicle_Age col to < 1 Year for 0, 1-2 Year for 1 & > 2 Years for 2}     
        """
        logging.info("Mapping columns")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        df['Vehicle_Damage'] = df['Vehicle_Damage'].map( {'No': 0, 'Yes': 1} ).astype(int)
        df['Vehicle_Age'] = df['Vehicle_Age'].map( {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2} ).astype(int)
        return df
    
    def _tranform_policy_sales_channel(self, df, indicator):
        """
        From the indicator value, if 1 then train_data else test_data
        If train_data then create a new_artifact called allowed_policy_sales_channel.json
        If test data use the same artifact to transform the data
        Here we check if the policy_sales_channel value_count() > 1000 if so we keep that value else we 
        replace it with 0  
        """
        logging.info("Transforming policy_sales_channel column")
        filtered_policy_sales = None
        if indicator == 1:
            policy_sales_counts = df['Policy_Sales_Channel'].value_counts()
            filtered_policy_sales = policy_sales_counts[policy_sales_counts > 1000].to_dict()
            save_json_data(self.data_transformation_config.filtered_policy_sales_path, filtered_policy_sales)
        else:
            filtered_policy_sales = load_json_data(self.data_transformation_config.filtered_policy_sales_path)
        
        df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].map(lambda x: x if x in filtered_policy_sales else 0).astype(int)
        return df

    def _convert_to_cat_columns(self, df):
        """Converts columns to categorical type."""
        logging.info("Converting columns to categorical type")
        df['Region_Code'] = df['Region_Code'].astype(int).astype('category')
        df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('category')
        return df

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            input_feature_train_df = self._map_columns(input_feature_train_df)
            input_feature_train_df = self._tranform_policy_sales_channel(input_feature_train_df, 1)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._convert_to_cat_columns(input_feature_train_df)

            input_feature_test_df = self._map_columns(input_feature_test_df)
            input_feature_test_df = self._tranform_policy_sales_channel(input_feature_test_df, 0)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._convert_to_cat_columns(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                filtered_policy_sales_path=self.data_transformation_config.filtered_policy_sales_path
            )

        except Exception as e:
            raise MyException(e, sys) from e