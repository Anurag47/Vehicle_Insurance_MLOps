import sys
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame

class VehicleData:
    def __init__(self,
                Gender,
                Age,
                Driving_License,
                Region_Code,
                Previously_Insured,
                Annual_Premium,
                Policy_Sales_Channel,
                Vintage,
                Vehicle_Age,
                Vehicle_Damage
                ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Damage = Vehicle_Damage

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            logging.info(f"Converted vehicle data to Dict - {vehicle_input_dict}")
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_vehicle_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logging.info("Converts VehicleData class input to dictionary")

        try:
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage]
            }

            logging.info("Created vehicle data dict")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class VehicleDataClassifier:
    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig(),
                 ) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)
        
    def _map_columns(self, df):
        """
        Map Gender column to 0 for Female and 1 for Male.
        Map Vehicle_Damage col to 0 for No and 1 for Yes
        Map Vehicle_Age col to < 1 Year for 0, 1-2 Year for 1 & > 2 Years for 2}     
        """
        logging.info("Mapping Gender columns")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        logging.info("Mapping Vehicle Damage columns")
        df['Vehicle_Damage'] = df['Vehicle_Damage'].map( {'No': 0, 'Yes': 1} ).astype(int)
        logging.info("Mapping Vehicle Age columns")
        df['Vehicle_Age'] = df['Vehicle_Age'].map( {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2} ).astype(int)
        return df

    def _convert_to_cat_columns(self, df):
        """Converts columns to categorical type."""
        logging.info("Converting columns to categorical type")
        df['Region_Code'] = df['Region_Code'].astype(int).astype('category')
        df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('category')
        return df

    def predict(self, dataframe) -> str:
        """
        This is the method of VehicleDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")

            dataframe = self._map_columns(dataframe)
            dataframe = self._convert_to_cat_columns(dataframe)

            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)