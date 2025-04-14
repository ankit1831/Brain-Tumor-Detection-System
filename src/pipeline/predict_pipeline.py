import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Variance: int,
        Standard_Deviation: int,
        Entropy:int,
        Skewness: int,
        Kurtosis: int,
        Contrast: int,
        Energy: int,
        ASM: int,
        Homogeneity: int,
        Dissimilarity: int):

        self.Variance = Variance

        self.Standard_Deviation= Standard_Deviation

        self.Entropy = Entropy

        self.Skewness = Skewness

        self.Kurtosis = Kurtosis

        self.Contrast = Contrast

        self.Energy = Energy

        self.ASM = ASM

        self.Homogeneity = Homogeneity

        self.Dissimilarity = Dissimilarity

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Variance": [self.Variance],
                "Standard_Deviation": [self.Standard_Deviation],
                "Entropy": [self.Entropy],
                "Kurtosis": [self.Kurtosis],
                "Skewness": [self.Skewness],
                "Contrast": [self.Contrast],
                "Energy": [self.Energy],
                "ASM": [self.ASM],
                "Homogeneity": [self.Homogeneity],
                "Dissimilarity": [self.Dissimilarity],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

