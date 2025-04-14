import os
import sys
import matplotlib 
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier



from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,fbeta_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "SVC": SVC(),
                "GaussianNB": GaussianNB(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                
            }
            '''
            params={
                

                "Logistic Regression": {
                    #'penalty':['l1', 'l2', 'elasticnet'],
                    'C':[100,10,1.0,0.1,0.01],
                    #'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                },

                "SVC":{
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf','sigmoid','poly','linear']

                },

                "GaussianNB": {},

                "KNeighborsClassifier":{},

                "Random Forest":{
                    'criterion':['entropy', 'log_loss', 'gini'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },


                "Decision Tree": {
                    'criterion':['gini','entropy', 'log_loss'],
                    'splitter':['best','random'],
                    'max_depth':[1,2,3,4,5],
                    'max_features':['sqrt','log2']
                },
                
                "Gradient Boosting":{
                    'loss':['log_loss', 'exponential'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },


                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    #'loss':['algorithm', 'estimator', 'learning_rate', 'n_estimators', 'random_state'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                
            }
            '''    
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)#,param=params)
           
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<60:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            classification_repor=confusion_matrix(y_test,predicted)
            accuracy = accuracy_score(y_test, predicted)*100


            

            return accuracy,best_model,classification_repor
            



            
        except Exception as e:
            raise CustomException(e,sys)