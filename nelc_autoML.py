import optuna
import pandas as pd
import numpy as np
from lightgbm import *
from tqdm import tqdm
import logging
import gc
import os
from datetime import timedelta
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,r2_score,f1_score
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.neighbors import *
import dill as pickle
import warnings
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import json
from datetime import datetime
import optuna.visualization as vis
from sklearn.impute import KNNImputer,SimpleImputer
logging.basicConfig(
    # filename='logs.log', 
    # filemode='a', 
    # format='%(asctime)s - %(levelname)s : %(message)s', 
    level=logging.INFO
    )
def log_message(level, message):
    if level.lower() == 'info':
        print("[INFO]: ",message)
        logging.info(message)
    elif level.lower() == 'error':
        print("[ERROR]: ",message)
        logging.error(message)
    elif level.lower() == 'warning':
        print("[WARNING]: ",message)
        logging.warning(message)
    elif level.lower() == 'debug':
        print("[DEBUG]: ",message)
        logging.debug(message)
    else:
        logging.critical('Unsupported logging level: ' + level)

class AutoML:
    def __init__(self,data:pd.DataFrame,interpretability:float,target_column:str,data_preprocessing:bool=False):
        self.data = data.copy()
        self.interpretability = interpretability
        self.target_column = target_column
        self.data_preprocessing = data_preprocessing
        if self.target_column not in list(self.data.columns):
            raise RuntimeError(f"the target column should be on of the data columns : {list(self.data.columns)}")
        self.features_num = []
        self.features_cat = []
        self.features_date = []
        self.features_bool = []
        self.ml_algorithms = pd.read_csv('ml_algorithms.csv')
        self.ml_algorithms_parameters = json.load(open('ml_algorithms_parameters.json','r'))
        for col in self.data.columns:
            if str(self.data[col].dtype) in ['float32','float64','int32','int64']:
                self.features_num.append(col)
            elif 'datetime' in str(self.data[col].dtype):
                self.features_date.append(col)
            else:
                self.features_cat.append(col)
        log_message('debug',self.data.info())
        self.encoders = {}
        log_message('info',list(self.data[self.target_column].unique()))
        self.refine_the_data()
        log_message('info',list(self.data[self.target_column].unique()))
        log_message('debug',self.data.info())
        if 'float' in str(self.data[self.target_column].dtype) or 'int' in str(self.data[self.target_column].dtype):
            self.features_num.remove(self.target_column)
        elif 'bool' in str(self.data[self.target_column].dtype):
            self.features_bool.remove(self.target_column)
        else:
            self.features_cat.remove(self.target_column)
        self.original_data = self.data
            
    def refine_the_data(self):
        log_message('debug','Start casting to numarical features')
        # from catagorical data to numarical
        new_cols = []
        for col in tqdm(self.features_cat):
            try:
                self.data[col] = self.data[col].astype('float64')
                self.features_num.append(col)
                new_cols.append(col)
            except:
                continue
        for col in new_cols:
            self.features_cat.remove(col)
            
        # from catagorical data to dates
        log_message('debug','Start casting to date features')
        new_cols = []
        for col in tqdm(self.features_cat):
            try:
                warnings.simplefilter(action='ignore', category=UserWarning)
                self.data[col] = pd.to_datetime(self.data[col])
                self.features_date.append(col)
                new_cols.append(col)
            except Exception as e:
                continue
        for col in new_cols:
            self.features_cat.remove(col)

        # Calculate the n unique
        cols_nunique = self.data[self.features_cat].nunique()
        cols_nunique_bool = cols_nunique[cols_nunique==2]
        cols_nunique_id = cols_nunique[cols_nunique==self.data.shape[0]]
        cols_nunique_alot = cols_nunique[cols_nunique>100]
        
        # from catagorical to bool
        log_message('debug','Start casting to bool features')
        for col in tqdm(cols_nunique_bool.index):
            self.encoders[col] = {}
            uniques = list(self.data[col].unique())
            for i,j in enumerate(uniques):
                self.encoders[col][j] = bool(i)
            self.data[col] = self.data[col].map(self.encoders[col])
            self.features_cat.remove(col)
            self.features_bool.append(col)
            
        # from numarical to bool
        cols_nunique = self.data[self.features_num].nunique()
        cols_nunique_bool = cols_nunique[cols_nunique==2]
        for col in tqdm(cols_nunique_bool.index):
            self.encoders[col] = {}
            uniques = list(self.data[col].unique())
            for i,j in enumerate(uniques):
                self.encoders[col][j] = bool(i)
            self.data[col] = self.data[col].map(self.encoders[col])
            self.features_num.remove(col)
            self.features_bool.append(col)

        # drop ids columns and columns with a lot of unique values
        self.data = self.data.drop(columns=list(cols_nunique_id.index)+list(cols_nunique_alot.index))
        for col in set(list(cols_nunique_id.index)+list(cols_nunique_alot.index)):
            self.features_cat.remove(col)
            
        # Add date features
        if self.features_date:
            log_message('debug','Start creating date features')
        for col in tqdm(self.features_date):
            self.data[col+'_'+'day_of_year'] = self.data[col].dt.day_of_year
            self.features_num.append(col+'_'+'day_of_year')
            self.data[col+'_'+'quarter'] = self.data[col].dt.quarter
            self.features_num.append(col+'_'+'quarter')
            self.data[col+'_'+'day_of_week'] = self.data[col].dt.day_of_week 
            self.features_num.append(col+'_'+'day_of_week')
            self.data[col+'_'+'days_in_month'] = self.data[col].dt.days_in_month 
            self.features_num.append(col+'_'+'days_in_month')
            self.data[col+'_'+'day'] = self.data[col].dt.day
            self.features_num.append(col+'_'+'day')
            self.data[col+'_'+'month'] = self.data[col].dt.month
            self.features_num.append(col+'_'+'month')
            self.data[col+'_'+'year'] = self.data[col].dt.year
            self.features_num.append(col+'_'+'year')
            self.data[col+'_'+'hour'] = self.data[col].dt.hour
            self.features_num.append(col+'_'+'hour')
            self.data[col+'_'+'day_of_year'+'_sin'] = (np.pi *self.data[col+'_'+'day_of_year'] / 183).apply(lambda x:np.sin(x))
            self.features_num.append(col+'_'+'day_of_year'+'_sin')
            self.data[col+'_'+'hour'+'_sin'] = (np.pi *self.data[col+'_'+'hour'] / 12).apply(lambda x:np.sin(x))
            self.features_num.append(col+'_'+'hour'+'_sin')
        # remove nan columns
        nan_columns = list(self.data.isna().all()[self.data.isna().all()].index)
        self.data = self.data.dropna(axis=1,how='all')
        for col in nan_columns:
            if col in self.features_cat:
                self.features_cat.remove(col)
            elif col in self.features_num:
                self.features_num.remove(col)
            elif col in self.features_bool:
                self.features_bool.remove(col)
        self.data = self.data.drop(columns=self.features_date)

        # Fill nan columns
        # self.data[self.features_cat] = self.data[self.features_cat].fillna('UNK')
        # self.data[self.features_num] = self.data[self.features_num].fillna(0)
        # self.data[self.features_bool] = self.data[self.features_bool].fillna(False)
        
        # Drop constant columns
        constant_columns = list(self.data[self.features_num].std(axis=0)[self.data[self.features_num].std(axis=0)==0].index)
        for col in constant_columns:
            self.features_num.remove(col)
        if constant_columns:
            self.data = self.data.drop(columns=constant_columns)
        
        # Define the task
        target_nunique = self.data[self.target_column].nunique()
        if target_nunique == 2:
            self.task = 'binary_classification'
        elif str(self.data[self.target_column].dtype) == 'object':
            self.task = 'multi_classification'
        elif str(self.data[self.target_column].dtype) in ['float64','float32','int64','int32']:
            self.task = 'regression'
        log_message('debug',f'The selected task is {self.task}')
    
    def preprocess(self,type_num,type_cat,type_cat_target,type_num_target):
        self.X = self.data.drop(columns=self.target_column)
        self.y = self.data[self.target_column]
        X_cat = self.X[self.features_cat].copy()
        X_num = self.X[self.features_num].copy()
        X_bool = self.X[self.features_bool].copy().values
        num_pars = {}
        if type_num == 'min_max':
            X_min = np.nanmin(X_num.values,axis=0)
            X_max = np.nanmax(X_num.values,axis=0)
            new_X_num = (X_num - X_min)/(X_max - X_min)
            num_pars = {'X_min':X_min,'X_max':X_max}
        elif type_num == 'standard':
            X_mean = np.nanmean(X_num.values,axis=0)
            X_std = np.nanstd(X_num.values,axis=0)
            new_X_num = (X_num - X_mean)/X_std
            num_pars = {'X_mean':X_mean,'X_std':X_std}
        else:
            raise RuntimeError("wrong preprocessing type")
            
        if type_cat == 'label':
            for col in self.features_cat:
                le = LabelEncoder()
                le.fit(X_cat[col])
                X_cat.loc[:,col] = le.transform(X_cat[col])
                self.encoders[col] = le
            X_cat = X_cat.values
       
        elif type_cat == 'one_hot':
            one_hots = []
            for col in self.features_cat:
                le = OneHotEncoder(handle_unknown='ignore')
                le.fit(X_cat[col].values.reshape(-1, 1))
                one_hots.append(le.transform(X_cat[col].values.reshape(-1, 1)).toarray())
                self.encoders[col] = le
            X_cat = np.concatenate(one_hots,axis=1)
        else:
            raise RuntimeError("wrong preprocessing type")
            
        if self.task == 'regression':
            if type_num_target == 'min_max':
                y_min = self.y.copy().min(axis=0)
                y_max = self.y.copy().max(axis=0)
                new_y = (self.y.copy() - y_min)/(y_max - y_min)
                new_y = new_y.to_numpy()
                num_pars['y_min']=y_min
                num_pars['y_max']=y_max
                
            elif type_num_target == 'standard':
                y_mean = self.y.copy().mean(axis=0)
                y_std = self.y.copy().std(axis=0)
                new_y = (self.y.copy() - y_mean)/y_std
                new_y = new_y.to_numpy()
                num_pars['y_std']=y_std
                num_pars['y_mean']=y_mean

        elif self.task == 'multi_classification' or self.task == 'binary_classification':
            if type_cat_target == 'label':
                le = LabelEncoder()
                le.fit(self.y.copy())
                new_y= le.transform(self.y.copy()).reshape(-1, 1)
                self.encoders['target'] = le
            elif type_cat_target == 'one_hot':
                le = OneHotEncoder(handle_unknown='ignore')
                le.fit(self.y.copy().values.reshape(-1, 1))
                new_y = le.transform(self.y.copy().values.reshape(-1, 1)).toarray()
                self.encoders['target'] = le
        return np.concatenate([X_cat,new_X_num,X_bool],axis=1),new_y,num_pars

    def missing_values_handler(self,drop,handling_type_num='',handling_type_cat=''):
        if drop:
            self.data = self.data.copy().dropna()
            self.imputer_num = None
            self.imputer_cat_and_bool = None
            
        if handling_type_num != 'KNN' and self.features_num and handling_type_num:
            imputer = SimpleImputer(strategy=handling_type_num)
            self.data.loc[:,self.features_num] = imputer.fit_transform(self.data.loc[:,self.features_num])
            self.imputer_num = imputer

        if handling_type_cat and self.features_cat + self.features_bool and handling_type_cat:
            imputer = SimpleImputer(strategy=handling_type_cat)
            self.data.loc[:,self.features_cat + self.features_bool] = imputer.fit_transform(self.data.loc[:,self.features_cat + self.features_bool])
            self.imputer_cat_and_bool = imputer

        if handling_type_num == 'KNN' and self.features_num:
            imputer = KNNImputer(n_neighbors=7)
            self.data.loc[:,self.features_num] = imputer.fit_transform(self.data.loc[:,self.features_num])
            self.imputer_num = imputer
        self.data = self.data.dropna()

    def postprocess(self,y,params,type_num,type_cat,inference):
        if self.task == 'regression':
            if type_num == 'min_max':
                y_min = params['y_min']
                y_max = params['y_max']
                new_y = y * (y_max - y_min) + y_min
                
            elif type_num == 'standard':
                y_mean = params['y_mean']
                y_std = params['y_std']
                new_y = (y * y_std) + y_mean

        elif self.task == 'multi_classification' or self.task == 'binary_classification':
            if inference:
                if type_cat == 'label':
                    le = self.encoders['target']
                    new_y= le.inverse_transform(y)

                elif type_cat == 'one_hot':
                    le = self.encoders['target']
                    new_y = le.inverse_transform(y.reshape(-1, 1).astype(int))
            else:
                new_y = y.copy()
        return new_y
    
    def evaluate(self,y_true,y_pred):
        if self.task == 'multi_classification':
            score = f1_score(y_true=y_true,y_pred=y_pred,average='weighted')
        elif self.task == 'binary_classification':
            score = f1_score(y_true=y_true,y_pred=y_pred)
        elif self.task == 'regression':
            score = r2_score(y_true=y_true,y_pred=y_pred)
        return score
        
    def train(self,trial):
        self.data = self.original_data.copy()
        try:
            if self.task == 'regression':
                algorithms_list = self.ml_algorithms[self.ml_algorithms.regression == 1]['algorithm'].tolist()
                ml_algorithm = trial.suggest_categorical('ml_algorithm', algorithms_list)
                ml_algorithm_type = self.ml_algorithms[self.ml_algorithms.algorithm==ml_algorithm]['type'].item()
            elif self.task == 'binary_classification':
                algorithms_list = self.ml_algorithms[self.ml_algorithms.binary_classification == 1]['algorithm'].tolist()
                ml_algorithm = trial.suggest_categorical('ml_algorithm', algorithms_list)
                ml_algorithm_type = self.ml_algorithms[self.ml_algorithms.algorithm==ml_algorithm]['type'].item()
            elif self.task == 'multi_classification':
                algorithms_list = self.ml_algorithms[self.ml_algorithms.multi_classification == 1]['algorithm'].tolist()
                ml_algorithm = trial.suggest_categorical('ml_algorithm', algorithms_list)
                ml_algorithm_type = self.ml_algorithms[self.ml_algorithms.algorithm==ml_algorithm]['type'].item()
            
            if ml_algorithm_type in ['linear','svm','knn']:
                type_cat = 'one_hot'
                if self.task == 'binary_classification' or self.task == 'multi_classification':
                    type_cat_target = 'label'
                else:
                    type_cat_target = None
            else:
                type_cat = trial.suggest_categorical('type_cat', ['one_hot','label'])
                if self.task == 'binary_classification' or self.task == 'multi_classification':
                    type_cat_target = 'label'
                else:
                    type_cat_target = None

            if ml_algorithm in ['ComplementNB','MultinomialNB','CategoricalNB']:
                type_num = 'min_max'
                if self.task == 'regression':
                    type_num_target = 'min_max'
                else:
                    type_num_target = None
            else:
                type_num = trial.suggest_categorical('type_num', ['min_max','standard'])
                if self.task == 'regression':
                    type_num_target = trial.suggest_categorical('type_num_target', ['min_max','standard'])
                else:
                    type_num_target = None
            log_message('info',list(self.data[self.target_column].unique()))
            if self.data.isnull().sum().sum() > 0:
                log_message('debug',"Opps!! there are nulls in your data")

                if not self.data.copy().dropna().empty:
                    handling_type_drop = trial.suggest_categorical('handling_type_drop', [True,False])
                else:
                    handling_type_drop = False
                
                if not handling_type_drop:
                    handling_type_num = trial.suggest_categorical('handling_type_num', ['mean','median','most_frequent','KNN'])
                    handling_type_cat = 'most_frequent'
                else:
                    handling_type_num = None
                    handling_type_cat = None

                self.missing_values_handler(handling_type_cat=handling_type_cat,handling_type_num=handling_type_num,drop=handling_type_drop)
            else:
                self.imputer_cat_and_bool = None
                self.imputer_num = None
            
            log_message('info',list(self.data[self.target_column].unique()))

            X,y,pars = self.preprocess(type_num=type_num,
                                       type_cat=type_cat,
                                       type_cat_target=type_cat_target,
                                       type_num_target=type_num_target)  
            if X.shape[0] > 5000:
                log_message('info','The data is huge, we will train on a subset of the data')
                n = 5000
                indexes = np.random.choice(X.shape[0], n, replace=False)  
                X = X[indexes,:]
                y = y[indexes]
            parameters = self.ml_algorithms_parameters[ml_algorithm]
            trial_parameters = {}
            for key,val in parameters.items():
                if ml_algorithm == 'GradientBoostingClassifier' and key == 'loss' and self.task == 'multi_classification':
                    trial_parameters[key] = 'log_loss'
                    continue
                if ml_algorithm == 'LinearSVC' and key=='penalty' and trial_parameters['loss'] == 'hinge':
                    trial_parameters[key] = 'l2'
                    continue
                if key == 'oob_score' and trial_parameters['bootstrap']==False:
                    trial_parameters[key] = False  
                    continue                                  
                if key == 'n_jobs':
                    trial_parameters[key] = val
                    continue
                if ml_algorithm == 'TweedieRegressor' and key=='power':
                    trial_parameters[key] = 0
                    continue
                if ml_algorithm == 'ExtraTreesRegressor' and type_num == 'standard' and key == 'criterion':
                    trial_parameters[key] = trial.suggest_categorical(ml_algorithm+"_"+key,["squared_error", "absolute_error", "friedman_mse"])
                    continue
                if isinstance(val[0],bool):
                    trial_parameters[key] = trial.suggest_categorical(ml_algorithm+"_"+key,val)
                elif isinstance(val[0],int):
                    trial_parameters[key] = trial.suggest_int(ml_algorithm+"_"+key,val[0],val[1])
                elif isinstance(val[0],float):
                    trial_parameters[key] = trial.suggest_float(ml_algorithm+"_"+key,val[0],val[1])
                else:
                    trial_parameters[key] = trial.suggest_categorical(ml_algorithm+"_"+key,val)
            model = eval(ml_algorithm)
            model = model(**trial_parameters)
            if self.task != 'regression':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y ,random_state=42,shuffle=True)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,shuffle=True)

            log_message("debug","Fit "+ml_algorithm)
            log_message("debug","Trial_parameters: "+str(trial_parameters))
            model.fit(X_train,y_train)
            log_message("debug","Fited "+ml_algorithm)
            y_pred = model.predict(X_test)
            y_pred = self.postprocess(y_pred,pars,type_num_target,type_cat_target,False)
            y_test = self.postprocess(y_test,pars,type_num_target,type_cat_target,False)
            score = self.evaluate(y_pred=y_pred,y_true=y_test)
            log_message("info","Score "+str(score))
            return score
        except Exception as e:
            log_message('error',e)
            log_message('error',trial_parameters)
            return 0
    
    def optimize(self,n_trials):
        for i in range(n_trials):
            log_message("debug","Asking")
            trial = self.study.ask()
            log_message("debug","Training")  # Generate a trial suggestion
            value = self.train(trial)  # Evaluate the objective function
            log_message("debug","Telling")
            self.study.tell(trial, value)
            log_message("debug","Finished")
            log_message("debug","="*10)
            gc.collect()
            yield i
    
    def init_study(self):
        log_message('debug',f'The optimization phase started')
        self.study = optuna.create_study(direction='maximize')

    def final_training(self):
        self.best_trial = self.study.best_trial
        self.best_params = self.study.best_params
        log_message('info',f"The best parameters:\n{self.best_params}")
        self.score = self.study.best_value
        log_message('info',f'The optimized model achieved {self.score} score')
        log_message('debug',f'The fitting phase started')
        temp_parameters = self.best_params
        ml_algorithm = temp_parameters['ml_algorithm']
        ml_algorithm_type = self.ml_algorithms[self.ml_algorithms.algorithm==ml_algorithm]['type'].item()
        if ml_algorithm_type in ['linear','svm','knn']:
            self.type_cat = 'one_hot'
            if self.task == 'binary_classification' or self.task == 'multi_classification':
                self.type_cat_target = 'label'
            else:
                self.type_cat_target = None
        else:
            self.type_cat = temp_parameters['type_cat']
            del temp_parameters['type_cat']
            if self.task == 'binary_classification' or self.task == 'multi_classification':
                self.type_cat_target = 'label'
            else:
                self.type_cat_target = None   
            
        if ml_algorithm in ['ComplementNB','MultinomialNB','CategoricalNB']:
            self.type_num = 'min_max'
            if self.task == 'regression':
                self.type_num_target = 'min_max'
            else:
                self.type_num_target = None
        else:
            self.type_num = temp_parameters['type_num']
            del temp_parameters['type_num']
            if self.task == 'regression':
                self.type_num_target = temp_parameters['type_num_target']
                del temp_parameters['type_num_target']
            else:
                self.type_num_target = None
                   
        del temp_parameters['ml_algorithm'] 
        temp_parameters2 = {}
        for key,val in temp_parameters.items():
            temp_parameters2[key.replace(ml_algorithm+"_","").strip()] = val
        temp_parameters = temp_parameters2
        self.handling_type_num = temp_parameters.get('handling_type_num')
        self.handling_type_cat = 'most_frequent'
        self.handling_type_drop = temp_parameters.get('handling_type_drop')
        if self.handling_type_num:
            del temp_parameters['handling_type_num']
        if self.handling_type_drop != None:
            del temp_parameters['handling_type_drop']
        self.missing_values_handler(handling_type_cat=self.handling_type_cat,
                                    handling_type_num=self.handling_type_num,
                                    drop=self.handling_type_drop)
        X,y,pars = self.preprocess(type_num=self.type_num,
                                   type_cat=self.type_cat,
                                   type_cat_target=self.type_cat_target,
                                   type_num_target=self.type_num_target)

        self.numarical_preprocessing_parameters = pars
        model = eval(ml_algorithm)
        model = model(**temp_parameters)
        model.fit(X,y)
        self.model = model
        return vis.plot_optimization_history(self.study)
           
    def save(self,model_name):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        with open(f"Models/{model_name}_model.pkl", 'wb') as f:
            pickle.dump(self.model,f)
        with open(f"Models/{model_name}_tuner.pkl", 'wb') as f:
            pickle.dump(self,f)
        log_message('debug',f'The model has been saved successfuly, model path is {model_name}_model/tuner.pkl')

class Module():
    def __init__(self,automl:AutoML):
        self.preprocessing_parameters = automl.numarical_preprocessing_parameters
        self.model = automl.model
        self.features_cat = automl.features_cat
        self.features_num = automl.features_num
        self.features_bool = automl.features_bool
        self.features_date = automl.features_date
        self.encoders = automl.encoders
        self.type_num = automl.type_num
        self.type_cat = automl.type_cat
        self.type_cat_target = automl.type_cat_target
        self.type_num_target = automl.type_num_target
        self.task = automl.task
        self.imputer_num = automl.imputer_num
        self.imputer_cat_and_bool = automl.imputer_cat_and_bool
    
    def predict(self,data:pd.DataFrame):
        output=None
        try:
            for col in tqdm(self.features_date ):
                data[col+'_'+'day_of_year'] = data[col].dt.day_of_year
                data[col+'_'+'quarter'] = data[col].dt.quarter
                data[col+'_'+'day_of_week'] = data[col].dt.day_of_week 
                data[col+'_'+'days_in_month'] = data[col].dt.days_in_month 
                data[col+'_'+'day'] = data[col].dt.day
                data[col+'_'+'month'] = data[col].dt.month
                data[col+'_'+'year'] = data[col].dt.year
                data[col+'_'+'hour'] = data[col].dt.hour
                data[col+'_'+'day_of_year'+'_sin'] = (np.pi *data[col+'_'+'day_of_year'] / 183).apply(lambda x:np.sin(x))
                data[col+'_'+'hour'+'_sin'] = (np.pi *data[col+'_'+'hour'] / 12).apply(lambda x:np.sin(x))
            log_message("debug",data.info())
            for col in self.features_bool:
                data[col] = data[col].map(self.encoders[col])
            log_message("debug",data.info())
            if self.imputer_cat_and_bool != None:
                data.loc[:,self.features_cat+self.features_bool] = self.imputer_cat_and_bool.transform(data.loc[:,self.features_cat+self.features_bool])

            if self.imputer_num != None:
                data.loc[:,self.features_num] = self.imputer_num.transform(data.loc[:,self.features_num])
            log_message("debug",data.info())
            X_cat = data[self.features_cat].copy()
            X_num = data[self.features_num].copy()
            X_bool = data[self.features_bool].copy().values
            if self.type_num == 'min_max':
                X_min = self.preprocessing_parameters['X_min']
                X_max = self.preprocessing_parameters['X_max']
                new_X_num = (X_num - X_min)/(X_max - X_min)
    
            elif self.type_num == 'standard':
                X_mean = self.preprocessing_parameters['X_mean']
                X_std = self.preprocessing_parameters['X_std']
                new_X_num = (X_num - X_mean)/X_std
        
            else:
                raise RuntimeError("wrong preprocessing type")
                
            if self.type_cat == 'label':
                for col in self.features_cat:
                    le = self.encoders[col]
                    X_cat.loc[:,col] = le.transform(X_cat[col])

                X_cat = X_cat.values
        
            elif self.type_cat == 'one_hot':
                one_hots = []
                for col in self.features_cat:
                    le = self.encoders[col]
                    one_hots.append(le.transform(X_cat[col].values.reshape(-1, 1)).toarray())
                X_cat = np.concatenate(one_hots,axis=1)
            else:
                raise RuntimeError("wrong preprocessing type")
            input_data = np.concatenate([X_cat,new_X_num,X_bool],axis=1)
            output = self.model.predict(input_data)
            if self.task == 'regression':
                if self.type_num_target == 'min_max':
                    y_min = self.preprocessing_parameters['y_min']
                    y_max = self.preprocessing_parameters['y_max']
                    new_output = output * (y_max - y_min) + y_min
                    
                elif self.type_num_target == 'standard':
                    y_mean = self.preprocessing_parameters['y_mean']
                    y_std = self.preprocessing_parameters['y_std']
                    new_output = (output * y_std ) + y_mean

            elif self.task == 'multi_classification' or self.task == 'binary_classification':
                if self.type_cat_target == 'label':
                    le = self.encoders['target']
                    new_output = le.inverse_transform(output)

                elif self.type_cat_target == 'one_hot':
                    le = self.encoders['target']
                    new_output = le.inverse_transform(output.reshape(-1, 1))

            return new_output
        except Exception as e:
            log_message('error',e)
            log_message('error',data.isna().sum()[data.isna().sum()>0])
            if output:
                log_message('error',output)
                log_message('error',self.encoders['target'].classes_)