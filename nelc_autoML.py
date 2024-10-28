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
from sklearn.impute import KNNImputer
# logging.basicConfig(filename='logs.log', filemode='a', format='%(asctime)s - %(levelname)s : %(message)s', level=logging.DEBUG)
def log_message(level, message):
    if level.lower() == 'info':
        logging.info(message)
    elif level.lower() == 'error':
        logging.error(message)
    elif level.lower() == 'warning':
        logging.warning(message)
    elif level.lower() == 'debug':
        logging.debug(message)
    else:
        logging.critical('Unsupported logging level: ' + level)

class AutoML:
    def __init__(self,data:pd.DataFrame,interpretability:float,target_column:str,data_preprocessing:bool=False):
        self.data = data
        self.interpretability = interpretability
        self.target_column = target_column
        self.data_preprocessing = data_preprocessing
        if self.target_column not in list(self.data.columns):
            raise RuntimeError(f"the target column should be on of the data columns : {list(self.data.columns)}")
        self.features_num = []
        self.features_cat = []
        self.features_date = []
        self.features_bool = []
        self.ml_algorithms = pd.read_csv('ml_algorithms.csv',sep='\t')
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
        self.refine_the_data()
        log_message('debug',self.data.info())
        if 'float' in str(self.data[self.target_column].dtype) or 'int' in str(self.data[self.target_column].dtype):
            self.features_num.remove(self.target_column)
        elif 'bool' in str(self.data[self.target_column].dtype):
            self.features_bool.remove(self.target_column)
        else:
            self.features_cat.remove(self.target_column)
            
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
            self.data[col] = self.data[col].astype(bool)
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
        
        # Define the taask
        if str(self.data[self.target_column].dtype) == 'bool':
            self.task = 'binary_classification'
        elif str(self.data[self.target_column].dtype) == 'object':
            self.task = 'multi_classification'
        elif str(self.data[self.target_column].dtype) in ['float64','float32','int64','int32']:
            self.task = 'regression'
        log_message('debug',f'The selected task is {self.task}')
    
    def preprocess(self,type_num,type_cat):
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
            if type_num == 'min_max':
                y_min = self.y.copy().min(axis=0)
                y_max = self.y.copy().max(axis=0)
                new_y = (self.y.copy() - y_min)/(y_max - y_min)
                new_y = new_y.to_numpy()
                num_pars['y_min']=y_min
                num_pars['y_max']=y_max
                
            elif type_num == 'standard':
                y_mean = self.y.copy().mean(axis=0)
                y_std = self.y.copy().std(axis=0)
                new_y = (self.y.copy() - y_mean)/y_std
                new_y = new_y.to_numpy()
                num_pars['y_std']=y_std
                num_pars['y_mean']=y_mean

        elif self.task == 'multi_classification' or self.task == 'binary_classification':
            # if type_cat == 'label':
            le = LabelEncoder()
            le.fit(self.y.copy())
            new_y= le.transform(self.y.copy())
            self.encoders['target'] = le
            # elif type_cat == 'one_hot':
            #     le = OneHotEncoder()
            #     le.fit(self.y.copy().values.reshape(-1, 1))
            #     new_y = le.transform(self.y.copy().values.reshape(-1, 1)).toarray()
            #     self.encoders['target'] = le
        return np.concatenate([X_cat,new_X_num,X_bool],axis=1),new_y,num_pars

    def missing_values_handler(self,X,y,handling_type):
        if handling_type=="drop":
            new_data = self.data.dropna()
            if new_data.empty:
                handling_type = "target_based_filling"
            else:
                self.data = new_data
                return
        if handling_type=="target_based_filling":
            new_data = self.data.copy()
            missing_columns = new_data.isna().sum()
            missing_columns = missing_columns[missing_columns>0].index
            targets = list(new_data[self.target_column].unique())
            for col in missing_columns:
                for target in targets:
                    indexes = new_data[new_data[self.target_column]==target][col].index
                    if not indexes.empty:
                        if col in self.features_num:
                            col_mean = new_data.loc[indexes,col].mean()
                            new_data.loc[indexes,col] = new_data.loc[indexes,col].fillna(col_mean)
                        elif col in self.features_cat or col in self.features_date:
                            maximum_count_col = new_data.loc[indexes,col].value_counts().sort_values().tolist()[0]
                            new_data.loc[indexes,col] = new_data.loc[indexes,col].fillna(maximum_count_col)
                    else:
                        pass
            new_data = new_data.dropna()
            if new_data.empty:
                handling_type = "KNN"
            else:
                self.data = new_data
        
        if handling_type == "KNN":
            new_data = np.concatenate([X,np.expand_dims(y,axis=1)],axis=1)
            imputer = KNNImputer(n_neighbors=3)
            new_data = imputer.fit_transform(new_data)
            new_X = new_data[:,:-1]
            new_y = new_data[:,-1]
            return new_X,new_y
        
        if handling_type=="sampler_filling":
            pass
        if handling_type=="generator_filling":
            pass
    
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
                    print(y.reshape(-1, 1).shape)
                    print(y)
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
            else:
                type_cat = trial.suggest_categorical('type_cat', ['one_hot','label'])

            if ml_algorithm in ['ComplementNB','MultinomialNB','CategoricalNB']:
                type_num = 'min_max'
            else:
                type_num = trial.suggest_categorical('type_num', ['min_max','standard'])

            handling_type = trial.suggest_categorical('handling_type', ['drop','target_based_filling','KNN'])
            handling_type = "KNN"
            X,y,pars = self.preprocess(type_num,type_cat)
            X,y = self.missing_values_handler(X,y,handling_type=handling_type)
            if X.shape[0] > 5000:
                log_message('info','The data is huge, we will train on a subset of the data')
                n = 5000
                indexes = np.random.choice(X.shape[0], n, replace=False)  
                X = X[indexes,:]
                y = y[indexes]
            parameters = self.ml_algorithms_parameters[ml_algorithm]
            trial_parameters = {}
            for key,val in parameters.items():
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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)
            print("fit ",ml_algorithm)
            print(trial_parameters)
            model.fit(X_train,y_train)
            print("fited ",ml_algorithm)
            y_pred = model.predict(X_test)
            y_pred = self.postprocess(y_pred,pars,type_num,type_cat,False)
            y_test = self.postprocess(y_test,pars,type_num,type_cat,False)
            score = self.evaluate(y_pred=y_pred,y_true=y_test)
            print(score)
            return score
        except Exception as e:
            log_message('error',e)
            log_message('error',trial_parameters)
            return None
    
    def optimize(self,n_trials):
        for i in range(n_trials):
            print("ask")
            trial = self.study.ask()
            print("train")  # Generate a trial suggestion
            value = self.train(trial)  # Evaluate the objective function
            print("tell")
            self.study.tell(trial, value)
            print("finish")
            print("="*10)
            gc.collect()
            yield i
    
    def init_study(self):
        log_message('debug',f'The optimization phase started')
        self.study = optuna.create_study(direction='maximize')

    def final_training(self):
        self.best_trial = self.study.best_trial
        self.best_params = self.study.best_params
        self.score = self.study.best_value
        log_message('info',f'The optimized model achieved {self.score} score')
        log_message('debug',f'The fitting phase started')
        temp_parameters = self.best_params
        ml_algorithm = temp_parameters['ml_algorithm']
        ml_algorithm_type = self.ml_algorithms[self.ml_algorithms.algorithm==ml_algorithm]['type'].item()
        if ml_algorithm_type in ['linear','svm','knn']:
            self.type_cat = 'one_hot'
        else:
            self.type_cat = temp_parameters['type_cat']
            del temp_parameters['type_cat']
        if ml_algorithm in ['ComplementNB','MultinomialNB','CategoricalNB']:
            self.type_num = 'min_max'
        else:
            self.type_num = temp_parameters['type_num']
            del temp_parameters['type_num']
        del temp_parameters['ml_algorithm']
        del temp_parameters['handling_type']
        temp_parameters2 = {}
        for key,val in temp_parameters.items():
            temp_parameters2[key.replace(ml_algorithm+"_","").strip()] = val
        temp_parameters = temp_parameters2
        X,y,pars = self.preprocess(self.type_num,self.type_cat)
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
        self.task = automl.task
    
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
            
            for col in self.features_bool:
                data[col] = data[col].map(self.encoders[col])
            
            # Fill nan columns
            data[self.features_cat] = data[self.features_cat].fillna('UNK')
            data[self.features_num] = data[self.features_num].fillna(0)
            data[self.features_bool] = data[self.features_bool].fillna(False)

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
                if self.type_num == 'min_max':
                    y_min = self.preprocessing_parameters['y_min']
                    y_max = self.preprocessing_parameters['y_max']
                    new_output = output * (y_max - y_min) + y_min
                    
                elif self.type_num == 'standard':
                    y_mean = self.preprocessing_parameters['y_mean']
                    y_std = self.preprocessing_parameters['y_std']
                    new_output = (output * y_std ) + y_mean

            elif self.task == 'multi_classification' or self.task == 'binary_classification':
                if self.type_cat == 'label':
                    le = self.encoders['target']
                    new_output = le.inverse_transform(output)

                elif self.type_cat == 'one_hot':
                    le = self.encoders['target']
                    new_output = le.inverse_transform(output.reshape(-1, 1))

            return new_output
        except Exception as e:
            log_message('error',e)
            log_message('error',data)
            if output:
                log_message('error',output)
                log_message('error',self.encoders['target'].classes_)