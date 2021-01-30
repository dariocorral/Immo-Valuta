#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:07:42 2020

@author: dariocorral
"""

from preprocessing import Preprocessing
from utils import Utils

#Ignore Warnings Console Messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning
)

import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
#import streamlit as st
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn import tree
import category_encoders as ce
from category_encoders.wrapper import NestedCVWrapper
import matplotlib.pyplot as plt
import time
import math


class Model():
    """
    Model Class
    """
    
    def __init__(self):
        
        self.__utils = Utils()
        self.__preprocessing = Preprocessing()
        self.model_fit = self.fit()
        self.__preprocessing.execute()
        
    @property
    def dataset_preprocessed(self):
        """
        Returns
        -------
        DataFrame Houses Preprocessed

        """
        
        df = pd.read_csv("data/houses_clean.csv")
        
        df.drop('Unnamed: 0',axis = 1,inplace = True)
        
        return df
        

        
    def get_prepared_df(self):
       """
       Prepare dataframe for modelling

       Parameters
       ----------
       df : Dataframe Data

       Returns
       -------
       Array to Model 

       """
       
       df = self.dataset_preprocessed
       
              
       #Bins
       #bins_list = np.arange(300,3600,100).tolist()
       bins_list = np.arange(25000,4500000,25000).tolist()

       #bins_list.append(0)
       #bins_list.append(int(math.ceil(df['priceByArea'].max() / 100.0) * 100.0))
       #bins_list.sort()

       #df["priceByAreaBins"] = pd.cut(df['priceByArea'],bins_list,labels = False)
       df["priceBins"] = pd.cut(df['price'],bins_list,labels = False)

       
       #Property Type Union
       df['propertyType'] = np.where((df['propertyType']=='studio') |
                                     (df['propertyType']=='duplex'),
                                     'flat',df['propertyType'])
       
       #Select Features
       df = df[['price','size','propertyType', 'district',
                            'status',
                'roomsCat','bathroomsCat','box_posto_auto','hasTerrace',
                'hasGarden','hasSwimmingPool']]
       
       return df
    
    @property
    def labels_dataset(self):
        """
        Returns
        -------
        Labels Dataset Numpy Array

        """
        df = self.get_prepared_df()
        labels = np.array(df['price'])
        
        return labels

    @property
    def features_dataset(self):
        """
        Returns
        -------
        Features Dataset Numpy Array

        """
        df = self.get_prepared_df()
        features= df.drop('price', axis = 1)
        features = np.array(features) 
        
        return features
    
    @property
    def features_list(self):
        """
        Returns
        -------
        Features List

        """
        df = self.get_prepared_df()
        features= df.drop('price', axis = 1)
        # Saving feature names for later use
        feature_list = list(features.columns)
        
        return feature_list
    
    @property
    def n_features(self):
        """
        Returns
        -------
        Number of features

        """
        return len(self.features_list)
    
    @property
    def cat_index(self):
        """
        Returns
        -------
        Index position categorical columns

        """    
        df = self.get_prepared_df()
        df.drop('price',axis = 1,inplace = True)
        categorical_features_indices = np.where(
            (df.dtypes != np.int)&(df.dtypes != np.float))[0]
        
        index = categorical_features_indices.reshape(1,-1).tolist()[0]
        
        return index
        
    def search_best_rf(self,n_trees = 2500,
                       saveStats = True):
        """
        Seach Best Random Forest Model
  
        Parameters
         ----------
        df : DataFrame prepared (method prepared_data)
  
        Returns
        -------
        JSON File (model_params_rf.json).
  
        """
        #Process Time
        start = time.time()
        
        #Datasets
        features = self.features_dataset
        labels = self.labels_dataset
                
        #Generate random state
        #min_samples_split_values to test        
        max_features_list = np.arange(0.33,0.55,0.02).tolist()
        max_features_list = [ round(elem, 2) for elem in max_features_list ]
            
        max_features_list.append('sqrt')
        max_features_list.append('auto')
        
        #Get max n_trees
        max_n_trees = self.depth_of_trees.max()[0]
        max_depth_list = np.arange(int(max_n_trees/3),
                                   max_n_trees,
                                   1).tolist()
        max_depth_list.append(None)
        
        #Min Sample leaf
        min_samples_leaf_list = np.arange(0.01,0.36,0.01).tolist()
        min_samples_leaf_list = [ round(elem, 2) for elem in min_samples_leaf_list ]
        min_samples_leaf_list.append(None)
        
        
        ccp_list = np.arange(0.001,0.036,0.001).tolist()
        ccp_list = [ round(elem, 5) for elem in ccp_list ]

        ccp_list = [0.001]
        #min_samples_leaf_list.append(None)
        
        param_grid = {#"max_features":max_features_list,
                      #"max_depth":max_depth_list,
                      "min_impurity_decrease":min_samples_leaf_list}
        
        #RF Model to test
        rf = RandomForestRegressor(
                          bootstrap = True,
                          oob_score = True,
                          n_estimators = n_trees,
                          max_depth=9,
                          max_features = 0.33,
                          random_state=7)
        
        #Encoder
        encoder = ce.GLMMEncoder(cols=self.cat_index)
        

        #Encoder Cv
        cv_encoder = NestedCVWrapper(
            feature_encoder= encoder,
            cv=5,shuffle=True,random_state=7)
        
        
        #Apply Transform to all datasets
        feat_tsf = cv_encoder.fit_transform(features,labels)
        
        
        #Define and execute pipe        
        grid_cv= GridSearchCV(rf,param_grid,verbose = 3)
                        
        grid_cv.fit(feat_tsf,labels)
        
        df_results = pd.DataFrame(grid_cv.cv_results_)
        
        #Save CV Results
        if saveStats:
            
            df_results.to_csv('data/cv_hyperparams_model.csv')
                
        
        best_max_depth = (
            df_results.loc[
                df_results['rank_test_score']==1]['param_max_depth'].values[0])

        model_params = {
            "bootstrap":True,
             "max_depth" : best_max_depth,
             "max_features":0.33,
             "oob_score": True,
             "n_estimators" : n_trees
            }
        
        # Serializing json  
        json_object = json.dumps(model_params, indent = 2) 
      
        # Writing .json 
        with open("model_params_rf.json", "w") as outfile: 
            outfile.write(json_object)
        
        #End Time
        end = time.time()
        time_elapsed = round((end - start)/60,1)

        return ('Time elapsed minutes: %1.f' % 
                    (time_elapsed))
    
    #@st.cache
    def fit(self):
        """
        Returns
        -------
        Fit Best Params Model

        """
        file = 'model_params_rf.json'
        
        # Opening JSON file 
        with open(file, 'r') as openfile: 
      
        # Reading from json file 
            best_model_params = json.load(openfile) 
        
        #Encoder    
        encoder = ce.GLMMEncoder(cols=self.cat_index)
                
        #Encoder Cv
        cv_encoder = NestedCVWrapper(
            feature_encoder= encoder,
            cv=5,shuffle=True)
        
        #Datasets
        features = self.features_dataset
        labels = self.labels_dataset
        
        #Shuffle & apply cat encoder
        pipe = Pipeline(steps=[ 
                ('catEncoder', cv_encoder),
                ('rf',RandomForestRegressor(
                          **best_model_params,
                          random_state = 7
                          ))
                ])

        
        #Fit & Metrics
        pipe.fit(features,labels)
        
        oob_score = (pipe['rf'].oob_score_)*100
        
        print("OOB Score: %.2f" % oob_score)
        
        return pipe
    
    @property
    def oob_score(self):
        """
        Returns
        -------
        Best Model OOB Score

        """
        
        return self.model_fit['rf'].oob_score_
    
    #@st.cache
    def fit_predict(self,
            size,
            propertyType,
            district,
            status,
            rooms, 
            bathrooms,
            box_posto_auto,
            hasGarden,
            hasTerrace,
            hasSwimmingPool
            ):
        """
        
        Parameters
        ----------
        district : str (category)
        status : str (category)
        rooms : int
        bathrooms : int
        box_posto_auto : Bool(1,0)
        garden : Bool(1,0)
        terrace : Bool(1,0)
        hasSwimmingPool : Bool(1,0)

        Returns
        -------
        Prediction : Best Model Prediction

        """
        file = 'model_params_rf.json'
        
        # Opening JSON file 
        with open(file, 'r') as openfile: 
      
        # Reading from json file 
            best_model_params = json.load(openfile) 
        
        """
        #Avg Price Zone
        avg_price_zone_df = self.dataset_preprocessed[['district','avgPriceZone']]

        avg_price_zone_df = avg_price_zone_df.drop_duplicates()       
        
        avgPriceZone = avg_price_zone_df.loc[
            avg_price_zone_df['district']==district]['avgPriceZone'].values[0]
        """
        
        #Rooms Logic
        if rooms >= 4:
            roomsCat = 4
        else:
            roomsCat = rooms
            
        #Rooms Logic
        if bathrooms >= 2:
            bathroomsCat = 2
        else:
            bathroomsCat = bathrooms
            
        #Avg Price Zone
        #avgPriceZone = self.avg_price_district(district)

        array = np.array([
            size,
            propertyType,
            district,
            status,
            roomsCat,
            bathroomsCat,
            box_posto_auto,
            hasGarden,
            hasTerrace,
            hasSwimmingPool
                    ]).reshape(1,-1)
        
        
        #Encoder    
        encoder = ce.GLMMEncoder(cols=self.cat_index)
        
        #Encoder CV KFold
        cv_encoder = NestedCVWrapper(
            encoder,
            cv=5,shuffle=True, random_state=7)
        
        #Datasets
        features = self.features_dataset
        labels = self.labels_dataset
        
        #Apply Transform to all datasets
        feat_tsf = cv_encoder.fit_transform(features,labels,array)
        
        #RF Regressor
        rf = RandomForestRegressor(
                          **best_model_params,
                          random_state = 7
                          )

        #Fit & Metrics
        rf.fit(feat_tsf[0],labels)
        
        #OOB Score
        oob_score = (rf.oob_score_)*100
        
        print("OOB Score: %.2f" % oob_score)

        #Prediction
        prediction = rf.predict(feat_tsf[1])[0]

        return prediction#int(math.floor(prediction / 5) * 5)
    
    @property
    def permutation_importance(self):
        """
        Permutation Features Importance 
        
        Returns
        -------
        
        Graph Permutation Importance
        """
        
        file = 'model_params_rf.json'
        
        # Opening JSON file 
        with open(file, 'r') as openfile: 
      
        # Reading from json file 
            best_model_params = json.load(openfile) 
        
        #Datasets
        features = self.features_dataset
        labels = self.labels_dataset
        
        #Encoder    
        encoder = ce.GLMMEncoder(cols=self.cat_index)
        
        #Encoder CV KFold
        cv_encoder = NestedCVWrapper(
            encoder,
            cv=5,shuffle=True, random_state=None)
        
        #Datasets
        features = self.features_dataset
        labels = self.labels_dataset
        
        #Apply Transform to all datasets
        feat_tsf = cv_encoder.fit_transform(features,labels)

        #Encoder Cv
        cv_encoder = NestedCVWrapper(
            feature_encoder= encoder,
            cv=5,shuffle=True)
        
        #RF Regressor
        rf = RandomForestRegressor(
                          **best_model_params,
                          random_state = 7
                          )
        #Fit
        rf.fit(feat_tsf,labels)

        #Permutation importance
        result = permutation_importance(rf, 
                                        feat_tsf,
                                        self.labels_dataset, 
                                        n_repeats=10,
                                random_state=7, n_jobs=2)
        
        sorted_idx = result.importances_mean.argsort()
        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=self.get_prepared_df().iloc[:,1:].columns[sorted_idx])
        ax.set_title("Permutation Importances")
        fig.tight_layout()
        
        return plt.show()
        
    
    def plot_tree(self,tree_number = 0):
        """
        

        Parameters
        ----------
        number : Int. Tree to plot. The default is 0.

        Returns
        -------
        Tree Image

        """
        model_rf = self.model_fit['rf']
        
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,30), dpi=800)
        tree.plot_tree(model_rf.estimators_[tree_number],
               feature_names = self.features_list, 
               class_names='price',
               filled = True);
        fig.savefig('data/rf_individualtree.png')
        
        return fig
    
    def feature_imp(self):
        """
        Feature Importance Model Method

        Returns
        -------
        Dataframe with features Importance

        """
        df = (
        pd.DataFrame({"ft":self.features_list,
                      'imp':self.model_fit['rf'].feature_importances_}
                         ))
        
        df.sort_values(by='imp', ascending = False, inplace = True)
        
        return df
    
    @property
    def depth_of_trees(self):
        """
        
        Returns
        -------
        Dataframe with Trees depth

        """
        
        file = 'model_params_rf.json'
        
        # Opening JSON file 
        with open(file, 'r') as openfile: 
      
        # Reading from json file 
            best_model_params = json.load(openfile) 

        #Rf without depth limit
        rf = RandomForestRegressor(**best_model_params
                          )
        
        #Encoder for categorical columns
        encoder = ce.GLMMEncoder(cols=self.cat_index)
        
        #Pipeline
        pipe = Pipeline(steps=[ 
                ('catEncodder',encoder),
                ('rf', rf)])

        pipe.fit(self.features_dataset,self.labels_dataset)
        
        #Get depth of trees
        max_depth_list = []
        
        for i in rf.estimators_:
            
            max_depth_list.append(i.get_depth())
            
        print("Max depht: %i trees" % max(max_depth_list)) 
       
        return  pd.DataFrame(max_depth_list,columns=['trees'])
    
    def train_test_samples(self, features, labels, test_size=0.20,
                           random_state=None):
        
        features  = self.features_dataset
        labels = self.labels_dataset
        
        #Encoder    
        encoder = ce.GLMMEncoder(cols=self.cat_index)
        
        #Encoder CV KFold
        cv_encoder = NestedCVWrapper(
            encoder,
            cv=5,shuffle=True, random_state=7)
        
        #Datasets
        features = self.features_dataset
        labels = self.labels_dataset
        
        #Apply Transform to all datasets
        feat_tsf = cv_encoder.fit_transform(features,labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            feat_tsf,labels, test_size = test_size, random_state = random_state)
        
        return X_train, X_test, y_train, y_test
    
    def avg_price_district(self,district):
        
        df = self.dataset_preprocessed
        
        df = df.groupby('district').mean()['priceByArea']
        
        return int(df.loc[df.index==district].values[0])
        
    
        

