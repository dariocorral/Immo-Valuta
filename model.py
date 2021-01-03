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

import pandas as pd
import numpy as np
import streamlit as st
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn import tree
import category_encoders as ce
from category_encoders.wrapper import NestedCVWrapper
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
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
       
       
       #Cat 25m2
       
       df['size'] = (
          df['size'].apply(lambda x: int(math.floor(x / 10.0) * 10.0))
          )
       
       
       #Drop columns
       df.drop(['parkingSpacePrice','avgPriceZone','hasAirConditioning',
        'hasBoxRoom', 'hasTerrace', 'hasGarden','priceByArea',
        'hasParkingSpace', 'parkingSpacePrice', 'statusInt','propertyTypeInt',
        'rooms','bathrooms','floor','hasLift'
                                ],axis= 1,
                inplace = True)
       
       #Sorted
       df = df[['price', 'propertyType', 'size', 'district', 'status', 
                'roomsCat','bathroomsCat','floorCat','box_posto_auto',
                'garden_terrace','liftCat','hasSwimmingPool'
        ]]
       
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
        max_depth_list = np.arange(int(max_n_trees/2),
                                   max_n_trees,
                                   3).tolist()
        max_depth_list.append(None)
        
        param_grid = {"max_features":max_features_list,
                      "max_depth":max_depth_list}
        
        #RF Model to test
        rf = RandomForestRegressor(
                          bootstrap = True,
                          oob_score = True,
                          n_estimators = n_trees,
                          random_state=7)
        
        #Encoder
        encoder = ce.GLMMEncoder(cols=self.cat_index)

        #Encoder Cv
        cv_encoder = NestedCVWrapper(
            encoder,
            cv=KFold(n_splits=5, shuffle=True, random_state=7)
            )

        #Define and execute pipe        
        pipe = Pipeline(steps=[ 
                        ('catEncoder', cv_encoder),
                       ('grid', GridSearchCV(rf,param_grid,verbose = 3))
                           ])
                        
        pipe.fit(features,labels)
        
        df_results = pd.DataFrame(pipe['grid'].cv_results_)
        
        #Save CV Results
        if saveStats:
            
            df_results.to_csv('data/cv_hyperparams_model.csv')
                
        best_max_features = (
            df_results.loc[
                df_results['rank_test_score']==1]['param_max_features'].values[0])
        
        best_max_depth = (
            df_results.loc[
                df_results['rank_test_score']==1]['param_max_depth'].values[0])

        model_params = {
            "bootstrap":True,
             "max_depth" : best_max_depth,
             "oob_score": True,
             "max_features": best_max_features,
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
            encoder,
            cv=KFold(n_splits=5, shuffle=True,random_state=7))
        
        #Datasets
        features = self.features_dataset
        labels = self.labels_dataset

        #Shuffle & apply cat encoder
        pipe = Pipeline(steps=[ 
                ('catEncoder', cv_encoder),
                ('rf',RandomForestRegressor(
                          **best_model_params,
                          random_state = 7))
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
        
        return self.model_fit.oob_score_
    
    #@st.cache
    def predict(self,
            size,
            propertyType,
            district,
            status,
            rooms, 
            bathrooms,
            box_posto_auto,
            garden_terrace,
            floor,
            lift,
            swimming_pool
            ):
        """
        
        Parameters
        ----------
        propertyType : str (category)
        size: int (m2)
        district : str (category)
        status : str (category)
        rooms : int
        bathrooms : int
        floor: int
        box_posto_auto : Bool(1,0)
        garden_terrace : Bool(1,0)
        Lift : Bool(1,0)
        hasSwimmingPool : Bool(1,0)

        Returns
        -------
        Prediction : Best Model Prediction

        """
        #Logic FloorCat
        if floor <= 1:
            floorCat = 0
        else:
            floorCat = 1
        
        #Logic LiftCat
        if lift == 1 and floor >=2:
            liftCat = 1
        else:
            liftCat = 0
            
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

        array = np.array([
            propertyType,
            size,
            district, 
            status, 
            roomsCat,
            bathroomsCat,
            floorCat,
            box_posto_auto,
            garden_terrace,
            liftCat,
            swimming_pool
                    ]).reshape(1,-1)
        
        
        value = self.model_fit.predict(array)[0]
        
        return int(math.floor(value / 5000.0) * 5000.0)
    
    @property
    def permutation_importance(self):
        """
        Permutation Features Importance 
        
        """
        
        model_fit = self.model_fit
        
        #Encoder
        encoder = ce.GLMMEncoder(cols=self.cat_index)

        #Shuffle & apply cat encoder
        pipe = Pipeline(steps=[ 
                ('shuffle',shuffle(random_state=7)),
                ('catEncoder', encoder)
                           ])
        #Datasets
        features = self.features_dataset
        labels = self.labels_dataset
                
        #Apply piipe
        features = pipe.fit_transform(features,labels)
        
        result = permutation_importance(model_fit, 
                                        features,
                                        labels, 
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
        #Rf without depth limit
        rf = RandomForestRegressor(n_estimators=1000,
                          max_depth = None,
                          random_state = 7)
        
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

