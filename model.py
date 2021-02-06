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
import streamlit as st
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import (train_test_split,
                            HalvingRandomSearchCV)
from sklearn.inspection import permutation_importance
from sklearn import tree
import category_encoders as ce
from category_encoders.wrapper import NestedCVWrapper
import matplotlib.pyplot as plt
import time
import math

@st.cache(allow_output_mutation=True)
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
       
                     
       df['size']= df['size'].apply(lambda x: math.ceil(x / 5.0) * 5.0)

       
       #Property Type Union
       df['propertyType'] = np.where((df['propertyType']=='studio') |
                                     (df['propertyType']=='duplex'),
                                     'flat',df['propertyType'])
       
       #Select Features
       df = df[['price','size','propertyType', 'district',
                            'status','roomsCat',
                'bathroomsCat']]#'box_posto_auto','hasTerrace',
                #'hasGarden','hasSwimmingPool']],
       
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
    def feat_tsf_dataset(self):
        """
        Returns
        -------
        Features Dataset Numpy Array with Category Encoders

        """
        features = self.features_dataset
        labels = self.labels_dataset
        
        #Encoder
        encoder = ce.GLMMEncoder(cols=self.cat_index)
        

        #Encoder Cv
        cv_encoder = NestedCVWrapper(
            feature_encoder= encoder,
            cv=5,shuffle=True,random_state=7)
        
        
        #Apply Transform to all datasets
        feat_tsf = cv_encoder.fit_transform(features,labels)
        
        return feat_tsf

    
    
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
        feat_tsf = self.feat_tsf_dataset
        labels = self.labels_dataset
                
        #Generate random state
        #min_samples_split_values to test        
        max_features_list = np.arange(0.20,0.66,0.01).tolist()
        max_features_list = [ round(elem, 2) for elem in max_features_list ]
            
        max_features_list.append('sqrt')
        max_features_list.append('auto')
        
        #Get max n_trees
        max_n_trees = self.depth_of_trees.max()[0]
        max_depth_list = np.arange(int(max_n_trees/4),
                                   max_n_trees,
                                   1).tolist()
        max_depth_list.append(None)
        
        #min_impurity_decrease
        min_impurity_decrease_list = np.arange(0.01,0.26,0.01).tolist()
        min_impurity_decrease_list = [ round(elem, 2) for elem in min_impurity_decrease_list ]
        
        #min_samples_leaf_list.append(None)
        
        param_grid = {"max_features":max_features_list,
                      "max_depth":max_depth_list,
                      "min_impurity_decrease":min_impurity_decrease_list}
        
        #RF Model to test
        rf = RandomForestRegressor(
                          bootstrap = True,
                          oob_score = True,
                          n_estimators = n_trees,
                          random_state=7)
        
        
        #Define and execute pipe        
        grid_cv= HalvingRandomSearchCV(estimator=rf,
                                     param_distributions=param_grid,
                                     random_state=7,
                                     max_resources='auto',
                                     verbose = 3).fit(feat_tsf,labels)
                        
        
        df_results = pd.DataFrame(grid_cv.cv_results_)
        
        #Save CV Results
        if saveStats:
            
            df_results.to_csv('data/cv_hyperparams_model.csv')
                
        
        print ("Best Params:")    
        print(grid_cv.best_params_)
        
        print("Saving model in 'model_params.joblib'")
        # Writing joblibfile with best model 
        dump(grid_cv.best_estimator_, 'model_params.joblib')
        
        #End Time
        end = time.time()
        time_elapsed = round((end - start)/60,1)

        return ('Time elapsed minutes: %1.f' % 
                    (time_elapsed))
    
    def fit(self):
        """
        Returns
        -------
        Fit Best Params Model

        """
        
        
        #Datasets
        feat_tsf = self.feat_tsf_dataset
        labels = self.labels_dataset
        
        #Load Best Model
        rf = load('model_params.joblib')        
        
        #Fit & Metrics
        rf.fit(feat_tsf,labels)
        
        oob_score = (rf.oob_score_)*100
        
        print("OOB Score: %.2f" % oob_score)
        
        return rf
    
    @property
    def oob_score(self):
        """
        Returns
        -------
        Best Model OOB Score

        """
        
        return self.model_fit.oob_score_
    
    @property
    def params(self):
        """
        Returns
        -------
        Best Model Params

        """
        
        return self.model_fit.get_params()

    def predict(self,
            size,
            propertyType,
            district,
            status,
            rooms, 
            bathrooms,
            #box_posto_auto,
            #hasGarden,
            #hasTerrace,
            #hasSwimmingPool
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
        
        """
        #Avg Price Zone
        avg_price_zone_df = self.dataset_preprocessed[['district','avgPriceZone']]

        avg_price_zone_df = avg_price_zone_df.drop_duplicates()       
        
        avgPriceZone = avg_price_zone_df.loc[
            avg_price_zone_df['district']==district]['avgPriceZone'].values[0]
        """
        
        #Rooms Category
        roomsCat = self.roomsCategory(rooms)
            
        #Bathrooms Logic
        bathroomsCat = self.bathroomsCategory(bathrooms)
        
        #Array for prediction
        array = np.array([
            size,
            propertyType,
            district,
            status,
            roomsCat,
            bathroomsCat,
            #box_posto_auto,
            #hasGarden,
            #hasTerrace,
            #hasSwimmingPool
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
        
        #Prediction
        prediction = self.model_fit.predict(feat_tsf[1])[0]

        return prediction
    
    @property
    def permutation_importance(self):
        """
        Permutation Features Importance 
        
        Returns
        -------
        
        Graph Permutation Importance
        """
        
        #Datasets
        feat_tsf = self.feat_tsf_dataset
        labels = self.labels_dataset
        
        rf = load('model_params.joblib')        
        
        #Fit
        rf.fit(feat_tsf,labels)

        #Permutation importance
        result = permutation_importance(rf, 
                                        feat_tsf,
                                        labels, 
                                        n_repeats=10,
                                random_state=7, n_jobs=2)
        
        df = (
        pd.DataFrame({"ft":self.features_list,
                      'imp_mean':result.importances_mean,
                     'imp_dsvt':result.importances_std}
                         ))
        
        df.sort_values(by='imp_mean', ascending = False, inplace = True)
        
        
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
        model_rf = self.model_fit
        
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
                      'imp':self.model_fit.feature_importances_}
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

        #Get depth of trees
        max_depth_list = []
        
        for i in self.model_fit.estimators_:
            
            max_depth_list.append(i.get_depth())
            
        print("Max depht: %i trees" % max(max_depth_list)) 
       
        return  pd.DataFrame(max_depth_list,columns=['trees'])
    
    def train_test_samples(self, features, labels, test_size=0.20,
                           random_state=None):
        
        feat_tsf  = self.feat_tsf_dataset
        labels = self.labels_dataset
        
                
        X_train, X_test, y_train, y_test = train_test_split(
            feat_tsf,labels, test_size = test_size, random_state = random_state)
        
        return X_train, X_test, y_train, y_test
    
    def avg_price_district(self,district):
        
        df = self.dataset_preprocessed
        
        df = df.groupby('district').mean()['priceByArea']
        
        return int(df.loc[df.index==district].values[0])
    
    @property
    def propertyTypeList(self):
        
        propertyTypelist = ['Appartamento', 'Attico',
                    'Villa', 'Casa Rustica']
    
        return propertyTypelist
    
    def propertyTypeConverter(self,propertyType):
        """
        Parameters
        ----------
        propertyType : Str  Selected Property Type.
    
        Returns
        -------
        Property Type str
    
        """
        #Options list
        propertyTypelist = self.propertyTypeList
        
        #Lower elements list
        propertyTypelist = [i.lower() for i in propertyTypelist]
        
        #Assertion
        assert propertyType.lower() in propertyTypelist
        
        #Default Value
        propertyTypeOutput = 'flat'
        
        if propertyType.lower() == 'appartamento':
        
            propertyTypeOutput = 'flat'
        
        elif propertyType.lower() == 'attico':
        
            propertyTypeOutput = 'penthouse'
    
        elif propertyType.lower() == 'villa':
        
            propertyTypeOutput = 'chalet'
        
        elif propertyType.lower() == 'casa rustica':
        
            propertyTypeOutput = 'countryHouse'
    
        return propertyTypeOutput

    @property
    def statusList(self):
        
        status_it = ['Da ristrutturare','Buono',
                      'Nuova Costruzione']

        return status_it

    def statusConverter(self,status):
        """
        Parameters
        ----------
        status : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #Options list
        statusList = self.statusList
        
        #Lower elements list
        statusList = [i.lower() for i in statusList]
        
        #Assertion
        assert status.lower() in statusList
        
        #Default Value
        statusOutput = 'renew'
        
        #Mapping status
        if status.lower() == 'Da ristrutturare':
            
            statusOutput = 'renew'
            
        elif status.lower() == 'Buono':
            
            statusOutput = 'good'
            
        elif status.lower() == 'Nuova Costruzione':
        
            statusOutput = 'newdevelopment'
            
        return statusOutput
    
    def roomsCategory(self,rooms):
        """
        Parameters
        ----------
        rooms : Int
            Rooms.

        Returns
        -------
        Rooms Category.

        """
        
        #Rooms Logic
        if rooms >= 4:
            roomsCat = 4
        else:
            roomsCat = rooms
            
        return roomsCat

    def bathroomsCategory(self,bathrooms):
        """
        Parameters
        ----------
        rooms : Int
            Bathrooms.

        Returns
        -------
        Bathrooms Category.

        """
        
        if bathrooms >= 2:
            bathroomsCat = 2
        else:
            bathroomsCat = bathrooms
        
        return bathroomsCat