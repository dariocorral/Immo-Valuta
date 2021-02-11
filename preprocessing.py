#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 10:03:13 2020

@author: dariocorral
"""


import pandas as pd
import numpy as np
from utils import Utils




class Preprocessing(object):
    
    """
    Class to preprocessing data
    
    """
    
    def __init__(self):
        
        self.__utils = Utils()
    
    
    def execute(self, saveCSV = True):
        
        """
        ETL Process Execution
        """
    
        #Datasets Varese houses Test
        df_varese = pd.read_csv("data/varese_houses.csv")
        
        
        df_varese['district'] = 'VA - ' + df_varese['district']
        
        df_monza = pd.read_csv("data/monza_houses.csv")
        
        df_monza ['district'] = 'MO - ' + df_monza['district']
        
        #Df total
        df = df_varese.append(df_monza)
        
        #Drop first column
        df.drop('Unnamed: 0',axis = 1,inplace=True)
        
        #Apply unnest json columns
        column_features_exp = self.__utils.unnest_column_json(df, 'features')
        column_parking_exp = self.__utils.unnest_column_json(df, 'parkingSpace')
        
        #Add columns to Df
        df[[*column_features_exp.columns]]=column_features_exp
        df[[*column_parking_exp.columns]]=column_parking_exp
        
        #Select Column important
        df = df[['price','propertyType','size','floor','rooms','bathrooms',
                 'district','latitude','longitude','status','hasLift','hasAirConditioning', 
                 'hasBoxRoom', 'hasTerrace','hasGarden', 'hasSwimmingPool', 
                 'hasParkingSpace','isParkingSpaceIncludedInPrice', 
                 'parkingSpacePrice']]
        
        #Price + parking Price
        df['price'] = np.where(
            (df['isParkingSpaceIncludedInPrice']==False)&
            (df['hasParkingSpace']==True),
            df['price']+df['parkingSpacePrice'],
            df['price']
            )
        
        #Avg Price District
        df['priceByArea'] = df['price']/df['size']
        
        #Floor to Int
        df.floor.replace(['nan','en','bj','st','ss'],0,inplace = True)
        df.floor.fillna(0,inplace = True)
        df.floor = df.floor.astype(int)
        
        #District /Status as Category
        df.district = df.district.astype('category')
        
        #Fill NaN hasLift
        df.hasLift.fillna(False,inplace = True)
        
        avg_price_district_df = (
            pd.DataFrame(df.groupby(['district']).mean()
                         ['priceByArea'].astype(int))).reset_index()
        
        df = df.merge(avg_price_district_df, left_on='district', 
                       right_on='district')
        
        df.rename(columns={'priceByArea_x':'priceByArea',
                           'priceByArea_y':'avgPriceZone'},inplace = True)
        
        #Status NaN to renew or Good
        df['status'] = np.where( 
                    (df['priceByArea'] < df['avgPriceZone'])
                    &(df['status'].isnull())
                        ,'renew',
                        np.where(
                            (df['priceByArea'] >= df['avgPriceZone'])
                            &(df['status'].isnull()),
                                'good',
                                    df['status']
                                ))
        
        #Add renew Ligth
        df['status'] = np.where( 
                    (df['priceByArea'] < df['avgPriceZone'])&
                    (df['status']=='good'),
                    'renew', df['status']
                        )
        
        
        #Status as category
        df.status = df.status.astype('category')

        
        df = df[['price', 'propertyType', 'size','priceByArea',  'floor', 'rooms',
                 'bathrooms','district', 'latitude','longitude','avgPriceZone',
                 'status', 'hasLift','hasAirConditioning', 'hasBoxRoom',
                 'hasTerrace', 'hasGarden',
                 'hasSwimmingPool','hasParkingSpace','parkingSpacePrice'
                 ]]
        
        df['priceByArea'] = df['priceByArea'].astype(int)
        
        #Change boolean columns to 1/0
        df[['hasLift','hasAirConditioning', 'hasBoxRoom',
               'hasTerrace', 'hasGarden', 'hasSwimmingPool', 
       'hasParkingSpace']]=(
              df[['hasLift','hasAirConditioning', 
                  'hasBoxRoom','hasTerrace', 'hasGarden', 
                  'hasSwimmingPool', 'hasParkingSpace']].astype(int))
       
        #status Int
        statusInt=(df.groupby('status').mean()['priceByArea']).rank().astype(int)
       
        df['statusInt'] = pd.merge(df['status'],statusInt.reset_index(),
                      on='status',how='left').iloc[:,-1:]
                       
        #Box Posto Auto Union
        df['box_posto_auto'] = np.where(
           ((df['hasBoxRoom'] ==1)|(df['hasParkingSpace'] == 1)),
           1,0)
       
        #Property Type
        df['propertyType'] = df['propertyType'].astype('category')
       
        propertyTypeInt=(
            df.groupby('propertyType').mean()['priceByArea']).rank().astype(int)
       
        df['propertyTypeInt'] = pd.merge(df['propertyType'],
                    propertyTypeInt.reset_index(),on='propertyType',
                    how='left').iloc[:,-1:]
       
        #Garden / Terrace Union
        df['garden_terrace'] = np.where(
           (df['hasTerrace'] == 1)|
           (df['hasGarden'] ==1),
           1,0)
        
        #Floor 
        df['floorCat']=np.where(
                 (df['floor']<=1)
                    ,0,1)
        #Lift Cat
        df['liftCat'] = np.where(
                 (df['hasLift'] == 1)  & (df['floor'] >= 2)
                    ,1,0)
        
        #Rooms Cat
        df['roomsCat'] = np.where(
                 (df['rooms'] >=4 )  
                    ,4,df['rooms'])
        
        #Rooms Cat
        df['bathroomsCat'] = np.where(
                 (df['bathrooms'] >= 2)  
                    ,2,df['bathrooms'])
       
        if saveCSV:
            
            df.to_csv("data/houses_clean.csv")
        
        return df

