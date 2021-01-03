#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:51:27 2020

@author: Dario Corral
"""

import requests
import pandas as pd
import json
import time


class Api(object):
    """
    Class to get data from Idealista API
    
    """

    #Key & Host
    rapid_api_key = "753477ba1amsh01854eb07e8d826p199d93jsna40f6bbac65d"
    rapid_api_host = "idealista2.p.rapidapi.com"
    
    #Urls
    url_autocomplete = "https://idealista2.p.rapidapi.com/auto-complete"
    url_list = "https://idealista2.p.rapidapi.com/properties/list"
    
    
    def get_response_message(self, json_data):
        """
        

        Returns
        -------
        None.

        """ 
        
        try:
            
            return json_data['message']
            
        except:
            
            pass
        
        return
    
    def get_location(self, city = 'Varese'):
        """
        
    
        Parameters
        ----------
        city : Str, optional
            DESCRIPTION. The default is 'Varese'.
    
        Returns
        -------
        DataFrame with Location List.
    
        """
    
        #querystring = {"prefix":"varese","country":"it"}
        
        
        
        querystring = {"prefix":city.lower(),"country":"it"}
        
        headers = {
            'x-rapidapi-key': self.rapid_api_key,
            'x-rapidapi-host': self.rapid_api_host
            }
    
        response = requests.request("GET", self.url_autocomplete, 
                                    headers=headers, params=querystring)
    
    
        json_data = json.loads(response.text)
        
        json_message = self.get_response_message(json_data)
        
        if json_message is not None:
            
            raise Exception(json_message)
            
        else:
    
            df_locations = pd.DataFrame.from_dict(json_data['locations'])
            
            return df_locations
    
    
    def get_properties(self,city = 'Varese', saveCSV = True):
        """
        
    
        Parameters
        ----------
        city : str, optional
            DESCRIPTION. The default is 'Varese'.
            
        saveCSV: Bool
    
        Returns
        -------
        Dataframe.
    
        """
    
    
        df_locations = self.get_location(city = city.lower())
    
        
        querystring = {"operation":"sales","locationName":str(df_locations.name[0]),
                       "locationId":str(df_locations.locationId[0]),"maxItems":"1000",
                       "country":"it","locale":"en","sort":"asc","numPage":"1"}
    
    
        headers = {
            'x-rapidapi-key': self.rapid_api_key,
            'x-rapidapi-host': self.rapid_api_host,
            'Content-Type': 'application/json;charset=UTF-8'
            }
    
        response = requests.request("GET", self.url_list, headers=headers, 
                                    params=querystring)
    
    
        json_data_list = json.loads(response.text)
    
    
        df_houses = pd.DataFrame()    
    
    
        for i in  range(1,json_data_list['totalPages']+1):
            
            
            
            querystring = {"operation":"sales",
                           "locationName":str(df_locations.name[0]),
                       "locationId":str(df_locations.locationId[0]),
                       "maxItems":"40",
                       "country":"it","locale":"en","sort":"asc","numPage":i}
            
            try:
            
                response = requests.request("GET", self.url_list, 
                                        headers=headers, params=querystring)
            
                json_data_house = json.loads(response.text)
                
                json_message = self.get_response_message(json_data_house)
        
                if json_message is not None:
                    
                    print (json_message)
                    
                    return

                df_house_raw = pd.DataFrame.from_dict(
                                    json_data_house['elementList'])
                
                df_houses = df_houses.append(df_house_raw)
                
                #Wait 5 seconds
                time.sleep(5)
        
            except:
                
                print("Error fetching Data")
                continue
            
        
        if saveCSV:
            
            df_houses.to_csv('data/'+str(city.lower())+'_houses.csv')
    
    
        return df_houses
