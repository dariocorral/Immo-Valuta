#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:50:41 2020

@author: dariocorral
"""

import pandas as pd
import json
import io

class Utils(object):
    
    """
    Utils methods
    
    """
    
    def unnest_column_json(self, df, column):
        
        """
        Unnest columns from Idealista source
        """
        
        #Change Quotes for JSon
        df[column]= df[column].astype(str).str.replace("\'",'\"')
        
        if column == 'features':
        
            df[column].replace('nan','{"hasAirConditioning":false}',inplace = True)
        
        elif column =='parkingSpace':
            
            df[column].replace('nan','{"hasParkingSpace":false}',inplace = True)

        df[column] = df[column].str.replace('True','true')
        df[column]  = df[column].str.replace('False','false')
        
        type_column = df[column].to_list()
        
        type_series = pd.DataFrame()
        
        for i in range(0, len(type_column)):
            
            
            json_dict = json.load(io.StringIO(type_column[i]))
        
            type_series = type_series.append(pd.DataFrame(json_dict,index=[i]))    
        
        
        if column =='parkingSpace':
            
            type_series['parkingSpacePrice'].fillna(0,inplace = True)
        
        type_series.fillna(False,inplace = True)
        
        return type_series