#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:29:46 2020

@author: dariocorral
"""
import streamlit as st
import math
from model import Model



def app_run():
    """
    Returns
    -------
    Run App
    
    """
    #load Module class
    # Create a text element and let the reader know the data is loading.
    
    with st.spinner(text='In progress'):
    #data_load_state = st.text('Loading data & model...')
    # Notify the reader that the data was successfully loaded.
        __model = Model()
        st.success('Model Ready')
    
    #Dataset & zones
    df = __model.dataset_preprocessed
    
    #Parking Space Price
    parking_price_df =(
        df.loc[df['parkingSpacePrice']>10000].
            groupby('district').mean()['parkingSpacePrice'])
    
    #Title & Header Image
    st.title('Valuta la tua casa a Varese')
    
    st.subheader("Scopri il valore di mercato della tua casa "
             "comodo e facile con un click")
    
    st.image('./data/Header Varese.jpg',use_column_width=True)
    
    st.subheader ("Usiamo un algoritmo di Machine Learning su %s immobili a Varese"
         % len(df))
    
    #Parameters
    st.subheader("Imposta i parametri dell'immobile")
    
    #Zone
    zonesList = df['district'].unique().tolist()
    
    district = st.selectbox('Zona', zonesList, index = 0)
    
    #Property Type List
    propertyTypelist = __model.propertyTypeList

    propertyType = st.selectbox('Tipo', propertyTypelist, index = 0)
    
    #Conversiont to model variables
    propertyType = __model.propertyTypeConverter(propertyType)
    
    #Rest of parameters  
    size = st.number_input('Metri Quadri',
              min_value=10, 
              max_value=5000,
              value = 100
              )
    
    rooms = st.slider('Locali',
            min_value = 1,
            max_value =  10,
            value = 3)
        
    #Conversiont to model variables
    #roomsCat = __model.roomsCategory(rooms)
    if rooms >= 4:
        roomsCat = 4
    else:
        roomsCat = rooms

    #Bathrooms
    bathrooms = st.slider('Bagni',
            min_value = 1,
            max_value = 10,
            value = 2
                          )
    #Conversiont to model variables
    if bathrooms >= 2:
            bathroomsCat = 2
    else:
        bathroomsCat = bathrooms

    #Status italiano
    status_it = __model.statusList
    
    status = st.radio('Stato',status_it, index = 1)
    
    #Conversiont to model variables
    statusOutput = 'good'

    if status == "Da ristrutturare":
        
        statusOutput = 'renew'
        
    elif status == "Buono":
        
        statusOutput = 'good'
        
    elif status == "Nuova Costruzione":
    
        statusOutput = 'newdevelopment'
    #Extra Feautures
    #parkingBox = st.checkbox('Posto Auto - Box', value = 0)
        
    #garden = st.checkbox('Giardino- Terrazzo', value = 0)
    
    #swimming_pool = st.checkbox('Piscina', value = 0)
    
    #Parking Space District Selected
    
    try:
    
        parking_space = int(
            parking_price_df.loc[parking_price_df.index==district].values[0])
        
    except:
        
        parking_space = 0
            
    #Button to value    
    button = st.button('Valuta')
        
    if button:
        
        value_model = __model.predict( 
            size,
            propertyType,
            district,
            statusOutput,
            roomsCat,
            bathroomsCat,
            ) 
    
        value = int(math.ceil((value_model ) / 5000.0) * 5000.0)
        
        st.write("Valore di mercato")
        st.write("{:,.0f}€".format(value))
        
        if parking_space > 0:
            
            st.write("Prezzo medio Posto auto di Varese - %s " % district)
            st.write("{:,.0f}€".format(parking_space))
        
        
        
    
if __name__ == "__main__":
    
    app_run()
    
