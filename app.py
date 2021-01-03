#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:29:46 2020

@author: dariocorral
"""
import streamlit as st
import math
from model import Model



def run_app():
    """
    

    Returns
    -------
    None.

    """
    #load Module class
    # Create a text element and let the reader know the data is loading.
    
    with st.spinner(text='In progress'):
    #data_load_state = st.text('Loading data & model...')
    # Notify the reader that the data was successfully loaded.
        model = Model()
        st.success('Model Ready')
    
    #Dataset & zones
    df = model.dataset_preprocessed

    #Title & Header Image
    st.title('Valuta la tua casa a Varese')
    
    st.subheader("Scopri il valore di mercato degli immobili "
             "secondo i siti web piú importanti.")
    
    st.image('./data/Header Varese.jpg',use_column_width=True)
    
    st.subheader ("Usiamo un algoritmo di Machine Learning su %s immobili a Varese"
         % len(df))

    #Parameters
    st.subheader("Imposta i parametri dell'immobile")
    
    
        
    propertyTypelist = ['Appartamento', 'Attico', 'Studio','Duplex',
                    'Villa', 'Casa Rustica']
    
    
    zonesList = df['district'].unique().tolist()
    
    zone = st.selectbox('Zona', zonesList, index = 0)
    
    propertyType = st.selectbox('Tipo', propertyTypelist, index = 0)
    
    if propertyType == 'Appartamento':
    
        propertyType = 'flat'
    
    elif propertyType == 'Attico':
    
        propertyType = 'penthouse'

    elif propertyType == 'Studio':
    
        propertyType = 'studio'
        
    elif propertyType == 'Duplex':
    
        propertyType = 'duplex'
        
    elif propertyType == 'Villa':
    
        propertyType = 'chalet'
    
    elif propertyType == 'Casa Rustica':
    
        propertyType = 'countryHouse'



    #Rest of parameters  
    size = st.number_input('Metri Quadri',
              min_value=int(math.floor(df['size'].quantile(0.01) / 25) * 25), 
              max_value=int(math.ceil(df['size'].quantile(0.99) / 35) * 25),
              value = int(math.ceil(df['size'].mode().values[0] / 25) * 25)
              )
    
    
    rooms = st.slider('Locali',
            min_value = 1,
            max_value = int(math.ceil(df['rooms'].quantile(0.99) / 5) * 5),
            value = int(df['rooms'].mode().values[0])
        )
    
    #floor
    if propertyType in ('chalet','countryHouse'):
        
        floor = 0
        
    elif propertyType == 'penthouse':
        
        floor = st.slider('Piano',
                min_value = 1,
                max_value = int(math.ceil(df['floor'].quantile(0.99) / 5) * 5),
                value = 4
                )
    else:
        
        floor = st.slider('Piano',
                min_value = 0,
                max_value = int(math.ceil(df['floor'].quantile(0.99) / 5) * 5),
                value = 1
            )
    
    bathrooms = int(df['bathrooms'].mode().values[0])
    
    bathrooms = st.slider('Bagni',
            min_value = 0,
            max_value = int(math.ceil(df['bathrooms'].quantile(0.99) / 5) * 5),
            value = int(df['bathrooms'].mode().values[0])
                          )
    
    #Status italiano
    status_it = ['Da ristrutturare','Buono',
                          'Nuova Costruzione']
    
    
    status = st.radio('Stato',status_it, index = 1)
    #Mapping status
    if status == 'Da ristrutturare':
        
        status = 'renew'
        
    elif status == 'Buono':
        
        status = 'good'
        
    elif status == 'Nuova Costruzione':
    
        status = 'newdevelopment'

    lift = st.checkbox('Ascensore', value = 0)
    
    parking = st.checkbox('Posto Auto', value = 0)
    
    box = st.checkbox('Box', value = 0) 
    
    terrace = st.checkbox('Terrazo', value = 0)
    
    garden = st.checkbox('Giardino', value = 0)
    
    air_conditioning = 0
    air_conditioning = st.checkbox('Aria Condizionanta',value = 0)
    
    swimming_pool = st.checkbox('Piscina', value = 0)
            
    
    #Button to value    
    button = st.button('Valuta')
        
    if button:
        
        value_model = model.predict( 
                    rooms,
                    bathrooms,
                    floor,
                    zone,
                    status,
                    int(lift),
                    int(air_conditioning),
                    int(box),
                    int(terrace),
                    int(garden),
                    int(swimming_pool),
                    int(parking),
                    propertyType
                    ) 

        
        value = int(math.floor((value_model * size) / 5000.0) * 5000.0)
        
        st.write("%i €" % value)
    
if __name__ == "__main__":
    
    run_app()

