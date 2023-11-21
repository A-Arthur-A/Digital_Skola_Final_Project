import streamlit as st
import pandas as pd
import numpy as np
import haversine as hs
from haversine import Unit
import datetime as dt


# import ml package
import joblib
import os

attribute_info = """
                 - pickup date : YYYY-MM-DD UTC
                 - pickup time : hh:mm UTC
                 - pickup longitude : Pickup Location, Longitude
                 - pickup longitude : Pickup Location, Latitude
                 - pickup longitude : dropoff Location, Longitude
                 - pickup longitude : dropoff Location, Latitude
                 - passenger count : 1 - 7
                 """

now = dt.datetime.now()


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def run_ml_app():
    st.subheader("ML Section")
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    pickup_dt = st.date_input("Pickup Date", value = now)
    pickup_tm = st.time_input("Pickup Time", value = now)
    picklong = st.number_input("Pickup Location Longitude :",-75.000000,-73.000000,-73.981880,0.001000,"%.6f")
    picklat = st.number_input("Pickup Location Latitude :",40.500000,41.500000,40.752805,0.001000,"%.6f")
    droplong = st.number_input("Dropoff Location Longitude :",-75.000000,-73.000000,-73.981010,0.001000,"%.6f")
    droplat = st.number_input("Dropoff Location Latitude :",40.500000,41.500000,40.752958,0.001000,"%.6f")
    passcount = st.number_input("Passenger count :",0,7,1)

    
    #Jarak perjalanan
    loc1=(picklat, picklong)
    loc2=(droplat, droplong)
    distance = hs.haversine(loc1,loc2,unit=Unit.KILOMETERS)
    
    with st.expander("Your Selected Options"):
        ml_data = {"pickup_longitude" : [picklong],
                "pickup_latitude" : [picklat],
                "dropoff_longitude" : [droplong],
                "dropoff_latitude" : [droplat],
                "passenger_count" : [passcount],
                "year" : [pickup_dt.year],
                "month" : [pickup_dt.month],
                "day" : [pickup_dt.day],
                "weekday" : [pickup_dt.weekday()],
                "hour" : [pickup_tm.hour],
                "Distance_in_Km" : [distance]}
        
    df_new = pd.DataFrame(ml_data)
    st.write(df_new)
    
    model_reg, scaler = joblib.load('model_with_scaler.joblib')
    scaled_data = scaler.transform(df_new)
    
    df_scld = pd.DataFrame(scaled_data, columns=df_new.columns)
    prediction_log = model_reg.predict(df_scld)
    prediction_reg = np.exp(prediction_log)-1
    st.write(prediction_reg)

  
    # prediction section
    st.subheader("Prediction result")
    st.write("Fare amount :", prediction_reg[0])
    
