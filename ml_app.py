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
                 - Pickup date : YYYY-MM-DD UTC
                 - Pickup time : hh:mm UTC
                 - Pickup longitude : Pickup Location, Longitude
                 - Pickup latitude : Pickup Location, Latitude
                 - Dropoff longitude : dropoff Location, Longitude
                 - Dropoff latitude : dropoff Location, Latitude
                 - Passenger count : 1 - 7
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
    passcount = st.number_input("Number of Passenger :",0,7,1)

    
    #Jarak perjalanan
    loc1=(picklat, picklong)
    loc2=(droplat, droplong)
    distance = hs.haversine(loc1,loc2,unit=Unit.KILOMETERS)
    ori_data = {"Date" : [pickup_dt],
                "Time" : [(pickup_tm.hour, pickup_tm.minute)],
                "Pickup Location" : [(picklat, picklong)],
                "Dropoff Location" : [(droplat, droplong)],
                "Number of Passenger" : [passcount]}
                
    df_ori = pd.DataFrame(ori_data)
    df_ori["Number of Passenger"] = df_ori["Number of Passenger"].astype(int)
    st.table(df_ori)
    dfl = pd.DataFrame({"lat" : [picklat], "long" : [picklong]})
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
                "Distance_in_Km" : [np.log1p(distance)]}
    df_new = pd.DataFrame(ml_data)
    
    model_reg, scaler = joblib.load('model_with_scaler.joblib')
    scaled_data = scaler.transform(df_new)
    
    df_scld = pd.DataFrame(scaled_data, columns=df_new.columns)
    prediction_log = model_reg.predict(df_scld)
    prediction_reg = np.exp(prediction_log)-1
    
  
    # prediction section
    st.subheader("Prediction result")
    st.write("Fare amount :", prediction_reg[0])
    st.write("Distance :", distance, "km")
    st.write("Pickup Location :")
    
    arr1 = np.array([[picklat, picklong]])
    df1 = pd.DataFrame(
    arr1,
    columns=['lat', 'lon'])
    st.map(df1)

    st.write("Dropoff Location :")
    
    arr2 = np.array([[droplat, droplong]])
    df2 = pd.DataFrame(
    arr2,
    columns=['lat', 'lon'])
    st.map(df2)
