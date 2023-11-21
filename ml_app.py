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

dep = {'Sales & Marketing':1, 'Operations':2, 'Technology':3, 'Analytics':4,
       'R&D':5, 'Procurement':6, 'Finance':7, 'HR':8, 'Legal':9}
edu = {'Below Secondary':1, "Bachelor's":2, "Master's & above":3}
rec = {'referred':1, 'sourcing':2, 'others':3}
gen = {'m':1, 'f':2}
reg = {'region_1':1,'region_2':2,'region_3':3,'region_4':4,'region_5':5,
       'region_6':6,'region_7':7,'region_8':8,'region_9':9,'region_10':10,
       'region_11':11,'region_12':12,'region_13':13,'region_14':14,'region_15':15,
       'region_16':16,'region_17':17,'region_18':18,'region_19':19,'region_20':20,
       'region_21':21,'region_22':22,'region_23':23,'region_24':24,'region_25':25,
       'region_26':26,'region_27':27,'region_28':28,'region_29':29,'region_30':30,
       'region_31':31,'region_32':32,'region_33':33,'region_34':34}
now = dt.datetime.now()


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
          
def load_scaler(scaler_file):
    loaded_scaler = joblib.load(open(os.path.join(scaler_file), 'rb'))
    return loaded_scaler

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
    department = st.selectbox('Department', ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 
                                             'R&D', 'Procurement', 'Finance', 'HR', 'Legal'])
    region = st.selectbox('Region', ['region_1','region_2','region_3','region_4','region_5', 'region_6','region_7',
                                     'region_8','region_9','region_10','region_11','region_12',
                                     'region_13','region_14','region_15','region_16','region_17','region_18','region_19',
                                     'region_20','region_21','region_22','region_23','region_24','region_25','region_26',
                                     'region_27','region_28','region_29','region_30','region_31','region_32','region_33',
                                     'region_34'])
    education = st.selectbox('Education', ["Below Secondary", "Bachelor's", "Master's & above"])
    gender = st.radio('Gender', ['m','f'])
    recruitment = st.selectbox("Recruitment Channel", ["referred", "sourcing", "others"])
    training = st.number_input("No of Training", 1, 10)
    age = st.number_input("Age",10,60)
    rating = st.number_input("Previous Year Rating",1,5)
    service = st.number_input("Length of Service",1,37)
    awards = st.radio("Awards Won", [0,1])
    avg_training = st.number_input("Average Training Score",0,100)
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
        result = {
             'Department':department,
            'Region':region,
            'education':education,
            'gender':gender,
            'recruitment_channel':recruitment,
            'no_of_trainings':training,
            'age':age,
            'previous_year_rating':rating,
            'length_of_service':service,
            'awards_won':awards,
            'avg_training_score':avg_training,
        }

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
    
    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'R&D', 'Procurement', 'Finance', 'HR', 'Legal']:
            res = get_value(i, dep)
            encoded_result.append(res)
        elif i in ['region_1','region_2','region_3','region_4','region_5', 'region_6','region_7',
                                    'region_8','region_9','region_10','region_11','region_12',
                                    'region_13','region_14','region_15','region_16','region_17','region_18','region_19',
                                    'region_20','region_21','region_22','region_23','region_24','region_25','region_26',
                                    'region_27','region_28','region_29','region_30','region_31','region_32','region_33',
                                    'region_34']:
            res = get_value(i, reg)
            encoded_result.append(res)
        elif i in ["Below Secondary", "Bachelor's", "Master's & above"]:
            res = get_value(i, edu)
            encoded_result.append(res)
        elif i in ['m','f']:
            res = get_value(i, gen)
            encoded_result.append(res)
        elif i in ["referred", "sourcing", "others"]:
            res = get_value(i, rec)
            encoded_result.append(res)
    
    st.write(encoded_result)
    # prediction section
    st.subheader("Prediction result")
    st.write("Fare amount :", prediction_reg[0,0])
    
