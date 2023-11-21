import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#3872fb;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Uber Fare Amount Prediction App </h1>
		    <h4 style="color:white;text-align:center;">Analyst Team </h4>
		    </div>
            """

desc_temp = """
            ### Fare Amount Prediction App
            This app will be used by the driver to predict the fare amount of the order
            #### Data Source
            - https://github.com/A-Arthur-A/Digital_Skola_Final_Project/raw/main/uber.csv
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Regression Section
            """

def main():

    stc.html(html_temp)
    
    menu = ['Home', 'Fare Amount Prediction']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.markdown(desc_temp)
    elif choice == "Fare Amount Prediction":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
