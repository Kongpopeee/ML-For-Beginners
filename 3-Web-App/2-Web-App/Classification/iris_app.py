import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 

# loading in the model to predict on the data 
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

def welcome(): 
    return 'welcome all'

def prediction(sepal_length, sepal_width, petal_length, petal_width): 
    # Make the prediction
    prediction = classifier.predict( 
        [[float(sepal_length), float(sepal_width), 
          float(petal_length), float(petal_width)]])

    # Convert numeric prediction to class name
    iris_class = {
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    }

    # Return the class name based on the predicted index
    return iris_class[int(prediction[0])]  # Ensure we get the first element as an integer

def main(): 
    st.title("Iris Flower Prediction") 

    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1> 
    </div> 
    """

    st.markdown(html_temp, unsafe_allow_html = True) 

    sepal_length = st.number_input(
        "Sepal Length",
        min_value=0.00,
        max_value=10.00,
        value=5.00,
        step=0.01,
        format="%.2f"
    )

    sepal_width = st.number_input(
        "Sepal Width",
        min_value=0.00,
        max_value=5.00,
        value=2.50,
        step=0.01,
        format="%.2f"
    )

    petal_length = st.number_input(
        "Petal Length",
        min_value=0.00,
        max_value=7.00,
        value=3.50,
        step=0.01,
        format="%.2f"
    )

    petal_width = st.number_input(
        "Petal Width",
        min_value=0.00,
        max_value=3.00,
        value=1.50,
        step=0.01,
        format="%.2f"
    )

    result ="" 

    if st.button("Predict"): 
        result = prediction(sepal_length, sepal_width, petal_length, petal_width) 
    st.success('This flower is classified as {}'.format(result)) 

if __name__=='__main__': 
    main()
