import streamlit as st
import joblib
import numpy as np
saved_model = joblib.load('xgboost_model.joblib')

st.title("Iris Flower Classification")
SepalLength = st.number_input("Sepal Length")
SepalWidth = st.number_input("Sepal Width")
PetalLength = st.number_input("Petal Length")
PetalWidth = st.number_input("Petal Width")

if st.button("Predict"):
    features = np.array([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
    prediction = saved_model.predict(features)
    
    if prediction==0:
        st.write('The given flower is Iris-Sentosa')
    elif prediction==1:
        st.write('The given flower is Iris-versicolor')
    else:
        st.write('The given flower is Iris-virginica')
