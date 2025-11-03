import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('house_price_model.pkl')
encoder = joblib.load('encoder.pkl')

st.title("House Price Prediction")


property_type = st.selectbox("Property Type", ["House", "Flat", "Upper Portion", "Lower Portion", "Room", "Farm House", 'Penthouse']) 
city = st.selectbox("City", ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"])    
purpose = st.selectbox("Purpose", ["Sale", "Rent"])
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
baths = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
area = st.number_input("Area in Marla", min_value=1, max_value=1000, value=10)

# Prepare input for the model
input_df = pd.DataFrame([[property_type, city, purpose]],
                        columns=['property_type', 'city', 'purpose'])

encoded_input = encoder.transform(input_df).toarray()
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(input_df.columns))

final_input = pd.concat([pd.DataFrame([[baths, bedrooms, area]], columns=['baths', 'bedrooms', 'Area_in_Marla']), encoded_df], axis=1)

# Make prediction
if st.button("Predict Price"):
    pred = model.predict(final_input)
    st.success(f"Predicted House Price: {np.expm1(pred[0]):,.2f}")
