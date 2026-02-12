import streamlit as st
import joblib
import numpy as np

# 1. Load the model and the encoder
model = joblib.load('iris_model.pkl')
le = joblib.load('label_encoder.pkl')

# 2. Set up the Website Title
st.title("🌸 Iris Species Predictor")
st.write("Adjust the sliders below to predict the type of Iris flower.")

# 3. Create sliders for user input
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

# 4. Predict Button
if st.button("Predict Species"):
    # Arrange inputs into the correct format
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Translate number (0, 1, 2) back to name
    species_name = le.inverse_transform(prediction)
    
    # Display result
    st.success(f"The predicted species is: **{species_name[0]}**")
