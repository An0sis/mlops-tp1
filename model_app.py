import streamlit as st
import joblib

prediction = ""
model = joblib.load("regression.joblib")
st.number_input("Size (in sqft)", key="size", value=0)
st.number_input("Number of rooms", key="nb_rooms", value = 0)
st.checkbox("Garden",key="garden", value = False)
if st.button("Predict"):
    prediction = model.predict([[st.session_state.size, st.session_state.nb_rooms, st.session_state.garden]])[0]
st.write("Predicted price:", prediction, "€")