import numpy as np
import pickle
import streamlit as st
import sklearn

# Load the model
pickle_in = open("linear_regression.pkl", "rb")
linear_regression = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_medical_insurance_cost(age, sex, bmi, children, smoker, region):
    prediction = linear_regression.predict([[age, sex, bmi, children, smoker, region]])
    return prediction

def main():
    st.title("Welcome to our website!")
    
    # HTML for the title
    html_temp = """
    <div style="background-color:tomato;padding:10px;border-radius:10px;margin-bottom:10px">
    <h3 style="color:white;text-align:center;">Medical Insurance Premium Cost Prediction</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    
    try:
        with st.form(key="insurance_form"):
            age = st.number_input("Age", min_value=1, max_value=100, value=1, step=1)
            sex = st.radio("Sex", ["Male", "Female"])
            bmi = st.number_input("BMI", value=25.0, step=0.1)
            children = st.number_input("Number of Children", min_value=0, value=0, step=1)
            smoker = st.radio("Smoker", ["Yes", "No"])
            region = st.radio("Region (Hemisphere)", ["North-East", "North-West", "South-East", "South-West"])

            submit_button = st.form_submit_button(label='Predict')
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    result = 0.0
    default_used = False
    if submit_button:
        try:
            # Convert sex and smoker inputs to numerical values
            sex = 1 if sex == "Male" else 0
            smoker = 1 if smoker == "Yes" else 0
            region_dict = {"North-East": 0, "North-West": 1, "South-East": 2, "South-West": 3}
            region = region_dict[region]

            # Make prediction
            prediction = predict_medical_insurance_cost(age, sex, bmi, children, smoker, region)[0]

            # Ensure the result is not less than 1000
            if prediction < 1000:
                result = 1000
                default_used = True
            else:
                result = prediction
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Display the result message if prediction is successful
    raw_text = u"\u20B9"  # Unicode for Indian Rupee symbol
    if default_used:
        st.success(f'The approximate medical insurance premium cost per year is {raw_text} {result:.2f}. Default value is used because the predicted value was less than 1000.')
    else:
        st.success(f'The approximate medical insurance premium cost per year is {raw_text} {result:.2f}.')

    if st.button("About"):
        st.text("Built by Bhavesh Dwaram")
        st.text("NIE Mysuru")

if __name__ == '__main__':
    main()
