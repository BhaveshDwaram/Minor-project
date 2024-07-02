import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the models
with open("polynomial_regression.pkl", "rb") as f:
    polynomial_regression = pickle.load(f)

with open("family_floater.pkl", "rb") as f:
    family_floater_regression = pickle.load(f)

def welcome():
    return "Welcome All"

def predict_medical_insurance_cost(model, age, sex, bmi, children, smoker, region, parents=0):
    scaler = StandardScaler()
    if model == "Family Floater":
        features = np.array([[age, sex, bmi, children, smoker, region, parents]])
        prediction = family_floater_regression.predict(features)
    else:  # Polynomial regression
        features = np.array([[age, sex, bmi, children, smoker, region]])
        new_features_df = pd.DataFrame(features, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        new_features_df[['age', 'bmi']] = scaler.fit_transform(new_features_df[['age', 'bmi']])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        new_poly_features = poly.fit_transform(new_features_df[['age', 'bmi']])
        poly_feature_names = poly.get_feature_names_out(['age', 'bmi'])
        new_poly_df = pd.DataFrame(new_poly_features, columns=poly_feature_names)
        new_X_poly = pd.concat([new_features_df.drop(columns=['age', 'bmi']), new_poly_df], axis=1)
        prediction = polynomial_regression.predict(new_X_poly)
    
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

    model_type = st.selectbox("Choose Model Type", ["Polynomial", "Family Floater"])

    with st.form(key="insurance_form"):
        age = st.number_input("Age", min_value=1, max_value=100, value=1, step=1)
        sex = st.radio("Sex", ["Male", "Female"])
        bmi = st.number_input("BMI", value=25.0, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.radio("Smoker", ["Yes", "No"])
        region = st.radio("Region (Hemisphere)", ["North-East", "North-West", "South-East", "South-West"])

        if model_type == "Family Floater":
            parents = st.number_input("Number of Parents", min_value=0, max_value=2, value=0, step=1)
        else:
            parents = 0

        submit_button = st.form_submit_button(label='Predict')
    
    result = 0.0
    if submit_button:
        # Convert inputs
        sex = 1 if sex == "Male" else 0
        smoker = 1 if smoker == "Yes" else 0
        region_dict = {"North-East": 0, "North-West": 1, "South-East": 2, "South-West": 3}
        region = region_dict[region]

        # Validate inputs
        valid_input = True
        if not (1 <= age <= 100):
            st.error("Age must be between 1 and 100.")
            valid_input = False
        if not (0 <= bmi <= 100):
            st.error("BMI must be between 0 and 100.")
            valid_input = False
        if not (0 <= children <= 10):
            st.error("Number of children must be between 0 and 10.")
            valid_input = False
        if model_type == "Family Floater" and not (0 <= parents <= 2):
            st.error("Number of parents must be between 0 and 2.")
            valid_input = False

        if valid_input:
            try:
                # Make prediction
                prediction = predict_medical_insurance_cost(model_type, age, sex, bmi, children, smoker, region, parents)[0]

                # Ensure the result is not less than 1000
                if prediction < 1000:
                    result = 1000
                    default_used = True
                else:
                    result = prediction
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            result = 0.0

        # Display the result message if prediction is successful
        raw_text = u"\u20B9"  # Unicode for Indian Rupee symbol
        if result == 0.0:
            st.error('Invalid input values. Prediction could not be made.')
        else:
            if result == 1000:
                st.success(f'The approximate medical insurance premium cost per year is {raw_text} {result:.2f}. Default value is used because the predicted value was less than 1000.')
            else:
                st.success(f'The approximate medical insurance premium cost per year is {raw_text} {result:.2f}.')

    if st.button("About"):
        st.text("Built by Bhavesh Dwaram")
        st.text("NIE Mysuru")

if __name__ == '__main__':
    main()
