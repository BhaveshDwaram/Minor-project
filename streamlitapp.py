import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the models
with open("polynomial_regression.pkl", "rb") as f:
    polynomial_regression = pickle.load(f)

with open("family_floater.pkl", "rb") as f:
    family_floater = pickle.load(f)

def welcome():
    return "Welcome All"

def predict_medical_insurance_cost(model, age, sex, bmi, children, smoker, region, parents=0):
    scaler = StandardScaler()

    if model == "Family Floater":
        features = np.array([[age, sex, bmi, children, smoker, region, parents]])
        new_features_df = pd.DataFrame(features, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'parents'])
        new_features_df[['age', 'bmi']] = scaler.fit_transform(new_features_df[['age', 'bmi']])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        new_poly_features = poly.fit_transform(new_features_df[['age', 'bmi']])
        poly_feature_names = poly.get_feature_names_out(['age', 'bmi'])
        new_poly_df = pd.DataFrame(new_poly_features, columns=poly_feature_names)
        new_X_poly = pd.concat([new_features_df.drop(columns=['age', 'bmi']), new_poly_df], axis=1)
        prediction = family_floater.predict(new_X_poly)

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

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #000000;
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            background-color: #ff5733;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h3 {
            color: white;
            text-align: center;
        }
        .form-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: box-shadow 0.3s ease;
        }
        .form-container:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .form-container input, .form-container select, .form-container textarea {
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .form-container input:focus, .form-container select:focus, .form-container textarea:focus {
            border-color: #ff5733;
            box-shadow: 0 0 8px rgba(255, 87, 51, 0.6);
        }
        .result-box {
            background-color: #28a745;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            margin-top: 20px;
            margin-bottom: 20px;
            animation: fadeIn 0.5s ease-in-out;
        }
        .about-box {
            background-color: #007bff;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            animation: fadeIn 0.5s ease-in-out;
        }
        .footer {
            text-align: center;
            font-size: 30px;
            margin-top: 20px;
            color: #888;
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        /* Welcome Section */
        .welcome-section {
            background:#000000;
            padding: 10px;
            border-radius: 10px;
            color: white;
            text-align: left;
            margin-bottom: 30px;
            border :solid;
            border-color :#ff5733;
        }
        .welcome-section h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-align: center;
            color : white;
        }
        .welcome-section p {
            font-size: 1.2rem;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    # Welcome Section
    st.markdown("""
    <div class="welcome-section" >
        <h1>Welcome To Our  Website !</h1>
        <p>
            Predicting medical insurance premiums is essential for several reasons. 
            It allows insurance companies to accurately assess the financial risks associated with insuring individuals. 
            By leveraging predictive models, insurers can determine premiums that align closely with the expected costs of providing healthcare coverage to policyholders. 
            Also, accurate premium prediction promotes fairness and transparency in the insurance market. 
            It ensures that individuals are charged premiums that reflect their specific risk profiles based on factors such as age, medical history, lifestyle choices, and other relevant demographics. 
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header" id="prediction-model"><h3>Medical Insurance Premium Cost Prediction</h3></div>', unsafe_allow_html=True)

    model_type = st.selectbox("Choose Model Type", ["Polynomial Model", "Family Floater"])

    height_unit = st.selectbox("Height Unit", ["centimeter", "feet and inches"])
    
    with st.form(key="insurance_form", clear_on_submit=False):
        
        age = st.number_input("Age (in years)", min_value=1, max_value=100, value=1, step=1)
        sex = st.radio("Sex", ["Male", "Female"])
        weight = st.number_input("Weight (in kg)", value=70.0, step=0.1)
        
        if height_unit == "cm":
            height = st.number_input("Height (in cm)", value=170.0, step=0.1)
        else:
            col1, col2 = st.columns(2)
            with col1:
                feet = st.number_input("Height (Feet)", min_value=0, max_value=8, value=5, step=1)
            with col2:
                inches = st.number_input("Height (Inches)", min_value=0.0, max_value=11.9, value=8.0, step=0.1)
            height = feet * 30.48 + inches * 2.54  # Convert feet and inches to cm

        bmi = weight / (height / 100) ** 2  # Calculate BMI

        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.radio("Are you a Smoker ?", ["Yes", "No"])
        region = st.radio("Select the Region", ["North-East", "North-West", "South-East", "South-West"])

        if model_type == "Family Floater":
            parents = st.number_input("Number of Parents", min_value=0, max_value=2, value=0, step=1)
        else:
            parents = 0

        submit_button = st.form_submit_button(label='Predict insurance premium')

        st.markdown('</div>', unsafe_allow_html=True)
    
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
                st.markdown(f'<div class="result-box">The approximate medical insurance premium cost per year is {raw_text} {result:.2f}. (Default value is used)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box">The approximate medical insurance premium cost per year is {raw_text} {result:.2f}.</div>', unsafe_allow_html=True)

    if st.button("Built-By"):
        st.markdown('<div class="about-box">Built by Animesh Kumar ,Bhavesh Dwaram,Nishanth Hebbar, Damodhar <br>NIE Mysuru</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Medical Insurance Prediction App</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()