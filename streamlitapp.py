import numpy as np
import pickle
import streamlit as st
import sklearn

# Load the models
with open("linear_regression.pkl", "rb") as file:
    linear_regression = pickle.load(file)

with open("linear_regression_case_2.pkl", "rb") as file:
    linear_regression_case_2 = pickle.load(file)

with open("polynomial_regression.pkl", "rb") as file:
    polynomial_regression = pickle.load(file)

with open("family_floater.pkl", "rb") as file:
    family_floater = pickle.load(file)

# Dictionary to store the models
models = {
    "Linear Regression": linear_regression,
    "Polynomial Regression": polynomial_regression,
    "Linear Regression Case 2": linear_regression_case_2,
    "Family Floater": family_floater
}


def predict_medical_insurance_cost(model, features):
    prediction = model.predict([features])
    return prediction


def main():
    st.title("Welcome to our Medical Insurance Prediction Website")

    if 'step' not in st.session_state:
        st.session_state.step = 0

    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    def next_step():
        st.session_state.step += 1
        st.experimental_rerun()

    def prev_step():
        st.session_state.step -= 1
        st.experimental_rerun()

    def reset_form():
        st.session_state.form_data = {}
        st.session_state.step = 0
        st.experimental_rerun()

    if st.session_state.step == 0:
        st.write("""
            ## Welcome to the Medical Insurance Prediction Website
            This tool helps you predict the annual medical insurance premium cost based on various factors. 
            You can choose from different models to get the prediction.
        """)
        if st.button("Try Our Product"):
            next_step()

    elif st.session_state.step == 1:
        st.write("### Choose a Model")
        model_choice = st.selectbox("Choose the model", list(models.keys()),
                                    index=list(models.keys()).index(st.session_state.form_data.get('model', 'Linear Regression')))
        st.session_state.form_data['model'] = model_choice
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back"):
                prev_step()
        with col2:
            if st.button("Next"):
                next_step()

    elif st.session_state.step == 2:
        model_choice = st.session_state.form_data['model']
        with st.form(key="insurance_form"):
            age = st.number_input("Age", min_value=1, max_value=100,
                                  value=st.session_state.form_data.get('age', 1), step=1)
            sex = st.radio("Sex", ["Male", "Female"], index=["Male", "Female"].index(
                st.session_state.form_data.get('sex', "Male")))
            bmi = st.number_input("BMI", value=st.session_state.form_data.get('bmi', 25.0), step=0.1)
            children = st.number_input("Number of Children", min_value=0,
                                       value=st.session_state.form_data.get('children', 0), step=1)
            smoker = st.radio("Smoker", ["Yes", "No"], index=["Yes", "No"].index(
                st.session_state.form_data.get('smoker', "No")))
            region = st.radio("Region (Hemisphere)", ["North-East", "North-West", "South-East", "South-West"],
                              index=["North-East", "North-West", "South-East", "South-West"].index(
                                  st.session_state.form_data.get('region', "North-East")))

            if model_choice == "Family Floater":
                parents = st.number_input("Number of Parents", min_value=0, max_value=2,
                                          value=st.session_state.form_data.get('parents', 0), step=1)
                st.session_state.form_data['parents'] = parents

            st.session_state.form_data.update({
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'children': children,
                'smoker': smoker,
                'region': region
            })

            submit_button = st.form_submit_button(label='Predict')

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back"):
                prev_step()
        with col2:
            if st.button("Reset"):
                reset_form()

        if submit_button:
            # Validation
            error = None
            if not (1 <= age <= 100):
                error = "Age must be between 1 and 100."
            elif not (0 <= bmi <= 100):
                error = "BMI must be between 0 and 100."
            elif not (0 <= children <= 10):
                error = "Number of children must be between 0 and 10."
            elif model_choice == "Family Floater" and not (0 <= parents <= 2):
                error = "Number of parents must be between 0 and 2."

            if error:
                st.error(error)
            else:
                st.session_state.step += 1
                st.experimental_rerun()

    elif st.session_state.step == 3:
        model_choice = st.session_state.form_data['model']
        age = st.session_state.form_data['age']
        sex = 1 if st.session_state.form_data['sex'] == "Male" else 0
        bmi = st.session_state.form_data['bmi']
        children = st.session_state.form_data['children']
        smoker = 1 if st.session_state.form_data['smoker'] == "Yes" else 0
        region_dict = {"North-East": 0, "North-West": 1, "South-East": 2, "South-West": 3}
        region = region_dict[st.session_state.form_data['region']]
        parents = st.session_state.form_data.get('parents', 0)

        # Prepare features based on selected model
        if model_choice == "Family Floater":
            features = [age, sex, bmi, children, smoker, region, parents]
        else:
            features = [age, sex, bmi, children, smoker, region]

        model = models[model_choice]
        prediction = predict_medical_insurance_cost(model, features)[0]

        # Ensure the result is not less than 1000
        default_used = False
        if prediction < 1000:
            result = 1000
            default_used = True
        else:
            result = prediction

        raw_text = u"\u20B9"  # Unicode for Indian Rupee symbol
        if default_used:
            st.success(
                f'The approximate medical insurance premium cost per year is {raw_text} {result:.2f}. Default value is used because the predicted value was less than 1000.')
        else:
            st.success(
                f'The approximate medical insurance premium cost per year is {raw_text} {result:.2f}.')

        # Display model accuracy and plot some graphs (for demonstration purposes)
        st.write(f"### Model Accuracy: (This is a placeholder, replace with actual accuracy)")

        st.write("### Sample Graphs")
        st.line_chart(np.random.randn(10, 2))
        st.bar_chart(np.random.randn(10, 2))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back"):
                prev_step()
        with col2:
            if st.button("Reset"):
                reset_form()
                

    if st.button("About"):
        st.text("Built by Bhavesh Dwaram")
        st.text("NIE Mysuru")


if __name__ == '__main__':
    main()
