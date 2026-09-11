
import os
import streamlit as st
import pandas as pd
import joblib

# Load the trained model committed to the repository
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_model_v1.joblib"
)

model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction")

st.write("""
This application predicts whether a customer is likely to purchase
the Wellness Tourism Package based on customer and interaction details.
""")

# Customer details
age = st.number_input("Age", 18, 100, 30)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    1, 20, 2
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

number_of_trips = st.number_input(
    "Number of Trips",
    0, 50, 3
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    0, 10, 0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    0.0, 1000000.0, 25000.0
)

# Customer interaction details
pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

number_of_followups = st.number_input(
    "Number of Followups",
    0, 20, 3
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    0.0, 100.0, 15.0
)

# Create input DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch
}])

# Prediction
if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Likely to Purchase"
        st.success(
            f"The model predicts that the customer is **{result}**."
        )
    else:
        result = "Unlikely to Purchase"
        st.info(
            f"The model predicts that the customer is **{result}**."
        )
