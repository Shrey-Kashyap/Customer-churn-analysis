import streamlit as st
import pandas as pd
import joblib 
from sklearn.preprocessing import RobustScaler
# Function to input customer data
def input_customer_data():
    CreditScore = st.number_input('Credit Score', min_value=0, max_value=1000, value=500)
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.number_input('Age', min_value=18, max_value=100, value=30)
    Tenure = st.number_input('Tenure (years)', min_value=0, max_value=10, value=5)
    Balance = st.number_input('Balance', min_value=0.0, value=0.0, step=1000.0)
    NumOfProducts = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
    HasCrCard = st.selectbox('Has Credit Card', [0, 1])
    IsActiveMember = st.selectbox('Is Active Member', [0, 1])
    EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
    Geography = st.selectbox('Geography', ['Germany', 'Spain', 'France'])
    
    # One-hot encoding for Geography
    Geography_Germany = 1 if Geography == 'Germany' else 0
    Geography_Spain = 1 if Geography == 'Spain' else 0

    data = {
        'CreditScore': CreditScore,
        'Gender': 1 if Gender == 'Male' else 0,  # Assuming Male = 1, Female = 0
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Geography_Germany': Geography_Germany,
        'Geography_Spain': Geography_Spain
    }

    return pd.DataFrame(data, index=[0])

# Streamlit app layout
st.title("Customer Churn Analysis")

st.header("Please Enter customer details:")

customer_data = input_customer_data()

st.subheader("Customer Data")


model=joblib.load("ChurnPredictor")
rs=joblib.load("ChurnScaler")

def on_button_click():
    st.write(customer_data)
    ans=model.predict(rs.transform([customer_data.loc[0, :].values]))
    st.write("Predicted : Will Exit" if ans==1 else "Predicted : Will not Exit")

# Create a button
if st.button('Predict'):
    on_button_click()



