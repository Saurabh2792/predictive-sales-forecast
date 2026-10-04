import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and prep data
df = pd.read_csv('data/dataset.csv')
X = df[['Advertising_Spend', 'Store_Traffic', 'Holiday']]
y = df['Sales']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Dashboard Interface
st.title("Predictive Sales Forecaster")
st.write("Input daily metrics to forecast retail sales.")

# Inputs
spend = st.number_input("Advertising Spend ($)", min_value=0, max_value=5000, value=500)
traffic = st.number_input("Estimated Store Traffic", min_value=0, max_value=2000, value=150)
holiday = st.selectbox("Is it a Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction Logic
if st.button("Forecast Sales"):
    prediction = model.predict([[spend, traffic, holiday]])
    st.success(f"📈 Estimated Sales: ${prediction[0]:.2f}")

st.subheader("Historical Sales Data")
st.dataframe(df)