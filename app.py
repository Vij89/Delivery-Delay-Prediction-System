import pandas as pd
import streamlit as st
import joblib

st.title("Delivery Delay Prediction System")
st.write("Enter shipment details to predict late-delivery risk.")

# Load trained model and preprocessing files
model = joblib.load("delivery_delay_model.pkl")
preprocessor = joblib.load("delivery_preprocessor.pkl")
input_options = joblib.load("delivery_input_options.pkl")

with st.form("shipment_form"):
    order_type = st.selectbox("Order Type", input_options["Type"])
    scheduled_days = st.number_input(
        "Scheduled Shipping Days", min_value=0, value=3
    )
    category = st.selectbox("Category", input_options["Category Name"])
    segment = st.selectbox(
        "Customer Segment", input_options["Customer Segment"]
    )
    market = st.selectbox("Market", input_options["Market"])
    region = st.selectbox("Order Region", input_options["Order Region"])
    shipping = st.selectbox("Shipping Mode", input_options["Shipping Mode"])
    discount = st.number_input("Order Item Discount", min_value=0.0, value=0.0)
    quantity = st.number_input("Order Item Quantity", min_value=1, value=1)
    sales = st.number_input("Sales", min_value=0.0, value=100.0)

    submitted = st.form_submit_button("Predict Delivery Risk")

if submitted:
    shipment = pd.DataFrame([{
        "Type": order_type,
        "Days for shipment (scheduled)": scheduled_days,
        "Category Name": category,
        "Customer Segment": segment,
        "Market": market,
        "Order Region": region,
        "Shipping Mode": shipping,
        "Order Item Discount": discount,
        "Order Item Quantity": quantity,
        "Sales": sales
    }])

    processed = preprocessor.transform(shipment)
    prediction = model.predict(processed)[0]

    if prediction == 1:
        st.warning("Prediction: Late delivery risk")
    else:
        st.success("Prediction: Not classified as late")