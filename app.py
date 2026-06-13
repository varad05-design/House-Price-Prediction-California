import streamlit as st
import pandas as pd
import sklearn
import joblib

# Must be the first Streamlit command
st.set_page_config(
    page_title="California Real Estate Predictor",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('final_model.joblib')
    return scaler, model

scaler, model = load_assets()

# Main Title and Description
st.title('🏡 California Real Estate Price Predictor')
st.markdown("""
Welcome to the **California Real Estate Price Predictor**! 
Enter the details of the neighborhood below to get an estimated median house price.
""")
st.divider()

def user_input_features():
    st.subheader("📍 Location Details")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input('Latitude', 32.0, 42.0, 35.6, 0.1, help="Geographical latitude")
    with col2:
        longitude = st.number_input('Longitude', -124.0, -114.0, -119.5, 0.1, help="Geographical longitude")

    st.subheader("🏠 Property Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        house_age = st.slider("House Age (Years)", 1, 52, 25, help="Median age of a house within a block")
    with col4:
        avg_rooms = st.number_input('Avg Number of Rooms', 2.0, 10.0, 5.0, 0.5, help="Average number of rooms within a block")
    with col5:
        avg_bedrms = st.number_input('Avg Number of Bedrooms', 1.0, 5.0, 1.0, 0.5, help="Average number of bedrooms within a block")

    st.subheader("👥 Demographics & Economics")
    col6, col7, col8 = st.columns(3)
    with col6:
        med_inc = st.number_input("Median Income ($10k)", 1.0, 15.0, 3.5, 0.1, help="Median income for households within a block of houses (measured in tens of thousands of US Dollars)")
    with col7:
        population = st.number_input('Block Population', 500, 5000, 1500, 100, help="Total number of people residing within a block")
    with col8:
        avg_occup = st.number_input('Avg House Occupancy', 1.0, 10.0, 2.5, 0.25, help="Average number of household members")

    data = {
        'HouseAge': house_age,
        'MedInc': med_inc,
        'AveRooms': avg_rooms,
        'AveBedrms': avg_bedrms,
        'Population': population,
        'AveOccup': avg_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }
    return data

user_inputs = user_input_features()

input_df = pd.DataFrame(user_inputs, index=[0])

# Create the 'rooms_per_person' feature.
if input_df['Population'][0] > 0:
    input_df['rooms_per_person'] = input_df['AveRooms'][0] / input_df['Population'][0]
else:
    input_df['rooms_per_person'] = 0

# Create the 'bedrooms_per_room' feature.
if input_df['AveRooms'][0] > 0:
    input_df['bedrooms_per_room'] = input_df['AveBedrms'][0] / input_df['AveRooms'][0]
else:
    input_df['bedrooms_per_room'] = 0

final_feature_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                       'Latitude', 'Longitude', 'rooms_per_person', 'bedrooms_per_room']

input_df = input_df[final_feature_order]

st.divider()

# Prediction Section
st.subheader("🔮 Price Prediction")
st.markdown("Review your inputs and click the button to see the model's prediction.")

predict_btn = st.button('Predict Price', type='primary', use_container_width=True)

if predict_btn:
    with st.spinner("Calculating estimated price..."):
        scaled_input = scaler.transform(input_df[final_feature_order])
        prediction = model.predict(scaled_input)
        predicted_price = prediction[0]
        final_price = predicted_price * 100000
        
    st.success("Prediction Complete!")
    
    # Display the result using a nice metric card
    st.metric(label="Estimated Median House Price", value=f"${final_price:,.0f}")
else:
    st.info("Click the 'Predict Price' button above to generate a prediction.")

with st.expander("View Raw Input Data"):
    st.dataframe(input_df, hide_index=True)