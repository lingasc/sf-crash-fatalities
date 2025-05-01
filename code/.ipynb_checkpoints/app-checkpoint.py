import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import os
import traceback

# Set page config
st.set_page_config(
    page_title="Traffic Fatality Risk Explorer",
    page_icon="🚗",
    layout="wide"
)

# Sunset melody colors
sunset_melody = ['#252D6E', '#6F5B96', '#B3698A', '#F08A7B', '#FFA477']

# Custom CSS for sunset melody theme
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F08A7B;
        color: white;
    }
    h1, h2, h3 {
        color: #252D6E;
    }
    .stButton>button {
        background-color: #6F5B96;
        color: white;
    }
    .stButton>button:hover {
        background-color: #B3698A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for safe folium display
def safe_folium_display(map_obj, width=1000, height=600):
    """Safely display a folium map with error handling."""
    try:
        folium_static(map_obj, width=width, height=height)
        return True
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")
        st.warning("Displaying a fallback map visualization instead.")
        return False

# Load the data and model
@st.cache_data
def load_data():
    try:
        if os.path.exists('../data/fatality_data_processed.csv'):
            df = pd.read_csv('../data/fatality_data_processed.csv')
            return df
        elif os.path.exists('../data/cleaned_fatalities.csv'):
            df = pd.read_csv('../data/cleaned_fatalities.csv')
            return df
        else:
            st.error("No data file found. Please run train_model.py first.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        if os.path.exists('../code/fatality_risk_model.pkl'):
            with open('../code/fatality_risk_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.error("Model file not found. Please run train_model.py first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_shap_values():
    try:
        if os.path.exists('../code/shap_values.pkl') and os.path.exists('../code/feature_names.pkl'):
            with open('../code/shap_values.pkl', 'rb') as f:
                shap_values, X_test_processed = pickle.load(f)
            with open('../code/feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            return shap_values, X_test_processed, feature_names
        else:
            st.warning("SHAP values files not found. Feature importance visualization will not be available.")
            return None, None, None
    except Exception as e:
        st.warning(f"Error loading SHAP values: {str(e)}")
        return None, None, None

# Create a function to generate the risk map
def create_risk_map(df, time_filter=None, day_filter=None, category_filter=None):
    try:
        # Center the map on San Francisco
        sf_center = [37.7749, -122.4194]
        m = folium.Map(location=sf_center, zoom_start=13, tiles='cartodbpositron')
        
        # Filter data based on user selections
        filtered_df = df.copy()
        if time_filter and time_filter != 'All':
            filtered_df = filtered_df[filtered_df['time_of_day'] == time_filter]
        if day_filter and len(day_filter) > 0:
            filtered_df = filtered_df[filtered_df['day_of_week'].isin(day_filter)]
        if category_filter and len(category_filter) > 0:
            filtered_df = filtered_df[filtered_df['collision_category'].isin(category_filter)]
        
        # Add the heatmap layer
        heat_data = [[row['latitude'], row['longitude'], 1] for _, row in filtered_df.iterrows()]
        HeatMap(heat_data, 
                radius=15, 
                gradient={0.2: sunset_melody[0], 0.4: sunset_melody[1], 
                         0.6: sunset_melody[2], 0.8: sunset_melody[3], 
                         1.0: sunset_melody[4]},
                min_opacity=0.5,
                blur=10).add_to(m)
        
        # Add markers for each fatality with popup information
        for _, row in filtered_df.iterrows():
            # Convert all values to strings to avoid the AttributeError
            popup_text = f"""
            <b>Date:</b> {str(row.get('collision_date', 'N/A'))}<br>
            <b>Time:</b> {str(row.get('time_of_day', 'N/A'))}<br>
            <b>Category:</b> {str(row.get('collision_category', 'N/A'))}<br>
            <b>Age:</b> {str(row.get('age', 'N/A'))}<br>
            <b>Sex:</b> {str(row.get('sex', 'N/A'))}<br>
            """
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color=sunset_melody[0],
                fill=True,
                fill_color=sunset_melody[0],
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

# Function to prepare input for model prediction
def prepare_model_input(latitude, longitude, time_of_day, day_of_week, collision_category):
    try:
        input_data = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'collision_hour': [12],  # Default to noon
            'collision_month': [6],  # Default to June
            'time_of_day': [time_of_day],
            'day_of_week': [day_of_week],
            'collision_category': [collision_category]
        })
        return input_data
    except Exception as e:
        st.error(f"Error preparing model input: {str(e)}")
        return None

# Function to create a fallback map using plotly
def create_fallback_map(df, time_filter=None, day_filter=None, category_filter=None):
    try:
        # Filter data based on user selections
        filtered_df = df.copy()
        if time_filter and time_filter != 'All':
            filtered_df = filtered_df[filtered_df['time_of_day'] == time_filter]
        if day_filter and len(day_filter) > 0:
            filtered_df = filtered_df[filtered_df['day_of_week'].isin(day_filter)]
        if category_filter and len(category_filter) > 0:
            filtered_df = filtered_df[filtered_df['collision_category'].isin(category_filter)]
        
        # Create a scatter mapbox
        fig = px.scatter_mapbox(
            filtered_df, 
            lat="latitude", 
            lon="longitude", 
            color="collision_category" if "collision_category" in filtered_df.columns else None,
            color_discrete_sequence=sunset_melody,
            zoom=11, 
            height=600,
            hover_data=["time_of_day", "day_of_week", "age", "sex"]
        )
        fig.update_layout(mapbox_style="carto-positron")
        return fig
    except Exception as e:
        st.error(f"Error creating fallback map: {str(e)}")
        return None

# Load data and model
df = load_data()
model = load_model()
shap_values, X_test_processed, feature_names = load_shap_values()

# Check if data and model loaded successfully
data_loaded = not df.empty
model_loaded = model is not None
shap_loaded = shap_values is not None

# App title and description
st.title("Traffic Fatality Risk Explorer")
st.markdown("Analyze and predict traffic fatality risks in San Francisco using the sunset melody theme.")

# Show loading status
if not data_loaded:
    st.error("❌ Data failed to load")
else:
    st.success(f"✅ Data loaded successfully: {len(df)} records")

if not model_loaded:
    st.error("❌ Model failed to load")
else:
    st.success("✅ Model loaded successfully")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Risk Map", "Risk Prediction", "Risk Factors"])

with tab1:
    st.header("Fatality Risk Map")
    
    # Sidebar filters for the map
    st.sidebar.header("Map Filters")
    
    time_filter = st.sidebar.selectbox(
        'Time of Day',
        ['All', 'Morning (5am-12pm)', 'Afternoon (12pm-5pm)', 'Night (9pm-5am)']
    )
    
    # Get unique days of week from the data
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_filter = st.sidebar.multiselect('Day of Week', days, default=days)
    
    # Get unique collision categories from the data
    if data_loaded and 'collision_category' in df.columns:
        categories = df['collision_category'].unique().tolist()
        category_filter = st.sidebar.multiselect('Collision Category', categories, default=categories)
    else:
        categories = []
        category_filter = []
        st.sidebar.warning("Collision categories not available")
    
    # Create and display the map
    if data_loaded:
        try:
            with st.spinner("Generating risk map..."):
                risk_map = create_risk_map(df, time_filter, day_filter, category_filter)
                
                if risk_map is not None:
                    # Try to display with folium_static
                    if not safe_folium_display(risk_map, width=1000, height=600):
                        # If folium display fails, use the fallback
                        st.write("Using fallback map visualization:")
                        fallback_map = create_fallback_map(df, time_filter, day_filter, category_filter)
                        if fallback_map is not None:
                            st.plotly_chart(fallback_map, use_container_width=True)
                        else:
                            st.error("Both primary and fallback map visualizations failed.")
                else:
                    st.error("Failed to create map.")
        except Exception as e:
            st.error(f"Error in map tab: {str(e)}")
            st.error(traceback.format_exc())
    else:
        st.warning("Map cannot be displayed because data failed to load.")

with tab2:
    st.header("Risk Prediction")
    
    if data_loaded and model_loaded:
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Location")
                latitude = st.number_input("Latitude", value=37.7749, format="%.5f")
                longitude = st.number_input("Longitude", value=-122.4194, format="%.5f")
            
            with col2:
                st.subheader("Conditions")
                time_of_day = st.selectbox(
                    'Time of Day',
                    ['Morning (5am-12pm)', 'Afternoon (12pm-5pm)', 'Night (9pm-5am)']
                )
                day_of_week = st.selectbox('Day of Week', days)
                if categories:
                    collision_category = st.selectbox('Collision Category', categories)
                else:
                    collision_category = "Pedestrian"  # Default if categories not available
            
            if st.button("Predict Risk"):
                with st.spinner("Calculating risk..."):
                    # Prepare input for model
                    input_data = prepare_model_input(latitude, longitude, time_of_day, day_of_week, collision_category)
                    
                    if input_data is not None:
                        try:
                            # Get prediction
                            risk_score = model.predict_proba(input_data)[0][1]  # Assuming binary classification
                            
                            # Display result with appropriate color from sunset melody
                            if risk_score < 0.2:
                                color = sunset_melody[0]
                                risk_level = "Very Low"
                            elif risk_score < 0.4:
                                color = sunset_melody[1]
                                risk_level = "Low"
                            elif risk_score < 0.6:
                                color = sunset_melody[2]
                                risk_level = "Moderate"
                            elif risk_score < 0.8:
                                color = sunset_melody[3]
                                risk_level = "High"
                            else:
                                color = sunset_melody[4]
                                risk_level = "Very High"
                            
                            st.markdown(f"<h2 style='color:{color}'>Risk Level: {risk_level}</h2>", unsafe_allow_html=True)
                            
                            # Create a gauge chart for the risk score
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = risk_score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Risk Score", 'font': {'color': color, 'size': 24}},
                                gauge = {
                                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': color},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [0, 0.2], 'color': sunset_melody[0]},
                                        {'range': [0.2, 0.4], 'color': sunset_melody[1]},
                                        {'range': [0.4, 0.6], 'color': sunset_melody[2]},
                                        {'range': [0.6, 0.8], 'color': sunset_melody[3]},
                                        {'range': [0.8, 1], 'color': sunset_melody[4]},
                                    ],
                                }
                            ))
                            
                            st.plotly_chart(fig)
                            
                            # Add risk factors explanation
                            st.subheader("Risk Factors for This Location")
                            st.write(f"• Time of Day: {time_of_day} is a {'higher' if time_of_day == 'Night (9pm-5am)' else 'moderate'} risk period")
                            st.write(f"• Collision Category: {collision_category} collisions are {'common' if collision_category == 'Pedestrian' else 'less common'} in this area")
                            st.write(f"• Day of Week: {day_of_week}s have {'higher' if day_of_week in ['Friday', 'Saturday'] else 'typical'} risk patterns")
                            
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
                    else:
                        st.error("Could not prepare input data for prediction.")
        except Exception as e:
            st.error(f"Error in prediction tab: {str(e)}")
    else:
        st.warning("Risk prediction is not available because data or model failed to load.")

with tab3:
    st.header("Risk Factors Analysis")
    
    if data_loaded:
        try:
            # Create a summary of risk factors
            st.subheader("Key Risk Factors")
            
            # Feature importance visualization
            if shap_loaded:
                try:
                    st.write("The chart below shows the importance of different factors in predicting fatality risk:")
                    
                    # Create a summary plot of feature importance
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(
                        shap_values, 
                        X_test_processed,
                        feature_names=feature_names,
                        plot_type="bar",
                        color=sunset_melody[2],
                        show=False
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error displaying SHAP values: {str(e)}")
                    st.warning("Feature importance visualization could not be displayed.")
            else:
                st.warning("SHAP values not available. Feature importance visualization cannot be displayed.")
            
            # Show collision statistics
            st.subheader("Collision Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    # Collision category distribution
                    if 'collision_category' in df.columns:
                        category_counts = df['collision_category'].value_counts().reset_index()
                        category_counts.columns = ['Category', 'Count']
                        
                        fig = px.bar(
                            category_counts, 
                            x='Category', 
                            y='Count',
                            title='Fatalities by Collision Category',
                            color_discrete_sequence=sunset_melody
                        )
                        st.plotly_chart(fig)
                    else:
                        st.warning("Collision category data not available.")
                except Exception as e:
                    st.error(f"Error creating collision category chart: {str(e)}")
            
            with col2:
                try:
                    # Time of day distribution
                    if 'time_of_day' in df.columns:
                        time_counts = df['time_of_day'].value_counts().reset_index()
                        time_counts.columns = ['Time of Day', 'Count']
                        
                        fig = px.pie(
                            time_counts, 
                            names='Time of Day', 
                            values='Count',
                            title='Fatalities by Time of Day',
                            color_discrete_sequence=sunset_melody
                        )
                        st.plotly_chart(fig)
                    else:
                        st.warning("Time of day data not available.")
                except Exception as e:
                    st.error(f"Error creating time of day chart: {str(e)}")
            
            # Additional statistics
            try:
                if 'age' in df.columns and 'collision_category' in df.columns:
                    st.subheader("Age Distribution by Collision Category")
                    
                    # Calculate age statistics by collision category
                    age_stats = df.groupby('collision_category')['age'].agg(['mean', 'median', 'min', 'max']).reset_index()
                    age_stats.columns = ['Collision Category', 'Mean Age', 'Median Age', 'Min Age', 'Max Age']
                    age_stats = age_stats.round(1)
                    
                    st.dataframe(age_stats)
                    
                    # Create a box plot
                    fig = px.box(
                        df, 
                        x='collision_category', 
                        y='age',
                        title='Age Distribution by Collision Category',
                        color='collision_category',
                        color_discrete_sequence=sunset_melody
                    )
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error creating age distribution visualization: {str(e)}")
        
        except Exception as e:
            st.error(f"Error in risk factors tab: {str(e)}")
    else:
        st.warning("Risk factors cannot be displayed because data failed to load.")

# Footer
st.markdown("---")
st.markdown("Traffic Fatality Risk Explorer | Powered by Sunset Melody Theme")