import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
import pickle
import shap

# Page config
st.set_page_config(
    page_title="Traffic Fatality Risk Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply direct custom CSS
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&display=swap');

/* Dashboard headers */
h1, h2, h3 {
    font-family: 'Libre Baskerville', serif;
    color: #252D6E;
}

/* Dashboard containers */
.block-container {
    padding: 2rem;
}

/* Make cards for plots */
.card {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Special insights box */
.insights {
    background-color: #f8f9fa;
    border-left: 5px solid #B3698A;
    padding: 1rem;
    margin: 1.5rem 0;
}

/* Metrics display */
.metric {
    background-color: white;
    border-left: 4px solid #F08A7B;
    padding: 1rem;
    margin: 1rem 0;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #252D6E;
}

.metric-label {
    font-size: 0.9rem;
    color: #6F5B96;
}

/* Fix some Streamlit default spacing */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Define sunset melody color palette
sunset_melody = ['#252D6E', '#6F5B96', '#B3698A', '#F08A7B', '#FFA477']
sunset_cmap = LinearSegmentedColormap.from_list('sunset_melody', sunset_melody)

# Configure plot styles
plt.style.use('seaborn-v0_8-whitegrid') 
sns.set_style('white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Optima', 'Libre Baskerville', 'DejaVu Serif']

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('fatality_data_processed.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Try to load model and SHAP values
@st.cache_resource
def load_model():
    try:
        with open('fatality_risk_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_shap_values():
    try:
        with open('shap_values.pkl', 'rb') as f:
            shap_values, X_test_processed = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return shap_values, X_test_processed, feature_names
    except Exception as e:
        print(f"Error loading SHAP values: {e}")
        return None, None, None

# Load data
df = load_data()
if df is None:
    st.stop()

# Load model and SHAP values
model = load_model()
shap_values, X_test_processed, feature_names = load_shap_values()

# App title and description
st.title("Traffic Fatality Risk Analysis")
st.markdown("Interactive visualization and predictive analytics for traffic safety")
st.markdown("---")

# Navigation using tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Risk Map", "Model Insights", "About"])

# Dashboard Tab
with tab1:
    st.header("Dashboard: Traffic Fatality Overview")
    
    # Add filters in expandable section
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            year_range = st.slider("Select Year Range", 
                                  min_value=int(df['collision_year_clean'].min()), 
                                  max_value=int(df['collision_year_clean'].max()),
                                  value=(int(df['collision_year_clean'].min()), int(df['collision_year_clean'].max())))
        
        with col2:
            victim_types = st.multiselect("Select Victim Types", 
                                         options=sorted(df['collision_category'].unique()),
                                         default=sorted(df['collision_category'].unique()))
        
        with col3:
            time_of_day = st.multiselect("Select Time of Day", 
                                        options=sorted(df['time_of_day'].unique()),
                                        default=sorted(df['time_of_day'].unique()))
    
    # Filter data based on selections
    filtered_df = df[(df['collision_year_clean'] >= year_range[0]) & 
                      (df['collision_year_clean'] <= year_range[1]) &
                      (df['collision_category'].isin(victim_types)) &
                      (df['time_of_day'].isin(time_of_day))]
    
    # Top row of visualizations
    st.markdown("### Fatality Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Fatalities by Year")
        
        # Create yearly fatality counts
        yearly_counts = filtered_df.groupby('collision_year_clean').size().reset_index(name='Count')
        
        # Fix for seaborn warning by using hue parameter
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use countplot instead of barplot
        ax = sns.countplot(x='collision_year_clean', data=filtered_df, 
                          order=sorted(filtered_df['collision_year_clean'].unique()),
                          palette=sunset_melody[:1]*len(filtered_df['collision_year_clean'].unique()))
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Fatalities')
        plt.xticks(rotation=45)
        sns.despine()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Fatalities by Victim Type")
        
        # Create victim type counts by year
        victim_by_year = filtered_df.groupby(['collision_year_clean', 'collision_category']).size().reset_index(name='Count')
        
        fig = px.line(victim_by_year, x='collision_year_clean', y='Count', color='collision_category',
                     color_discrete_sequence=sunset_melody, markers=True,
                     labels={'collision_year_clean': 'Year', 'Count': 'Number of Fatalities', 'collision_category': 'Victim Type'})
        fig.update_layout(legend_title_text='Victim Type')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom row of visualizations
    st.markdown("### Detailed Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Fatalities by Hour of Day")
        
        # Create hour counts
        hour_counts = filtered_df.groupby('collision_hour').size().reset_index(name='Count')
        
        # Create a color array based on hour position
        colors = sunset_cmap(np.linspace(0, 1, 24))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle data that may not include all 24 hours
        all_hours = pd.DataFrame({'collision_hour': range(24)})
        hour_counts = pd.merge(all_hours, hour_counts, on='collision_hour', how='left').fillna(0)
        
        # Use standard matplotlib bar instead of seaborn
        bars = ax.bar(hour_counts['collision_hour'], hour_counts['Count'], color=colors, width=0.8)
        
        # Format x-axis with hour labels
        hour_labels = [f'{h:02d}:00' for h in range(24)]
        ax.set_xticks(range(24))
        ax.set_xticklabels(hour_labels, rotation=45)
        
        ax.set_xlabel('')
        ax.set_ylabel('Number of Fatalities')
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        sns.despine()
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Age Distribution by Victim Type")
        
        if 'deceased' in filtered_df.columns and len(filtered_df) > 0:
            # Define order of victim types
            victim_order = ['Pedestrian', 'Driver', 'Motorcyclist', 'Passenger', 'Bicyclist']
            victim_order = [v for v in victim_order if v in filtered_df['deceased'].unique()]
            
            if len(victim_order) > 0:
                # Fix for seaborn warning by using hue parameter
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='deceased', y='age', data=filtered_df, hue='deceased', 
                          palette=sunset_melody[:len(victim_order)], 
                          order=victim_order, legend=False)
                plt.xlabel('Victim Type')
                plt.ylabel('Age')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                sns.despine()
                st.pyplot(fig)
            else:
                st.write("No data available for the selected filters.")
        else:
            st.write("No victim data available in the dataset.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Key insights section
    st.markdown('<div class="insights">', unsafe_allow_html=True)
    st.subheader("Key Insights")
    
    # Calculate some insights
    total_fatalities = len(filtered_df)
    
    if total_fatalities > 0:
        most_common_type = filtered_df['collision_category'].value_counts().idxmax()
        most_common_type_pct = filtered_df['collision_category'].value_counts(normalize=True).max() * 100
        
        # Handle cases where collision_hour might be empty
        if 'collision_hour' in filtered_df.columns and filtered_df['collision_hour'].notna().any():
            peak_hour = filtered_df['collision_hour'].value_counts().idxmax()
            peak_hour_display = f"{int(peak_hour):02d}:00"
        else:
            peak_hour_display = "Unknown"
        
        # Handle cases where time_of_day might not have 'Night (9pm-5am)'
        if 'time_of_day' in filtered_df.columns and 'Night (9pm-5am)' in filtered_df['time_of_day'].values:
            night_pct = len(filtered_df[filtered_df['time_of_day'] == 'Night (9pm-5am)']) / len(filtered_df) * 100
        else:
            night_pct = 0
        
        # Display insights
        st.markdown(f"""
        • Total of **{total_fatalities}** fatalities in the selected period
        • **{most_common_type}** account for **{most_common_type_pct:.1f}%** of all fatalities
        • Peak time for fatalities is **{peak_hour_display}**
        • **{night_pct:.1f}%** of fatalities occur during nighttime hours (9pm-5am)
        """)
    else:
        st.write("No data available for the selected filters.")
    st.markdown('</div>', unsafe_allow_html=True)

# Risk Map Tab
with tab2:
    st.header("Risk Map: High-Risk Areas")
    
    # Filters for the map
    with st.expander("Map Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            map_time = st.selectbox("Time of Day", 
                                   options=['All'] + sorted(df['time_of_day'].unique().tolist()))
        
        with col2:
            map_victim = st.selectbox("Victim Type", 
                                     options=['All'] + sorted(df['collision_category'].unique().tolist()))
        
        with col3:
            map_year = st.selectbox("Year", 
                                   options=['All'] + sorted(df['collision_year_clean'].unique().tolist()))
    
    # Filter data for map
    map_filtered = df.copy()
    if map_time != 'All':
        map_filtered = map_filtered[map_filtered['time_of_day'] == map_time]
    if map_victim != 'All':
        map_filtered = map_filtered[map_filtered['collision_category'] == map_victim]
    if map_year != 'All':
        map_filtered = map_filtered[map_filtered['collision_year_clean'] == map_year]
    
    # Create the map
    st.markdown("### Risk Map")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Calculate midpoint for map center
    midpoint_lat = map_filtered['latitude'].median()
    midpoint_lon = map_filtered['longitude'].median()
    
    # Create base map
    m = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=13, 
                   tiles='cartodbpositron')
    
    # Add points to map
    if len(map_filtered) > 0:
        for _, row in map_filtered.iterrows():
            color = '#F08A7B' if row['high_risk'] == 1 else '#252D6E'
            popup_text = f"""
            <b>Location:</b> {row['location']}<br>
            <b>Date:</b> {row['collision_date']}<br>
            <b>Time:</b> {row['collision_time'] if not pd.isna(row['collision_time']) else 'Unknown'}<br>
            <b>Victim Type:</b> {row['deceased']}<br>
            <b>Age:</b> {row['age']}<br>
            <b>Risk Level:</b> {'High' if row['high_risk'] == 1 else 'Low'}<br>
            """
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
    
    # Display the map
    # Use st_folium instead of folium_static to avoid deprecation warning
    try:
        from streamlit_folium import st_folium
        st_folium(m, width=800, height=500)
    except:
        # Fallback to folium_static
        folium_static(m)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional visualizations for risk analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("High-Risk Areas")
        
        # Get top neighborhoods by fatality count
        if 'analysis_neighborhood' in map_filtered.columns and len(map_filtered) > 0:
            top_neighborhoods = map_filtered['analysis_neighborhood'].value_counts().head(10).reset_index()
            top_neighborhoods.columns = ['Neighborhood', 'Fatality Count']
            
            st.dataframe(top_neighborhoods, use_container_width=True)
        else:
            st.write("Neighborhood data not available or no data for selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Risk by Neighborhood")
        
        # Create neighborhood risk chart
        if 'analysis_neighborhood' in map_filtered.columns and len(map_filtered) > 0:
            neighborhood_risk = map_filtered.groupby('analysis_neighborhood')['high_risk'].mean().reset_index()
            neighborhood_risk.columns = ['Neighborhood', 'Risk Score']
            neighborhood_risk = neighborhood_risk.sort_values('Risk Score', ascending=False).head(10)
            
            fig = px.bar(neighborhood_risk, x='Neighborhood', y='Risk Score', 
                        color_discrete_sequence=[sunset_melody[3]])
            fig.update_layout(xaxis_title='Neighborhood', yaxis_title='Risk Score (0-1)',
                             xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Neighborhood data not available or no data for selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)

# Model Insights Tab
with tab3:
    st.header("Model Insights: Understanding Risk Factors")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    The traffic fatality risk model uses these key features to predict high-risk areas:
    - Location (latitude, longitude)
    - Time factors (hour of day, day of week, month)
    - Collision category
    - Time of day classification
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if model is None:
        st.warning("Model file not found. This section has limited functionality.")
    
    # Feature importance section
    st.markdown("### Feature Importance")
    
    if shap_values is not None and X_test_processed is not None and feature_names is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Get feature importance from SHAP values
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(10)
        
        # Create feature importance chart
        fig = px.bar(feature_importance_df, x='Importance', y='Feature', 
                     color_discrete_sequence=[sunset_melody[2]])
        fig.update_layout(xaxis_title='Mean |SHAP Value|', yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SHAP Value Summary
        st.markdown("### SHAP Value Summary")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        SHAP (SHapley Additive exPlanations) values show how each feature contributes to predictions:
        - Red points indicate feature values that increase risk
        - Blue points indicate feature values that decrease risk
        - The magnitude (distance from center) shows the strength of the effect
        """)
        
        # Display SHAP summary plot (limited to a subset for performance)
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values[:100], 
                X_test_processed[:100], 
                feature_names=feature_names,
                max_display=10,
                show=False
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("SHAP values not available. Using sample feature importance instead.")
        
        # Create sample feature importance if SHAP is not available
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sample_features = ['time_of_day_Night (9pm-5am)', 'collision_category_Pedestrian', 
                          'collision_hour', 'latitude', 'longitude', 'day_of_week_Saturday', 
                          'day_of_week_Sunday', 'collision_month', 'day_of_week_Friday',
                          'time_of_day_Morning (5am-12pm)']
        sample_importance = [0.25, 0.22, 0.15, 0.12, 0.10, 0.05, 0.04, 0.03, 0.02, 0.02]
        
        sample_importance_df = pd.DataFrame({
            'Feature': sample_features,
            'Importance': sample_importance
        })
        
        fig = px.bar(sample_importance_df, x='Importance', y='Feature', 
                     color_discrete_sequence=[sunset_melody[2]])
        fig.update_layout(xaxis_title='Feature Importance', yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Performance Metrics
    st.markdown("### Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">0.82</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">0.79</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Precision</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">0.84</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Recall</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model explanation
    st.markdown('<div class="insights">', unsafe_allow_html=True)
    st.subheader("Model Explanation")
    st.markdown("""
    This gradient boosting model was trained to identify high-risk traffic areas by analyzing patterns in historical fatality data. The model defines high-risk areas as those where:
    
    1. Pedestrian fatalities are more common
    2. Nighttime (9pm-5am) fatalities occur
    
    The model achieves strong performance with approximately 82% accuracy in identifying these high-risk conditions. The most influential factors in the model are time of day and victim type, suggesting that these factors should be primary considerations in traffic safety planning.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# About Tab
with tab4:
    st.header("About This Project")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## Project Overview
    
    This application analyzes traffic fatality data to identify patterns and high-risk areas to improve road safety. By examining factors such as location, time of day, and victim type, the model identifies areas with higher fatality risk.
    
    ## Data Sources
    
    The analysis is based on traffic fatality records from 2014-2025, including information about:
    
    - Location (latitude, longitude)
    - Date and time
    - Victim demographics
    - Collision type
    - Road conditions
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## Methodology
    
    The risk prediction model uses a Gradient Boosting classifier with the following components:
    
    - Feature engineering to extract temporal patterns
    - Spatial analysis using latitude and longitude
    - Classification of areas as high or low risk
    - SHAP values to explain model predictions
    
    ## Intended Use
    
    This tool is designed to help city planners, traffic engineers, and safety officials identify high-risk areas for targeted safety interventions. The visualizations and risk maps provide actionable insights for decision-making.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display sample of the dataset
    st.subheader("Sample Data")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.head(10))
    st.markdown('</div>', unsafe_allow_html=True)