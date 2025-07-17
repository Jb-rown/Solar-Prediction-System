import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import datetime
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Solar Energy Production Prediction System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        color: #28A745;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #E8F5E8;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data for demonstration
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic solar power data for demonstration"""
    np.random.seed(42)
    
    # Generate time series data
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate synthetic weather features
    data = {
        'DATE_TIME': dates,
        'AMBIENT_TEMPERATURE': np.random.normal(25, 8, n_samples),
        'MODULE_TEMPERATURE': np.random.normal(35, 10, n_samples),
        'IRRADIATION': np.random.uniform(0, 1000, n_samples),
        'HUMIDITY': np.random.uniform(20, 80, n_samples),
        'WIND_SPEED': np.random.uniform(0, 15, n_samples)
    }
    
    # Generate synthetic power output based on realistic relationships
    df = pd.DataFrame(data)
    df['HOUR'] = pd.to_datetime(df['DATE_TIME']).dt.hour
    df['MONTH'] = pd.to_datetime(df['DATE_TIME']).dt.month
    df['DAY_OF_YEAR'] = pd.to_datetime(df['DATE_TIME']).dt.dayofyear
    
    # Realistic power generation based on irradiation and temperature
    df['DC_POWER'] = (
        df['IRRADIATION'] * 0.05 +  # Primary factor
        df['AMBIENT_TEMPERATURE'] * 0.1 +  # Secondary factor
        np.sin(df['HOUR'] * np.pi / 12) * 20 +  # Daily pattern
        np.random.normal(0, 5, n_samples)  # Noise
    )
    
    # Ensure no negative power values
    df['DC_POWER'] = np.maximum(df['DC_POWER'], 0)
    
    # AC power is typically slightly less than DC power
    df['AC_POWER'] = df['DC_POWER'] * 0.95
    
    return df

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    """Load and prepare the solar power data"""
    df = generate_synthetic_data(2000)
    
    # Feature engineering
    df['HOUR'] = pd.to_datetime(df['DATE_TIME']).dt.hour
    df['MONTH'] = pd.to_datetime(df['DATE_TIME']).dt.month
    df['DAY_OF_YEAR'] = pd.to_datetime(df['DATE_TIME']).dt.dayofyear
    df['TEMP_IRRADIATION'] = df['AMBIENT_TEMPERATURE'] * df['IRRADIATION']
    df['IS_DAYTIME'] = ((df['HOUR'] >= 6) & (df['HOUR'] <= 18)).astype(int)
    
    return df

# Train model
@st.cache_resource
def train_model(df):
    """Train the Random Forest model"""
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 
                'HUMIDITY', 'WIND_SPEED', 'HOUR', 'MONTH', 'DAY_OF_YEAR', 
                'TEMP_IRRADIATION', 'IS_DAYTIME']
    
    X = df[features]
    y = df['DC_POWER']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, X_test, y_test, y_pred, features

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">☀️ Solar Energy Production Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### About this System")
        st.markdown("""
        This application predicts solar energy production based on weather conditions.
        
        **Features:**
        - Real-time prediction based on weather inputs
        - Interactive visualizations
        - Model performance metrics
        - SDG 7 alignment: Affordable and Clean Energy
        """)
        
        st.markdown("### How to Use")
        st.markdown("""
        1. Adjust the weather parameters in the left column
        2. Click 'Predict' to get power output prediction
        3. View visualizations and insights in the right column
        """)
    
    # Load data and train model
    df = load_and_prepare_data()
    model, rmse, r2, X_test, y_test, y_pred, features = train_model(df)
    
    # Layout: Three columns
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Column 1: Input Parameters
    with col1:
        st.markdown('<h3 class="sub-header">🌡️ Input Weather Parameters</h3>', 
                    unsafe_allow_html=True)
        
        # Weather input sliders
        temperature = st.slider(
            "Ambient Temperature (°C)", 
            min_value=0.0, 
            max_value=50.0, 
            value=25.0,
            help="Average ambient temperature"
        )
        
        module_temp = st.slider(
            "Module Temperature (°C)", 
            min_value=0.0, 
            max_value=60.0, 
            value=35.0,
            help="Solar panel surface temperature"
        )
        
        irradiation = st.slider(
            "Solar Irradiation (W/m²)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=500.0,
            help="Solar irradiation intensity"
        )
        
        humidity = st.slider(
            "Humidity (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0,
            help="Relative humidity"
        )
        
        wind_speed = st.slider(
            "Wind Speed (m/s)", 
            min_value=0.0, 
            max_value=20.0, 
            value=5.0,
            help="Wind speed"
        )
        
        # Time inputs
        st.markdown("**⏰ Time Parameters**")
        hour = st.slider("Hour of Day", 0, 23, 12)
        month = st.selectbox("Month", range(1, 13), index=5)
        day_of_year = st.slider("Day of Year", 1, 365, 150)
    
    # Column 2: Prediction Results
    with col2:
        st.markdown('<h3 class="sub-header">🔮 Prediction Results</h3>', 
                    unsafe_allow_html=True)
        
        # Prepare input data
        temp_irradiation = temperature * irradiation
        is_daytime = 1 if 6 <= hour <= 18 else 0
        
        input_data = pd.DataFrame({
            'AMBIENT_TEMPERATURE': [temperature],
            'MODULE_TEMPERATURE': [module_temp],
            'IRRADIATION': [irradiation],
            'HUMIDITY': [humidity],
            'WIND_SPEED': [wind_speed],
            'HOUR': [hour],
            'MONTH': [month],
            'DAY_OF_YEAR': [day_of_year],
            'TEMP_IRRADIATION': [temp_irradiation],
            'IS_DAYTIME': [is_daytime]
        })
        
        # Make prediction
        if st.button("🚀 Predict Power Output", type="primary"):
            prediction = model.predict(input_data)[0]
            
            st.markdown(f'''
            <div class="prediction-result">
                ⚡ {prediction:.2f} kW
            </div>
            ''', unsafe_allow_html=True)
            
            # Additional insights
            if prediction > 30:
                st.success("High power output expected! ☀️")
            elif prediction > 15:
                st.info("Moderate power output expected 🌤️")
            else:
                st.warning("Low power output expected ⛅")
        
        # Model Performance Metrics
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("**📊 Model Performance**")
        st.metric("RMSE", f"{rmse:.2f} kW")
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("Accuracy", f"{r2*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SDG Impact
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**🌍 SDG 7 Impact**")
        st.markdown("This system supports **Affordable and Clean Energy** by optimizing solar power planning and reducing fossil fuel dependency.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 3: Visualizations
    with col3:
        st.markdown('<h3 class="sub-header">📈 Visualizations & Insights</h3>', 
                    unsafe_allow_html=True)
        
        # Tab layout for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Performance", "🔍 Feature Importance", "📈 Historical Data"])
        
        with tab1:
            # Prediction vs Actual scatter plot
            fig_scatter = px.scatter(
                x=y_test, 
                y=y_pred,
                labels={'x': 'Actual Power (kW)', 'y': 'Predicted Power (kW)'},
                title="Predicted vs Actual Power Output"
            )
            fig_scatter.add_shape(
                type="line",
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max(),
                line=dict(color="red", dash="dash")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Residuals histogram
            residuals = y_test - y_pred
            fig_residuals = px.histogram(
                residuals,
                nbins=30,
                title="Prediction Residuals Distribution",
                labels={'value': 'Residuals (kW)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        with tab2:
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Solar Power Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature correlation heatmap
            correlation_matrix = df[features + ['DC_POWER']].corr()
            fig_heatmap = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab3:
            # Historical power output time series
            fig_timeseries = px.line(
                df.head(500),
                x='DATE_TIME',
                y='DC_POWER',
                title="Historical Solar Power Output"
            )
            st.plotly_chart(fig_timeseries, use_container_width=True)
            
            # Power output by hour of day
            hourly_avg = df.groupby('HOUR')['DC_POWER'].mean().reset_index()
            fig_hourly = px.bar(
                hourly_avg,
                x='HOUR',
                y='DC_POWER',
                title="Average Power Output by Hour of Day"
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Solar Energy Production Prediction System | Supporting SDG 7: Affordable and Clean Energy</p>
        <p>Built with Streamlit | Data Science for Sustainable Development</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()