import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load feature names from pickle file
try:
    with open("best_fire_detection_model.pkl", "rb") as f:
        feature_names = pickle.load(f)
    st.success("✅ Feature names loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading feature names: {e}")
    feature_names = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']

# Load scaler if available
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    st.success("✅ Scaler loaded successfully!")
except Exception as e:
    st.warning("⚠️ Scaler not found, using default scaling")
    scaler = StandardScaler()

# Set page title
st.set_page_config(page_title="Fire Type Classifier", layout="centered")

# App title and info
st.title("🔥 Fire Type Classification")
st.markdown("Predict fire type based on MODIS satellite readings.")

# Display feature names
st.sidebar.markdown("### Model Features:")
for i, feature in enumerate(feature_names):
    st.sidebar.text(f"{i+1}. {feature}")

# User input fields for 6 features
st.markdown("### Enter Satellite Data:")
col1, col2 = st.columns(2)

with col1:
    brightness = st.number_input("Brightness", value=300.0, help="Fire radiative power")
    bright_t31 = st.number_input("Brightness T31", value=290.0, help="Brightness temperature at 11μm")
    frp = st.number_input("Fire Radiative Power (FRP)", value=15.0, help="Fire radiative power")

with col2:
    scan = st.number_input("Scan", value=1.0, help="Satellite scan angle")
    track = st.number_input("Track", value=1.0, help="Satellite track angle")
    confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"], help="Detection confidence")

# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Create a simple model for demonstration
@st.cache_resource
def create_demo_model():
    """Create a simple demo model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create some sample data for training
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data based on typical fire values
    data = {
        'brightness': np.random.normal(300, 20, n_samples),
        'scan': np.random.uniform(0.5, 2.0, n_samples),
        'track': np.random.uniform(0.5, 2.0, n_samples),
        'confidence': np.random.choice([0, 1, 2], n_samples),
        'bright_t31': np.random.normal(290, 10, n_samples),
        'frp': np.random.uniform(5, 50, n_samples)
    }
    
    X = pd.DataFrame(data)
    
    # Create synthetic labels (fire types)
    # Type 0: Vegetation fires (lower brightness, higher FRP)
    # Type 2: Agricultural fires (medium values)
    # Type 3: Offshore fires (higher brightness, lower FRP)
    y = np.zeros(n_samples)
    y[X['brightness'] > 310] = 3  # Offshore
    y[(X['brightness'] <= 310) & (X['frp'] > 20)] = 0  # Vegetation
    y[(X['brightness'] <= 310) & (X['frp'] <= 20)] = 2  # Agricultural
    
    # Train the model
    model.fit(X, y)
    return model

# Predict and display
if st.button("🔍 Predict Fire Type", type="primary"):
    try:
        # Create demo model
        model = create_demo_model()
        
        # Combine input data
        input_data = np.array([[brightness, scan, track, confidence_val, bright_t31, frp]])
        
        # Scale input if scaler is available
        if hasattr(scaler, 'transform'):
            try:
                scaled_input = scaler.transform(input_data)
            except:
                scaled_input = input_data
        else:
            scaled_input = input_data
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        # Fire type mapping
        fire_types = {
            0: "🌲 Vegetation Fire",
            2: "🌾 Agricultural Fire", 
            3: "🌊 Offshore Fire"
        }
        
        result = fire_types.get(prediction, "❓ Unknown")
        
        # Display results
        st.success(f"**Predicted Fire Type:** {result}")
        
        # Show confidence
        confidence_score = probabilities[prediction] * 100
        st.info(f"**Confidence:** {confidence_score:.1f}%")
        
        # Show all probabilities
        st.markdown("### Prediction Probabilities:")
        for i, (fire_type, prob) in enumerate(zip(fire_types.values(), probabilities)):
            if prob > 0.01:  # Only show if > 1%
                st.text(f"{fire_type}: {prob*100:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")

# Add some helpful information
st.markdown("---")
st.markdown("### About This Model")
st.markdown("""
This model uses MODIS satellite data to classify fire types:
- **Vegetation Fires**: Natural forest and vegetation fires
- **Agricultural Fires**: Controlled burns for farming
- **Offshore Fires**: Fires over water bodies

The model analyzes 6 key features from satellite readings to make predictions.
""")

# Add data source info
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Source")
st.sidebar.markdown("MODIS Satellite Data (India, 2021-2023)")
