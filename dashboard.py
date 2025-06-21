import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras import layers
import os
from datetime import datetime, timedelta
import time

# Configure TensorFlow to avoid conflicts with Streamlit
tf.config.experimental.set_memory_growth = True
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Import functions from our model.py
from model import (
    load_data, split_data, make_windows, make_train_test_splits,
    create_model, make_preds, evaluate_preds, mean_absolute_scaled_error
)

# Configure Streamlit page
st.set_page_config(
    page_title="BitPredictor Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #f7931a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">₿ BitPredictor Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Interactive Bitcoin Price Prediction using Machine Learning**")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("🔧 Model Configuration")
window_size = st.sidebar.slider("Window Size (days)", min_value=3, max_value=30, value=7, 
                                help="Number of previous days to use for prediction")
horizon = st.sidebar.slider("Prediction Horizon", min_value=1, max_value=7, value=1,
                           help="Number of days to predict into the future")
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=200, value=50,
                          help="Number of training iterations")

# Load data
@st.cache_data
def load_bitcoin_data():
    """Load and cache Bitcoin data"""
    try:
        timesteps, prices = load_data()
        return timesteps, prices
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Train model
def train_model(window_size, horizon, epochs):
    """Train and cache the model"""
    timesteps, prices = load_bitcoin_data()
    if timesteps is None or prices is None:
        return None, None, None, None, None
    
    # Create windowed data
    full_windows, full_labels = make_windows(prices, window_size=window_size, horizon=horizon)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    
    # Create and train model
    tf.keras.backend.clear_session()  # Clear any previous sessions
    
    # Create model directly to avoid naming conflicts
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(horizon, activation="linear")
    ])
    
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["mae"]
    )
    
    # Train with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Training Progress: Epoch {epoch + 1}/{epochs} - Loss: {logs.get("loss", 0):.4f}')
    
    model.fit(
        x=train_windows,
        y=train_labels,
        epochs=epochs,
        verbose=0,
        batch_size=128,
        validation_data=(test_windows, test_labels),
        callbacks=[StreamlitCallback()]
    )
    
    progress_bar.empty()
    status_text.empty()
    
    return model, train_windows, test_windows, train_labels, test_labels

def main():
    # Load data
    timesteps, prices = load_bitcoin_data()
    
    if timesteps is None or prices is None:
        st.error("Failed to load Bitcoin data. Please check your internet connection.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🤖 Model Training", "📊 Predictions", "📋 Performance"])
    
    with tab1:
        st.header("Bitcoin Price Overview")
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'Date': timesteps,
            'Price': prices
        })
        
        # Display key statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${prices[-1]:,.2f}")
        
        with col2:
            price_change = prices[-1] - prices[-2]
            st.metric("24h Change", f"${price_change:,.2f}", f"{price_change:+.2f}")
        
        with col3:
            st.metric("All-time High", f"${np.max(prices):,.2f}")
        
        with col4:
            st.metric("All-time Low", f"${np.min(prices):,.2f}")
        
        # Interactive price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Price'],
            mode='lines',
            name='Bitcoin Price',
            line=dict(color='#f7931a', width=2)
        ))
        
        fig.update_layout(
            title="Bitcoin Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(df, x='Price', nbins=50, title="Price Distribution")
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Calculate daily returns
            df['Daily_Return'] = df['Price'].pct_change() * 100
            fig_returns = px.histogram(df, x='Daily_Return', nbins=50, title="Daily Returns Distribution (%)")
            fig_returns.update_layout(showlegend=False)
            st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab2:
        st.header("Model Training")
        
        # Check if model with same parameters already exists
        model_key = f"model_w{window_size}_h{horizon}_e{epochs}"
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    model, train_windows, test_windows, train_labels, test_labels = train_model(
                        window_size, horizon, epochs
                    )
                    
                    if model is not None:
                        st.success("✅ Model training completed successfully!")
                        
                        # Store model results in session state
                        st.session_state.model = model
                        st.session_state.test_windows = test_windows
                        st.session_state.test_labels = test_labels
                        st.session_state.timesteps = timesteps
                        st.session_state.prices = prices
                        st.session_state.model_key = model_key
                    else:
                        st.error("❌ Model training failed. Please try again.")
                except Exception as e:
                    st.error(f"❌ Model training failed with error: {str(e)}")
                    st.error("Please try refreshing the page and training again.")
        
        # Display model architecture
        st.subheader("Model Architecture")
        st.code(f"""
Model Configuration:
- Window Size: {window_size} days
- Prediction Horizon: {horizon} day(s)
- Training Epochs: {epochs}

Architecture:
- Dense Layer: 128 units, ReLU activation
- Output Layer: {horizon} unit(s), Linear activation
- Loss Function: Mean Absolute Error (MAE)
- Optimizer: Adam
        """)
    
    with tab3:
        st.header("Model Predictions")
        
        # Add info about multi-step predictions
        if horizon > 1:
            st.info(f"📝 **Note**: This model predicts {horizon} days ahead. The charts below show the first day's prediction for comparison with actual prices. Multi-step predictions are more challenging and typically have higher error rates.")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
        else:
            # Make predictions
            model = st.session_state.model
            test_windows = st.session_state.test_windows
            test_labels = st.session_state.test_labels
            timesteps = st.session_state.timesteps
            prices = st.session_state.prices
            
            # Generate predictions
            predictions = make_preds(model, test_windows)
            
            # Create prediction visualization
            test_dates = timesteps[-len(test_windows):]
            actual_prices = test_labels[:, 0] if len(test_labels.shape) > 1 else test_labels
            
            # Handle multi-step predictions by taking the mean or first prediction
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # For multi-step predictions, use the first prediction or mean
                predictions_to_plot = predictions[:, 0]  # Use first prediction
                horizon_note = f" (showing 1st day of {horizon}-day prediction)"
            else:
                predictions_to_plot = predictions
                horizon_note = ""
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'Predictions vs Actual Prices{horizon_note}', 'Prediction Error'),
                vertical_spacing=0.1
            )
            
            # Add actual vs predicted prices
            fig.add_trace(
                go.Scatter(x=test_dates, y=actual_prices, name='Actual Price', 
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=test_dates, y=predictions_to_plot, name='Predicted Price', 
                          line=dict(color='red', width=2, dash='dash')),
                row=1, col=1
            )
            
            # Add error plot
            error = actual_prices - predictions_to_plot
            fig.add_trace(
                go.Scatter(x=test_dates, y=error, name='Prediction Error', 
                          line=dict(color='green', width=1)),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Error (USD)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions table
            st.subheader("Recent Predictions")
            recent_data = pd.DataFrame({
                'Date': test_dates[-10:],
                'Actual Price': actual_prices[-10:],
                'Predicted Price': predictions_to_plot[-10:],
                'Error': error[-10:],
                'Error %': (error[-10:] / actual_prices[-10:]) * 100
            })
            
            st.dataframe(recent_data.round(2), use_container_width=True)
    
    with tab4:
        st.header("Model Performance Metrics")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
        else:
            # Calculate performance metrics
            model = st.session_state.model
            test_windows = st.session_state.test_windows
            test_labels = st.session_state.test_labels
            
            predictions = make_preds(model, test_windows)
            actual_prices = test_labels[:, 0] if len(test_labels.shape) > 1 else test_labels
            
            # Handle multi-step predictions for evaluation
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predictions_for_eval = predictions[:, 0]  # Use first prediction for evaluation
            else:
                predictions_for_eval = predictions
            
            # Evaluate predictions
            results = evaluate_preds(y_true=actual_prices, y_pred=predictions_for_eval)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"${results['mae']:.2f}")
                st.metric("Root Mean Squared Error (RMSE)", f"${results['rmse']:.2f}")
            
            with col2:
                st.metric("Mean Squared Error (MSE)", f"{results['mse']:,.0f}")
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{results['mape']:.2f}%")
            
            with col3:
                st.metric("Mean Absolute Scaled Error (MASE)", f"{results['mase']:.4f}")
                accuracy = max(0, 100 - results['mape'])
                st.metric("Model Accuracy", f"{accuracy:.1f}%")
            
            # Performance interpretation
            st.subheader("Performance Interpretation")
            
            if results['mase'] < 1.0:
                st.success("🎉 **Excellent!** The model performs better than a naive forecast.")
            elif results['mase'] < 1.5:
                st.warning("⚠️ **Good** performance, but there's room for improvement.")
            else:
                st.error("❌ **Poor** performance. Consider adjusting model parameters.")
            
            # Error analysis
            st.subheader("Error Analysis")
            
            error = actual_prices - predictions_for_eval
            error_df = pd.DataFrame({
                'Absolute Error': np.abs(error),
                'Squared Error': error ** 2,
                'Percentage Error': (error / actual_prices) * 100
            })
            
            fig_error = px.box(error_df, title="Error Distribution Analysis")
            st.plotly_chart(fig_error, use_container_width=True)

if __name__ == "__main__":
    main() 