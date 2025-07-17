"""
Smart Agriculture System with AI-IoT Integration
Author: Student
Date: July 2025

This module demonstrates a conceptual smart agriculture system that integrates
AI models with IoT sensors for crop yield prediction and farm management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json
from datetime import datetime, timedelta

class SmartAgricultureSystem:
    """
    AI-driven IoT system for smart agriculture and crop yield prediction.
    """
    
    def __init__(self):
        self.sensors = self.define_sensor_requirements()
        self.ai_model = None
        self.sensor_data = None
        
    def define_sensor_requirements(self):
        """
        Define the IoT sensors needed for the smart agriculture system.
        
        Returns:
            dict: Sensor specifications and requirements
        """
        sensors = {
            "environmental_sensors": {
                "soil_moisture": {
                    "type": "Capacitive Soil Moisture Sensor",
                    "range": "0-100%",
                    "accuracy": "±3%",
                    "sampling_rate": "Every 30 minutes",
                    "purpose": "Monitor soil water content for irrigation decisions"
                },
                "soil_temperature": {
                    "type": "DS18B20 Temperature Sensor",
                    "range": "-55°C to 125°C",
                    "accuracy": "±0.5°C",
                    "sampling_rate": "Every 30 minutes",
                    "purpose": "Monitor soil temperature for crop growth optimization"
                },
                "air_temperature": {
                    "type": "DHT22 Temperature/Humidity Sensor",
                    "range": "-40°C to 80°C",
                    "accuracy": "±0.5°C",
                    "sampling_rate": "Every 15 minutes",
                    "purpose": "Monitor ambient temperature for climate control"
                },
                "humidity": {
                    "type": "DHT22 Humidity Sensor",
                    "range": "0-100% RH",
                    "accuracy": "±2-5% RH",
                    "sampling_rate": "Every 15 minutes",
                    "purpose": "Monitor air humidity for disease prevention"
                },
                "light_intensity": {
                    "type": "BH1750 Light Sensor",
                    "range": "0-65535 lux",
                    "accuracy": "±20%",
                    "sampling_rate": "Every 30 minutes",
                    "purpose": "Monitor light conditions for photosynthesis optimization"
                }
            },
            "chemical_sensors": {
                "soil_ph": {
                    "type": "pH Sensor Module",
                    "range": "0-14 pH",
                    "accuracy": "±0.1 pH",
                    "sampling_rate": "Every 2 hours",
                    "purpose": "Monitor soil acidity for nutrient availability"
                },
                "nitrogen_level": {
                    "type": "NPK Sensor",
                    "range": "0-1999 mg/kg",
                    "accuracy": "±2%",
                    "sampling_rate": "Every 4 hours",
                    "purpose": "Monitor nitrogen content for fertilizer management"
                },
                "phosphorus_level": {
                    "type": "NPK Sensor",
                    "range": "0-1999 mg/kg",
                    "accuracy": "±2%",
                    "sampling_rate": "Every 4 hours",
                    "purpose": "Monitor phosphorus content for root development"
                },
                "potassium_level": {
                    "type": "NPK Sensor",
                    "range": "0-1999 mg/kg",
                    "accuracy": "±2%",
                    "sampling_rate": "Every 4 hours",
                    "purpose": "Monitor potassium content for disease resistance"
                }
            },
            "imaging_sensors": {
                "crop_camera": {
                    "type": "Raspberry Pi Camera Module V3",
                    "resolution": "12MP",
                    "features": "Auto-focus, HDR",
                    "sampling_rate": "Every 6 hours",
                    "purpose": "Visual crop health monitoring and growth tracking"
                },
                "multispectral_camera": {
                    "type": "NDVI Camera",
                    "bands": "NIR, Red, Green",
                    "resolution": "5MP",
                    "sampling_rate": "Weekly",
                    "purpose": "Crop health assessment and stress detection"
                }
            },
            "weather_sensors": {
                "rainfall": {
                    "type": "Tipping Bucket Rain Gauge",
                    "resolution": "0.2mm",
                    "accuracy": "±2%",
                    "sampling_rate": "Continuous",
                    "purpose": "Monitor precipitation for irrigation planning"
                },
                "wind_speed": {
                    "type": "Anemometer",
                    "range": "0-70 m/s",
                    "accuracy": "±3%",
                    "sampling_rate": "Every 15 minutes",
                    "purpose": "Monitor wind conditions for pesticide application"
                }
            }
        }
        
        return sensors
    
    def generate_synthetic_sensor_data(self, days=365, crop_type="tomato"):
        """
        Generate synthetic IoT sensor data for demonstration.
        
        Args:
            days (int): Number of days to simulate
            crop_type (str): Type of crop being monitored
            
        Returns:
            pd.DataFrame: Synthetic sensor data
        """
        np.random.seed(42)
        
        # Create date range
        dates = pd.date_range(start='2024-01-01', periods=days*24, freq='H')
        
        # Generate synthetic data based on seasonal patterns
        data = []
        
        for i, date in enumerate(dates):
            # Seasonal variations
            day_of_year = date.dayofyear
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            
            # Daily variations
            hour_factor = np.sin(2 * np.pi * date.hour / 24)
            
            record = {
                'timestamp': date,
                'day_of_year': day_of_year,
                'hour': date.hour,
                
                # Environmental sensors
                'soil_moisture': 45 + 15 * season_factor + 5 * np.random.normal(0, 1),
                'soil_temperature': 20 + 10 * season_factor + 3 * hour_factor + np.random.normal(0, 2),
                'air_temperature': 22 + 12 * season_factor + 5 * hour_factor + np.random.normal(0, 2),
                'humidity': 60 + 20 * season_factor + 10 * np.random.normal(0, 1),
                'light_intensity': max(0, 30000 + 20000 * hour_factor * (date.hour > 6 and date.hour < 18)),
                
                # Chemical sensors
                'soil_ph': 6.5 + 0.5 * np.random.normal(0, 1),
                'nitrogen_level': 80 + 20 * np.random.normal(0, 1),
                'phosphorus_level': 45 + 15 * np.random.normal(0, 1),
                'potassium_level': 120 + 30 * np.random.normal(0, 1),
                
                # Weather sensors
                'rainfall': max(0, 2 * season_factor + np.random.exponential(0.5)),
                'wind_speed': 3 + 2 * np.random.normal(0, 1),
                
                # Growth stage (1-5 scale)
                'growth_stage': min(5, max(1, 1 + 4 * day_of_year / 365)),
                
                # Target variable: yield prediction (kg/hectare)
                'crop_yield': None  # Will be calculated
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Calculate crop yield based on environmental factors
        df['crop_yield'] = self.calculate_yield_target(df, crop_type)
        
        return df
    
    def calculate_yield_target(self, df, crop_type):
        """
        Calculate synthetic crop yield based on environmental factors.
        
        Args:
            df (pd.DataFrame): Sensor data
            crop_type (str): Type of crop
            
        Returns:
            np.array: Calculated yield values
        """
        # Base yield for the crop type
        base_yields = {
            'tomato': 50000,  # kg/hectare
            'corn': 8000,
            'wheat': 3000,
            'rice': 7000
        }
        
        base_yield = base_yields.get(crop_type, 40000)
        
        # Calculate yield based on optimal ranges
        optimal_ranges = {
            'soil_moisture': (50, 70),
            'soil_temperature': (18, 25),
            'air_temperature': (20, 28),
            'humidity': (50, 80),
            'soil_ph': (6.0, 7.0),
            'nitrogen_level': (70, 100),
            'phosphorus_level': (40, 60),
            'potassium_level': (100, 150)
        }
        
        yield_factors = []
        
        for _, row in df.iterrows():
            factor = 1.0
            
            for param, (min_val, max_val) in optimal_ranges.items():
                value = row[param]
                if min_val <= value <= max_val:
                    param_factor = 1.0
                else:
                    # Distance from optimal range
                    if value < min_val:
                        param_factor = max(0.3, value / min_val)
                    else:
                        param_factor = max(0.3, max_val / value)
                
                factor *= param_factor
            
            # Add some random variation
            factor *= (0.8 + 0.4 * np.random.random())
            
            yield_factors.append(factor)
        
        return base_yield * np.array(yield_factors)
    
    def prepare_ai_model_data(self, df):
        """
        Prepare data for AI model training.
        
        Args:
            df (pd.DataFrame): Raw sensor data
            
        Returns:
            tuple: (X, y) for machine learning
        """
        # Feature engineering
        feature_columns = [
            'soil_moisture', 'soil_temperature', 'air_temperature', 'humidity',
            'light_intensity', 'soil_ph', 'nitrogen_level', 'phosphorus_level',
            'potassium_level', 'rainfall', 'wind_speed', 'growth_stage'
        ]
        
        # Aggregate hourly data to daily averages
        daily_data = df.groupby(df['timestamp'].dt.date).agg({
            **{col: 'mean' for col in feature_columns},
            'crop_yield': 'first'
        }).reset_index()
        
        # Create additional features
        daily_data['temperature_range'] = daily_data['air_temperature'] - daily_data['soil_temperature']
        daily_data['moisture_deficit'] = 70 - daily_data['soil_moisture']  # Optimal moisture is ~70%
        daily_data['nutrient_balance'] = (daily_data['nitrogen_level'] + 
                                        daily_data['phosphorus_level'] + 
                                        daily_data['potassium_level']) / 3
        
        X = daily_data.drop(['timestamp', 'crop_yield'], axis=1)
        y = daily_data['crop_yield']
        
        return X, y
    
    def train_yield_prediction_model(self, X, y):
        """
        Train AI model for crop yield prediction.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target yield values
            
        Returns:
            dict: Model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model (suitable for IoT edge deployment)
        self.ai_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.ai_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.ai_model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'feature_importance': dict(zip(X.columns, self.ai_model.feature_importances_))
        }
        
        return metrics, X_test, y_test, y_pred
    
    def create_data_flow_diagram(self):
        """
        Create a conceptual data flow diagram for the smart agriculture system.
        
        Returns:
            dict: Data flow specification
        """
        data_flow = {
            "data_sources": {
                "iot_sensors": {
                    "environmental": ["soil_moisture", "temperature", "humidity", "light"],
                    "chemical": ["ph", "npk_levels"],
                    "imaging": ["crop_photos", "ndvi_images"],
                    "weather": ["rainfall", "wind_speed"]
                }
            },
            "data_processing": {
                "edge_gateway": {
                    "functions": [
                        "Data collection from sensors",
                        "Initial data validation",
                        "Data aggregation",
                        "Local storage backup"
                    ],
                    "technology": "Raspberry Pi 4 with LoRaWAN"
                },
                "data_preprocessing": {
                    "functions": [
                        "Data cleaning and validation",
                        "Feature engineering",
                        "Temporal aggregation",
                        "Anomaly detection"
                    ]
                }
            },
            "ai_processing": {
                "models": {
                    "yield_prediction": {
                        "algorithm": "Random Forest Regressor",
                        "inputs": "Environmental + Chemical + Growth stage data",
                        "output": "Predicted crop yield (kg/hectare)",
                        "update_frequency": "Daily"
                    },
                    "irrigation_optimization": {
                        "algorithm": "Decision Tree",
                        "inputs": "Soil moisture + Weather forecast + Crop stage",
                        "output": "Irrigation recommendations",
                        "update_frequency": "Every 6 hours"
                    },
                    "disease_detection": {
                        "algorithm": "CNN (Convolutional Neural Network)",
                        "inputs": "Crop images + Environmental conditions",
                        "output": "Disease risk assessment",
                        "update_frequency": "Daily"
                    }
                }
            },
            "output_systems": {
                "farmer_dashboard": {
                    "features": [
                        "Real-time sensor readings",
                        "Yield predictions",
                        "Irrigation recommendations",
                        "Alert notifications"
                    ]
                },
                "automated_systems": {
                    "irrigation": "Automatic watering based on AI recommendations",
                    "fertilization": "Nutrient application scheduling",
                    "alerts": "SMS/Email notifications for critical conditions"
                }
            },
            "data_storage": {
                "local": "Edge device storage for recent data",
                "cloud": "Historical data and model training",
                "backup": "Redundant storage for critical data"
            }
        }
        
        return data_flow
    
    def visualize_system_performance(self, metrics, X_test, y_test, y_pred):
        """
        Visualize the AI model performance and sensor data insights.
        
        Args:
            metrics (dict): Model performance metrics
            X_test, y_test, y_pred: Test data and predictions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted Yield
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Yield (kg/hectare)')
        axes[0, 0].set_ylabel('Predicted Yield (kg/hectare)')
        axes[0, 0].set_title(f'Yield Prediction Performance\nR² = {metrics["r2_score"]:.3f}')
        
        # 2. Feature Importance
        importance_df = pd.DataFrame(list(metrics['feature_importance'].items()), 
                                   columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        axes[0, 1].barh(importance_df['Feature'], importance_df['Importance'])
        axes[0, 1].set_title('Feature Importance for Yield Prediction')
        axes[0, 1].set_xlabel('Importance Score')
        
        # 3. Residuals Plot
        residuals = y_test - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Yield')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Plot')
        
        # 4. Correlation Heatmap of Key Features
        key_features = ['soil_moisture', 'soil_temperature', 'soil_ph', 'nitrogen_level']
        correlation_matrix = X_test[key_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('smart_agriculture_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to demonstrate the Smart Agriculture AI-IoT system.
    """
    print("🌾 Smart Agriculture AI-IoT System")
    print("=" * 50)
    
    # Initialize the system
    agri_system = SmartAgricultureSystem()
    
    # Display sensor requirements
    print("📡 IoT Sensor Requirements:")
    print("-" * 30)
    for category, sensors in agri_system.sensors.items():
        print(f"\n{category.upper()}:")
        for sensor_name, specs in sensors.items():
            print(f"  • {sensor_name}: {specs['type']}")
            print(f"    Purpose: {specs['purpose']}")
    
    # Generate synthetic sensor data
    print("\n📊 Generating synthetic sensor data...")
    sensor_data = agri_system.generate_synthetic_sensor_data(days=180, crop_type="tomato")
    print(f"Generated {len(sensor_data)} hourly sensor readings")
    
    # Prepare data for AI model
    print("\n🤖 Preparing data for AI model training...")
    X, y = agri_system.prepare_ai_model_data(sensor_data)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train AI model
    print("\n🚀 Training crop yield prediction model...")
    metrics, X_test, y_test, y_pred = agri_system.train_yield_prediction_model(X, y)
    
    print(f"Model Performance:")
    print(f"  Mean Absolute Error: {metrics['mae']:.2f} kg/hectare")
    print(f"  R² Score: {metrics['r2_score']:.3f}")
    
    # Display top features
    top_features = sorted(metrics['feature_importance'].items(), 
                         key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 Important Features:")
    for feature, importance in top_features:
        print(f"  • {feature}: {importance:.3f}")
    
    # Create data flow diagram
    print("\n🔄 Creating data flow diagram...")
    data_flow = agri_system.create_data_flow_diagram()
    
    # Visualize system performance
    print("\n📈 Generating system performance visualizations...")
    agri_system.visualize_system_performance(metrics, X_test, y_test, y_pred)
    
    # Save data flow diagram as JSON
    with open('smart_agriculture_dataflow.json', 'w') as f:
        json.dump(data_flow, f, indent=2)
    
    print("\n✅ Smart Agriculture AI-IoT system analysis completed!")
    print("📄 Data flow diagram saved as 'smart_agriculture_dataflow.json'")
    print("📊 Performance visualizations saved as 'smart_agriculture_analysis.png'")

if __name__ == "__main__":
    main()
