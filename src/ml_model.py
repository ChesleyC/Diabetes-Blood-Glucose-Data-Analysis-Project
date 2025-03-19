import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class GlucoseRecoveryPredictor:
    """
    Machine learning model for predicting blood glucose recovery time
    """

    def __init__(self, model_type="linear"):
        """
        Initialize the predictor

        Parameters:
            model_type (str): Model type, "linear" or "ridge"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None

    def _prepare_features(self, meal_events_df):
        """
        Prepare model features

        Parameters:
            meal_events_df (pandas.DataFrame): Meal events data

        Returns:
            tuple: (X, y, feature_names)
        """
        # Exclude samples with missing target variable
        df = meal_events_df.dropna(subset=['recovery_time_min'])

        if df.empty:
            print("No valid training samples (all samples are missing recovery time)")
            return pd.DataFrame(), pd.Series(), []

        # Define basic features
        base_features = [
            'pre_meal_cgm',  # Pre-meal glucose value
            'peak_cgm',      # Peak glucose value
            'rise_time_min', # Time to peak
            'rise_rate_mg_dl_min'  # Rise rate
        ]

        # Find features that actually exist in the data
        available_features = []
        for feature in base_features:
            if feature in df.columns:
                available_features.append(feature)

        if not available_features:
            print("Could not find valid predictive features")
            return pd.DataFrame(), pd.Series(), []

        # Process meal type (categorical variable)
        if 'meal_type' in df.columns and df['meal_type'].notna().any():
            # Create one-hot encoding for meal types
            meal_dummies = pd.get_dummies(df['meal_type'], prefix='meal')
            df = pd.concat([df, meal_dummies], axis=1)

            # Add meal type features
            meal_features = meal_dummies.columns.tolist()
            available_features.extend(meal_features)

        # Filter valid features (remove features with too many missing values)
        valid_features = []
        for feature in available_features:
            if feature in df.columns and df[feature].notna().sum() / len(df) >= 0.5:  # At least 50% non-missing
                valid_features.append(feature)

        if not valid_features:
            print("No features with sufficient non-missing values")
            return pd.DataFrame(), pd.Series(), []

        # For features with few missing values, perform simple imputation
        for feature in valid_features:
            if df[feature].isna().any():
                df[feature] = df[feature].fillna(df[feature].median())

        # Extract features and target variable
        X = df[valid_features]
        y = df['recovery_time_min']

        return X, y, valid_features

    def train(self, meal_events_df):
        """
        Train the model

        Parameters:
            meal_events_df (pandas.DataFrame): Meal events data

        Returns:
            self: Trained model
        """
        X, y, self.feature_names = self._prepare_features(meal_events_df)

        if X.empty or len(X) < 5:
            print("Insufficient training data")
            return self

        # Create and train the model
        if self.model_type == "linear":
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        elif self.model_type == "ridge":
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=1.0))
            ])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        try:
            self.model.fit(X, y)
            print(f"Successfully trained model using {len(self.feature_names)} features and {len(X)} samples")
        except Exception as e:
            print(f"Model training failed: {str(e)}")
            self.model = None

        return self

    def predict(self, meal_events_df):
        """
        Make predictions using the trained model

        Parameters:
            meal_events_df (pandas.DataFrame): Meal events data

        Returns:
            numpy.ndarray: Predicted recovery times
        """
        if self.model is None or not self.feature_names:
            print("Model has not been trained yet")
            return None

        # Prepare features
        df = meal_events_df.copy()

        # Process meal type (categorical variable)
        if 'meal_type' in df.columns and any(f.startswith('meal_') for f in self.feature_names):
            try:
                meal_dummies = pd.get_dummies(df['meal_type'], prefix='meal')
                df = pd.concat([df, meal_dummies], axis=1)
            except:
                print("Error processing meal types")
                for f in self.feature_names:
                    if f.startswith('meal_') and f not in df.columns:
                        df[f] = 0

        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Fill missing features with 0

        # Fill missing values
        X = df[self.feature_names].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)

        # Make predictions
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance

        Parameters:
            y_true (numpy.ndarray): True values
            y_pred (numpy.ndarray): Predicted values

        Returns:
            dict: Performance metrics
        """
        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            print("Not enough data for evaluation")
            return {}

        metrics = {}
        try:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        except Exception as e:
            print(f"Error calculating evaluation metrics: {str(e)}")
            return {}

        return metrics

    def plot_feature_importance(self, figsize=(10, 6)):
        """
        Plot feature importance

        Parameters:
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.model is None or self.feature_names is None:
            print("Model has not been trained yet")
            return None

        try:
            # Get feature coefficients
            coefs = self.model.named_steps['regressor'].coef_

            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': np.abs(coefs)
            })

            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)

            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_title('Feature Importance')
            ax.set_xlabel('Importance (Absolute Coefficient)')
            ax.set_ylabel('Feature')

            return fig
        except Exception as e:
            print(f"Error plotting feature importance: {str(e)}")
            return None

    def plot_prediction_vs_actual(self, y_true, y_pred, figsize=(8, 8)):
        """
        Plot predicted vs actual values

        Parameters:
            y_true (numpy.ndarray): True values
            y_pred (numpy.ndarray): Predicted values
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            print("Not enough data for plotting")
            return None

        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create scatter plot
            ax.scatter(y_true, y_pred, alpha=0.7)
            
            # Add perfect prediction line
            max_val = max(max(y_true), max(y_pred))
            min_val = min(min(y_true), min(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            # Set labels and title
            ax.set_xlabel('Actual Recovery Time (min)')
            ax.set_ylabel('Predicted Recovery Time (min)')
            ax.set_title('Predicted vs Actual Recovery Time')
            
            # Add metrics as text
            metrics = self.evaluate(y_true, y_pred)
            metrics_text = f"MAE: {metrics.get('mae', 'N/A'):.2f} min\n" \
                          f"RMSE: {metrics.get('rmse', 'N/A'):.2f} min\n" \
                          f"R²: {metrics.get('r2', 'N/A'):.2f}"
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend()
            plt.grid(True, alpha=0.3)
            
            return fig
        except Exception as e:
            print(f"Error plotting prediction vs actual: {str(e)}")
            return None


def leave_one_out_cross_validation(meal_events_df, model_type="linear"):
    """
    Perform leave-one-out cross-validation by patient

    Parameters:
        meal_events_df (pandas.DataFrame): Meal events data
        model_type (str): Model type, "linear" or "ridge"

    Returns:
        dict: Cross-validation results
    """
    if meal_events_df.empty or 'patient_id' not in meal_events_df.columns:
        print("Cannot perform cross-validation: Missing patient ID column")
        return None

    # Ensure target variable exists
    if 'recovery_time_min' not in meal_events_df.columns:
        print("Cannot perform cross-validation: Missing target variable 'recovery_time_min'")
        return None

    # Get unique patient IDs
    patient_ids = meal_events_df['patient_id'].unique()
    
    if len(patient_ids) < 2:
        print("Cannot perform cross-validation: Need at least 2 patients")
        return None

    print(f"Performing leave-one-patient-out cross-validation with {len(patient_ids)} patients")

    # Initialize results
    results = {
        'patient_id': [],
        'mae': [],
        'rmse': [],
        'r2': [],
        'sample_count': []
    }
    
    all_true = []
    all_pred = []

    # For each patient
    for patient_id in patient_ids:
        # Split data
        test_data = meal_events_df[meal_events_df['patient_id'] == patient_id]
        train_data = meal_events_df[meal_events_df['patient_id'] != patient_id]
        
        if test_data.empty or train_data.empty:
            print(f"Skipping patient {patient_id}: No valid test or train data")
            continue
            
        # Check if there's sufficient test data with valid target
        test_with_target = test_data.dropna(subset=['recovery_time_min'])
        if test_with_target.empty:
            print(f"Skipping patient {patient_id}: No test samples with valid target")
            continue
            
        # Train model
        predictor = GlucoseRecoveryPredictor(model_type=model_type)
        predictor.train(train_data)
        
        if predictor.model is None:
            print(f"Skipping patient {patient_id}: Model training failed")
            continue
            
        # Make predictions
        y_pred = predictor.predict(test_with_target)
        y_true = test_with_target['recovery_time_min'].values
        
        if y_pred is None:
            print(f"Skipping patient {patient_id}: Prediction failed")
            continue
            
        # Evaluate
        metrics = predictor.evaluate(y_true, y_pred)
        
        if not metrics:
            print(f"Skipping patient {patient_id}: Evaluation failed")
            continue
            
        # Store results
        results['patient_id'].append(patient_id)
        results['mae'].append(metrics['mae'])
        results['rmse'].append(metrics['rmse'])
        results['r2'].append(metrics['r2'])
        results['sample_count'].append(len(y_true))
        
        # Collect all predictions for overall evaluation
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    
    # Calculate overall performance
    if all_true and all_pred:
        predictor = GlucoseRecoveryPredictor(model_type=model_type)
        overall_metrics = predictor.evaluate(all_true, all_pred)
        results['overall'] = overall_metrics
    
    return results


def train_final_model(meal_events_df, model_type="linear"):
    """
    Train the final model on all data

    Parameters:
        meal_events_df (pandas.DataFrame): Meal events data
        model_type (str): Model type, "linear" or "ridge"

    Returns:
        GlucoseRecoveryPredictor: Trained model
    """
    if meal_events_df.empty:
        print("Cannot train final model: Empty data")
        return None
        
    # Ensure target variable exists
    if 'recovery_time_min' not in meal_events_df.columns:
        print("Cannot train final model: Missing target variable 'recovery_time_min'")
        return None
        
    # Check if there's sufficient data with valid target
    valid_data = meal_events_df.dropna(subset=['recovery_time_min'])
    if len(valid_data) < 10:
        print(f"Cannot train final model: Insufficient valid samples (found {len(valid_data)})")
        return None
        
    print(f"Training final model on {len(valid_data)} samples")
    
    # Train model
    predictor = GlucoseRecoveryPredictor(model_type=model_type)
    predictor.train(valid_data)
    
    return predictor