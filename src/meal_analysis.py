import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class MealAnalyzer:
    """
    Class for analyzing post-meal glucose responses
    """

    def __init__(self):
        """
        Initialize the analyzer
        """
        # Define normal blood glucose range (unit: mg/dL)
        self.normal_range = (70, 140)

        # Define blood glucose spike threshold (percent increase relative to pre-meal baseline)
        self.spike_threshold_percent = 30

        # Define minimum spike value (mg/dL)
        self.min_spike_value = 30

        # Possible dietary column names
        self.possible_dietary_columns = [
            'Dietary intake',
            'Diet',
            'Meal',
            'Dietary intake - Diet',
            'Food intake'
        ]

    def _find_dietary_column(self, df):
        """
        Find the dietary column in the data

        Parameters:
            df (pandas.DataFrame): Data table

        Returns:
            str: Found dietary column name, or None if not found
        """
        for col in self.possible_dietary_columns:
            if col in df.columns:
                return col

        # Try to find columns containing dietary or meal related terms
        for col in df.columns:
            col_lower = str(col).lower()
            if 'dietary' in col_lower or 'diet' in col_lower or 'meal' in col_lower:
                return col

        return None

    def analyze_meal_response(self, patient_df, meal_index, window_hours=4):
        """
        Analyze a single post-meal glucose response

        Parameters:
            patient_df (pandas.DataFrame): Patient data
            meal_index (int): Meal index
            window_hours (int): Post-meal observation window in hours

        Returns:
            dict: Post-meal response metrics
        """
        # Ensure Date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(patient_df['Date']):
            patient_df['Date'] = pd.to_datetime(patient_df['Date'])

        # Find dietary column
        dietary_col = self._find_dietary_column(patient_df)
        if dietary_col is None:
            print("Warning: No dietary data column found")
            return None

        # Find meal records
        meal_rows = patient_df[dietary_col].notna()
        if not meal_rows.any():
            print("No meal records found")
            return None

        meal_indices = patient_df[meal_rows].index.tolist()
        if meal_index >= len(meal_indices):
            print(f"Meal index out of range, there are {len(meal_indices)} meals")
            return None

        idx = meal_indices[meal_index]
        meal_time = patient_df.loc[idx, 'Date']
        meal_type = patient_df.loc[idx, 'meal_type'] if 'meal_type' in patient_df.columns and pd.notna(
            patient_df.loc[idx, 'meal_type']) else 'unknown'
        meal_content = patient_df.loc[idx, dietary_col]

        # Select data from 30 minutes before meal to window_hours after meal
        start_time = meal_time - timedelta(minutes=30)
        end_time = meal_time + timedelta(hours=window_hours)
        window_data = patient_df[(patient_df['Date'] >= start_time) & (patient_df['Date'] <= end_time)].copy()

        if window_data.empty or 'CGM (mg / dl)' not in window_data.columns:
            print("No CGM data in the observation window")
            return None

        # Calculate relative time (minutes)
        window_data['minutes_from_meal'] = (window_data['Date'] - meal_time).dt.total_seconds() / 60

        # Extract pre-meal glucose value
        pre_meal_data = window_data[window_data['minutes_from_meal'] <= 0]
        if pre_meal_data.empty:
            pre_meal_glucose = np.nan
        else:
            pre_meal_glucose = pre_meal_data['CGM (mg / dl)'].iloc[-1]  # Use the value closest to the meal time

        # Extract post-meal data
        post_meal_data = window_data[window_data['minutes_from_meal'] > 0]
        if post_meal_data.empty:
            print("No post-meal data")
            return None

        # Calculate peak and peak time
        peak_idx = post_meal_data['CGM (mg / dl)'].idxmax()
        peak_glucose = post_meal_data.loc[peak_idx, 'CGM (mg / dl)']
        peak_time_min = post_meal_data.loc[peak_idx, 'minutes_from_meal']

        # Determine if there is a significant spike
        is_spike = False
        if not np.isnan(pre_meal_glucose) and pre_meal_glucose > 0:
            increase_percent = ((peak_glucose - pre_meal_glucose) / pre_meal_glucose) * 100
            increase_absolute = peak_glucose - pre_meal_glucose
            is_spike = (increase_percent >= self.spike_threshold_percent) and (
                        increase_absolute >= self.min_spike_value)

        # Calculate rise rate (mg/dL/min)
        if not np.isnan(pre_meal_glucose) and peak_time_min > 0:
            rise_rate = (peak_glucose - pre_meal_glucose) / peak_time_min
        else:
            rise_rate = np.nan

        # Calculate decline rate (mg/dL/min)
        post_peak_data = post_meal_data[post_meal_data['minutes_from_meal'] > peak_time_min]
        if not post_peak_data.empty:
            # Find the lowest point after the peak
            lowest_idx = post_peak_data['CGM (mg / dl)'].idxmin()
            lowest_glucose = post_peak_data.loc[lowest_idx, 'CGM (mg / dl)']
            lowest_time_min = post_peak_data.loc[lowest_idx, 'minutes_from_meal']

            # Calculate decline rate
            if lowest_time_min > peak_time_min:
                decline_rate = (peak_glucose - lowest_glucose) / (lowest_time_min - peak_time_min)
            else:
                decline_rate = np.nan
        else:
            lowest_glucose = np.nan
            lowest_time_min = np.nan
            decline_rate = np.nan

        # Calculate recovery time (minutes)
        # Defined as the time when glucose returns to the upper limit of the normal range
        if peak_glucose > self.normal_range[1]:
            recovery_data = post_meal_data[(post_meal_data['minutes_from_meal'] > peak_time_min) &
                                           (post_meal_data['CGM (mg / dl)'] <= self.normal_range[1])]
            if not recovery_data.empty:
                recovery_idx = recovery_data.index[0]
                recovery_time_min = recovery_data.loc[recovery_idx, 'minutes_from_meal']
                recovery_time_from_peak = recovery_time_min - peak_time_min
            else:
                recovery_time_min = np.nan
                recovery_time_from_peak = np.nan
        else:
            recovery_time_min = 0  # Did not exceed normal range
            recovery_time_from_peak = 0

        # Calculate area under the curve (AUC)
        # Simplified as the integral of time and glucose
        if not post_meal_data.empty:
            auc = np.trapz(post_meal_data['CGM (mg / dl)'].values, post_meal_data['minutes_from_meal'].values)
        else:
            auc = np.nan

        # Summarize results
        results = {
            'meal_time': meal_time,
            'meal_type': meal_type,
            'meal_content': meal_content,
            'pre_meal_glucose': pre_meal_glucose,
            'peak_glucose': peak_glucose,
            'peak_time_min': peak_time_min,
            'is_spike': is_spike,
            'rise_rate': rise_rate,
            'lowest_glucose': lowest_glucose,
            'lowest_time_min': lowest_time_min,
            'decline_rate': decline_rate,
            'recovery_time_min': recovery_time_min,
            'recovery_time_from_peak': recovery_time_from_peak,
            'auc': auc
        }

        return results

    def analyze_all_meals(self, patient_df):
        """
        Analyze glucose responses for all meals of a patient

        Parameters:
            patient_df (pandas.DataFrame): Patient data

        Returns:
            pandas.DataFrame: Analysis results for all meals
        """
        # Find dietary column
        dietary_col = self._find_dietary_column(patient_df)
        if dietary_col is None:
            print("Warning: No dietary data column found")
            return pd.DataFrame()

        # Find all meal records
        meal_rows = patient_df[dietary_col].notna()
        if not meal_rows.any():
            print("No meal records found")
            return pd.DataFrame()

        meal_indices = patient_df[meal_rows].index.tolist()
        results = []

        for i, idx in enumerate(meal_indices):
            result = self.analyze_meal_response(patient_df, i)
            if result:
                # Add patient ID
                if 'source_file' in patient_df.columns:
                    result['patient_id'] = patient_df['source_file'].iloc[0].split('_')[0]

                # Add diabetes type
                if 'diabetes_type' in patient_df.columns:
                    result['diabetes_type'] = patient_df['diabetes_type'].iloc[0]

                results.append(result)

        return pd.DataFrame(results)

    def compare_meal_types(self, meal_results_df):
        """
        Compare glucose responses across different meal types

        Parameters:
            meal_results_df (pandas.DataFrame): Analysis results for all meals

        Returns:
            pandas.DataFrame: Average metrics for each meal type
        """
        if meal_results_df.empty or 'meal_type' not in meal_results_df.columns:
            print("No meal type data available for comparison")
            return pd.DataFrame()

        # Filter out unknown meal types and ensure there are multiple meal types
        valid_meals = meal_results_df[
            (meal_results_df['meal_type'].notna()) & 
            (meal_results_df['meal_type'] != 'unknown')
        ]
        
        if valid_meals.empty or len(valid_meals['meal_type'].unique()) <= 1:
            print("Insufficient meal type data for comparison")
            return pd.DataFrame()
            
        # Find all numeric columns that might be metrics
        numeric_cols = valid_meals.select_dtypes(include=['number']).columns.tolist()
        
        # Exclude non-metric columns
        exclude_cols = ['patient_id', 'meal_time', 'is_spike']
        available_metrics = [col for col in numeric_cols if col not in exclude_cols 
                            and 'diabetes_type' not in col and 'meal_type' not in col]
        
        if not available_metrics:
            print("No valid metrics found in the data")
            return pd.DataFrame()
        
        # Calculate mean for each metric by meal type and count samples
        summary = valid_meals.groupby('meal_type')[available_metrics].agg(['mean', 'std', 'count'])
        
        # Format for display - keep meal_type as index
        formatted_df = pd.DataFrame(index=summary.index)
        
        # Add all available metrics
        for metric in available_metrics:
            if (metric, 'mean') in summary.columns:
                formatted_df[metric] = summary[(metric, 'mean')].round(6)
        
        # Add count column based on first metric's count
        first_metric = available_metrics[0]
        formatted_df['count'] = summary[(first_metric, 'count')]
        
        return formatted_df

    def compare_diabetes_types(self, meal_events_df):
        """
        Compare glucose responses between different diabetes types

        Parameters:
            meal_events_df (pandas.DataFrame): Analysis results for all meals

        Returns:
            pandas.DataFrame: Comparison summary of diabetes types
        """
        if meal_events_df.empty or 'diabetes_type' not in meal_events_df.columns:
            print("No diabetes type data available for comparison")
            return pd.DataFrame()

        # Filter out rows without valid diabetes type
        valid_data = meal_events_df[meal_events_df['diabetes_type'].notna()]
        
        if valid_data.empty or len(valid_data['diabetes_type'].unique()) <= 1:
            print("Insufficient diabetes type data for comparison")
            return pd.DataFrame()
            
        # Define potential metrics to compare - using all available numeric columns
        # First, let's find all numeric columns that might be metrics
        numeric_cols = valid_data.select_dtypes(include=['number']).columns.tolist()
        
        # Exclude non-metric columns
        exclude_cols = ['patient_id', 'meal_time', 'is_spike']
        potential_metrics = [col for col in numeric_cols if col not in exclude_cols 
                            and 'diabetes_type' not in col and 'meal_type' not in col]
        
        if not potential_metrics:
            print("No valid metrics found in the data")
            return pd.DataFrame()
        
        # Calculate mean for each metric by diabetes type and count samples
        summary = valid_data.groupby('diabetes_type')[potential_metrics].agg(['mean', 'count'])
        
        # Format for display - keep diabetes_type as index
        formatted_df = pd.DataFrame(index=summary.index)
        
        # Add all available metrics
        for metric in potential_metrics:
            if (metric, 'mean') in summary.columns:
                formatted_df[metric] = summary[(metric, 'mean')].round(6)
        
        # Add count column based on first metric's count
        first_metric = potential_metrics[0]
        formatted_df['count'] = summary[(first_metric, 'count')]
        
        # Calculate p-values for comparison between diabetes types if there are two types
        if len(formatted_df) == 2:
            for metric in potential_metrics:
                if metric in valid_data.columns:
                    type1_data = valid_data[valid_data['diabetes_type'] == formatted_df.index[0]][metric].dropna()
                    type2_data = valid_data[valid_data['diabetes_type'] == formatted_df.index[1]][metric].dropna()
                    
                    if len(type1_data) > 0 and len(type2_data) > 0:
                        try:
                            _, p_value = stats.ttest_ind(type1_data, type2_data, equal_var=False)
                            formatted_df[f'{metric}_p_value'] = p_value
                        except:
                            # If t-test fails, skip this metric
                            pass
        
        return formatted_df