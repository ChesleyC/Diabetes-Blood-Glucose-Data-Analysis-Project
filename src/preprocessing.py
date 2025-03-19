import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataPreprocessor:
    """
    Class for preprocessing diabetes data
    """

    def __init__(self):
        """
        Initialize the preprocessor
        """
        # Define normal blood glucose range (unit: mg/dL)
        self.normal_range = (70, 140)  # Normal fasting blood glucose range

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

    def clean_glucose_data(self, df):
        """
        Clean glucose data, handling outliers and missing values

        Parameters:
            df (pandas.DataFrame): Original data table

        Returns:
            pandas.DataFrame: Cleaned data table
        """
        # Copy data to avoid modifying the original data
        df_clean = df.copy()

        # Identify CGM columns
        cgm_cols = [col for col in df_clean.columns if 'CGM' in str(col) or 'cgm' in str(col).lower()]
        if not cgm_cols and 'CGM (mg / dl)' not in df_clean.columns:
            # Try to create standard column name
            if 'CGM' in df_clean.columns:
                df_clean['CGM (mg / dl)'] = pd.to_numeric(df_clean['CGM'], errors='coerce')
                cgm_cols.append('CGM (mg / dl)')

        # Handle outliers and missing values in CGM columns
        for cgm_col in cgm_cols:
            # Convert non-numeric values to NaN
            df_clean[cgm_col] = pd.to_numeric(df_clean[cgm_col], errors='coerce')

            # Identify and handle outliers (e.g., values less than 20 or greater than 600)
            mask = (df_clean[cgm_col] < 20) | (df_clean[cgm_col] > 600)
            if mask.any():
                print(f"Found {mask.sum()} CGM outliers")
                df_clean.loc[mask, cgm_col] = np.nan

            # Use forward fill to handle short-term missing values
            df_clean[cgm_col] = df_clean[cgm_col].ffill(limit=2)

        # Identify CBG columns
        cbg_cols = [col for col in df_clean.columns if 'CBG' in str(col) or 'cbg' in str(col).lower()]
        if not cbg_cols and 'CBG (mg / dl)' not in df_clean.columns:
            if 'CBG' in df_clean.columns:
                df_clean['CBG (mg / dl)'] = pd.to_numeric(df_clean['CBG'], errors='coerce')
                cbg_cols.append('CBG (mg / dl)')

        # Handle outliers and missing values in CBG columns
        for cbg_col in cbg_cols:
            # Convert non-numeric values to NaN
            df_clean[cbg_col] = pd.to_numeric(df_clean[cbg_col], errors='coerce')

            # Identify and handle outliers
            mask = (df_clean[cbg_col] < 20) | (df_clean[cbg_col] > 600)
            if mask.any():
                print(f"Found {mask.sum()} CBG outliers")
                df_clean.loc[mask, cbg_col] = np.nan

        return df_clean

    def extract_meal_events(self, df, window_hours=4):
        """
        Extract post-meal glucose events

        Parameters:
            df (pandas.DataFrame): Data table
            window_hours (int): Number of hours in post-meal observation window

        Returns:
            pandas.DataFrame: Post-meal event data table
        """
        # Ensure data is sorted by time
        if 'Date' in df.columns:
            df = df.sort_values('Date')

        meal_events = []

        # Find dietary column
        dietary_col = self._find_dietary_column(df)
        if dietary_col is None:
            print("Warning: No dietary data column found")
            return pd.DataFrame()

        # Look for meal records
        meal_rows = df[dietary_col].notna()

        if not meal_rows.any():
            print("No meal records found")
            return pd.DataFrame()

        # Ensure there is a CGM column
        cgm_cols = [col for col in df.columns if 'CGM' in str(col) or 'cgm' in str(col).lower()]
        if not cgm_cols:
            print("No CGM data column found")
            return pd.DataFrame()

        cgm_col = cgm_cols[0]  # Use the first CGM column found

        for idx in df[meal_rows].index:
            meal_time = df.loc[idx, 'Date']
            meal_type = df.loc[idx, 'meal_type'] if 'meal_type' in df.columns and pd.notna(
                df.loc[idx, 'meal_type']) else 'unknown'
            meal_content = df.loc[idx, dietary_col]

            # Extract pre-meal glucose value (if available)
            pre_meal_mask = (df['Date'] >= meal_time - timedelta(minutes=30)) & (df['Date'] < meal_time)
            pre_meal_cgm = df.loc[pre_meal_mask, cgm_col].mean()

            # Extract blood glucose records within post-meal time window
            post_meal_mask = (df['Date'] > meal_time) & (df['Date'] <= meal_time + timedelta(hours=window_hours))
            post_meal_data = df.loc[post_meal_mask]

            if not post_meal_data.empty and cgm_col in post_meal_data.columns:
                # Calculate post-meal glucose peak
                peak_cgm = post_meal_data[cgm_col].max()
                peak_time = post_meal_data.loc[post_meal_data[cgm_col].idxmax(), 'Date'] if not post_meal_data[
                    cgm_col].isna().all() else np.nan

                # Calculate glucose rise rate (mg/dL/min)
                if not np.isnan(pre_meal_cgm) and not np.isnan(peak_cgm) and not pd.isna(peak_time):
                    rise_time = (peak_time - meal_time).total_seconds() / 60  # Rise time (minutes)
                    if rise_time > 0:
                        rise_rate = (peak_cgm - pre_meal_cgm) / rise_time
                    else:
                        rise_rate = np.nan
                else:
                    rise_rate = np.nan
                    rise_time = np.nan

                # Calculate time to return to normal range
                norm_mask = post_meal_data[cgm_col] <= self.normal_range[1]
                if norm_mask.any():
                    first_normal = post_meal_data[norm_mask]['Date'].min()
                    recovery_time = (first_normal - meal_time).total_seconds() / 60  # Recovery time (minutes)
                else:
                    recovery_time = np.nan

                # Create event record
                event = {
                    'patient_id': df['source_file'].iloc[0].split('_')[0] if 'source_file' in df.columns else 'unknown',
                    'meal_time': meal_time,
                    'meal_type': meal_type,
                    'meal_content': meal_content,
                    'pre_meal_cgm': pre_meal_cgm,
                    'peak_cgm': peak_cgm,
                    'peak_time': peak_time if not pd.isna(peak_time) else np.nan,
                    'rise_time_min': rise_time,
                    'rise_rate_mg_dl_min': rise_rate,
                    'recovery_time_min': recovery_time,
                    'diabetes_type': df['diabetes_type'].iloc[0] if 'diabetes_type' in df.columns else None
                }

                meal_events.append(event)

        return pd.DataFrame(meal_events)

    def calculate_glucose_variability(self, df):
        """
        Calculate glucose variability metrics

        Parameters:
            df (pandas.DataFrame): Data table

        Returns:
            dict: Glucose variability metrics
        """
        metrics = {}

        # Find CGM column
        cgm_cols = [col for col in df.columns if 'CGM' in str(col) or 'cgm' in str(col).lower()]
        if not cgm_cols:
            print("Warning: No CGM data column found")
            return metrics

        cgm_col = cgm_cols[0]  # Use the first CGM column found

        cgm_values = df[cgm_col].dropna()

        if len(cgm_values) > 0:
            # Calculate mean and standard deviation
            metrics['mean_glucose'] = cgm_values.mean()
            metrics['std_glucose'] = cgm_values.std()

            # Calculate coefficient of variation (CV)
            if metrics['mean_glucose'] > 0:
                metrics['cv_glucose'] = (metrics['std_glucose'] / metrics['mean_glucose']) * 100
            else:
                metrics['cv_glucose'] = np.nan

            # Calculate glucose value range
            metrics['min_glucose'] = cgm_values.min()
            metrics['max_glucose'] = cgm_values.max()
            metrics['range_glucose'] = metrics['max_glucose'] - metrics['min_glucose']

            # Calculate the proportion of time in hyperglycemia and hypoglycemia
            time_high = (cgm_values > self.normal_range[1]).mean() * 100
            time_low = (cgm_values < self.normal_range[0]).mean() * 100
            time_normal = 100 - time_high - time_low

            metrics['percent_time_high'] = time_high
            metrics['percent_time_low'] = time_low
            metrics['percent_time_in_range'] = time_normal

        return metrics

    def segment_by_time_of_day(self, df):
        """
        Segment data by time of day

        Parameters:
            df (pandas.DataFrame): Data table

        Returns:
            dict: Dictionary with time of day segments
        """
        segments = {}

        if 'Date' not in df.columns:
            print("Warning: No date column found, cannot segment by time of day")
            return segments

        # Morning (6am-11am)
        morning_mask = df['Date'].dt.hour.between(6, 10)
        segments['morning'] = df[morning_mask].copy()

        # Afternoon (11am-5pm)
        afternoon_mask = df['Date'].dt.hour.between(11, 16)
        segments['afternoon'] = df[afternoon_mask].copy()

        # Evening (5pm-11pm)
        evening_mask = df['Date'].dt.hour.between(17, 22)
        segments['evening'] = df[evening_mask].copy()

        # Night (11pm-6am)
        night_mask = ~(morning_mask | afternoon_mask | evening_mask)
        segments['night'] = df[night_mask].copy()

        return segments

    def preprocess_all_patients(self, patient_data_dict):
        """
        Preprocess data for all patients

        Parameters:
            patient_data_dict (dict): Dictionary with patient data

        Returns:
            tuple: (preprocessed_data, meal_events)
                - preprocessed_data (dict): Dictionary with preprocessed patient data
                - meal_events (pandas.DataFrame): Merged meal events for all patients
        """
        preprocessed_data = {}
        all_meal_events = []

        print(f"Preprocessing data for {len(patient_data_dict)} patients...")

        for patient_id, df in patient_data_dict.items():
            # Skip if dataframe is empty
            if df.empty:
                print(f"Empty data for patient {patient_id}, skipping")
                continue

            # Clean glucose data
            df_clean = self.clean_glucose_data(df)

            # Calculate glucose variability metrics
            variability_metrics = self.calculate_glucose_variability(df_clean)
            
            # Extract meal events
            patient_meal_events = self.extract_meal_events(df_clean)
            
            # Print condensed processing information
            meal_count = len(patient_meal_events) if not patient_meal_events.empty else 0
            if variability_metrics:
                print(f"Processing patient {patient_id}: Glucose variability: Mean={variability_metrics.get('mean_glucose', 'N/A'):.1f}, CV={variability_metrics.get('cv_glucose', 'N/A'):.1f}%, Found {meal_count} meal events")
            else:
                print(f"Processing patient {patient_id}: No glucose variability data, Found {meal_count} meal events")
            
            if not patient_meal_events.empty:
                all_meal_events.append(patient_meal_events)

            # Store preprocessed data
            preprocessed_data[patient_id] = df_clean

        # Merge all meal events
        merged_meal_events = pd.concat(all_meal_events, ignore_index=True) if all_meal_events else pd.DataFrame()

        return preprocessed_data, merged_meal_events