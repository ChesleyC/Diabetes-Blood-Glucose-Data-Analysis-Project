import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.font_manager as fm


class GlucoseVisualizer:
    """
    Glucose data visualization class
    """

    def __init__(self):
        """
        Initialize the visualizer
        """
        # Set default plot style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Use a standard font that should be available on most systems
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # Define normal blood glucose range (unit: mg/dL)
        self.normal_range = (70, 140)

        # Define color mapping
        self.meal_colors = {
            'breakfast': '#1f77b4',  # Blue
            'lunch': '#ff7f0e',      # Orange
            'dinner': '#2ca02c',     # Green
            'snack': '#d62728'       # Red
        }

        self.diabetes_colors = {
            'T1DM': '#1f77b4',  # Blue
            'T2DM': '#ff7f0e'   # Orange
        }

        # Possible CGM and CBG column names
        self.cgm_possible_cols = ['CGM (mg / dl)', 'CGM', 'CGM(mg/dl)', 'cgm']
        self.cbg_possible_cols = ['CBG (mg / dl)', 'CBG', 'CBG(mg/dl)', 'cbg']

    def _find_glucose_columns(self, df):
        """
        Find glucose columns in the data

        Parameters:
            df (pandas.DataFrame): Data table

        Returns:
            tuple: (cgm_col, cbg_col) - Found CGM and CBG column names
        """
        cgm_col = None
        cbg_col = None

        # Find CGM column
        for col in self.cgm_possible_cols:
            if col in df.columns:
                cgm_col = col
                break

        # If standard name not found, try to find columns containing 'cgm'
        if cgm_col is None:
            for col in df.columns:
                if 'cgm' in str(col).lower():
                    cgm_col = col
                    break

        # Find CBG column
        for col in self.cbg_possible_cols:
            if col in df.columns:
                cbg_col = col
                break

        # If standard name not found, try to find columns containing 'cbg'
        if cbg_col is None:
            for col in df.columns:
                if 'cbg' in str(col).lower():
                    cbg_col = col
                    break

        return cgm_col, cbg_col

    def plot_daily_glucose(self, patient_df, date=None, figsize=(12, 6)):
        """
        Plot daily glucose curve

        Parameters:
            patient_df (pandas.DataFrame): Patient data
            date (datetime.date, optional): Date to plot, defaults to first date in data
            figsize (tuple, optional): Figure size

        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        # Ensure Date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(patient_df['Date']):
            patient_df['Date'] = pd.to_datetime(patient_df['Date'])

        # If date not specified, use the first day in the data
        if date is None:
            date = patient_df['Date'].dt.date.min()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()

        # Filter data for the specified date
        day_data = patient_df[patient_df['Date'].dt.date == date]

        if day_data.empty:
            print(f"No data for date {date}")
            return None

        # Find CGM and CBG columns
        cgm_col, cbg_col = self._find_glucose_columns(day_data)

        if cgm_col is None and cbg_col is None:
            print("No glucose data columns found")
            return None

        # Create chart
        fig, ax = plt.subplots(figsize=figsize)

        # Plot CGM data
        if cgm_col is not None:
            ax.plot(day_data['Date'], day_data[cgm_col], label='CGM')

        # Plot CBG data (if available)
        if cbg_col is not None and day_data[cbg_col].notna().any():
            ax.scatter(
                day_data.loc[day_data[cbg_col].notna(), 'Date'],
                day_data.loc[day_data[cbg_col].notna(), cbg_col],
                color='red', s=50, label='CBG'
            )

        # Find dietary columns
        meal_cols = [col for col in day_data.columns if
                     'diet' in str(col).lower() or 'meal' in str(col).lower()]

        # Mark meal times
        if meal_cols:
            meal_col = meal_cols[0]
            meal_data = day_data[day_data[meal_col].notna()]
            for idx, row in meal_data.iterrows():
                meal_time = row['Date']
                meal_type = row['meal_type'] if 'meal_type' in row and pd.notna(row['meal_type']) else 'unknown'
                color = self.meal_colors.get(meal_type, 'gray')
                ax.axvline(x=meal_time, color=color, linestyle='--', alpha=0.7)
                try:
                    # Use English meal type labels
                    meal_label = meal_type.capitalize()
                    ax.text(
                        meal_time, ax.get_ylim()[1] * 0.95,
                        meal_label,
                        rotation=90, color=color
                    )
                except:
                    # Use simple label if error occurs
                    ax.text(
                        meal_time, ax.get_ylim()[1] * 0.95,
                        "Meal",
                        rotation=90, color=color
                    )

        # Plot normal range
        ax.axhspan(self.normal_range[0], self.normal_range[1], alpha=0.2, color='green', label='Normal Range')

        # Set chart properties
        try:
            ax.set_title(f'Blood Glucose Curve - {date}')
        except:
            ax.set_title('Blood Glucose Curve')
        ax.set_xlabel('Time')
        ax.set_ylabel('Blood Glucose (mg/dL)')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        return fig

    def plot_meal_response(self, patient_df, meal_index, window_hours=4, figsize=(10, 6)):
        """
        Plot post-meal glucose response curve

        Parameters:
            patient_df (pandas.DataFrame): Patient data
            meal_index (int): Index of the meal to analyze
            window_hours (int, optional): Post-meal observation window in hours
            figsize (tuple, optional): Figure size

        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        # Ensure Date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(patient_df['Date']):
            patient_df['Date'] = pd.to_datetime(patient_df['Date'])

        # Find dietary column
        meal_cols = [col for col in patient_df.columns if
                     'diet' in str(col).lower() or 'meal' in str(col).lower()]

        if not meal_cols:
            print("No meal record column found")
            return None

        meal_col = meal_cols[0]

        # Look for meal records
        meal_rows = patient_df[meal_col].notna()
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
        meal_content = patient_df.loc[idx, meal_col]

        # Select data from 30 minutes before meal to window_hours after meal
        start_time = meal_time - timedelta(minutes=30)
        end_time = meal_time + timedelta(hours=window_hours)
        window_data = patient_df[(patient_df['Date'] >= start_time) & (patient_df['Date'] <= end_time)]

        # Find CGM column
        cgm_col, _ = self._find_glucose_columns(window_data)

        if cgm_col is None:
            print("No CGM data column found")
            return None

        # Create chart
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate relative time (minutes)
        window_data['minutes_from_meal'] = (window_data['Date'] - meal_time).dt.total_seconds() / 60

        # Plot CGM data
        if cgm_col in window_data.columns:
            ax.plot(window_data['minutes_from_meal'], window_data[cgm_col], marker='o')

            # Mark meal time
            ax.axvline(x=0, color='red', linestyle='--', label='Meal Time')

            # Calculate peak and recovery time
            post_meal_data = window_data[window_data['minutes_from_meal'] > 0]
            if not post_meal_data.empty and not post_meal_data[cgm_col].isna().all():
                peak_idx = post_meal_data[cgm_col].idxmax()
                peak_glucose = post_meal_data.loc[peak_idx, cgm_col]
                peak_time = post_meal_data.loc[peak_idx, 'minutes_from_meal']
                
                # Mark peak point
                ax.scatter(peak_time, peak_glucose, color='red', s=100, zorder=5)
                ax.text(peak_time, peak_glucose, f' Peak: {peak_glucose:.1f} mg/dL\n ({peak_time:.0f} min)', 
                        verticalalignment='bottom')
                
                # Find recovery time (when glucose returns to normal range)
                norm_mask = (post_meal_data[cgm_col] <= self.normal_range[1]) & (post_meal_data['minutes_from_meal'] > peak_time)
                if norm_mask.any():
                    recovery_idx = post_meal_data[norm_mask].index[0]
                    recovery_glucose = post_meal_data.loc[recovery_idx, cgm_col]
                    recovery_time = post_meal_data.loc[recovery_idx, 'minutes_from_meal']
                    
                    # Mark recovery point
                    ax.scatter(recovery_time, recovery_glucose, color='green', s=100, zorder=5)
                    ax.text(recovery_time, recovery_glucose, 
                            f' Recovery: {recovery_time:.0f} min', 
                            verticalalignment='bottom')
            
            # Plot normal range
            ax.axhspan(self.normal_range[0], self.normal_range[1], alpha=0.2, color='green', label='Normal Range')
            
            # Title and meal info
            title = f'Post-meal Glucose Response - {meal_type.capitalize()}' if meal_type != 'unknown' else 'Post-meal Glucose Response'
            ax.set_title(title)
            
            # Add meal content as text in the chart
            if pd.notna(meal_content):
                ax.text(0.02, 0.02, f'Meal content: {meal_content}', 
                        transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            # Set axes labels
            ax.set_xlabel('Time (minutes from meal)')
            ax.set_ylabel('Blood Glucose (mg/dL)')
            ax.set_xlim(-30, window_hours * 60)
            ax.legend()
            
            return fig
        
        return None

    def compare_meal_types(self, meal_events_df, figsize=(12, 8)):
        """
        Compare glucose responses across different meal types

        Parameters:
            meal_events_df (pandas.DataFrame): Meal events data
            figsize (tuple, optional): Figure size

        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        if meal_events_df.empty or 'meal_type' not in meal_events_df.columns:
            print("No meal type data available for comparison")
            return None
            
        # Filter out unknown meal types and meals without recovery time
        valid_meals = meal_events_df[
            (meal_events_df['meal_type'].notna()) & 
            (meal_events_df['meal_type'] != 'unknown') &
            (meal_events_df['recovery_time_min'].notna())
        ]
        
        if valid_meals.empty:
            print("No valid meal data with known meal types and recovery times")
            return None
            
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Compare pre-meal glucose
        sns.boxplot(x='meal_type', y='pre_meal_cgm', data=valid_meals, hue='meal_type', palette=self.meal_colors, legend=False, ax=axes[0])
        axes[0].set_title('Pre-meal Glucose by Meal Type')
        axes[0].set_ylabel('Glucose (mg/dL)')
        axes[0].set_xlabel('Meal Type')
        
        # 2. Compare peak glucose
        sns.boxplot(x='meal_type', y='peak_cgm', data=valid_meals, hue='meal_type', palette=self.meal_colors, legend=False, ax=axes[1])
        axes[1].set_title('Peak Glucose by Meal Type')
        axes[1].set_ylabel('Glucose (mg/dL)')
        axes[1].set_xlabel('Meal Type')
        
        # 3. Compare glucose rise rate
        sns.boxplot(x='meal_type', y='rise_rate_mg_dl_min', data=valid_meals, hue='meal_type', palette=self.meal_colors, legend=False, ax=axes[2])
        axes[2].set_title('Glucose Rise Rate by Meal Type')
        axes[2].set_ylabel('Rise Rate (mg/dL/min)')
        axes[2].set_xlabel('Meal Type')
        
        # 4. Compare recovery time
        sns.boxplot(x='meal_type', y='recovery_time_min', data=valid_meals, hue='meal_type', palette=self.meal_colors, legend=False, ax=axes[3])
        axes[3].set_title('Recovery Time by Meal Type')
        axes[3].set_ylabel('Recovery Time (min)')
        axes[3].set_xlabel('Meal Type')
        
        plt.tight_layout()
        return fig

    def compare_diabetes_types(self, meal_events_df, figsize=(12, 6)):
        """
        Compare glucose responses between different diabetes types

        Parameters:
            meal_events_df (pandas.DataFrame): Meal events data
            figsize (tuple, optional): Figure size

        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        if meal_events_df.empty or 'diabetes_type' not in meal_events_df.columns:
            print("No diabetes type data available for comparison")
            return None
            
        # Filter out rows without valid diabetes type or recovery time
        valid_data = meal_events_df[
            (meal_events_df['diabetes_type'].notna()) & 
            (meal_events_df['recovery_time_min'].notna())
        ]
        
        if valid_data.empty:
            print("No valid data with known diabetes types and recovery times")
            return None
            
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Compare recovery time by diabetes type
        sns.boxplot(x='diabetes_type', y='recovery_time_min', data=valid_data, hue='diabetes_type', palette=self.diabetes_colors, legend=False, ax=axes[0])
        axes[0].set_title('Recovery Time by Diabetes Type')
        axes[0].set_ylabel('Recovery Time (minutes)')
        axes[0].set_xlabel('Diabetes Type')
        
        # 2. Compare rise rate by diabetes type
        sns.boxplot(x='diabetes_type', y='rise_rate_mg_dl_min', data=valid_data, hue='diabetes_type', palette=self.diabetes_colors, legend=False, ax=axes[1])
        axes[1].set_title('Glucose Rise Rate by Diabetes Type')
        axes[1].set_ylabel('Rise Rate (mg/dL/min)')
        axes[1].set_xlabel('Diabetes Type')
        
        plt.tight_layout()
        return fig