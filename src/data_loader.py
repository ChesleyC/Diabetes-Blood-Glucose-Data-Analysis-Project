import pandas as pd
import os
import re
import glob
from datetime import datetime
import numpy as np


class DiabetesDataLoader:
    """
    Class for loading and processing diabetes datasets
    """

    def __init__(self, data_dir="data"):
        """
        Initialize the data loader

        Parameters:
            data_dir (str): Data directory path
        """
        self.data_dir = data_dir
        self.t1dm_dir = os.path.join(data_dir, "Shanghai_T1DM")
        self.t2dm_dir = os.path.join(data_dir, "Shanghai_T2DM")
        self.t1dm_summary_path = os.path.join(data_dir, "Shanghai_T1DM_Summary.xlsx")
        self.t2dm_summary_path = os.path.join(data_dir, "Shanghai_T2DM_Summary.xlsx")

    def load_summary(self, diabetes_type="T1DM"):
        """
        Load summary data

        Parameters:
            diabetes_type (str): Diabetes type, "T1DM" or "T2DM"

        Returns:
            pandas.DataFrame: Summary data table
        """
        if diabetes_type == "T1DM":
            path = self.t1dm_summary_path
        elif diabetes_type == "T2DM":
            path = self.t2dm_summary_path
        else:
            raise ValueError("diabetes_type must be 'T1DM' or 'T2DM'")

        try:
            df = pd.read_excel(path)
            print(f"Successfully loaded {diabetes_type} summary data")
            return df
        except Exception as e:
            print(f"Error loading {diabetes_type} summary data: {str(e)}")
            return pd.DataFrame()

    def extract_patient_id(self, filename):
        """
        Extract patient ID from filename

        Parameters:
            filename (str): Filename

        Returns:
            str: Patient ID
        """
        base_name = os.path.basename(filename)
        match = re.match(r'(\d+)_\d+_\d+', base_name)
        if match:
            return match.group(1)
        else:
            print(f"Warning: Could not extract patient ID from filename: {base_name}")
            return None

    def load_patient_data(self, diabetes_type="T1DM"):
        """
        Load data for all patients

        Parameters:
            diabetes_type (str): Diabetes type, "T1DM" or "T2DM"

        Returns:
            dict: Dictionary with patient IDs as keys and DataFrames as values
        """
        if diabetes_type == "T1DM":
            folder_path = self.t1dm_dir
        elif diabetes_type == "T2DM":
            folder_path = self.t2dm_dir
        else:
            raise ValueError("diabetes_type must be 'T1DM' or 'T2DM'")

        data_dict = {}
        processed_files = 0
        skipped_files = 0
        error_files = 0

        if not os.path.exists(folder_path):
            print(f"Error: Folder path '{folder_path}' does not exist")
            return data_dict

        # Use glob to get all Excel files
        files = glob.glob(os.path.join(folder_path, "*.xls*"))
        print(f"Found {len(files)} files in {folder_path}")

        for filepath in files:
            filename = os.path.basename(filepath)

            # Skip temporary files (starting with ~ or ~$)
            if filename.startswith("~") or filename.startswith("~$"):
                print(f"Skipping temporary file: {filename}")
                skipped_files += 1
                continue

            try:
                # Choose engine based on file extension
                if filename.lower().endswith('.xlsx'):
                    df = pd.read_excel(filepath, engine="openpyxl")
                elif filename.lower().endswith('.xls'):
                    df = pd.read_excel(filepath, engine="xlrd")
                else:
                    print(f"Skipping file {filename}: Not in xls or xlsx format.")
                    skipped_files += 1
                    continue

                # Extract patient ID
                patient_id = self.extract_patient_id(filename)
                if patient_id is None:
                    skipped_files += 1
                    continue

                # Preprocess data
                df = self._preprocess_data(df, filename)

                # Add df to data_dict
                if patient_id not in data_dict:
                    data_dict[patient_id] = df
                else:
                    data_dict[patient_id] = pd.concat([data_dict[patient_id], df], ignore_index=True)

                processed_files += 1
                print(f"Successfully processed: {filename} (Patient ID: {patient_id})")

            except Exception as e:
                error_files += 1
                print(f"Error processing {filename}: {str(e)}")
                continue

        print(f"\nSummary:")
        print(f"  Processed: {processed_files} files")
        print(f"  Skipped: {skipped_files} files")
        print(f"  Errors: {error_files} files")
        print(f"  Total patients: {len(data_dict)}")

        return data_dict

    def _preprocess_data(self, df, filename):
        """
        Preprocess patient data

        Parameters:
            df (pandas.DataFrame): Original data table
            filename (str): Filename, used to record data source

        Returns:
            pandas.DataFrame: Preprocessed data table
        """
        # Add source file column
        df['source_file'] = filename

        # Process date/time columns
        if 'Date' in df.columns:
            # Try to convert date column to datetime type
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                df['timestamp'] = df['Date']  # Create a timestamp column
            except:
                print(f"Warning: Cannot convert date column to datetime type: {filename}")

        # Add diabetes type marker
        if '1' in filename.split('_')[0]:
            df['diabetes_type'] = 'T1DM'
        elif '2' in filename.split('_')[0]:
            df['diabetes_type'] = 'T2DM'

        # Normalize column names, remove leading/trailing spaces
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

        # Convert CGM values to numeric type
        cgm_cols = [col for col in df.columns if 'CGM' in str(col) or 'cgm' in str(col).lower()]
        for col in cgm_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert CBG values to numeric type
        cbg_cols = [col for col in df.columns if 'CBG' in str(col) or 'cbg' in str(col).lower()]
        for col in cbg_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Find meal-related column
        meal_cols = [col for col in df.columns if 
                    'diet' in str(col).lower() or 'meal' in str(col).lower()]

        # Identify meal times
        if meal_cols:
            df = self._identify_meal_times(df)  # Modified to only pass df parameter

        return df

    def _identify_meal_times(self, df, meal_col=None):
        """
        Identify meal times in the data

        Parameters:
            df (pandas.DataFrame): Data table
            meal_col (str, optional): Column name for meal content, if None will auto-detect

        Returns:
            pandas.DataFrame: Data table with meal type markers added
        """
        # Add meal_type column, default to NaN
        if 'meal_type' not in df.columns:
            df['meal_type'] = None  # Use None instead of np.nan

        # If meal_col not specified, try to find it
        if meal_col is None:
            meal_cols = [col for col in df.columns if
                         'diet' in str(col).lower() or 'meal' in str(col).lower()]
            if meal_cols:
                meal_col = meal_cols[0]

        if meal_col and meal_col in df.columns and 'Date' in df.columns:
            # Find rows where diet is not empty
            meal_rows = df[meal_col].notna()

            if meal_rows.any():
                # Add time markers for these rows
                for idx in df.loc[meal_rows].index:
                    time = df.loc[idx, 'Date'].time() if isinstance(df.loc[idx, 'Date'], datetime) else None

                    if time:
                        if time.hour >= 5 and time.hour < 10:
                            df.at[idx, 'meal_type'] = 'breakfast'
                        elif time.hour >= 10 and time.hour < 15:
                            df.at[idx, 'meal_type'] = 'lunch'
                        elif time.hour >= 17 and time.hour < 22:
                            df.at[idx, 'meal_type'] = 'dinner'
                        else:
                            df.at[idx, 'meal_type'] = 'snack'

        return df

    def merge_with_summary(self, patient_data, diabetes_type="T1DM"):
        """
        Merge patient data with summary information

        Parameters:
            patient_data (dict): Dictionary of patient data
            diabetes_type (str): Diabetes type, "T1DM" or "T2DM"

        Returns:
            dict: Dictionary of merged patient data
        """
        # Load summary data
        summary_df = self.load_summary(diabetes_type)
        
        # If summary data is empty, return the original data
        if summary_df.empty:
            print(f"No summary data available for {diabetes_type}, skipping merge")
            return patient_data
            
        merged_data = {}
        
        # Try to find the patient ID column in the summary data
        id_cols = [col for col in summary_df.columns if 'ID' in str(col) or 'id' in str(col).lower()]
        
        if not id_cols:
            print(f"Cannot find ID column in {diabetes_type} summary data, skipping merge")
            return patient_data
            
        id_col = id_cols[0]
        
        # Merge data for each patient
        for patient_id, df in patient_data.items():
            # Try to find the patient in summary data
            patient_summary = summary_df[summary_df[id_col].astype(str) == str(patient_id)]
            
            if not patient_summary.empty:
                # Add summary data as new columns
                for col in patient_summary.columns:
                    if col != id_col:  # Skip the ID column
                        # Add the value as a constant column
                        df[f'summary_{col}'] = patient_summary[col].values[0]
                        
                print(f"Merged summary data for patient {patient_id}")
            else:
                print(f"No summary data found for patient {patient_id}")
                
            # Store the updated dataframe
            merged_data[patient_id] = df
            
        return merged_data

    def load_all_data(self):
        """
        Load and process data for all diabetes types

        Returns:
            tuple: (T1DM data dictionary, T2DM data dictionary)
        """
        print("Loading T1DM data...")
        t1dm_data = self.load_patient_data("T1DM")
        
        if t1dm_data:
            print(f"Loaded data for {len(t1dm_data)} T1DM patients")
            t1dm_data = self.merge_with_summary(t1dm_data, "T1DM")
        else:
            print("No T1DM data loaded")
            
        print("\nLoading T2DM data...")
        t2dm_data = self.load_patient_data("T2DM")
        
        if t2dm_data:
            print(f"Loaded data for {len(t2dm_data)} T2DM patients")
            t2dm_data = self.merge_with_summary(t2dm_data, "T2DM")
        else:
            print("No T2DM data loaded")
            
        return t1dm_data, t2dm_data