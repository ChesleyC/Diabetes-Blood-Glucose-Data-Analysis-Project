import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import DiabetesDataLoader
from src.preprocessing import DataPreprocessor
from src.visualization import GlucoseVisualizer
from src.meal_analysis import MealAnalyzer
from src.ml_model import GlucoseRecoveryPredictor, leave_one_out_cross_validation, train_final_model


def main():
    # Create output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/figures", exist_ok=True)
    os.makedirs("output/data", exist_ok=True)

    print("===== Loading Data =====")
    loader = DiabetesDataLoader()
    t1dm_data, t2dm_data = loader.load_all_data()

    print(f"\nT1DM Patient Count: {len(t1dm_data)}")
    print(f"T2DM Patient Count: {len(t2dm_data)}")

    print("\n===== Preprocessing Data =====")
    preprocessor = DataPreprocessor()
    t1dm_processed, t1dm_meal_events = preprocessor.preprocess_all_patients(t1dm_data)
    t2dm_processed, t2dm_meal_events = preprocessor.preprocess_all_patients(t2dm_data)

    # Merge all meal events
    all_meal_events = pd.concat([t1dm_meal_events, t2dm_meal_events], ignore_index=True) if not (
                t1dm_meal_events.empty and t2dm_meal_events.empty) else pd.DataFrame()

    # Save processed data
    if not all_meal_events.empty:
        all_meal_events.to_csv("output/data/all_meal_events.csv", index=False)
        print(f"Saved {len(all_meal_events)} meal event records")
    else:
        print("No meal event data to save")

    print("\n===== Visualization Analysis =====")
    visualizer = GlucoseVisualizer()

    # Select the first T1DM patient for sample analysis
    if t1dm_processed:
        patient_id = list(t1dm_processed.keys())[0]
        patient_df = t1dm_processed[patient_id]

        # Plot daily glucose curve
        if 'Date' in patient_df.columns and not patient_df['Date'].isna().all():
            date = patient_df['Date'].dt.date.min()
            fig = visualizer.plot_daily_glucose(patient_df, date)
            if fig:
                fig.savefig(f"output/figures/daily_glucose_t1dm_{patient_id}.png")
                plt.close(fig)
                print(f"Saved daily glucose curve for T1DM patient {patient_id}")

    # Compare glucose responses across different meal types
    if not all_meal_events.empty and 'meal_type' in all_meal_events.columns:
        fig = visualizer.compare_meal_types(all_meal_events)
        if fig:
            fig.savefig("output/figures/meal_types_comparison.png")
            plt.close(fig)
            print("Saved comparison chart of different meal types")
    else:
        print("No meal event data available, cannot compare different meal types")

    # Compare glucose responses between T1DM and T2DM patients
    if not all_meal_events.empty and 'diabetes_type' in all_meal_events.columns:
        fig = visualizer.compare_diabetes_types(all_meal_events)
        if fig:
            fig.savefig("output/figures/diabetes_types_comparison.png")
            plt.close(fig)
            print("Saved comparison chart of different diabetes types")
    else:
        print("Not enough data for comparison")

    print("\n===== Post-Meal Analysis =====")
    analyzer = MealAnalyzer()

    # Analyze all meals for a single patient
    if t1dm_processed:
        patient_id = list(t1dm_processed.keys())[0]
        patient_df = t1dm_processed[patient_id]

        meal_results = analyzer.analyze_all_meals(patient_df)
        if not meal_results.empty:
            meal_results.to_csv(f"output/data/meal_results_t1dm_{patient_id}.csv", index=False)
            print(f"Saved post-meal analysis results for T1DM patient {patient_id}")

            # Compare different meal types
            meal_type_summary = analyzer.compare_meal_types(meal_results)
            if not meal_type_summary.empty:
                print("\nGlucose response comparison across meal types:")
                print(meal_type_summary)

    # Compare post-meal responses between T1DM and T2DM
    if not all_meal_events.empty and 'diabetes_type' in all_meal_events.columns:
        diabetes_comparison = analyzer.compare_diabetes_types(all_meal_events)
        if not diabetes_comparison.empty:
            print("\nGlucose response comparison between T1DM and T2DM patients:")
            print(diabetes_comparison)
            diabetes_comparison.to_csv("output/data/diabetes_type_comparison.csv")
            print("Saved comparison results of different diabetes types")
    else:
        print("Not enough data to compare diabetes types")

    print("\n===== Machine Learning Modeling =====")
    # Only train the model when there is sufficient data
    if not all_meal_events.empty and len(all_meal_events) > 10 and 'recovery_time_min' in all_meal_events.columns:
        # Leave-one-out cross-validation
        cv_results = leave_one_out_cross_validation(all_meal_events)
        if cv_results and len(cv_results.get('patient_id', [])) > 0:
            cv_df = pd.DataFrame({
                'patient_id': cv_results['patient_id'],
                'MAE': cv_results['mae'],
                'RMSE': cv_results['rmse'],
                'R2': cv_results['r2'],
                'samples': cv_results['sample_count']
            })

            cv_df.to_csv("output/data/cross_validation_results.csv", index=False)
            print("Saved cross-validation results")

            print("\nCross-validation performance:")
            print(f"Average MAE: {np.mean(cv_results['mae']):.2f} minutes")
            print(f"Average RMSE: {np.mean(cv_results['rmse']):.2f} minutes")
            print(f"Average R²: {np.mean(cv_results['r2']):.2f}")

            if 'overall' in cv_results:
                print("\nOverall performance:")
                print(f"Overall MAE: {cv_results['overall']['mae']:.2f} minutes")
                print(f"Overall RMSE: {cv_results['overall']['rmse']:.2f} minutes")
                print(f"Overall R²: {cv_results['overall']['r2']:.2f}")

        # Train final model
        final_model = train_final_model(all_meal_events)

        # Plot feature importance
        if final_model and final_model.model is not None:
            fig = final_model.plot_feature_importance()
            if fig:
                fig.savefig("output/figures/feature_importance.png")
                plt.close(fig)
                print("Saved feature importance plot")
    else:
        print("Insufficient data for machine learning modeling")

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()