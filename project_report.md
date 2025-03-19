# Diabetes Blood Glucose Data Analysis Project Implementation Report

## Abstract
This paper presents a comprehensive analysis of time-series data from diabetes patients, focusing on post-meal blood glucose responses. We utilize a publicly available dataset to explore glucose variability, meal-related glucose spikes, and recovery times. Our study employs advanced data processing techniques, visualization methods, and machine learning models to provide insights into the differences between Type 1 and Type 2 diabetes patients.

## Introduction
The management of diabetes requires continuous monitoring of blood glucose levels, especially in response to meals. This study aims to analyze the blood glucose response patterns of Type 1 and Type 2 diabetes patients using a dataset from Shanghai. We focus on understanding the dynamics of glucose spikes and recovery times post-meal, and we develop a predictive model for glucose recovery time.

## Definitions
- **Glucose Peak**: The highest glucose reading within 2 hours after a meal.
- **Recovery Time**: The time required for glucose levels to decrease from peak to near pre-meal levels (±20 mg/dL).
- **Spike Threshold**: A 30% increase in glucose levels relative to the pre-meal baseline, with a minimum increase of 30 mg/dL.

## Methodology
### Data Collection and Preprocessing
Data was collected from the Shanghai Diabetes dataset, which includes continuous glucose monitoring (CGM) data and meal records for both Type 1 and Type 2 diabetes patients. We processed the data to handle outliers, missing values, and to standardize glucose measurements.

### Feature Extraction
We extracted key features such as pre-meal glucose levels, peak glucose, rise rate, and recovery time. These features were used to analyze individual meal responses and to train a predictive model.

### Visualization
We employed various visualization techniques, including time-series plots and box plots, to illustrate glucose response patterns and to compare different meal types and diabetes types.

### Machine Learning Model
A linear regression model was developed to predict glucose recovery time using leave-one-subject-out cross-validation. The model was trained on features such as pre-meal glucose, meal type, and peak glucose.

## Experiments
### Individual Meal Analysis
We analyzed the glucose response for each meal, calculating metrics such as rise rate and recovery time. The analysis revealed that breakfast typically causes faster glucose rise rates, while snacks have shorter-lasting effects.

### Comparative Analysis
We compared glucose responses between Type 1 and Type 2 diabetes patients. The analysis confirmed that Type 1 diabetes patients generally have longer recovery times, aligning with clinical observations.

## Results
The visualization and analysis of glucose data provided insights into the variability and patterns of glucose responses. The machine learning model achieved an average mean absolute error (MAE) of 38.16 minutes in predicting recovery time.

## Discussion
Our findings highlight the differences in glucose response patterns between meal types and diabetes types. The predictive model's performance suggests that pre-meal glucose levels and meal type are significant predictors of recovery time.

## Conclusion
This study successfully analyzed and modeled post-meal glucose responses in diabetes patients. The results have potential implications for personalized diabetes management and treatment strategies.

## References
- [Dataset](https://figshare.com/articles/dataset/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/20444397?file=38259264)
- [Paper](https://www.nature.com/articles/s41597-023-01940-7)
