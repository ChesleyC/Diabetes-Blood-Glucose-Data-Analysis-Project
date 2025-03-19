### Project Overview

1. Readme.md

2. Overview of requirement
- time series project requirements english.txt
- /data/

3. Overview of result
- output_file_description.txt
- project_report.md
- /output/data/
- /output/figures/

4. Dive into the implementation
- /src/
- /main.py


### Project Structure

```
SugarFlow/
├── data/                      # Data files
│   ├── Shanghai_T1DM/         # T1DM data
│   ├── Shanghai_T2DM/         # T2DM data
│   ├── Shanghai_T1DM_Summary.xlsx
│   └── Shanghai_T2DM_Summary.xlsx
├── output/                    # Output results
│   ├── data/                  # Processed data
│   └── figures/               # Generated charts
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_loader.py         # Data loading module
│   ├── preprocessing.py       # Data preprocessing module
│   ├── visualization.py       # Data visualization module
│   ├── meal_analysis.py       # Post-meal analysis module
│   └── ml_model.py            # Machine learning model module
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
└── main.py                    # Main program entry
```


### Environment Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl (for .xlsx files)
- xlrd (for .xls files)

### Data Preparation

1. Download the dataset and extract it into the `data` folder
2. Ensure the file structure is as follows:
   - Excel files for T1DM patients in the `data/Shanghai_T1DM/` directory
   - Excel files for T2DM patients in the `data/Shanghai_T2DM/` directory
   - `Shanghai_T1DM_Summary.xlsx` and `Shanghai_T2DM_Summary.xlsx` in the `data/` directory

### Running the Analysis

Execute the main program:

```bash
python main.py
```