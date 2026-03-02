# Diabetes Prediction

## Introduction
Diabetes is a chronic metabolic disease, where the body in unable to produce or use insulin effectively, which is resulted 
by high blood sugar.

Traditional diagnosis requires medical tests and professional evaluations, however, a machine learning model can assist in:
- Identifying high-risk individuals early
- Supporting clinical decision makings
- Enabling accessible health-tech solutions

## Requirements

This project uses the Pima Indians Diabetes Dataset

Source:
Kagge link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

## Required Libraries
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

You can use 

pip install

command to download these dependencies.

## System Architecture

                          ┌─────────────────────────┐
                          │     Kaggle Dataset      │
                          │  (Pima Diabetes Data)   │
                          └─────────────┬───────────┘
                                        │
                                        ▼
                    ┌──────────────────────────────┐
                    │     Data Preprocessing       │
                    │  • Cleaning missing values   │
                    │  • Feature preparation       │
                    └─────────────┬────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │     ML Model Training        │
                    │  • Classification Model      │
                    │  • Evaluation Metrics        │
                    └─────────────┬────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │      Trained Model           │
                    └─────────────┬────────────────┘
                                  │
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
        ▼                                                   ▼
┌───────────────────────┐                        ┌────────────────────────┐
│  Streamlit Web App    │                        │   Model Evaluation     │
│  (User Interface)     │                        │  • Accuracy            │
│  • User Inputs        │                        │  • Confusion Matrix    │
│  • Form Handling      │                        │  • Classification Rep. │
└─────────────┬─────────┘                        └────────────────────────┘
              │
              ▼
     ┌───────────────────────┐
     │   Real-Time           │
     │   Prediction Output   │
     └───────────────────────┘


## Streamlit Integration

The ML pipeline is integrated into streamlit application inside Prediction.py
I did take assistance from Claude AI to integrate the Streamlit interface and improve the overall UI.

## How to run

1) Clone the repositiory using: 
git clone https://github.com/Vidhi405/Diabetes-Prediction.git

2) Install all the dependencies.

3) To run simply type: streamlit run Prediction.py

The application will open in your default browser.
