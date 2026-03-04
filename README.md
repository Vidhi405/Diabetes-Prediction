![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

# Diabetes Prediction

## Introduction
Diabetes is a chronic metabolic disease in which the body is unable to produce or effectively use insulin, resulting in high blood sugar levels.
Traditional diagnosis requires medical tests and professional evaluation. However, machine learning models can assist in:

- Identifying high-risk individuals at an early stage
- Supporting clinical decision-making
- Enabling accessible health-tech solutions

This project builds an end-to-end Machine Learning pipeline and integrates it with a Streamlit-based web application for real-time diabetes prediction.

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

## 🧠 System Architecture

```
                          ┌────────────────────────────┐
                          │       Kaggle Dataset       │
                          │   (Pima Diabetes Dataset)  │
                          └──────────────┬─────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────┐
                    │        Data Preprocessing        │
                    │  • Missing value handling        │
                    │  • Feature preparation           │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │        Model Training            │
                    │  • Classification Algorithm      │
                    │  • Evaluation Metrics            │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │          Trained Model           │
                    └──────────────┬───────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                             │
            ▼                                             ▼

   ┌──────────────────────────┐                ┌──────────────────────────┐
   │   Model Evaluation       │                │     Streamlit Web App    │
   │  • Accuracy              │                │  • User Input Form       │
   │  • Confusion Matrix      │                │  • Input Processing      │
   │  • Classification Report │                │  • Prediction Trigger    │
   └──────────────────────────┘                └─────────────┬────────────┘
                                                             │
                                                             ▼
                                              ┌──────────────────────────┐
                                              │    Real-Time Prediction  │
                                              │         Output           │
                                              └──────────────────────────┘
```


## Streamlit Integration

The ML pipeline is integrated into streamlit application inside Prediction.py
I did take assistance from Claude AI to integrate the Streamlit interface and improve the overall UI.

## How to run

1) Clone the repositiory using: 
git clone https://github.com/Vidhi405/Diabetes-Prediction.git

2) Install all the dependencies.

3) To run simply type: streamlit run Prediction.py

The application will open in your default browser.

## Areas for Improvement
What more I can do is improve the overall UI by learning more about Streamlit, to improve the layout, alignments and the visuals. I also want to improve the model accuracy and acceptance rate. What more I can do is implementing some kind of cross-validation for better evaluations.
Saving the trained model under the '.pkl' will result in faster inferences is another idea which I will be proceeding with next.
