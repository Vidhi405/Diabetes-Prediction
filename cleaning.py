# Cleaning the Dataset
import pandas as pd
import numpy as np

df=pd.read_csv("D:\\ProjectsTwo\\Diabetes\\diabetes train data.csv")

print("Original Shape: ", df.shape)

df.columns=(df.columns.str.lower().str.strip().str.replace(" ","_",regex=False).str.replace("-","_",regex=False))

if "id" in df.columns:
    df=df.drop(columns=["id"])

zero=["bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate","cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"]

for col in zero:
    if col in df.columns:
        df[col]=df[col].replace(0,np.nan)

numbers=df.select_dtypes(include="number").columns
category=df.select_dtypes(include="object").columns

for col in numbers:
    df[col]=df[col].fillna(df[col].median())

for col in category:
    df[col]=df[col].fillna(df[col].mode()[0])

if "age" in df.columns:
    df["age_group"]=pd.cut(df["age"],bins=[0,18,35,50,65,200], labels=["child","young_adult","adult","middle_age","senior"],right=False)

if "bmi" in df.columns:
    df["bmi_category"]=pd.cut(df["bmi"], bins=[0,18.5,25,30,1000],labels=["underweight","normal","overweight","obese"],right=False)

target=50000

small=df.sample(n=target,random_state=42)
small=small.reset_index(drop=True)

print("Sampled Dataset Shape: ",small.shape)

small.to_csv("Cleaned_DiabetesDataset.csv",index=False)
print("Saved Cleaned_DiabetesDataset.csv")

mess=small.copy()

if "bmi" in mess.columns:
    idx=mess.sample(frac=0.05, random_state=10).index
    mess.loc[idx,"bmi"]=np.nan

if "triglycerides" in mess.columns:
    out=mess.sample(n=15, random_state=20).index
    mess.loc[out,"triglycerides"]=mess["triglycerides"].median()*5

mess.to_csv("Messy.csv",index=False)
print("Saved messy.csv")

