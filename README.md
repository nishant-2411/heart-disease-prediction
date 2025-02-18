# Heart Disease Prediction using Logistic Regression

## Overview
This project develops a machine learning model to predict the likelihood of heart disease using **Logistic Regression**. The dataset includes various health-related attributes, and the model is trained to classify individuals as having heart disease or not.

## Dataset
The dataset used for this project is the **Heart Disease Dataset** from the UCI Machine Learning Repository. It includes features such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression induced by exercise
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia
- Target (1: Disease, 0: No Disease)

## Dependencies
To run this project, install the required Python libraries using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Implementation Steps
1. **Data Preprocessing**
   - Load the dataset using pandas.
   - Handle missing values if any.
   - Encode categorical variables if required.
   - Normalize or standardize the features if necessary.
   
2. **Exploratory Data Analysis (EDA)**
   - Visualize feature distributions.
   - Identify correlations using heatmaps.
   - Analyze the impact of different attributes on heart disease.
   
3. **Model Training**
   - Split the dataset into training and testing sets.
   - Train a **Logistic Regression** model.
   - Optimize hyperparameters if needed.
   
4. **Evaluation**
   - Calculate accuracy, precision, recall, and F1-score.
   - Plot confusion matrix and ROC curve.
   
## Usage
Run the following script to train and evaluate the model:
```bash
python train_model.py
```

## Results
- The model achieves an accuracy of approximately **XX%** (based on evaluation).
- Key risk factors for heart disease include **age, cholesterol levels, and maximum heart rate achieved**.

## Conclusion
This project successfully demonstrates heart disease prediction using **Logistic Regression**, highlighting the importance of feature selection and data preprocessing.
