# Loan Data Analysis Application

## Names
Matthew Lazur

Jacob Accardi

## Description

This repository contains a Streamlit web application for loan analysis and risk assessment. The application uses a trained logistic regression machine learning model to predict whether a loan application should be approved or denied based on various applicant characteristics including loan amount, FICO score, employment status, income, and financial history.

## Repository Contents

- `loan_data_analysis_app.py` - Main Streamlit application file
- `BUS458_model.pkl` - Trained logistic regression model (pickle file)
- `requirements.txt` - Python dependencies needed to run the application

## Background

This project was developed for **BUS458 Data to Decisions** class. In a separate notebook, we performed comprehensive data analysis and model development:

1. **Data Loading**: Read in training and test datasets
2. **Exploratory Data Analysis**: Conducted thorough exploratory analysis to understand the data structure, distributions, and relationships
3. **Model Training**: Trained both a logistic regression model and a decision tree model
4. **Model Tuning**: Tuned hyperparameters to optimize model performance
5. **Model Evaluation**: Evaluated both models using appropriate metrics
6. **Model Selection**: The logistic regression model demonstrated superior performance
7. **Model Deployment**: The trained logistic regression model was saved as a pickle file (`BUS458_model.pkl`) and integrated into this Streamlit application

The application allows users to input loan applicant information and receive real-time predictions about loan risk assessment.

## How to Run

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Navigate to the project directory
2. Run the Streamlit application:
   ```bash
   streamlit run loan_data_analysis_app.py
   ```
3. The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. Enter the loan applicant's details using the input fields:
   - **Reason**: Select the reason for the loan
   - **Requested Loan Amount**: Use the slider to specify the loan amount
   - **FICO Score**: Select the applicant's FICO credit score
   - **Employment Status**: Choose employment status
   - **Employment Sector**: Select the employment sector
   - **Monthly Gross Income**: Enter monthly gross income
   - **Monthly Housing Payment**: Enter monthly housing payment
   - **Ever Bankrupt or Foreclose**: Indicate if the applicant has ever been bankrupt or foreclosed
   - **Lender**: Select the lender type
2. Click the **"Evaluate Loan"** button
3. The application will display the prediction: either "Good Loan" 💲 or "Bad Loan" 🚫

