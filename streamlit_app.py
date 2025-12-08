import streamlit as st
import pickle
import pandas as pd
import sklearn

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Loan Analysis</b></h1>",
    unsafe_allow_html=True
)
# Load the trained model
with open("BUS458_model.pkl", "rb") as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Helper function to convert underscores to readable text
def format_label(value):
    return value.replace('_', ' ').title()

# Categorical inputs
reason_options = ['cover_an_unexpected_cost', 'credit_card_refinancing',
                  'home_improvement', 'major_purchase', 'other', 'debt_conslidation']
reason = st.selectbox("Reason (Reason)", reason_options, 
                      format_func=lambda x: x.replace('_', ' ').title().replace('Conslidation', 'Consolidation'))

employment_status = st.selectbox("Employment Status (Employment_Status)", ["full-time", "part-time", "unemployed"])

employment_sector_options = ['consumer_discretionary', 'information_technology', 'energy',
                            'consumer_staples', 'communication_services', 'materials',
                            'utilities', 'real_estate', 'health_care', 'industrials',
                            'financials', 'Unknown']
employment_sector = st.selectbox("Employment Sector (Employment_Sector)", employment_sector_options,
                                 format_func=format_label)


# Numeric inputs
requested_loan_amount = st.slider("Requested Loan Amount (Requested_Loan_Amount)", min_value=1000.0, max_value=500000.0, step=1000.0)
fico_score = st.slider("FICO Score (FICO_score)", min_value=300.0, max_value=850.0, step=1.0)
monthly_gross_income = st.slider("Monthly Gross Income (Monthly_Gross_Income)", min_value=0.0, max_value=50000.0, step=100.0)
monthly_housing_payment = st.number_input("Monthly Housing Payment (Monthly_Housing_Payment)", min_value=0, max_value=10000, step=100)
ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclose (Ever_Bankrupt_or_Foreclose)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "Reason": [reason],
    "Requested_Loan_Amount": [requested_loan_amount],
    "FICO_score": [fico_score],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt_or_foreclose]
})

numerical_cols = ['Requested_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income',
                         'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose']
        
numerical_data = input_data[numerical_cols]
#Add this

scaled_numerical = scaler.transform(numerical_data)
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_cols)

all_categories = {
    'Reason': ['cover_an_unexpected_cost', 'credit_card_refinancing',
               'home_improvement', 'major_purchase', 'other', 'debt_conslidation'],
    'Employment_Status': ['full_time', 'part_time', 'unemployed'],
    'Employment_Sector': ['consumer_discretionary', 'information_technology', 'energy',
                         'consumer_staples', 'communication_services', 'materials',
                         'utilities', 'real_estate', 'health_care', 'industrials',
                         'financials', 'unknown']
}

# Create empty DataFrame with ALL possible categorical columns
all_categorical_cols = []
for prefix, categories in all_categories.items():
    for cat in categories:
        all_categorical_cols.append(f'{prefix}_{cat}')

categorical_df = pd.DataFrame(0, index=[0], columns=all_categorical_cols)

# Set the selected values to 1
categorical_df[f'Reason_{reason}'] = 1
categorical_df[f'Employment_Status_{employment_status}'] = 1
categorical_df[f'Employment_Sector_{employment_sector}'] = 1

final_input = pd.concat([scaled_numerical_df, categorical_df], axis=1)
st.write(categorical_df.columns.tolist())
# Ensure all expected columns are present (fill missing columns with 0s)
model_columns = model.feature_names_in_  # Get the feature names used during training
for col in model_columns:
    if col not in final_input.columns:
        st.error(f"Missing column: {col}")
        final_input[col] = 0  # Add missing column with value 0

# Reorder columns to match the model's training data
final_input = final_input[model_columns]

# Predict button
if st.button("Evaluate Loan"):
    # Predict using the loaded model
    y_pred_proba_log = model.predict_proba(final_input)[:, 1]  # probability for ROC

    # Apply cutoff threshold
    threshold = 0.65
    y_pred_log = (y_pred_proba_log >= threshold).astype(int) 
    
    # Display result
    if y_pred_log == 0:
        st.write("The prediction is: **Bad Loan**  🚫")
    else:
        st.write("The prediction is: **Good Loan** 💲")




