import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import base64  # Add this import for base64 encoding

# Load the XGBoost model
model_1 = pickle.load(open('modelXGB.pkl', 'rb'))
def get_countid_enocde(test, cols, name):
    temp2 = test.groupby(cols)['case_id'].count().reset_index().rename(columns={'case_id': name})
    test = pd.merge(test, temp2, how='left', on=cols)
    test[name] = test[name].astype('float')
    test[name].fillna(np.median(temp2[name]), inplace=True)
    return test

def f_result(df, model):
    # Map column names to match with the expected names
    column_mapping = {
        'Type_of_Admission': 'Type of Admission',
        'Severity_of_illness': 'Severity of Illness'
        # Add other mappings here if needed
    }
    # Rename columns if needed
    df.rename(columns=column_mapping, inplace=True)
    # Encode categorical variables
    for column in ['Hospital_type_code', 'Hospital_region_code', 'Department',
                   'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
    # Apply get_countid_enocde function
    df = get_countid_enocde(df, ['patientid'], name='count_id_patient')
    df = get_countid_enocde(df, ['patientid', 'Hospital_region_code'], name='count_id_patient_hospitalCode')
    df = get_countid_enocde(df, ['patientid', 'Ward_Facility_Code'], name='count_id_patient_wardfacilityCode')
    # Drop unnecessary columns
    df = df.drop(['patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis=1)
    # Set index
    df.set_index('case_id', inplace=True, drop=True)
    # Convert DataFrame to numpy array
    df_array = df.values
    # Make predictions
    prediction_xgb = model.predict(df_array)
    # Map predictions to stay lengths
    stay_mapping = {0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50',
                    5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90',
                    9: '91-100', 10: 'More than 100 Days'}
    predicted_stay_transformed = [stay_mapping[val] for val in prediction_xgb]
    return predicted_stay_transformed
# Function to render HTML
def write_html(html):
    return st.markdown(html, unsafe_allow_html=True)
def main():
    if 'login' not in st.session_state:
        st.session_state.login = False

    if not st.session_state.login:
        login()
    else:
        render_web_app()
def login():
    st.title("Login")
    # Define correct credentials
    correct_user_id = "sriram_dr"
    correct_password = "sriram2003"
    # Get user inputs
    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    # Check if login button is clicked
    if st.button("Login"):
        # Check if credentials are correct
        if user_id == correct_user_id and password == correct_password:
            st.session_state.login = True
            st.success("Login successful!")
            render_web_app()  # Open web app after successful login
        else:
            st.error("Invalid user ID or password. Please try again.")

def render_web_app():
    st.title("Hospital")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Patient Length of Stay ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    case_id = st.text_input("case_id", "Type Here")
    Hospital_code = st.text_input("Hospital_code", "Type Here")
    # Provide options for Hospital_type_code
    Hospital_type_code_options = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    Hospital_type_code = st.selectbox("Hospital_type_code", Hospital_type_code_options)

    City_Code_Hospital = st.text_input("City_Code_Hospital", "Type Here")
    Hospital_region_code_options = ['X','Y','Z']
    Hospital_region_code = st.selectbox("Hospital_region_code", Hospital_region_code_options)
    Available_Extra_Rooms_in_Hospital = st.text_input("Available_Extra_Rooms_in_Hospital", "Type Here")
    Department_options = ['gynecology','anesthesia','radiotherapy','TB & Chest disease','surgery']
    Department = st.selectbox("Department", Department_options)
    Ward_Type_options = ['P', 'Q', 'R', 'S', 'T', 'U']
    Ward_Type = st.selectbox("Ward_Type", Ward_Type_options)
    Ward_Facility_Code_options = ['A','B','C','D','E','F']
    Ward_Facility_Code = st.selectbox("Ward_Facility_Code", Ward_Facility_Code_options)
    Bed_Grade_options = [1,2,3,4]
    Bed_Grade = st.selectbox("Bed_Grade", Bed_Grade_options)
    patientid = st.text_input("patientid", "Type Here")
    City_Code_Patient = st.text_input("City_Code_Patient", "Type Here")
    Type_of_Admission_options = ['Trauma','Emergency','Urgent']
    Type_of_Admission = st.selectbox("Type_of_Admission", Type_of_Admission_options)
    Severity_of_illness_options = ['Moderate','Minor','Extreme']
    Severity_of_illness = st.selectbox("Severity_of_illness", Severity_of_illness_options)
    Visitors_with_Patient = st.text_input("Visitors_with_Patient", "Type Here")
    Age_options = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100']
    Age = st.selectbox("Age", Age_options)
    Admission_Deposit = st.text_input("Admission_Deposit", "Type Here")
    result = ""
    if st.button("Predict"):
        # Create a dictionary with input values
        input_data = {
            'case_id': [case_id],
            'Hospital_code': [Hospital_code],
            'Hospital_type_code': [Hospital_type_code],
            'City_Code_Hospital': [City_Code_Hospital],
            'Hospital_region_code': [Hospital_region_code],
            'Available_Extra_Rooms_in_Hospital': [Available_Extra_Rooms_in_Hospital],
            'Department': [Department],
            'Ward_Type': [Ward_Type],
            'Ward_Facility_Code': [Ward_Facility_Code],
            'Bed_Grade': [Bed_Grade],
            'patientid': [patientid],
            'City_Code_Patient': [City_Code_Patient],
            'Type_of_Admission': [Type_of_Admission],
            'Severity_of_illness': [Severity_of_illness],
            'Visitors_with_Patient': [Visitors_with_Patient],
            'Age': [Age],
            'Admission_Deposit': [Admission_Deposit]
        }
        # Create a DataFrame from the input data
        input_df = pd.DataFrame(input_data)
        # Get the prediction
        result = f_result(input_df, model_1)  # Assuming model_1 is defined in main.py
    st.success('The output is {}'.format(result))
    # File uploader for CSV file
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV file
        input_df = pd.read_csv(uploaded_file)
        # Get the prediction
        result = f_result(input_df, model_1)
        # Display the predictions
        st.write(result)
        # Create a DataFrame with predictions
        predictions_df = pd.DataFrame({'case_id': input_df['case_id'], 'Predicted_Stay': result})
        # Download link for predictions CSV
        st.markdown(get_download_link(predictions_df), unsafe_allow_html=True)
    if st.button("About"):
        st.text("Built with Streamlit")
        st.text("Hospital")
def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

if __name__ == '__main__':
    main()
