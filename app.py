import streamlit as st
from main import f_result, model_1
import pandas as pd


def main():
    st.set_page_config(page_title="Main Project", page_icon="😈", layout="wide")
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
    if st.button("About"):
        st.text("Built with Streamlit")
        st.text("Hospital")


if __name__ == '__main__':
    main()
