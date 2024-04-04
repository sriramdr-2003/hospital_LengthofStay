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
    Hospital_type_code = st.text_input("Hospital_type_code ", "Type Here")
    City_Code_Hospital = st.text_input("City_Code_Hospital ", "Type Here")
    Hospital_region_code = st.text_input("Hospital_region_code", "Type Here")
    Available_Extra_Rooms_in_Hospital = st.text_input("Available_Extra_Rooms_in_Hospital", "Type Here")
    Department = st.text_input("Department", "Type Here")
    Ward_Type = st.text_input("Ward_Type", "Type Here")
    Ward_Facility_Code = st.text_input("Ward_Facility_Code", "Type Here")
    Bed_Grade = st.text_input("Bed_Grade", "Type Here")
    patientid = st.text_input("patientid ", "Type Here")
    City_Code_Patient = st.text_input("City_Code_Patient", "Type Here")
    Type_of_Admission = st.text_input("Type_of_Admission", "Type Here")
    Severity_of_illness = st.text_input("Severity_of_illness", "Type Here")
    Visitors_with_Patient = st.text_input("Visitors_with_Patient", "Type Here")
    Age = st.text_input("Age", "Type Here")
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
