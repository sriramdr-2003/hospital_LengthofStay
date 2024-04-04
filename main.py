import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost
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
