import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#load the trained model scaler pickle, onehot encoding
model=tf.keras.models.load_model('model.h5')

## load the encoder and scalers
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## streamlit app
st.title('Customer Churn Predication')

#user input

Geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Has Credit card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
	})

#one hot encode geography
geo_encoded=onehot_encoder_geo.transform([[Geography]]).toarray()


geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


#scaling the input data
input_data_scaled=scaler.transform(input_data)


#predict churn
predication=model.predict(input_data_scaled)
predication_proba=predication[0][0]

st.write(f'Churn Probability: {predication_proba:.2f}') 

if predication_proba>0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")