
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Get data
solar_data = pd.read_csv('./sonar_data.csv',header=None)
solar_data.head()

#prepare data
solar_data.groupby(60).mean()
solar_data.describe()

# Train test split
X = solar_data.drop(columns=60,axis=1)
y = solar_data[60]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,stratify=y, random_state=1)

# Train and Evualivate model
model = LogisticRegression()
model.fit(X_train,y_train)
training_prediction = model.predict(X_train)
print(accuracy_score(training_prediction,y_train))
test_prediction = model.predict(X_test)
print(accuracy_score(test_prediction,y_test))

# //////////////////////////////////////////////////////////////////////////










# UI CODE  

import streamlit as st
# Create Streamlit App:
#icon 
from PIL import Image
# Loading Image using PIL
im = Image.open('./assets/predictive.png')

#title of application 
st.set_page_config(
    layout="wide" , 
    page_title="Sonar Rock Vs Mine Prediction " ,
     page_icon = im)



# Create a navbar with a logo and a centered headline
st.markdown(
    """
    <style>
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
           background: #34e89e;  /* fallback for old browsers */
background: -webkit-linear-gradient(to right, #0f3443, #34e89e);  /* Chrome 10-25, Safari 5.1-6 */
background: linear-gradient(to right, #0f3443, #34e89e); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
            color: #fff;
        }
        .headline {
            text-align: center;
            color: #FFF;

    </style>
    """
    , unsafe_allow_html=True
)

# Navbar layout

st.markdown(
    """
    <div class="navbar">
        <h1 class="headline">Sonar Rock VS Mine Prediction Using Machine Learning</h1>
        <br>
    </div>
        <br>
        <br>


    """
    , unsafe_allow_html=True
)




# * Center Container
        
with st.container( height=300):
 # Create Streamlit App:
# Text input for user to enter data
 input_data = st.text_input('Enter Comma-Separated Values Here ( NOTE: Expecting 60 features as input.)')
# Predict and show result on button click
 if st.button('Predict'):
    # Prepare input data
    input_data_np_array = np.asarray(input_data.split(','), dtype=float)
    reshaped_input = input_data_np_array.reshape(1, -1)
    # Predict and show result
    prediction = model.predict(reshaped_input)
    if prediction[0] == 'R':
        st.subheader('This Object is Rock' ,divider='rainbow')
    else:
        st.subheader('The Object is Mine' , divider='rainbow')
 



 # Footer
        


# Background color for the footer
footer_color = "#000"  # Black color

# Custom HTML and CSS for the footer
footer_style = f"""
    <style>
        .footer {{
            background: #000428;  /* fallback for old browsers */
background: -webkit-linear-gradient(to right, #004e92, #000428);  /* Chrome 10-25, Safari 5.1-6 */
background: linear-gradient(to right, #004e92, #000428); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */

            color: white;
            padding: 10px;
            position: fixed;
            bottom: 0;
            width: 90vw;
            text-align: center;
        }}
    </style>
"""

# Display the custom HTML
st.markdown(footer_style, unsafe_allow_html=True)

# Your Streamlit app content goes here

# Display the footer
with st.markdown('<div class="footer">Design By : Tejas Bisen  | Developed By : Sohail Akhatar Ali </div>', unsafe_allow_html=True):
    pass

#bisen_tejas_