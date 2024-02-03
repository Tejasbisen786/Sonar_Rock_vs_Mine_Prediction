
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier





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
        
with st.container():
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
        st.subheader('This Object is Rock' )
    else:
        st.subheader('The Object is Mine')
 



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
with st.markdown('<div class="footer">Design and Developed By : Hvpm Final Year Students </div>', unsafe_allow_html=True):
    pass

#bisen_tejas_

#*********Feature v1.1 - Visualization Of Data *************




# ** COlumn Data**


col1, col2 = st.columns(2)

with col1:
   # Load the CSV file
  csv_file_path = './sonar_data.csv'  # Replace with the actual path to your CSV file
  df = pd.read_csv(csv_file_path, header=None, names=['Predicted Values'])

# Sidebar for user input or settings if needed
# For example, you can add filters, sliders, etc. here

# Display the dataset
  st.write("## Original  Dataset")
  st.write(df)
  
with col2:
   # Create visualizations
   # Example 1: Histogram
  st.write("## Histogram of Values")
  plt.figure(figsize=(10, 6))
  sns.histplot(df['Predicted Values'], kde=True)
  st.pyplot()


st.header("*****************Classification ************")

# Classify Data
col3, col4 = st.columns(2)

with col3:
   
# Load the CSV file
  csv_file_path = './sonar_data.csv'  # Replace with the actual path to your CSV file
  df = pd.read_csv(csv_file_path, header=None, names=['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                                                     'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
                                                     'feature11', 'feature12', 'feature13', 'feature14', 'feature15',
                                                     'feature16', 'feature17', 'feature18', 'feature19', 'feature20',
                                                     'feature21', 'feature22', 'feature23', 'feature24', 'feature25',
                                                     'feature26', 'feature27', 'feature28', 'feature29', 'feature30',
                                                     'feature31', 'feature32', 'feature33', 'feature34', 'feature35',
                                                     'feature36', 'feature37', 'feature38', 'feature39', 'feature40',
                                                     'feature41', 'feature42', 'feature43', 'feature44', 'feature45',
                                                     'feature46', 'feature47', 'feature48', 'feature49', 'feature50',
                                                     'feature51', 'feature52', 'feature53', 'feature54', 'feature55',
                                                     'feature56', 'feature57', 'feature58', 'feature59', 'feature60',
                                                     'label'])

  # Display the dataset
  st.write("## Original Dataset")
  st.write(df)


with col4:
  st.write("## Scatter Plot of Features 1 and 2")
  # Visualize the data using a scatter plot
  plt.figure(figsize=(10, 6))
  colors = {'R': 'red', 'M': 'blue'}
  plt.scatter(df['feature1'], df['feature2'], c=df['label'].map(colors))
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  st.pyplot()

  # Split the data into features (X) and labels (y)
  X = df.iloc[:, :-1]
  y = df['label']

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train a Random Forest classifier
  classifier = RandomForestClassifier(random_state=42)
  classifier.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = classifier.predict(X_test)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)

  st.write(f"Accuracy: {accuracy:.2%} |  Blue: Rock | Red : Mine ")



# removing warning from file
st.set_option('deprecation.showPyplotGlobalUse', False)


