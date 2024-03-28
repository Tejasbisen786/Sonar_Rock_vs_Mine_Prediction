import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np



st.set_page_config(
    layout="wide" , 
    page_title="Sonar Rock Vs Mine Prediction "
    )



# ** COlumn Data**


col1, col2 = st.columns(2)

with col1:
   # Load the CSV file
  csv_file_path = '../sonar_data.csv'  # Replace with the actual path to your CSV file
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
  csv_file_path = '../sonar_data.csv'  # Replace with the actual path to your CSV file
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


