
import requests
import base64
import time
import os
import pickle
import joblib
import keras_ocr
import easyocr
import torch
import cv2
import re
import string
import nltk
import mysql.connector
import subprocess
import pytesseract
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from streamlit_option_menu import option_menu
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from folium.plugins import MarkerCluster
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from flask import Flask, request, jsonify


#  ------------------------------------------ Streamlit Part ---------------------------------------------------------

st.set_page_config(layout= "wide")

st.markdown(
    f""" <style>.stApp {{
                    background:url("https://wallpapers.com/images/high/dark-purple-and-black-plain-75znhgkjjxu552fr.webp");
                    background-size:cover}}
                 </style>""",
    unsafe_allow_html=True
)

st.title(":green[**E-commerce-prediction-and-recommendation**]")


with st.sidebar:

    selected_page= option_menu("MENU",["Home", "DataFrame", "EDA", "Prediction", "Image-Analysis", "NLP-Preprocessing", "Recommendations"],
                        icons=["house", "list-task", "graph-up", "calculator", "image", "book", "film"],
                        menu_icon="cast",
                        default_index=0)

#  ------------------------------------------ Home ---------------------------------------------------------


if selected_page == "Home":
            
            st.divider()

            st.markdown("""
                        ## :blue[Project Overview]

                        - **DataFrame Analysis:** Delve into detailed insights and outcomes derived from our ML models.
                        View a comprehensive breakdown of model performances, encompassing accuracy, precision, recall, and F1-score, neatly presented in a tabular format. 
                        - **Exploratory Data Analysis (EDA):** Embark on a visual expedition through our dataset, uncovering concealed patterns and trends via interactive plots and charts.
                        - **Prediction Engine:** Witness the potency of predictive analytics as we anticipate the likelihood of visitor conversion with real-time prediction probabilities.
                        
                        **:blue[Image Analysis:]**
                        
                        - Employ a spectrum of image preprocessing methods.
                        - Extract and exhibit text from images for further examination.


                        **:blue[NLP Text Preprocessing:]**
                        
                        - Observe the metamorphosis of raw text data through techniques such as stemming and lowercase conversion.
                        - Explore sentiment analysis with vibrant bar chart visualizations, delivering valuable insights into customer emotions.
                        - Immerse yourself in word cloud visualization to gain a comprehensive view of your textual data.

                        **:blue[Recommendations System:]**

                        - Uncover universal recommendations developed from bespoke movie datasets.
                        - Receive tailored suggestions personalized to individual user tastes.
                        """)


def generate_model_table(model_name, accuracy, precision, recall, f1):

    model_table = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })

    return model_table

#  ------------------------------------------ DataFrame ---------------------------------------------------------


if selected_page == "DataFrame":

    st.divider()

    st.title(":blue[DataFrame]")
    st.write("")

    path = "C:/Users/HP/OneDrive/Desktop/Final Project/classification_data.csv"
    df = pd.read_csv(path)
    st.dataframe(df)

    # Placeholder values, replace with actual metrics

    model_metrics = [
        ('Random Forest', 0.9932, 0.9955, 0.9918, 0.9936),
        ('Logistic Regression', 0.7546, 0.7313, 0.8549, 0.7883),
        ('Support Vector Machine', 0.7225, 0.7336, 0.7550, 0.7441),
        ('K-Nearest Neighbors', 0.9686, 0.9671, 0.9745, 0.9708),
        ('Decision Tree', 0.9898, 0.9914, 0.9896, 0.9905)
    ]

    # Create a list to store DataFrames for each model

    model_tables = []

    # Generate model tables and store in the list

    for model_name, accuracy, precision, recall, f1 in model_metrics:
        model_table = generate_model_table(model_name, accuracy, precision, recall, f1)
        model_tables.append(model_table)

    # Concatenate DataFrames for all models
        
    df_results = pd.concat(model_tables, ignore_index=True)

    # Display the DataFrame using Streamlit

    st.divider()

    st.title(":red[Algorithm results]")
    st.dataframe(df_results)

    st.divider()

#  ------------------------------------------ EDA ---------------------------------------------------------

if selected_page == "EDA":

    st.title(":blue[Exploratory Data Analysis (EDA)]")

    st.divider()

    # Load the CSV file into a DataFrame

    path = "C:/Users/HP/OneDrive/Desktop/Final Project/classification_data.csv"
    df = pd.read_csv(path)

    # Define columns

    col1, col2, col3 = st.columns(3)

    # Visualizations in col

    with col1:

        # Histogram for 'count_session'

        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.histplot(df['count_session'], bins=30, kde=True, color='blue', ax=ax1)
        plt.title('Distribution of count_session')
        plt.xlabel('count_session')
        plt.ylabel('Frequency')
        st.pyplot(fig1)

        # Count plot for 'channelGrouping'

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.countplot(x='channelGrouping', data=df, palette='viridis', ax=ax2)
        plt.title('Count of channelGrouping')
        plt.xlabel('channelGrouping')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # Visualizations in col2

    with col2:

        # Bar chart for total visitors vs. buyed visitors
        
        total_visitors = df['has_converted'].count()
        buyed_visitors = df['has_converted'].sum()

        fig3, ax3 = plt.subplots(figsize=(10, 8.4))
        sns.barplot(x=['Total Visitors', 'Buyed Visitors'], y=[total_visitors, buyed_visitors], palette='viridis', ax=ax3)
        ax3.set_ylabel('Count')
        ax3.set_title('Total Visitors vs. Buyed Visitors')

        # Show the chart

        st.pyplot(fig3)

        # Scatter Plot

        fig4, ax4 = plt.subplots(figsize=(10, 8.4))
        sns.scatterplot(x='time_on_site', y='transactionRevenue', data=df, ax=ax4)
        plt.title('Scatter Plot: time_on_site vs. transactionRevenue')
        plt.xlabel('Time on Site')
        plt.ylabel('Transaction Revenue')
        st.pyplot(fig4)

    # Visualizations in col3

    with col3:

        # Pie Chart

        fig5, ax5 = plt.subplots(figsize=(3.6, 5))
        df['has_converted'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Pastel1'))
        plt.title('Distribution of Sessions by Conversion Status', y=1.02)
        st.pyplot(fig5)


        # Boxplots for numerical variables

        fig6, ax6 = plt.subplots(figsize=(5.6, 4))
        sns.boxplot(x='time_on_site', data=df, ax=ax6)
        plt.title('Boxplot: time_on_site')
        st.pyplot(fig6)

    st.divider()



bin_mapping = {
    'avg_session_time_binned': {'low': [0, 500], 'medium': [500, 1000], 'high': [1000, np.inf]},
    'count_hit_binned': {'low': [0, 100], 'medium': [100, 200], 'high': [200, np.inf]},
    'num_interactions_binned': {'low': [0, 1000], 'medium': [1000, 5000], 'high': [5000, np.inf]}
}

avg_session_time_options = list(bin_mapping['avg_session_time_binned'].keys())
count_hit_options = list(bin_mapping['count_hit_binned'].keys())
num_interactions_options = list(bin_mapping['num_interactions_binned'].keys())

df1 = pd.read_csv('C:/Users/HP/OneDrive/Desktop/Final Project/df1.csv')  

sessionQualityDim_options = df1['sessionQualityDim'].unique().tolist()
single_page_rate_options = df1['single_page_rate'].unique().tolist()


def predict(model, features):

    # Separate numerical and categorical columns

    numerical_cols = ['sessionQualityDim', 'single_page_rate']
    categorical_cols = ['avg_session_time', 'count_hit', 'num_interactions']
    
    # Preprocess numerical columns

    numerical_features = features[numerical_cols]
    
    # Preprocess categorical columns

    categorical_features = features[categorical_cols]

    # Perform one-hot encoding

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features_encoded = one_hot_encoder.fit_transform(categorical_features)
    
    # Combine preprocessed numerical and categorical features

    processed_features = np.hstack((numerical_features, categorical_features_encoded.toarray()))

    # Impute missing values

    imputer = SimpleImputer(strategy='mean')  
    processed_features = imputer.fit_transform(processed_features)

    # Predict using the model

    prediction = model.predict(processed_features)
    prediction_proba = model.predict_proba(processed_features)[:, 1]  

    return prediction, prediction_proba

def load_model(filename):

    with open(filename, 'rb') as file:

        model = pickle.load(file)

    return model

model = load_model('C:/Users/HP/OneDrive/Desktop/Final Project/random_forest_model.pkl')


#  ------------------------------------------ Prediction ---------------------------------------------------------

if selected_page == "Prediction":


    st.title(":blue[Prediction]")

    selected_num_interactions = st.selectbox(':green[Select Number of Interactions]', options=num_interactions_options)
    selected_avg_session_time = st.selectbox(':green[Select Average Session Time]', options=avg_session_time_options)
    selected_sessionQualityDim = st.selectbox(':green[Select Session Quality Dimension]', options=sessionQualityDim_options)
    selected_count_hit = st.selectbox(':green[Select Count Hit]', options=count_hit_options)
    selected_single_page_rate = st.selectbox(':green[Select Single Page Rate]', options=single_page_rate_options)

    if st.button(':orange[Predict]'):

        features = np.array([[selected_sessionQualityDim, selected_avg_session_time, selected_single_page_rate, selected_count_hit, selected_num_interactions]])
        
        feature_names = ['sessionQualityDim', 'avg_session_time', 'single_page_rate', 'count_hit', 'num_interactions']

        features_df = pd.DataFrame(features, columns=feature_names)

        prediction, prediction_proba = predict(model, features_df)

        # Display prediction results
        
        with st.container():

            st.write(':blue[Prediction:]', ":green[Yes]" if prediction == 1 else ":red[No]")
            st.write(':orange[Prediction Probability:]', prediction_proba)

            if prediction == 1:

                st.balloons()
        

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'
# C:\Program Files\Tesseract-OCR\tesseract.exe


def preprocess_resize(original_image, target_size=(200, 200)):

    resized_image = original_image.resize(target_size)

    return resized_image

def preprocess_contrast(original_image, factor=2.0):

    contrast_image = ImageEnhance.Contrast(original_image).enhance(factor)

    return contrast_image

def preprocess_brightness(original_image, factor=1.5):

    brightness_image = ImageEnhance.Brightness(original_image).enhance(factor)

    return brightness_image

def preprocess_rotation(original_image, rotation_angle=45):

    rotated_image = ImageOps.exif_transpose(original_image.rotate(rotation_angle))

    return rotated_image

def preprocess_flip(original_image, flip_type="horizontal"):

    if flip_type == "horizontal":

        flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

    elif flip_type == "vertical":

        flipped_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)

    else:

        flipped_image = original_image

    return flipped_image

def preprocess_crop(original_image, coordinates=(50, 50, 150, 150)):

    cropped_image = original_image.crop(coordinates)

    return cropped_image

def preprocess_grayscale(original_image):

    grayscale_image = original_image.convert("L")

    return grayscale_image

def preprocess_edge_detection(original_image):

    img_array = np.array(original_image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    processed_image = ImageOps.grayscale(Image.fromarray(edges))

    return processed_image

def preprocess_text_extraction(original_image):

    img_array = np.array(original_image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    extracted_text = pytesseract.image_to_string(gray_image)

    return extracted_text

def preprocess_color_space_conversion(original_image, color_space="HSV"):

    if color_space == "HSV":

        converted_image = original_image.convert("HSV")
    else:

        converted_image = original_image.convert("RGB")

    if color_space == "HSV":

        converted_image = converted_image.convert("RGB")

    return converted_image

def preprocess_histogram_equalization(original_image):

    if original_image.mode == 'RGBA':

        original_image = original_image.convert('RGB')

    equalized_image = ImageOps.equalize(original_image)

    return equalized_image

def preprocess_image_filtering(original_image, filter_type="gaussian"):

    if filter_type == "gaussian":

        filtered_image = original_image.filter(ImageFilter.GaussianBlur(radius=2))

    elif filter_type == "median":

        filtered_image = original_image.filter(ImageFilter.MedianFilter(size=3))

    else:

        filtered_image = original_image

    return filtered_image

def preprocess_text_extraction(original_image):
     
     img_array = np.array(original_image)
     gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
     extracted_text = pytesseract.image_to_string(gray_image)

     if not extracted_text:
         
         return "No text found."

     return extracted_text

    
#  ------------------------------------------ Image-Analysis ---------------------------------------------------------


if selected_page == "Image-Analysis":

    st.divider()

    st.title(":blue[Image-Analysis]")

    uploaded_image = st.file_uploader(":green[Upload Image]", type=["png", "jpeg", "jpg"])

    if uploaded_image is not None:

        original_image = Image.open(uploaded_image)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(original_image, caption="Original Image", use_column_width=True)

        processed_resized_image = preprocess_resize(original_image)
        processed_contrast_image = preprocess_contrast(original_image, factor=2.0)
        processed_brightness_image = preprocess_brightness(original_image, factor=1.5)
        processed_rotation_image = preprocess_rotation(original_image, rotation_angle=45)
        processed_flip_horizontal_image = preprocess_flip(original_image, flip_type="horizontal")
        processed_crop_image = preprocess_crop(original_image, coordinates=(50, 50, 150, 150))
        processed_grayscale_image = preprocess_grayscale(original_image)
        processed_edge_image = preprocess_edge_detection(original_image)
        processed_color_space_image = preprocess_color_space_conversion(original_image, color_space="HSV")
        processed_equalized_image = preprocess_histogram_equalization(original_image)
        processed_filtered_image = preprocess_image_filtering(original_image, filter_type="gaussian")


        with col1:

            st.image(processed_resized_image, caption="Resized Image", use_column_width=True)
            st.image(processed_crop_image, caption="Cropped Image", use_column_width=True)
            st.image(processed_grayscale_image, caption="Grayscale Conversion Result", use_column_width=True)

        with col2:

            st.image(processed_contrast_image, caption="Contrast Adjustment Result", use_column_width=True)
            st.image(processed_brightness_image, caption="Brightness Adjustment Result", use_column_width=True)
            st.image(processed_color_space_image, caption="Color Space Conversion Result", use_column_width=True)
            st.image(processed_equalized_image, caption="Histogram Equalization Result", use_column_width=True)

        with col3:

            st.image(processed_rotation_image, caption="Rotation Result", use_column_width=True)
            st.image(processed_flip_horizontal_image, caption="Horizontal Flip Result", use_column_width=True)
            st.image(processed_edge_image, caption="Edge Detection Result", use_column_width=True)
            st.image(processed_filtered_image, caption="Image Filtering Result", use_column_width=True)
        
        

        extracted_text = preprocess_text_extraction(original_image)

        st.write(":green[Extracted Text:]")
        st.write(extracted_text)

        st.divider()


#  ------------------------------------------ NLP-Preprocessing ---------------------------------------------------------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')


# NLP preprocessing functions

def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]

    return ' '.join(filtered_text)

def perform_stemming(text):

    ps = PorterStemmer()
    words = word_tokenize(text)
    stemmed_text = [ps.stem(word) for word in words]

    return ' '.join(stemmed_text)

def perform_lemmatization(text):

    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(lemmatized_text)

def remove_punctuation_special_chars(text):

    return ' '.join(char for char in text if char not in string.punctuation)

def remove_numbers(text):

    return ' '.join(char for char in text if not char.isdigit())

def get_part_of_speech(text):

    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    return pos_tags

def perform_sentiment_analysis(text):

    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)

    if sentiment_score['compound'] >= 0.05:

        sentiment = "Positive"

    elif sentiment_score['compound'] <= -0.05:

        sentiment = "Negative"

    else:

        sentiment = "Neutral"

    return sentiment, sentiment_score

def generate_sentiment_bar_chart(sentiment_score):

    fig, ax = plt.subplots()

    # Calculate y-values for the bar chart

    positive_value = max(0, sentiment_score['pos'])
    negative_value = max(0, -sentiment_score['neg'])
    neutral_value = max(0, sentiment_score['neu'])
    
    # Specify colors for each bar

    colors = ['green', 'red', 'blue']

    sns.barplot(x=['Positive', 'Negative', 'Neutral'], y=[positive_value, negative_value, neutral_value], palette=colors, ax=ax)
    st.pyplot(fig)

def generate_word_cloud(text):

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot() 

if selected_page == "NLP-Preprocessing":

    st.divider()

    st.title(":blue[NLP-Preprocessing]")

    # Get user input

    user_text = st.text_area(":green[Enter your text:]")

    if st.button(":orange[Process Text]"):

        results = []

        processed_text = user_text
        results.append((':green[Original Text:]', processed_text))

        processed_text = processed_text.lower()
        results.append((':green[Lowercasing:]', processed_text))

        processed_text = remove_stopwords(processed_text)
        results.append((':green[Remove Stopwords:]', processed_text))

        processed_text = perform_stemming(processed_text)
        results.append((':green[Stemming:]', processed_text))

        processed_text = perform_lemmatization(processed_text)
        results.append((':green[Lemmatization:]', processed_text))

        processed_text = remove_punctuation_special_chars(processed_text)
        results.append((':green[Remove Punctuation and Special Characters:]', processed_text))

        processed_text = remove_numbers(processed_text)
        results.append((':green[Remove Numbers:]', processed_text))

        processed_text = get_part_of_speech(processed_text)
        results.append((':green[Port of Speech:]', processed_text))

        # Display results
        for step, result in results:

            st.subheader(step)
            st.write(result)

            st.divider()


    st.title(":blue[Sentiment Analysis]")

    if st.button(":orange[Perform Sentiment Analysis]"):

        sentiment, sentiment_score = perform_sentiment_analysis(user_text)

        # Display results

        st.subheader(":blue[Sentiment:]")
        st.write(sentiment)

        st.subheader(":orange[Sentiment Scores:]")
        st.write(f":green[Positive:] {sentiment_score['pos']:.2f}")
        st.write(f":red[Negative:] {sentiment_score['neg']:.2f}")
        st.write(f"Neutral: {sentiment_score['neu']:.2f}")
        st.write(f":blue[Compound:] {sentiment_score['compound']:.2f}")

        # Generate and display the sentiment bar chart

        generate_sentiment_bar_chart(sentiment_score)


    def generate_word_cloud(text):

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.divider()

    if st.button(":orange[Generate Word Cloud]"):

        generate_word_cloud(user_text) 

    st.divider()


#  ------------------------------------------ Recommendations ---------------------------------------------------------


if selected_page == "Recommendations":

    st.divider()

    st.title(":blue[Recommendations]")

    def recommend(movie):

        index = movies_data[movies_data['title'] == movie].index

        if not index.empty:

            index = index[0]
            distances = sorted(list(enumerate(similarity_data[index])), reverse=True, key=lambda x: x[1])
            recommended_movie_names = []

            for i in distances[1:6]:

                recommended_movie_names.append(movies_data.iloc[i[0]]['title'])

            return recommended_movie_names
        
        else:

            st.warning("Movie not found in the database.")

            return []

    movies_data = pd.DataFrame.from_dict(pickle.load(open('C:/Users/HP/OneDrive/Desktop/Final Project/movies.pkl', 'rb')))
    similarity_data = pickle.load(open('C:/Users/HP/OneDrive/Desktop/Final Project/similarity.pkl', 'rb'))

    st.header(':red[Movie Recommendation System]')

    movie_list = movies_data['title'].values
    selected_movie = st.selectbox(
        ":blue[Type or select a movie from the dropdown]",
        movie_list
    )

    if st.button(':green[Show Recommendation]'):

        recommended_movie_names = recommend(selected_movie)
        st.write(":blue[Recommended Movies:]")

        for movie_name in recommended_movie_names:

            st.write(f"- {movie_name}")

        st.divider()


