import pickle
import string

import nltk
import streamlit as st
import time
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download('stopwords')

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
# Define function to make prediction
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)
def predict(text):
    # Insert your prediction function here
    transformed_sms=transform_text(text)
    vector_input=tfidf.transform([transformed_sms])
    result=model.predict(vector_input)[0]
    return result

# Define Streamlit app
def app():
    # Set page title
    st.set_page_config(page_title="My Streamlit App")

    # Set page header
    # st.header("Text Prediction App")
    st.markdown("<h1 style='color: #ff6600;'>Email/SMS Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 5px solid #ff6600;'>", unsafe_allow_html=True)

    # Add text input
    text = st.text_input("Enter your text:")

    # Add prediction button with animation
    if st.button("Predict"):
        with st.spinner("Making prediction..."):
            time.sleep(2) # Replace this with your actual prediction function
            prediction = predict(text)

            # Display prediction result with animation
            # st.write(prediction, " is your prediction!")
            if prediction==1:
                # st.header("Spam")
                st.markdown("<h2 style='text-align: center; color: #ff6600;'>Spam</h2>", unsafe_allow_html=True)
                st.warning("This is a danger alert!")
            else:
                st.markdown("<h2 style='text-align: center; color: #4BB543 ;'>Not Spam</h2>", unsafe_allow_html=True)
                st.balloons()


if __name__ == '__main__':
    app()
