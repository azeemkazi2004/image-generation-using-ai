import streamlit as st
import requests

st.set_page_config(page_title="Image Generator", layout="wide")
st.title("bikes Image Generator")

prompt = st.text_input("Enter your prompt", "")

if st.button("Generate Image"):
    if prompt.strip() == "":
        st.error("Please enter a prompt")
    else:
        # Pollinations API URL
        url = "https://image.pollinations.ai/prompt/" + prompt

        # Request the image
        response = requests.get(url)

        if response.status_code == 200:
            st.image(response.content, use_column_width=True)
        else:
            st.error("Error generating image. Try again!")
