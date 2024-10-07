import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import gc

# Function to handle image upload/capture
def handle_image_input():
    st.sidebar.title("Choose Image Input")
    input_type = st.sidebar.radio("How would you like to provide the image?", ("Upload Image", "Capture Image"))
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            return Image.open(uploaded_file)
    elif input_type == "Capture Image":
        captured_image = st.camera_input("Capture an image using your webcam")
        if captured_image is not None:
            return Image.open(captured_image)
    return None

# Load models
@st.cache_resource
def load_models():
    return {
        "T5": pipeline("text2text-generation", model="t5-base"),
        "GPT-2": pipeline("text-generation", model="gpt2"),
        "CLIP": pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32"),
        "ViT": pipeline("image-classification", model="google/vit-base-patch16-224"),
        "Sentiment Analysis": pipeline("sentiment-analysis"),
        "Question Answering": pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    }

models = load_models()

# Task: Text Generation using T5
def perform_t5_text_generation():
    user_input = st.text_input("Enter a prompt for text generation (T5):")
    if user_input:
        with st.spinner("Generating text..."):
            result = models["T5"](user_input)
            st.write(f"Generated Text: {result[0]['generated_text']}")

# Task: Text Generation using GPT-2
def perform_gpt2_text_generation():
    user_input = st.text_input("Enter a prompt for text generation (GPT-2):")
    if user_input:
        with st.spinner("Generating text..."):
            result = models["GPT-2"](user_input, max_length=50)
            st.write(f"Generated Text: {result[0]['generated_text']}")

# Task: Image Classification using CLIP
def perform_clip_image_classification():
    uploaded_image = handle_image_input()
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Classifying image..."):
            result = models["CLIP"](uploaded_image, candidate_labels=["a photo of a cat", "a photo of a dog", "a person"])
            st.write(f"Classification Result: {result}")

# Task: Image Classification using ViT
def perform_vit_image_classification():
    uploaded_image = handle_image_input()
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Classifying image..."):
            result = models["ViT"](uploaded_image)
            st.write(f"Classification Result: {result[0]['label']} (Confidence: {result[0]['score']:.2f})")

# Task: Sentiment Analysis
def perform_sentiment_analysis():
    user_input = st.text_input("Enter a sentence for sentiment analysis:")
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            result = models["Sentiment Analysis"](user_input)
            st.write(f"Sentiment: {result[0]['label']} (Confidence: {result[0]['score']:.2f})")

# Task: Question Answering
def perform_question_answering():
    context = st.text_area("Enter context for question answering:")
    question = st.text_input("Enter your question:")
    if context and question:
        with st.spinner("Answering the question..."):
            result = models["Question Answering"](question=question, context=context)
            st.write(f"Answer: {result['answer']}")

# Main app logic
st.title("Multitasking App with Hugging Face Models")

task = st.sidebar.selectbox("Select a task", ("T5 Text Generation", "GPT-2 Text Generation", "CLIP Image Classification", "ViT Image Classification", "Sentiment Analysis", "Question Answering"))


if task == "T5 Text Generation":
    perform_t5_text_generation()
elif task == "GPT-2 Text Generation":
    perform_gpt2_text_generation()
elif task == "CLIP Image Classification":
    perform_clip_image_classification()
elif task == "ViT Image Classification":
    perform_vit_image_classification()
elif task == "Sentiment Analysis":
    perform_sentiment_analysis()
elif task == "Question Answering":
    perform_question_answering()

# Call garbage collection at the end to free up memory
gc.collect()
