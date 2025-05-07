import streamlit as st
import os
import sys
from pathlib import Path

# Add the project root to the path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

# Import modules from the project
try:
    from src.tasks.text_generation import TextGenerator
    from src.models.hub_interface import HuggingFaceHubInterface
    #from src.tasks.text_generation import summarize_text
    #from src.tasks.text_generation import analyze_sentiment
    #from src.tasks.text_generation import extract_entities
    #from src.tasks.image_generation import generate_image
    #from src.tasks.image_captioning import caption_image
    from src.utils.config import load_config
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.info("Make sure you run this app from the project root directory")
    sys.exit(1)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "config", "app_config.yaml")
default_config = {
    "text_generation": {
        "available_models": ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"]
    },
    "text_summarization": {
        "available_models": ["facebook/bart-large-cnn", "t5-base"]
    },
    "sentiment_analysis": {
        "available_models": ["facebook/bart-large-mnli", "roberta-large-mnli"]
    },
    "ner": {
        "available_models": ["dbmdz/bert-large-cased-finetuned-conll03-english"]
    },
    "image_generation": {
        "available_models": ["stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5"]
    },
    "image_captioning": {
        "available_models": ["Salesforce/blip-image-captioning-base"]
    },
    "object_detection": {
        "available_models": ["facebook/detr-resnet-50"]
    }
}
config = load_config(config_path, default_config)

tg = TextGenerator(HuggingFaceHubInterface())
    
def set_page_config():
    """Configure the Streamlit page"""
    st.set_page_config(
        page_title="SteloyAI Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("SteloyAI Platform")
    st.sidebar.markdown("Your unified AI platform for NLP and vision tasks")

def render_nlp_section():
    """Render the NLP section of the app"""
    st.header("Natural Language Processing")
    
    nlp_task = st.selectbox(
        "Select NLP Task",
        ["Text Generation", "Text Summarization", "Sentiment Analysis", "Named Entity Recognition"]
    )
    
    if nlp_task == "Text Generation":
        st.subheader("Text Generation")
        prompt = st.text_area("Enter your prompt:", height=150)
        model = st.selectbox("Select Model", config["text_generation"]["available_models"])
        max_length = st.slider("Maximum Length", min_value=10, max_value=1000, value=200)
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        
        if st.button("Generate Text"):
            if prompt:
                with st.spinner("Generating text..."):
                    try:
                        generated_text = tg.generate(
                            prompt=prompt,
                            model_id=model,
                            max_length=max_length,
                            temperature=temperature
                        )
                        st.success("Text Generated Successfully!")
                        st.markdown("### Generated Text")
                        st.write(generated_text)
                    except Exception as e:
                        st.error(f"Error generating text: {e}")
            else:
                st.warning("Please enter a prompt.")
    
    elif nlp_task == "Text Summarization":
        st.subheader("Text Summarization")
        text_to_summarize = st.text_area("Enter text to summarize:", height=250)
        model = st.selectbox("Select Model", config["text_summarization"]["available_models"])
        max_length = st.slider("Maximum Summary Length", min_value=10, max_value=500, value=100)
        
        if st.button("Summarize Text"):
            if text_to_summarize:
                with st.spinner("Summarizing text..."):
                    try:
                        summary = summarize_text(
                            text=text_to_summarize,
                            model=model,
                            max_length=max_length
                        )
                        st.success("Text Summarized Successfully!")
                        st.markdown("### Summary")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Error summarizing text: {e}")
            else:
                st.warning("Please enter text to summarize.")
    
    elif nlp_task == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        text_for_sentiment = st.text_area("Enter text for sentiment analysis:", height=150)
        model = st.selectbox("Select Model", config["sentiment_analysis"]["available_models"])
        
        if st.button("Analyze Sentiment"):
            if text_for_sentiment:
                with st.spinner("Analyzing sentiment..."):
                    try:
                        sentiment_result = analyze_sentiment(
                            text=text_for_sentiment,
                            model=model
                        )
                        st.success("Sentiment Analysis Completed!")
                        
                        # Display results in a more visual way
                        st.markdown("### Sentiment Analysis Results")
                        
                        # Create columns for better layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Sentiment", sentiment_result["sentiment"])
                        
                        with col2:
                            st.metric("Confidence", f"{sentiment_result['confidence']:.2f}")
                        
                        # Show detailed breakdown if available
                        if "details" in sentiment_result:
                            st.subheader("Detailed Breakdown")
                            st.json(sentiment_result["details"])
                    except Exception as e:
                        st.error(f"Error analyzing sentiment: {e}")
            else:
                st.warning("Please enter text for sentiment analysis.")
    
    elif nlp_task == "Named Entity Recognition":
        st.subheader("Named Entity Recognition")
        text_for_ner = st.text_area("Enter text for entity extraction:", height=150)
        model = st.selectbox("Select Model", config["ner"]["available_models"])
        
        if st.button("Extract Entities"):
            if text_for_ner:
                with st.spinner("Extracting entities..."):
                    try:
                        entities = extract_entities(
                            text=text_for_ner,
                            model=model
                        )
                        st.success("Entity Extraction Completed!")
                        
                        st.markdown("### Extracted Entities")
                        
                        # Group entities by type
                        entity_types = {}
                        for entity in entities:
                            entity_type = entity["type"]
                            if entity_type not in entity_types:
                                entity_types[entity_type] = []
                            entity_types[entity_type].append(entity)
                        
                        # Display entities by type
                        for entity_type, entities_of_type in entity_types.items():
                            with st.expander(f"{entity_type} ({len(entities_of_type)})"):
                                for entity in entities_of_type:
                                    st.write(f"• {entity['text']} ({entity['confidence']:.2f})")
                    except Exception as e:
                        st.error(f"Error extracting entities: {e}")
            else:
                st.warning("Please enter text for entity extraction.")

def render_vision_section():
    """Render the vision section of the app"""
    st.header("Computer Vision")
    
    vision_task = st.selectbox(
        "Select Vision Task",
        ["Image Generation", "Image Captioning", "Object Detection"]
    )
    
    if vision_task == "Image Generation":
        st.subheader("Image Generation")
        prompt = st.text_area("Enter image prompt:", height=100)
        model = st.selectbox("Select Model", config["image_generation"]["available_models"])
        
        col1, col2 = st.columns(2)
        with col1:
            width = st.slider("Width", min_value=256, max_value=1024, value=512, step=64)
        with col2:
            height = st.slider("Height", min_value=256, max_value=1024, value=512, step=64)
        
        guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
        num_inference_steps = st.slider("Inference Steps", min_value=10, max_value=100, value=50, step=5)
        
        if st.button("Generate Image"):
            if prompt:
                with st.spinner("Generating image..."):
                    try:
                        image_path = generate_image(
                            prompt=prompt,
                            model=model,
                            width=width,
                            height=height,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps
                        )
                        st.success("Image Generated Successfully!")
                        st.image(image_path, caption="Generated Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
            else:
                st.warning("Please enter a prompt for image generation.")
    
    elif vision_task == "Image Captioning":
        st.subheader("Image Captioning")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        model = st.selectbox("Select Model", config["image_captioning"]["available_models"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Generate Caption"):
                with st.spinner("Generating caption..."):
                    try:
                        # Save the uploaded file temporarily
                        temp_path = os.path.join("/tmp", uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Generate caption
                        caption = caption_image(
                            image_path=temp_path,
                            model=model
                        )
                        st.success("Caption Generated Successfully!")
                        st.markdown(f"### Caption")
                        st.write(caption)
                        
                        # Remove temporary file
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error generating caption: {e}")
    
    elif vision_task == "Object Detection":
        st.subheader("Object Detection")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        model = st.selectbox("Select Model", config["object_detection"]["available_models"])
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect Objects"):
                with st.spinner("Detecting objects..."):
                    try:
                        # Save the uploaded file temporarily
                        temp_path = os.path.join("/tmp", uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Detect objects
                        result_image_path, detected_objects = detect_objects(
                            image_path=temp_path,
                            model=model,
                            confidence_threshold=confidence_threshold
                        )
                        
                        st.success("Object Detection Completed!")
                        
                        # Display image with bounding boxes
                        st.markdown("### Detection Results")
                        st.image(result_image_path, caption="Detection Results", use_column_width=True)
                        
                        # Display detected objects
                        st.markdown("### Detected Objects")
                        for obj in detected_objects:
                            st.write(f"• {obj['class']} (Confidence: {obj['confidence']:.2f})")
                        
                        # Remove temporary files
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error detecting objects: {e}")

def render_about_section():
    """Render information about the project"""
    st.header("About SteloyAI Platform")
    
    st.markdown("""
    ## SteloyAI Platform
    
    A unified platform for natural language processing and computer vision tasks.
    
    ### Features:
    
    #### NLP Capabilities:
    - Text Generation - Create coherent and contextually relevant text
    - Text Summarization - Extract key information from documents
    - Sentiment Analysis - Understand emotions and opinions in text
    - Named Entity Recognition - Identify and classify entities in text
    
    #### Computer Vision Capabilities:
    - Image Generation - Create images from text descriptions
    - Image Captioning - Generate descriptive captions for images
    - Object Detection - Identify and locate objects within images
    
    ### Getting Started
    
    Check out the [GitHub repository](https://github.com/steloybits/steloyai) for detailed documentation and setup instructions.
    """)

def main():
    set_page_config()
    
    # Navigation in sidebar
    page = st.sidebar.radio(
        "Navigate",
        ["NLP", "Computer Vision", "About"]
    )
    
    # Add model info to sidebar
    with st.sidebar.expander("Model Information"):
        st.info("This application uses various AI models for different tasks. Select a task to see available models.")
    
    # Add GitHub link to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("[View on GitHub](https://github.com/steloybits/steloyai)")
    
    # Display the appropriate section based on selection
    if page == "NLP":
        render_nlp_section()
    elif page == "Computer Vision":
        render_vision_section()
    else:
        render_about_section()

if __name__ == "__main__":
    main()
