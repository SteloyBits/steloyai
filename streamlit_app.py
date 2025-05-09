import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="SteloyAI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state if not present
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Add the project root to the path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

# Check PyTorch installation and CUDA availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
except ImportError:
    st.error("PyTorch is not installed. Please install it using: pip install torch")
    sys.exit(1)
except Exception as e:
    st.error(f"Error initializing PyTorch: {e}")
    sys.exit(1)

# Import modules from the project
try:
    from src.tasks.text_generation import TextGenerator
    from src.models.hub_interface import HuggingFaceHubInterface
    from src.tasks.text_generation import summarize_text
    from src.tasks.text_generation import analyze_sentiment
    from src.tasks.text_generation import extract_entities
    from src.tasks.image_processing import generate_image
    from src.tasks.image_processing import caption_image
    from src.utils.config import load_config
    from src.utils.model_verification import get_verified_config
    from src.utils.prompt_manager import PromptManager
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.info("Make sure you run this app from the project root directory")
    sys.exit(1)

# Initialize session state for chat management
if 'prompt_manager' not in st.session_state:
    history_dir = os.path.join(os.path.dirname(__file__), "chat_history")
    st.session_state.prompt_manager = PromptManager(history_dir=history_dir)
    if not st.session_state.prompt_manager.current_session:
        st.session_state.prompt_manager.create_new_session()

# Load configuration with verified models
try:
    config_path = os.path.join(os.path.dirname(__file__), "config", "app_config.yaml")
    config = get_verified_config()
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    sys.exit(1)

# Initialize text generator with prompt manager
try:
    tg = TextGenerator(HuggingFaceHubInterface(), st.session_state.prompt_manager)
except Exception as e:
    st.error(f"Failed to initialize text generator: {e}")
    sys.exit(1)

# Add device information to session state
if 'device' not in st.session_state:
    st.session_state.device = DEVICE

# Add custom CSS with theme support and animated toggle
st.markdown(f"""
    <style>
    /* Theme variables */
    :root {{
        --primary-color: #1E88E5;
        --secondary-color: #64B5F6;
        --background-color: {'#F5F7FA' if st.session_state.theme == 'light' else '#1E1E1E'};
        --text-color: {'#2C3E50' if st.session_state.theme == 'light' else '#FFFFFF'};
        --card-background: {'#FFFFFF' if st.session_state.theme == 'light' else '#2D2D2D'};
        --border-color: {'#E0E0E0' if st.session_state.theme == 'light' else '#404040'};
        --hover-color: {'#F0F0F0' if st.session_state.theme == 'light' else '#3D3D3D'};
    }}
    
    /* Theme toggle container */
    .theme-toggle-container {{
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 999;
    }}
    
    /* Theme toggle button */
    .theme-toggle {{
        width: 60px;
        height: 30px;
        background: var(--card-background);
        border-radius: 15px;
        padding: 5px;
        position: relative;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    
    .theme-toggle:hover {{
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    /* Toggle icons */
    .theme-toggle i {{
        font-size: 16px;
        transition: all 0.3s ease;
    }}
    
    .theme-toggle .moon {{
        color: #f1c40f;
        transform: {'translateX(0)' if st.session_state.theme == 'light' else 'translateX(30px)'};
        opacity: {'1' if st.session_state.theme == 'light' else '0'};
    }}
    
    .theme-toggle .sun {{
        color: #f39c12;
        transform: {'translateX(0)' if st.session_state.theme == 'dark' else 'translateX(-30px)'};
        opacity: {'1' if st.session_state.theme == 'dark' else '0'};
    }}
    
    /* Toggle slider */
    .theme-toggle::before {{
        content: '';
        position: absolute;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: var(--primary-color);
        left: {'3px' if st.session_state.theme == 'light' else '33px'};
        transition: all 0.3s ease;
    }}
    
    /* Global styles */
    .stApp {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: var(--card-background);
        padding: 1rem;
        border-right: 1px solid var(--border-color);
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: var(--text-color);
        font-weight: 600;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: var(--secondary-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Select boxes */
    .stSelectbox {{
        background-color: var(--card-background);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }}
    
    /* Text areas */
    .stTextArea>div>div>textarea {{
        background-color: var(--card-background);
        color: var(--text-color);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }}
    
    /* Slider styling */
    .stSlider > div > div > div {{
        background-color: var(--primary-color);
    }}
    
    /* Slider value styling */
    .stSlider > div > div > div > div {{
        background-color: transparent !important;
        color: var(--text-color) !important;
        font-weight: 500;
    }}
    
    /* Slider track styling */
    .stSlider > div > div > div > div > div {{
        background-color: var(--primary-color) !important;
    }}
    
    /* Slider thumb styling */
    .stSlider > div > div > div > div > div > div {{
        background-color: var(--primary-color) !important;
        border: 2px solid var(--card-background) !important;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: var(--card-background);
        color: var(--text-color);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }}
    
    /* Metrics */
    .stMetric {{
        background-color: var(--card-background);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* File uploader */
    .stFileUploader>div>div {{
        background-color: var(--card-background);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }}
    
    /* Success/Error messages */
    .stSuccess, .stError {{
        border-radius: 8px;
        padding: 1rem;
    }}
    
    /* Chat messages */
    .chat-message {{
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        background-color: var(--card-background);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* Custom container for better spacing */
    .custom-container {{
        padding: 2rem;
        background-color: var(--card-background);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }}
    </style>
    
    <!-- Theme toggle button -->
    <div class="theme-toggle-container">
        <div class="theme-toggle" onclick="document.querySelector('#theme-toggle').click()">
            <i class="moon">🌙</i>
            <i class="sun">☀️</i>
        </div>
    </div>
""", unsafe_allow_html=True)

# Add hidden button for theme toggle
st.markdown('<div style="display: none;"><button id="theme-toggle"></button></div>', unsafe_allow_html=True)

def toggle_theme():
    """Toggle between light and dark theme"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

def set_page_config():
    """Configure the Streamlit page"""
    st.sidebar.title("🤖 SteloyAI Platform")
    
    # Add JavaScript for theme toggle
    st.markdown("""
        <script>
        document.getElementById('theme-toggle').addEventListener('click', function() {
            // The actual toggle is handled by the Python function
            // This is just for the UI interaction
        });
        </script>
    """, unsafe_allow_html=True)

def render_chat_management():
    """Render chat management section in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Management")
    
    # Search functionality with custom styling
    search_query = st.sidebar.text_input("🔍 Search chats", "")
    if search_query:
        sessions = st.session_state.prompt_manager.search_sessions(search_query)
    else:
        sessions = st.session_state.prompt_manager.get_session_list()
    
    # New chat and clear buttons with custom styling
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("✨ New Chat", use_container_width=True):
            st.session_state.prompt_manager.create_new_session()
            st.rerun()
    with col2:
        if st.button("🗑️", help="Clear messages in current chat"):
            if st.session_state.prompt_manager.clear_current_session():
                st.rerun()
    
    # List of existing chats with custom styling
    st.sidebar.markdown("### 📚 Recent Chats")
    
    if not sessions:
        st.sidebar.info("No chat history available")
        return
        
    # Sort sessions by last updated
    sessions.sort(key=lambda x: x['last_updated'], reverse=True)
    
    # Display each session with custom styling
    for session in sessions:
        with st.sidebar.expander(f"💭 {session['title']} ({session['message_count']} messages)"):
            # Format the title with date
            last_updated = datetime.fromisoformat(session['last_updated'])
            created_at = datetime.fromisoformat(session['created_at'])
            
            # Display session info with custom styling
            st.markdown(f"""
                <div style='color: #666; font-size: 0.9em;'>
                    📅 Created: {created_at.strftime('%Y-%m-%d %H:%M')}<br>
                    🔄 Last updated: {last_updated.strftime('%Y-%m-%d %H:%M')}
                </div>
            """, unsafe_allow_html=True)
            
            # Get session stats
            stats = st.session_state.prompt_manager.get_session_stats(session['id'])
            if stats:
                st.markdown(f"""
                    <div style='color: #666; font-size: 0.9em;'>
                        📊 Total messages: {stats['total_messages']}<br>
                        👤 User messages: {stats['user_messages']}<br>
                        🤖 Assistant messages: {stats['assistant_messages']}
                    </div>
                """, unsafe_allow_html=True)
            
            # Session actions with custom styling
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button("🔄 Switch", key=f"switch_{session['id']}", use_container_width=True):
                    st.session_state.prompt_manager.switch_session(session['id'])
                    st.rerun()
            
            with col2:
                new_title = st.text_input("✏️ New title", session['title'], key=f"title_{session['id']}")
                if new_title != session['title']:
                    st.session_state.prompt_manager.update_session_title(session['id'], new_title)
                    st.rerun()
            
            with col3:
                if st.button("🗑️", key=f"delete_{session['id']}", use_container_width=True):
                    st.session_state.prompt_manager.delete_session(session['id'])
                    st.rerun()

def render_nlp_section():
    """Render the NLP section of the app"""
    st.header("Natural Language Processing")
    
    nlp_task = st.selectbox(
        "Select NLP Task",
        ["Text Generation", "Text Summarization", "Sentiment Analysis", "Named Entity Recognition"]
    )
    
    if nlp_task == "Text Generation":
        st.subheader("Text Generation")
        
        # Display current chat title
        if st.session_state.prompt_manager.current_session:
            st.markdown(f"### Current Chat: {st.session_state.prompt_manager.current_session.title}")
        
        prompt = st.text_area("Enter your prompt:", height=150)
        
        # Check if there are available models
        if not config["text_generation"]["available_models"]:
            st.error("No text generation models are currently available. Please try again later.")
            return
            
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Select Model", config["text_generation"]["available_models"])
            max_length = st.slider("Maximum Length", min_value=10, max_value=1000, value=200)
        with col2:
            temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
            style = st.selectbox("Response Style", ["default", "einstein", "scholar"])
        
        if st.button("Generate Text"):
            if prompt:
                with st.spinner("Generating text..."):
                    try:
                        generated_text = tg.generate(
                            prompt=prompt,
                            model_id=model,
                            max_length=max_length,
                            temperature=temperature,
                            style=style
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
    
    # Add theme toggle button handler
    if st.session_state.get('_theme_toggle_clicked', False):
        toggle_theme()
        st.session_state._theme_toggle_clicked = False
    
    render_chat_management()
    
    # Navigation in sidebar with custom styling
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📱 Navigation",
        ["NLP", "Computer Vision", "About"],
        label_visibility="collapsed"
    )
    
    # Add model info to sidebar with custom styling
    with st.sidebar.expander("ℹ️ Model Information"):
        st.info("""
            This application uses various AI models for different tasks. 
            Select a task to see available models.
        """)
    
    # Add GitHub link to sidebar with custom styling
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='text-align: center;'>
            <a href='https://github.com/steloybits/steloyai' target='_blank'>
                <img src='https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?style=for-the-badge&logo=github'/>
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    # Display the appropriate section based on selection
    if page == "NLP":
        render_nlp_section()
    elif page == "Computer Vision":
        render_vision_section()
    else:
        render_about_section()

if __name__ == "__main__":
    main()
