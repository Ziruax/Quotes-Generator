import streamlit as st
import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd
import numpy as np
import cv2
import io
import base64
import time
import zipfile
import tempfile
import json
from datetime import datetime
import re
import threading
import queue

# Define paths - use relative paths for Streamlit Cloud compatibility
DATA_DIR = os.path.join("data")
FONTS_DIR = os.path.join("fonts")
IMAGES_DIR = os.path.join("images")
TEMP_DIR = os.path.join("temp")

# Create directories if they don't exist
for directory in [DATA_DIR, FONTS_DIR, IMAGES_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Import custom modules
from image_scraper import ImageScraper
from quote_processor import QuoteProcessor
from text_overlay import TextOverlay
from watermark import WatermarkProcessor
from seo_optimizer import SEOOptimizer
from streamlit_ui import StreamlitUI

# Main application class
class QuoteImageGenerator:
    def __init__(self):
        """Initialize the application components"""
        self.image_scraper = ImageScraper(IMAGES_DIR)
        self.quote_processor = QuoteProcessor(FONTS_DIR)
        self.text_overlay = TextOverlay()
        self.watermark_processor = WatermarkProcessor()
        self.seo_optimizer = SEOOptimizer()
        self.ui = StreamlitUI(self)
        
    def run(self):
        """Run the main application"""
        # Create sidebar for configuration
        self.ui.create_sidebar()
        
        # Create main content area
        self.ui.create_main_content()

# Initialize and run the application
if __name__ == "__main__":
    st.set_page_config(
        page_title="Quote Image Generator",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 75, 75, 0.1);
        border-bottom-color: #FF4B4B;
    }
    .stButton>button {
        border-radius: 4px;
        height: 2.5rem;
    }
    .stTextInput>div>div>input {
        border-radius: 4px;
    }
    .stSelectbox>div>div>div {
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    app = QuoteImageGenerator()
    app.run()
