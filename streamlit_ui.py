import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
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

class StreamlitUI:
    """
    Class for building the Streamlit UI with real-time preview functionality.
    Enhanced with improved UI/UX features.
    """
    
    def __init__(self, app):
        """
        Initialize the Streamlit UI
        
        Args:
            app: Main application instance
        """
        self.app = app
        
        # Initialize state variables
        if 'current_images' not in st.session_state:
            st.session_state.current_images = []
        if 'current_quotes' not in st.session_state:
            st.session_state.current_quotes = []
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0
        if 'preview_image' not in st.session_state:
            st.session_state.preview_image = None
        if 'text_position' not in st.session_state:
            st.session_state.text_position = None
        if 'watermark_applied' not in st.session_state:
            st.session_state.watermark_applied = False
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = ""
        
    def create_sidebar(self):
        """Create the sidebar with configuration options"""
        with st.sidebar:
            st.header("✨ Configuration")
            
            # Create tabs for better organization
            tab1, tab2, tab3, tab4 = st.tabs(["Images", "Quotes", "Styling", "Export"])
            
            with tab1:
                self._create_image_tab()
                
            with tab2:
                self._create_quote_tab()
                
            with tab3:
                self._create_styling_tab()
                
            with tab4:
                self._create_export_tab()
    
    def _create_image_tab(self):
        """Create the image source tab"""
        st.subheader("Image Source")
        image_source = st.radio(
            "Select Image Source",
            ["Unsplash", "Pexels", "Pixabay", "Upload Your Own"],
            key="image_source",
            horizontal=True
        )
        
        if image_source != "Upload Your Own":
            search_keyword = st.text_input("Search Keyword", "nature", key="search_keyword")
            
            col1, col2 = st.columns(2)
            with col1:
                aspect_ratio = st.selectbox(
                    "Aspect Ratio",
                    ["Landscape (16:9)", "Portrait (9:16)", "Square (1:1)"],
                    key="aspect_ratio"
                )
            with col2:
                num_images = st.slider("Number of Images", 1, 10, 3, key="num_images")
            
            st.button("🔍 Search Images", key="search_button", use_container_width=True, 
                      on_click=self._search_images_callback)
        else:
            uploaded_images = st.file_uploader(
                "Upload Images", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True,
                key="uploaded_images"
            )
            
            if uploaded_images:
                st.button("Process Uploaded Images", key="process_images_button", use_container_width=True,
                         on_click=self._process_uploaded_images_callback, args=(uploaded_images,))
    
    def _create_quote_tab(self):
        """Create the quote input tab"""
        st.subheader("Quote Input")
        quote_input_method = st.radio(
            "Quote Input Method",
            ["Direct Input", "Upload CSV/Excel"],
            key="quote_input_method",
            horizontal=True
        )
        
        if quote_input_method == "Direct Input":
            quote_text = st.text_area(
                "Enter Quotes (one per line)",
                "Success is not final, failure is not fatal: it is the courage to continue that counts. - Winston Churchill",
                key="quote_text",
                height=150
            )
            
            st.button("Process Quotes", key="process_quotes_button", use_container_width=True,
                     on_click=self._process_quotes_text_callback, args=(quote_text,))
        else:
            uploaded_file = st.file_uploader(
                "Upload Quote File",
                type=["csv", "xlsx", "xls"],
                key="quote_file"
            )
            
            if uploaded_file is not None:
                file_type = "csv" if uploaded_file.name.endswith(".csv") else "excel"
                st.button("Process File", key="process_file_button", use_container_width=True,
                         on_click=self._process_quotes_file_callback, args=(uploaded_file, file_type))
        
        # Font Configuration
        st.subheader("Font Settings")
        font_selection = st.selectbox(
            "Font Family",
            ["Default", "Custom"],
            key="font_selection"
        )
        
        if font_selection == "Custom":
            uploaded_font = st.file_uploader(
                "Upload Font",
                type=["ttf", "otf"],
                key="uploaded_font"
            )
            
            if uploaded_font is not None:
                st.button("Add Font", key="add_font_button", use_container_width=True,
                         on_click=self._add_custom_font_callback, args=(uploaded_font,))
    
    def _create_styling_tab(self):
        """Create the styling tab"""
        # Text Positioning
        st.subheader("Text Positioning")
        text_positioning = st.radio(
            "Text Positioning Method",
            ["AI (Automatic)", "Manual"],
            key="text_positioning",
            horizontal=True
        )
        
        # Text Effects
        st.subheader("Text Effects")
        text_effect = st.selectbox(
            "Text Effect",
            ["None", "Outline", "Shadow", "Glow"],
            key="text_effect"
        )
        
        text_color = st.color_picker(
            "Text Color",
            "#FFFFFF",
            key="text_color"
        )
        
        background_blur = st.slider(
            "Background Blur",
            0, 10, 0,
            key="background_blur"
        )
        
        # Watermark Settings
        st.subheader("Watermark")
        use_watermark = st.checkbox(
            "Add Watermark",
            value=False,
            key="use_watermark"
        )
        
        if use_watermark:
            watermark_type = st.radio(
                "Watermark Type",
                ["Image", "Text"],
                key="watermark_type",
                horizontal=True
            )
            
            if watermark_type == "Image":
                watermark_file = st.file_uploader(
                    "Upload Watermark",
                    type=["png", "jpg", "jpeg"],
                    key="watermark_file"
                )
                
                if watermark_file is not None:
                    st.button("Load Watermark", key="load_watermark_button", use_container_width=True,
                             on_click=self._load_watermark_callback, args=(watermark_file,))
            else:
                watermark_text = st.text_input(
                    "Watermark Text",
                    "© Your Brand",
                    key="watermark_text"
                )
                
                st.button("Create Text Watermark", key="create_text_watermark_button", use_container_width=True,
                         on_click=self._create_text_watermark_callback, args=(watermark_text,))
            
            if st.session_state.get('watermark_loaded', False):
                watermark_position = st.selectbox(
                    "Watermark Position",
                    ["top-left", "top-right", "bottom-left", "bottom-right", "center"],
                    index=3,  # Default to bottom-right
                    key="watermark_position"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    watermark_opacity = st.slider(
                        "Opacity",
                        0.1, 1.0, 0.7,
                        key="watermark_opacity"
                    )
                with col2:
                    watermark_size = st.slider(
                        "Size",
                        0.05, 0.5, 0.2,
                        key="watermark_size"
                    )
                
                watermark_rotation = st.slider(
                    "Rotation",
                    0, 360, 0,
                    key="watermark_rotation"
                )
                
                st.button("Apply Watermark", key="apply_watermark_button", use_container_width=True,
                         on_click=self._apply_watermark_callback)
        
        # Dark Mode Toggle
        st.subheader("UI Settings")
        dark_mode = st.checkbox(
            "Dark Mode",
            value=st.session_state.dark_mode,
            key="dark_mode_toggle",
            on_change=self._toggle_dark_mode
        )
    
    def _create_export_tab(self):
        """Create the export tab"""
        st.subheader("Export Settings")
        seo_keywords = st.text_input(
            "SEO Keywords (comma separated)",
            "motivational,quote,inspiration",
            key="seo_keywords"
        )
        
        image_quality = st.slider(
            "Image Quality",
            70, 100, 90,
            key="image_quality"
        )
        
        # Add metadata checkbox
        add_metadata = st.checkbox(
            "Add SEO Metadata",
            value=True,
            key="add_metadata",
            help="Embed metadata in images for better SEO"
        )
    
    def create_main_content(self):
        """Create the main content area with preview and controls"""
        st.title("✨ Advanced Quote Image Generator")
        st.markdown("""
        Generate beautiful quote images with intelligent text placement, multi-language support, 
        watermarking, and advanced customization options.
        """)
        
        # Display processing status if any
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Preview", "Batch Processing", "Export"])
        
        with tab1:
            self._create_preview_tab()
            
        with tab2:
            self._create_batch_tab()
            
        with tab3:
            self._create_export_tab()
    
    def _create_preview_tab(self):
        """Create the preview tab content"""
        # Display current image and quote
        if st.session_state.current_images and st.session_state.current_quotes:
            # Create columns for layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display preview image if available, otherwise display current image
                if st.session_state.preview_image is not None:
                    st.image(st.session_state.preview_image, use_column_width=True)
                else:
                    current_image = st.session_state.current_images[st.session_state.current_image_index]
                    st.image(current_image, use_column_width=True)
                
                # Image navigation
                nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
                
                with nav_col1:
                    st.button("◀️ Previous", key="prev_image_button", on_click=self._navigate_images, args=(-1,))
                
                with nav_col2:
                    st.write(f"Image {st.session_state.current_image_index + 1} of {len(st.session_state.current_images)}")
                
                with nav_col3:
                    st.button("Next ▶️", key="next_image_button", on_click=self._navigate_images, args=(1,))
            
            with col2:
                # Quote selection
                st.subheader("Select Quote")
                
                for i, quote in enumerate(st.session_state.current_quotes):
                    quote_text = quote['text']
                    if quote['author']:
                        quote_text += f" - {quote['author']}"
                    
                    # Truncate long quotes for display
                    if len(quote_text) > 50:
                        display_text = quote_text[:47] + "..."
                    else:
                        display_text = quote_text
                    
                    if st.button(display_text, key=f"quote_button_{i}"):
                        self._generate_preview(i)
                
                # Manual positioning
                if st.session_state.get('text_positioning') == "Manual":
                    st.subheader("Manual Text Position")
                    
                    # Get image dimensions
                    if st.session_state.current_images:
                        img = st.session_state.current_images[st.session_state.current_image_index]
                        img_width, img_height = img.size
                        
                        # Position sliders
                        x_pos = st.slider(
                            "X Position",
                            0, img_width, img_width // 2,
                            key="text_x_position"
                        )
                        
                        y_pos = st.slider(
                            "Y Position",
                            0, img_height, img_height // 2,
                            key="text_y_position"
                        )
                        
                        st.session_state.text_position = (x_pos, y_pos)
                        
                        st.button("Update Position", key="update_position_button", on_click=self._generate_preview)
        else:
            st.info("Please select an image source and add quotes to get started.")
    
    def _create_batch_tab(self):
        """Create the batch processing tab content"""
        st.subheader("Batch Processing")
        
        if st.session_state.current_images and st.session_state.current_quotes:
            st.write(f"Ready to process {len(st.session_state.current_images)} images with {len(st.session_state.current_quotes)} quotes.")
            
            # Batch options
            st.write("Batch Processing Options:")
            
            batch_mode = st.radio(
                "Processing Mode",
                ["One quote per image", "All quotes on each image"],
                key="batch_mode",
                horizontal=True
            )
            
            # Add progress tracking
            if 'batch_progress' not in st.session_state:
                st.session_state.batch_progress = 0
                
            if st.session_state.batch_progress > 0:
                st.progress(st.session_state.batch_progress, "Processing images...")
            
            st.button("Generate All", key="generate_all_button", use_container_width=True, 
                     on_click=self._batch_generate_callback, args=(batch_mode,))
        else:
            st.info("Please select an image source and add quotes to enable batch processing.")
    
    def _create_export_tab(self):
        """Create the export tab content"""
        st.subheader("Export Options")
        
        if hasattr(st.session_state, 'generated_images') and st.session_state.generated_images:
            # Display generated images
            st.write(f"{len(st.session_state.generated_images)} images generated and ready for export.")
            
            # Show a few sample images
            sample_size = min(3, len(st.session_state.generated_images))
            cols = st.columns(sample_size)
            
            for i in range(sample_size):
                with cols[i]:
                    st.image(st.session_state.generated_images[i], use_column_width=True)
            
            # Export options
            export_format = st.radio(
                "Export Format",
                ["Individual JPG Files", "ZIP Archive"],
                key="export_format",
                horizontal=True
            )
            
            optimize_images = st.checkbox(
                "Optimize Images for Web",
                value=True,
                key="optimize_images"
            )
            
            st.button("Export Images", key="export_button", use_container_width=True,
                     on_click=self._export_images_callback, args=(export_format, optimize_images))
        else:
            st.info("Generate images in the Preview or Batch Processing tab first.")
    
    # Callback methods for UI interactions
    def _search_images_callback(self):
        """Callback for searching images"""
        source = st.session_state.image_source.lower()
        keyword = st.session_state.search_keyword
        
        # Convert aspect ratio to format expected by scraper
        aspect_ratio = st.session_state.aspect_ratio
        if aspect_ratio == "Landscape (16:9)":
            ratio = "landscape"
        elif aspect_ratio == "Portrait (9:16)":
            ratio = "portrait"
        else:
            ratio = "square"
        
        num_images = st.session_state.num_images
        
        # Update processing status
        st.session_state.processing_status = f"Searching for {keyword} images on {source}..."
        
        # Search for images
        image_paths = self.app.image_scraper.search_images(
            keyword, source, ratio, num_images
        )
        
        # Load images
        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image: {e}")
        
        # Update session state
        st.session_state.current_images = images
        st.session_state.current_image_index = 0
        st.session_state.preview_image = None
        
        # Clear processing status
        st.session_state.processing_status = f"Found {len(images)} images for '{keyword}'!"
    
    def _process_uploaded_images_callback(self, uploaded_images):
        """Callback for processing uploaded images"""
        images = []
        
        # Update processing status
        st.session_state.processing_status = "Processing uploaded images..."
        
        for uploaded_image in uploaded_images:
            try:
                # Read image
                img = Image.open(uploaded_image)
                
                # Check resolution
                width, height = img.size
                if width < 1000:
                    st.warning(f"Image {uploaded_image.name} has low resolution ({width}x{height}). Results may not be optimal.")
                
                images.append(img)
            except Exception as e:
                st.error(f"Error processing {uploaded_image.name}: {e}")
        
        # Update session state
        if images:
            st.session_state.current_images = images
            st.session_state.current_image_index = 0
            st.session_state.preview_image = None
            
            # Update processing status
            st.session_state.processing_status = f"Loaded {len(images)} images!"
        else:
            st.session_state.processing_status = "No valid images found in the uploaded files."
    
    def _process_quotes_text_callback(self, text):
        """Callback for processing quotes from text"""
        # Process quotes
        quotes = self.app.quote_processor.process_quotes_from_text(text)
        
        # Update session state
        st.session_state.current_quotes = quotes
        
        # Update processing status
        st.session_state.processing_status = f"Processed {len(quotes)} quotes!"
    
    def _process_quotes_file_callback(self, file, file_type):
        """Callback for processing quotes from file"""
        try:
            # Read file content
            file_content = file.read()
            
            # Process quotes
            quotes = self.app.quote_processor.process_quotes_from_file(file_content, file_type)
            
            # Update session state
            st.session_state.current_quotes = quotes
            
            # Update processing status
            st.session_state.processing_status = f"Processed {len(quotes)} quotes from {file.name}!"
        except Exception as e:
            st.session_state.processing_status = f"Error processing file: {e}"
    
    def _add_custom_font_callback(self, font_file):
        """Callback for adding custom font"""
        try:
            # Read font content
            font_content = font_file.read()
            
            # Add font
            success = self.app.quote_processor.add_custom_font(font_content, font_file.name)
            
            if success:
                st.session_state.processing_status = f"Added font: {font_file.name}"
            else:
                st.session_state.processing_status = f"Failed to add font: {font_file.name}"
        except Exception as e:
            st.session_state.processing_status = f"Error adding font: {e}"
    
    def _load_watermark_callback(self, watermark_file):
        """Callback for loading watermark image"""
        try:
            # Read watermark content
            watermark_content = watermark_file.read()
            
            # Load watermark
            success = self.app.watermark_processor.load_watermark(watermark_content)
            
            if success:
                st.session_state.watermark_loaded = True
                st.session_state.processing_status = f"Loaded watermark: {watermark_file.name}"
            else:
                st.session_state.processing_status = f"Failed to load watermark: {watermark_file.name}"
        except Exception as e:
            st.session_state.processing_status = f"Error loading watermark: {e}"
    
    def _create_text_watermark_callback(self, text):
        """Callback for creating text watermark"""
        try:
            # Create text watermark
            success = self.app.watermark_processor.create_text_watermark(text)
            
            if success:
                st.session_state.watermark_loaded = True
                st.session_state.processing_status = f"Created text watermark: {text}"
            else:
                st.session_state.processing_status = f"Failed to create text watermark"
        except Exception as e:
            st.session_state.processing_status = f"Error creating text watermark: {e}"
    
    def _apply_watermark_callback(self):
        """Callback for applying watermark settings"""
        # Update watermark settings
        settings = {
            'position': st.session_state.watermark_position,
            'opacity': st.session_state.watermark_opacity,
            'size': st.session_state.watermark_size,
            'rotation': st.session_state.watermark_rotation,
            'padding': 20  # Default padding
        }
        
        self.app.watermark_processor.update_settings(settings)
        
        # Mark watermark as applied
        st.session_state.watermark_applied = True
        
        # Regenerate preview if available
        if st.session_state.preview_image is not None:
            self._generate_preview()
        
        st.session_state.processing_status = "Watermark settings applied!"
    
    def _generate_preview(self, quote_index=None):
        """Generate preview image with selected quote"""
        if not st.session_state.current_images or not st.session_state.current_quotes:
            return
        
        # Get current image
        image = st.session_state.current_images[st.session_state.current_image_index]
        
        # Get quote
        if quote_index is not None:
            quote = st.session_state.current_quotes[quote_index]
        elif hasattr(st.session_state, 'current_quote_index'):
            quote = st.session_state.current_quotes[st.session_state.current_quote_index]
        else:
            quote = st.session_state.current_quotes[0]
            
        # Store current quote index
        if quote_index is not None:
            st.session_state.current_quote_index = quote_index
        
        # Convert hex color to RGB
        hex_color = st.session_state.text_color.lstrip('#')
        text_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Get text effect
        effect = st.session_state.text_effect.lower()
        if effect == "none":
            effect = "none"
        
        # Get text position method
        manual_position = st.session_state.text_positioning == "Manual"
        position = st.session_state.text_position if manual_position else None
        
        # Apply text overlay
        img_with_text = self.app.text_overlay.overlay_text(
            image, 
            quote, 
            position=position,
            text_color=text_color,
            effect=effect,
            background_blur=st.session_state.background_blur,
            manual_position=manual_position
        )
        
        # Apply watermark if enabled
        if st.session_state.get('watermark_applied', False):
            img_with_text = self.app.watermark_processor.apply_watermark(img_with_text)
        
        # Update preview
        st.session_state.preview_image = img_with_text
    
    def _navigate_images(self, direction):
        """Navigate between images"""
        # Calculate new index
        new_index = st.session_state.current_image_index + direction
        
        # Ensure index is within bounds
        if new_index < 0:
            new_index = len(st.session_state.current_images) - 1
        elif new_index >= len(st.session_state.current_images):
            new_index = 0
        
        # Update index
        st.session_state.current_image_index = new_index
        
        # Reset preview
        st.session_state.preview_image = None
    
    def _batch_generate_callback(self, mode):
        """Callback for batch generation"""
        if not st.session_state.current_images or not st.session_state.current_quotes:
            return
        
        # Initialize generated images list if not exists
        if 'generated_images' not in st.session_state:
            st.session_state.generated_images = []
        
        # Reset progress
        st.session_state.batch_progress = 0
        
        # Generate images based on mode
        generated_images = []
        
        if mode == "One quote per image":
            # Pair each image with a quote
            total_pairs = min(len(st.session_state.current_images), len(st.session_state.current_quotes))
            
            for i in range(total_pairs):
                # Update progress
                progress = (i + 1) / total_pairs
                st.session_state.batch_progress = progress
                
                # Get image and quote
                image = st.session_state.current_images[i % len(st.session_state.current_images)]
                quote = st.session_state.current_quotes[i % len(st.session_state.current_quotes)]
                
                # Generate image
                generated_image = self._generate_single_image(image, quote)
                generated_images.append(generated_image)
        else:  # All quotes on each image
            # Apply all quotes to each image
            total_combinations = len(st.session_state.current_images) * len(st.session_state.current_quotes)
            counter = 0
            
            for i, image in enumerate(st.session_state.current_images):
                for j, quote in enumerate(st.session_state.current_quotes):
                    # Update progress
                    counter += 1
                    progress = counter / total_combinations
                    st.session_state.batch_progress = progress
                    
                    # Generate image
                    generated_image = self._generate_single_image(image, quote)
                    generated_images.append(generated_image)
        
        # Update session state
        st.session_state.generated_images = generated_images
        
        # Complete progress
        st.session_state.batch_progress = 1.0
        
        # Update processing status
        st.session_state.processing_status = f"Generated {len(generated_images)} images! Go to Export tab to download."
    
    def _generate_single_image(self, image, quote):
        """Generate a single image with quote and watermark"""
        # Convert hex color to RGB
        hex_color = st.session_state.text_color.lstrip('#')
        text_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Get text effect
        effect = st.session_state.text_effect.lower()
        if effect == "none":
            effect = "none"
        
        # Apply text overlay
        img_with_text = self.app.text_overlay.overlay_text(
            image, 
            quote, 
            text_color=text_color,
            effect=effect,
            background_blur=st.session_state.background_blur
        )
        
        # Apply watermark if enabled
        if st.session_state.get('watermark_applied', False):
            img_with_text = self.app.watermark_processor.apply_watermark(img_with_text)
        
        return img_with_text
    
    def _export_images_callback(self, format, optimize):
        """Callback for exporting images"""
        if not hasattr(st.session_state, 'generated_images') or not st.session_state.generated_images:
            st.session_state.processing_status = "No images to export!"
            return
        
        # Get SEO keywords
        keywords = st.session_state.seo_keywords.split(',')
        keywords = [k.strip() for k in keywords if k.strip()]
        
        # Get image quality
        quality = st.session_state.image_quality
        
        # Get metadata setting
        add_metadata = st.session_state.get('add_metadata', True)
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each image
            file_paths = []
            
            for i, image in enumerate(st.session_state.generated_images):
                # Get quote text for filename
                if hasattr(st.session_state, 'current_quotes') and i < len(st.session_state.current_quotes):
                    quote = st.session_state.current_quotes[i % len(st.session_state.current_quotes)]
                    quote_text = quote['text']
                    
                    # Generate metadata if enabled
                    if add_metadata:
                        metadata = self.app.seo_optimizer.generate_metadata(quote, keywords)
                        image = self.app.seo_optimizer.embed_metadata(image, metadata)
                else:
                    quote_text = f"quote_{i+1}"
                
                # Generate SEO-friendly filename
                filename = self.app.seo_optimizer.generate_filename(quote_text, keywords)
                
                # Optimize image if requested
                if optimize:
                    image = self.app.seo_optimizer.optimize_image(image, quality)
                
                # Save image
                file_path = os.path.join(temp_dir, f"{filename}.jpg")
                image.save(file_path, "JPEG", quality=quality)
                file_paths.append(file_path)
            
            # Export based on format
            if format == "ZIP Archive":
                # Create ZIP file
                zip_path = os.path.join(temp_dir, "quote_images.zip")
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file_path in file_paths:
                        zipf.write(file_path, os.path.basename(file_path))
                
                # Provide download link
                with open(zip_path, "rb") as f:
                    bytes_data = f.read()
                    st.download_button(
                        label="Download ZIP",
                        data=bytes_data,
                        file_name="quote_images.zip",
                        mime="application/zip"
                    )
            else:
                # Provide download links for individual files
                for i, file_path in enumerate(file_paths):
                    with open(file_path, "rb") as f:
                        bytes_data = f.read()
                        st.download_button(
                            label=f"Download Image {i+1}",
                            data=bytes_data,
                            file_name=os.path.basename(file_path),
                            mime="image/jpeg",
                            key=f"download_button_{i}"
                        )
    
    def _toggle_dark_mode(self):
        """Toggle dark mode"""
        st.session_state.dark_mode = not st.session_state.dark_mode
        self._apply_dark_mode(st.session_state.dark_mode)
    
    def _apply_dark_mode(self, enable):
        """Apply dark mode to the UI"""
        if enable:
            # Apply dark theme
            st.markdown("""
            <style>
            .stApp {
                background-color: #121212;
                color: #FFFFFF;
            }
            .stTabs [data-baseweb="tab-list"] {
                background-color: #1E1E1E;
            }
            .stTabs [data-baseweb="tab"] {
                color: #FFFFFF;
            }
            .stTabs [aria-selected="true"] {
                background-color: rgba(255, 75, 75, 0.2);
            }
            .stButton>button {
                background-color: #FF4B4B;
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            # Reset to light theme
            st.markdown("""
            <style>
            .stApp {
                background-color: #FFFFFF;
                color: #000000;
            }
            .stTabs [data-baseweb="tab-list"] {
                background-color: #F0F2F6;
            }
            .stTabs [data-baseweb="tab"] {
                color: #000000;
            }
            .stTabs [aria-selected="true"] {
                background-color: rgba(255, 75, 75, 0.1);
            }
            .stButton>button {
                background-color: #FFFFFF;
                color: #000000;
            }
            </style>
            """, unsafe_allow_html=True)
