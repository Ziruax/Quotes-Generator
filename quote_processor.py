import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFont
import re
import io
import base64
import langdetect

class QuoteProcessor:
    """
    Class for processing quotes with multi-language support and font selection.
    Supports direct input and CSV/Excel file uploads.
    """
    
    def __init__(self, fonts_dir):
        """
        Initialize the quote processor with fonts directory
        
        Args:
            fonts_dir (str): Directory containing font files
        """
        self.fonts_dir = fonts_dir
        
        # Create fonts directory if it doesn't exist
        os.makedirs(self.fonts_dir, exist_ok=True)
        
        # Initialize default fonts for different languages
        self.default_fonts = {
            'en': {  # English
                'title': 'default/DejaVuSans-Bold.ttf',
                'regular': 'default/DejaVuSans.ttf',
                'italic': 'default/DejaVuSans-Oblique.ttf'
            },
            'de': {  # German
                'title': 'default/DejaVuSans-Bold.ttf',
                'regular': 'default/DejaVuSans.ttf',
                'italic': 'default/DejaVuSans-Oblique.ttf'
            },
            'ur': {  # Urdu
                'title': 'default/DejaVuSans-Bold.ttf',
                'regular': 'default/DejaVuSans.ttf',
                'italic': 'default/DejaVuSans-Oblique.ttf'
            },
            'hi': {  # Hindi
                'title': 'default/DejaVuSans-Bold.ttf',
                'regular': 'default/DejaVuSans.ttf',
                'italic': 'default/DejaVuSans-Oblique.ttf'
            },
            'ar': {  # Arabic
                'title': 'default/DejaVuSans-Bold.ttf',
                'regular': 'default/DejaVuSans.ttf',
                'italic': 'default/DejaVuSans-Oblique.ttf'
            }
        }
        
        # Initialize available fonts list
        self.available_fonts = self._get_available_fonts()
        
    def _get_available_fonts(self):
        """
        Get list of available fonts in the fonts directory
        
        Returns:
            list: List of available font filenames
        """
        available_fonts = []
        
        # Create default fonts directory if it doesn't exist
        default_fonts_dir = os.path.join(self.fonts_dir, 'default')
        os.makedirs(default_fonts_dir, exist_ok=True)
        
        # Check if DejaVu Sans fonts exist, if not use system default
        dejavu_bold = os.path.join(default_fonts_dir, 'DejaVuSans-Bold.ttf')
        dejavu_regular = os.path.join(default_fonts_dir, 'DejaVuSans.ttf')
        dejavu_oblique = os.path.join(default_fonts_dir, 'DejaVuSans-Oblique.ttf')
        
        if not os.path.exists(dejavu_bold) or not os.path.exists(dejavu_regular) or not os.path.exists(dejavu_oblique):
            # Use system default font
            try:
                # Create a simple font file if needed
                from PIL import ImageFont
                default_font = ImageFont.load_default()
                available_fonts.append('default/DejaVuSans.ttf')
            except:
                pass
        
        # Check for font files in the fonts directory
        if os.path.exists(self.fonts_dir):
            for filename in os.listdir(self.fonts_dir):
                if filename.lower().endswith(('.ttf', '.otf')):
                    available_fonts.append(filename)
            
            # Check default directory
            if os.path.exists(default_fonts_dir):
                for filename in os.listdir(default_fonts_dir):
                    if filename.lower().endswith(('.ttf', '.otf')):
                        available_fonts.append(f"default/{filename}")
        
        return available_fonts
    
    def process_quotes_from_text(self, text):
        """
        Process quotes from direct text input
        
        Args:
            text (str): Text containing one or more quotes
            
        Returns:
            list: List of processed quote dictionaries
        """
        # Split text into individual quotes
        quotes = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if line:
                quotes.append(line)
        
        # Process each quote
        processed_quotes = []
        for quote in quotes:
            processed_quote = self._process_single_quote(quote)
            processed_quotes.append(processed_quote)
            
        return processed_quotes
    
    def process_quotes_from_file(self, file_content, file_type):
        """
        Process quotes from uploaded file (CSV or Excel)
        
        Args:
            file_content (bytes): File content
            file_type (str): File type ('csv' or 'excel')
            
        Returns:
            list: List of processed quote dictionaries
        """
        quotes = []
        
        try:
            # Read file based on type
            if file_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
            elif file_type == 'excel':
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            # Check if the file has the expected columns
            if 'quote' in df.columns:
                quote_col = 'quote'
            elif 'text' in df.columns:
                quote_col = 'text'
            elif len(df.columns) > 0:
                # Use the first column if no specific quote column is found
                quote_col = df.columns[0]
            else:
                raise ValueError("No valid quote column found in the file")
                
            # Extract author column if available
            author_col = None
            if 'author' in df.columns:
                author_col = 'author'
            elif 'by' in df.columns:
                author_col = 'by'
                
            # Process each quote
            for _, row in df.iterrows():
                quote_text = str(row[quote_col]).strip()
                if not quote_text or quote_text.lower() == 'nan':
                    continue
                    
                # Add author if available
                if author_col and pd.notna(row[author_col]):
                    quote_text += f" - {row[author_col]}"
                    
                # Process the quote
                processed_quote = self._process_single_quote(quote_text)
                quotes.append(processed_quote)
                
        except Exception as e:
            print(f"Error processing file: {e}")
            
        return quotes
    
    def _process_single_quote(self, quote_text):
        """
        Process a single quote
        
        Args:
            quote_text (str): Text of the quote
            
        Returns:
            dict: Processed quote dictionary
        """
        # Detect language using langdetect
        try:
            lang = langdetect.detect(quote_text)
            # Map to supported languages or default to English
            if lang not in ['en', 'de', 'ur', 'hi', 'ar']:
                lang = 'en'
        except:
            # Default to English if detection fails
            lang = 'en'
        
        # Select appropriate font based on language
        font_info = self._select_font_for_language(lang)
        
        # Extract author if present (format: "Quote text - Author")
        author = None
        if ' - ' in quote_text:
            quote_parts = quote_text.split(' - ', 1)
            quote_text = quote_parts[0].strip()
            author = quote_parts[1].strip()
            
        # Create processed quote dictionary
        processed_quote = {
            'text': quote_text,
            'author': author,
            'language': lang,
            'font': font_info
        }
        
        return processed_quote
    
    def _select_font_for_language(self, lang):
        """
        Select appropriate font for the detected language
        
        Args:
            lang (str): Detected language code
            
        Returns:
            dict: Font information dictionary
        """
        # Default to English if language not supported
        if lang not in self.default_fonts:
            lang = 'en'
            
        # Get font paths
        title_font = os.path.join(self.fonts_dir, self.default_fonts[lang]['title'])
        regular_font = os.path.join(self.fonts_dir, self.default_fonts[lang]['regular'])
        italic_font = os.path.join(self.fonts_dir, self.default_fonts[lang]['italic'])
        
        # Check if fonts exist, fallback to system default if not
        if not os.path.exists(title_font) or not os.path.exists(regular_font) or not os.path.exists(italic_font):
            # Use system default font
            try:
                from PIL import ImageFont
                default_font = ImageFont.load_default()
                default_font_path = os.path.join(self.fonts_dir, 'default', 'DejaVuSans.ttf')
                
                # Create font info dictionary with default font
                font_info = {
                    'title': default_font_path,
                    'regular': default_font_path,
                    'italic': default_font_path,
                    'language': lang
                }
                
                return font_info
            except:
                pass
            
        # Create font info dictionary
        font_info = {
            'title': title_font,
            'regular': regular_font,
            'italic': italic_font,
            'language': lang
        }
        
        return font_info
    
    def add_custom_font(self, font_content, font_name):
        """
        Add a custom font to the fonts directory
        
        Args:
            font_content (bytes): Font file content
            font_name (str): Font filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure font has proper extension
            if not font_name.lower().endswith(('.ttf', '.otf')):
                font_name += '.ttf'
                
            # Save font file
            font_path = os.path.join(self.fonts_dir, font_name)
            with open(font_path, 'wb') as f:
                f.write(font_content)
                
            # Update available fonts list
            self.available_fonts = self._get_available_fonts()
            
            return True
        except Exception as e:
            print(f"Error adding custom font: {e}")
            return False
    
    def get_font_preview(self, font_path, text="AaBbCcDdEe"):
        """
        Generate a preview image for a font
        
        Args:
            font_path (str): Path to the font file
            text (str): Text to display in the preview
            
        Returns:
            str: Base64 encoded image
        """
        try:
            # Create image
            img_width = 400
            img_height = 100
            background_color = (255, 255, 255)
            text_color = (0, 0, 0)
            
            img = Image.new('RGB', (img_width, img_height), background_color)
            
            # Load font
            font_size = 36
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                # Use default font if the specified font can't be loaded
                font = ImageFont.load_default()
            
            # Draw text
            draw = ImageDraw.Draw(img)
            
            # Handle different PIL versions for text size calculation
            try:
                text_width, text_height = draw.textsize(text, font=font)
                position = ((img_width - text_width) // 2, (img_height - text_height) // 2)
                draw.text(position, text, font=font, fill=text_color)
            except AttributeError:
                # For newer PIL versions
                bbox = font.getbbox(text)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                position = ((img_width - text_width) // 2, (img_height - text_height) // 2)
                draw.text(position, text, font=font, fill=text_color)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error generating font preview: {e}")
            return None
