import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import base64

class WatermarkProcessor:
    """
    Class for handling watermarking and branding features.
    Supports watermark upload, positioning, opacity adjustment, and application to multiple images.
    """
    
    def __init__(self):
        """Initialize the watermark processor"""
        # Default watermark settings
        self.default_settings = {
            'position': 'bottom-right',  # Options: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'
            'opacity': 0.7,              # 0.0 to 1.0
            'size': 0.2,                 # Relative to image size (0.0 to 1.0)
            'rotation': 0,               # Degrees (0 to 360)
            'padding': 20                # Padding from edge in pixels
        }
        
        # Current watermark settings
        self.settings = self.default_settings.copy()
        
        # Current watermark image
        self.watermark_image = None
        
    def load_watermark(self, watermark_content):
        """
        Load watermark image from content
        
        Args:
            watermark_content (bytes): Watermark image content
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load watermark image
            self.watermark_image = Image.open(io.BytesIO(watermark_content)).convert("RGBA")
            return True
        except Exception as e:
            print(f"Error loading watermark: {e}")
            return False
    
    def update_settings(self, settings):
        """
        Update watermark settings
        
        Args:
            settings (dict): New settings to apply
            
        Returns:
            dict: Updated settings
        """
        # Update settings
        for key, value in settings.items():
            if key in self.settings:
                self.settings[key] = value
                
        return self.settings
    
    def reset_settings(self):
        """
        Reset watermark settings to defaults
        
        Returns:
            dict: Default settings
        """
        self.settings = self.default_settings.copy()
        return self.settings
    
    def apply_watermark(self, image, custom_settings=None):
        """
        Apply watermark to image
        
        Args:
            image (PIL.Image): Image to apply watermark to
            custom_settings (dict, optional): Custom settings for this specific image
            
        Returns:
            PIL.Image: Image with watermark applied
        """
        # Check if watermark is loaded
        if self.watermark_image is None:
            return image
            
        # Make a copy of the image to avoid modifying the original
        img = image.copy()
        
        # Use custom settings if provided, otherwise use current settings
        settings = custom_settings if custom_settings else self.settings
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Resize watermark based on size setting
        watermark_size = int(min(img_width, img_height) * settings['size'])
        watermark = self._resize_watermark(watermark_size)
        
        # Apply rotation if needed
        if settings['rotation'] != 0:
            watermark = watermark.rotate(settings['rotation'], expand=True, resample=Image.BICUBIC)
        
        # Calculate position
        position = self._calculate_position(img.size, watermark.size, settings['position'], settings['padding'])
        
        # Apply opacity
        watermark = self._apply_opacity(watermark, settings['opacity'])
        
        # Create a new transparent image for the watermark
        watermark_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        
        # Paste the watermark onto the transparent layer
        watermark_layer.paste(watermark, position, watermark)
        
        # Convert image to RGBA if it's not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Composite the watermark layer onto the image
        img = Image.alpha_composite(img, watermark_layer)
        
        # Convert back to RGB if needed
        if image.mode == 'RGB':
            img = img.convert('RGB')
        
        return img
    
    def apply_watermark_to_multiple(self, images):
        """
        Apply watermark to multiple images using the same settings
        
        Args:
            images (list): List of PIL.Image objects
            
        Returns:
            list: List of images with watermark applied
        """
        return [self.apply_watermark(img) for img in images]
    
    def _resize_watermark(self, target_size):
        """
        Resize watermark while maintaining aspect ratio
        
        Args:
            target_size (int): Target size (max dimension)
            
        Returns:
            PIL.Image: Resized watermark
        """
        # Get watermark dimensions
        width, height = self.watermark_image.size
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions
        if width > height:
            new_width = target_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(new_height * aspect_ratio)
        
        # Resize watermark
        return self.watermark_image.resize((new_width, new_height), Image.LANCZOS)
    
    def _calculate_position(self, image_size, watermark_size, position, padding):
        """
        Calculate watermark position
        
        Args:
            image_size (tuple): Image dimensions (width, height)
            watermark_size (tuple): Watermark dimensions (width, height)
            position (str): Position ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
            padding (int): Padding from edge in pixels
            
        Returns:
            tuple: Position (x, y)
        """
        img_width, img_height = image_size
        watermark_width, watermark_height = watermark_size
        
        if position == 'top-left':
            return (padding, padding)
        elif position == 'top-right':
            return (img_width - watermark_width - padding, padding)
        elif position == 'bottom-left':
            return (padding, img_height - watermark_height - padding)
        elif position == 'bottom-right':
            return (img_width - watermark_width - padding, img_height - watermark_height - padding)
        elif position == 'center':
            return ((img_width - watermark_width) // 2, (img_height - watermark_height) // 2)
        else:
            # Default to bottom-right
            return (img_width - watermark_width - padding, img_height - watermark_height - padding)
    
    def _apply_opacity(self, image, opacity):
        """
        Apply opacity to watermark
        
        Args:
            image (PIL.Image): Watermark image
            opacity (float): Opacity (0.0 to 1.0)
            
        Returns:
            PIL.Image: Watermark with opacity applied
        """
        # Ensure opacity is within valid range
        opacity = max(0.0, min(1.0, opacity))
        
        # Make a copy of the image to avoid modifying the original
        img = image.copy()
        
        # Split the image into bands
        bands = list(img.split())
        
        # Apply opacity to alpha channel
        if len(bands) == 4:  # RGBA
            bands[3] = bands[3].point(lambda x: int(x * opacity))
            
        # Merge bands back together
        return Image.merge(img.mode, bands)
    
    def create_text_watermark(self, text, font_size=40, font_path=None, text_color=(255, 255, 255, 200)):
        """
        Create a text-based watermark
        
        Args:
            text (str): Watermark text
            font_size (int, optional): Font size
            font_path (str, optional): Path to font file, uses default if None
            text_color (tuple, optional): Text color (R, G, B, A)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a temporary image for text size calculation
            temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            draw = ImageDraw.Draw(temp_img)
            
            # Load font
            if font_path and os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    # Use default font if the specified font can't be loaded
                    font = ImageFont.load_default()
            else:
                # Use default font
                font = ImageFont.load_default()
                
            # Calculate text size
            try:
                # For older PIL versions
                text_width, text_height = draw.textsize(text, font=font)
            except AttributeError:
                # For newer PIL versions
                bbox = font.getbbox(text)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Create image for watermark
            watermark = Image.new('RGBA', (text_width + 20, text_height + 20), (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark)
            
            # Draw text
            try:
                # For older PIL versions
                draw.text((10, 10), text, font=font, fill=text_color)
            except TypeError:
                # For newer PIL versions
                draw.text((10, 10), text, font=font, fill=text_color[:3])
            
            # Set as watermark image
            self.watermark_image = watermark
            
            return True
        except Exception as e:
            print(f"Error creating text watermark: {e}")
            return False
    
    def get_watermark_preview(self, size=(200, 200), background_color=(200, 200, 200)):
        """
        Generate a preview image of the watermark
        
        Args:
            size (tuple, optional): Preview size (width, height)
            background_color (tuple, optional): Background color (R, G, B)
            
        Returns:
            str: Base64 encoded image
        """
        if self.watermark_image is None:
            return None
            
        try:
            # Create background image
            background = Image.new('RGB', size, background_color)
            
            # Resize watermark to fit preview
            preview_size = int(min(size) * 0.8)
            watermark = self._resize_watermark(preview_size)
            
            # Calculate position (center)
            position = ((size[0] - watermark.width) // 2, (size[1] - watermark.height) // 2)
            
            # Apply opacity
            watermark = self._apply_opacity(watermark, self.settings['opacity'])
            
            # Paste watermark onto background
            background.paste(watermark, position, watermark)
            
            # Convert to base64
            buffered = io.BytesIO()
            background.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error generating watermark preview: {e}")
            return None
