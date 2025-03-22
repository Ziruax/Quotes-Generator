import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import math

class TextOverlay:
    """
    Class for intelligent text overlay on images with AI-based positioning
    and contrast-based text placement.
    """
    
    def __init__(self):
        """Initialize the text overlay system"""
        # Define text effects
        self.text_effects = {
            'none': self._effect_none,
            'outline': self._effect_outline,
            'shadow': self._effect_shadow,
            'glow': self._effect_glow
        }
        
        # Define text alignment options
        self.text_alignments = ['left', 'center', 'right']
        
    def overlay_text(self, image, quote, position=None, text_size=None, 
                     alignment='center', effect='none', effect_options=None,
                     text_color=None, background_blur=0, manual_position=False):
        """
        Overlay text on image with intelligent positioning
        
        Args:
            image (PIL.Image): Image to overlay text on
            quote (dict): Quote dictionary with text, author, and font info
            position (tuple, optional): Manual position (x, y) if provided
            text_size (int, optional): Text size, auto-calculated if None
            alignment (str, optional): Text alignment ('left', 'center', 'right')
            effect (str, optional): Text effect ('none', 'outline', 'shadow', 'glow')
            effect_options (dict, optional): Options for the selected effect
            text_color (tuple, optional): Text color (R, G, B), auto-calculated if None
            background_blur (int, optional): Background blur amount (0-10)
            manual_position (bool, optional): Whether position is manually set
            
        Returns:
            PIL.Image: Image with text overlay
        """
        # Make a copy of the image to avoid modifying the original
        img = image.copy()
        
        # Apply background blur if requested
        if background_blur > 0:
            img = self._apply_background_blur(img, background_blur)
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Auto-calculate text size if not provided
        if text_size is None:
            # Base text size on image dimensions
            text_size = int(img_width * 0.05)  # 5% of image width
            
            # Adjust based on quote length
            quote_length = len(quote['text'])
            if quote_length > 100:
                text_size = int(text_size * 0.8)  # Reduce size for long quotes
            elif quote_length < 30:
                text_size = int(text_size * 1.2)  # Increase size for short quotes
        
        # Load fonts - with fallback to default
        try:
            title_font = ImageFont.truetype(quote['font']['title'], text_size)
        except:
            title_font = ImageFont.load_default()
            
        try:
            regular_font = ImageFont.truetype(quote['font']['regular'], text_size)
        except:
            regular_font = ImageFont.load_default()
            
        try:
            italic_font = ImageFont.truetype(quote['font']['italic'], int(text_size * 0.8))
        except:
            italic_font = ImageFont.load_default()
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Determine text color if not provided
        if text_color is None:
            text_color = self._determine_text_color(img)
        
        # Determine text position if not provided
        if position is None and not manual_position:
            position = self._determine_text_position_ai(img, quote['text'], title_font)
        elif position is None:
            # Default position (center)
            position = (img_width // 2, img_height // 2)
        
        # Calculate text dimensions
        quote_text = quote['text']
        author_text = quote['author'] if quote['author'] else None
        
        # Handle text alignment
        if alignment not in self.text_alignments:
            alignment = 'center'
        
        # Apply text effect
        if effect not in self.text_effects:
            effect = 'none'
            
        if effect_options is None:
            effect_options = {}
        
        # Draw the quote text
        self.text_effects[effect](
            img, draw, quote_text, position, title_font, 
            text_color, alignment, effect_options
        )
        
        # Draw the author text if available
        if author_text:
            # Calculate author position (below the quote text)
            try:
                # For older PIL versions
                text_height = title_font.getsize(quote_text)[1]
            except:
                # For newer PIL versions
                text_height = title_font.getbbox(quote_text)[3]
                
            author_y = position[1] + text_height + 20
            author_position = (position[0], author_y)
            
            # Draw author with italic font
            self.text_effects[effect](
                img, draw, f"- {author_text}", author_position, italic_font, 
                text_color, alignment, effect_options
            )
        
        return img
    
    def _determine_text_color(self, image):
        """
        Determine the best text color based on image brightness
        
        Args:
            image (PIL.Image): Image to analyze
            
        Returns:
            tuple: Text color (R, G, B)
        """
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Calculate average brightness
        if len(img_array.shape) == 3:
            # Color image
            avg_brightness = np.mean(img_array)
        else:
            # Grayscale image
            avg_brightness = np.mean(img_array)
        
        # Determine text color based on brightness
        if avg_brightness > 127:
            # Dark text for bright images
            return (0, 0, 0)
        else:
            # White text for dark images
            return (255, 255, 255)
    
    def _determine_text_position_ai(self, image, text, font):
        """
        Determine the best position for text using AI-based positioning
        
        Args:
            image (PIL.Image): Image to analyze
            text (str): Text to overlay
            font (PIL.ImageFont): Font to use
            
        Returns:
            tuple: Position (x, y)
        """
        # Get image dimensions
        img_width, img_height = image.size
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Calculate text dimensions
        try:
            # For older PIL versions
            text_width, text_height = font.getsize(text)
        except:
            # For newer PIL versions
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Detect faces (if any)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Create a mask for important areas (faces)
        mask = np.zeros_like(gray)
        
        # Mark faces as important areas
        for (x, y, w, h) in faces:
            # Add padding around faces
            padding = 20
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img_width, x + w + padding)
            y_end = min(img_height, y + h + padding)
            
            # Mark face area in mask
            mask[y_start:y_end, x_start:x_end] = 255
        
        # Detect edges for important objects
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges to make them more prominent
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Add edges to mask
        mask = cv2.bitwise_or(mask, dilated_edges)
        
        # Create grid of potential positions
        grid_size = 5
        positions = []
        scores = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate position
                x = int(img_width * (i + 0.5) / grid_size - text_width / 2)
                y = int(img_height * (j + 0.5) / grid_size - text_height / 2)
                
                # Ensure position is within image bounds
                x = max(10, min(x, img_width - text_width - 10))
                y = max(10, min(y, img_height - text_height - 10))
                
                # Create text region
                text_region = np.zeros_like(mask)
                text_region[y:y+text_height, x:x+text_width] = 255
                
                # Calculate overlap with important areas
                overlap = cv2.bitwise_and(mask, text_region)
                overlap_score = np.sum(overlap) / 255
                
                # Calculate contrast in region
                text_region_gray = gray[y:y+text_height, x:x+text_width]
                if text_region_gray.size > 0:
                    contrast_score = np.std(text_region_gray)
                else:
                    contrast_score = 0
                
                # Calculate distance from center
                center_x, center_y = img_width // 2, img_height // 2
                distance = math.sqrt((x + text_width/2 - center_x)**2 + (y + text_height/2 - center_y)**2)
                distance_score = 1 - (distance / (math.sqrt(img_width**2 + img_height**2) / 2))
                
                # Calculate final score (lower is better)
                # We want low overlap with important areas, high contrast, and close to center
                score = overlap_score - contrast_score * 0.5 - distance_score * 100
                
                positions.append((x, y))
                scores.append(score)
        
        # Find position with lowest score
        best_position = positions[scores.index(min(scores))]
        
        return best_position
    
    def _apply_background_blur(self, image, blur_amount):
        """
        Apply blur to the background for better text contrast
        
        Args:
            image (PIL.Image): Image to blur
            blur_amount (int): Blur amount (0-10)
            
        Returns:
            PIL.Image: Blurred image
        """
        # Limit blur amount
        blur_amount = max(0, min(10, blur_amount))
        
        # Apply Gaussian blur
        blur_radius = blur_amount * 2  # Convert to appropriate radius
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return blurred
    
    def _effect_none(self, img, draw, text, position, font, color, alignment, options):
        """
        Draw text with no special effect
        """
        x, y = position
        
        # Calculate text dimensions
        try:
            # For older PIL versions
            text_width, text_height = font.getsize(text)
        except:
            # For newer PIL versions
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Adjust position based on alignment
        if alignment == 'center':
            x = x - text_width // 2
        elif alignment == 'right':
            x = x - text_width
        
        draw.text((x, y), text, font=font, fill=color)
    
    def _effect_outline(self, img, draw, text, position, font, color, alignment, options):
        """
        Draw text with outline effect
        """
        x, y = position
        
        # Calculate text dimensions
        try:
            # For older PIL versions
            text_width, text_height = font.getsize(text)
        except:
            # For newer PIL versions
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Adjust position based on alignment
        if alignment == 'center':
            x = x - text_width // 2
        elif alignment == 'right':
            x = x - text_width
        
        # Get outline color and width
        outline_color = options.get('outline_color', (0, 0, 0) if color[0] > 127 else (255, 255, 255))
        outline_width = options.get('outline_width', 2)
        
        # Draw outline
        for offset_x in range(-outline_width, outline_width + 1):
            for offset_y in range(-outline_width, outline_width + 1):
                if offset_x == 0 and offset_y == 0:
                    continue
                draw.text((x + offset_x, y + offset_y), text, font=font, fill=outline_color)
        
        # Draw text
        draw.text((x, y), text, font=font, fill=color)
    
    def _effect_shadow(self, img, draw, text, position, font, color, alignment, options):
        """
        Draw text with shadow effect
        """
        x, y = position
        
        # Calculate text dimensions
        try:
            # For older PIL versions
            text_width, text_height = font.getsize(text)
        except:
            # For newer PIL versions
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Adjust position based on alignment
        if alignment == 'center':
            x = x - text_width // 2
        elif alignment == 'right':
            x = x - text_width
        
        # Get shadow color and offset
        shadow_color = options.get('shadow_color', (0, 0, 0))
        shadow_offset = options.get('shadow_offset', 3)
        
        # Draw shadow
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)
        
        # Draw text
        draw.text((x, y), text, font=font, fill=color)
    
    def _effect_glow(self, img, draw, text, position, font, color, alignment, options):
        """
        Draw text with glow effect
        """
        x, y = position
        
        # Calculate text dimensions
        try:
            # For older PIL versions
            text_width, text_height = font.getsize(text)
        except:
            # For newer PIL versions
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Adjust position based on alignment
        if alignment == 'center':
            x = x - text_width // 2
        elif alignment == 'right':
            x = x - text_width
        
        # Get glow color and radius
        glow_color = options.get('glow_color', (255, 255, 255) if color[0] < 127 else (0, 0, 0))
        glow_radius = options.get('glow_radius', 3)
        
        # Create a temporary image for the glow
        glow_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_img)
        
        # Draw text on temporary image
        glow_draw.text((x, y), text, font=font, fill=glow_color)
        
        # Apply blur to create glow
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=glow_radius))
        
        # Composite the glow onto the original image
        img.paste(glow_img, (0, 0), glow_img)
        
        # Draw text
        draw.text((x, y), text, font=font, fill=color)
    
    def adjust_text_for_manual_positioning(self, image, text, font, position, alignment='center'):
        """
        Adjust text position for manual positioning
        
        Args:
            image (PIL.Image): Image to overlay text on
            text (str): Text to overlay
            font (PIL.ImageFont): Font to use
            position (tuple): Position (x, y)
            alignment (str, optional): Text alignment ('left', 'center', 'right')
            
        Returns:
            tuple: Adjusted position (x, y)
        """
        x, y = position
        
        # Calculate text dimensions
        try:
            # For older PIL versions
            text_width, text_height = font.getsize(text)
        except:
            # For newer PIL versions
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Adjust position based on alignment
        if alignment == 'center':
            x = x - text_width // 2
        elif alignment == 'right':
            x = x - text_width
        
        # Ensure position is within image bounds
        img_width, img_height = image.size
        x = max(10, min(x, img_width - text_width - 10))
        y = max(10, min(y, img_height - text_height - 10))
        
        return (x, y)
