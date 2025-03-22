import os
import re
from PIL import Image
import io
import json
from datetime import datetime
import piexif

class SEOOptimizer:
    """
    Class for SEO optimization and export features.
    Handles image naming, metadata embedding, and image optimization.
    """
    
    def __init__(self):
        """Initialize the SEO optimizer"""
        pass
    
    def generate_filename(self, quote_text, keywords=None):
        """
        Generate SEO-friendly filename based on quote text and keywords
        
        Args:
            quote_text (str): Quote text
            keywords (list, optional): List of SEO keywords
            
        Returns:
            str: SEO-friendly filename
        """
        # Extract first few words from quote (max 8 words)
        words = quote_text.split()
        if len(words) > 8:
            words = words[:8]
        
        # Clean and join words
        filename = ' '.join(words)
        
        # Remove special characters and convert to lowercase
        filename = re.sub(r'[^\w\s-]', '', filename.lower())
        
        # Replace spaces with hyphens
        filename = re.sub(r'\s+', '-', filename)
        
        # Truncate if too long (max 50 characters)
        if len(filename) > 50:
            filename = filename[:50]
        
        # Add keywords if provided
        if keywords and len(keywords) > 0:
            # Select up to 3 keywords
            selected_keywords = keywords[:3]
            
            # Clean and join keywords
            keyword_str = '-'.join(selected_keywords)
            
            # Clean keyword string
            keyword_str = re.sub(r'[^\w\s-]', '', keyword_str.lower())
            keyword_str = re.sub(r'\s+', '-', keyword_str)
            
            # Combine filename and keywords
            filename = f"{filename}-{keyword_str}"
            
            # Truncate if too long (max 80 characters)
            if len(filename) > 80:
                filename = filename[:80]
        
        return filename
    
    def embed_metadata(self, image, metadata):
        """
        Embed metadata into image for better SEO
        
        Args:
            image (PIL.Image): Image to embed metadata into
            metadata (dict): Metadata to embed
            
        Returns:
            PIL.Image: Image with embedded metadata
        """
        try:
            # Convert image to JPEG if it's not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save image to a buffer
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            buffer.seek(0)
            
            # Create EXIF data
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            
            # Add metadata to EXIF
            if 'title' in metadata:
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = metadata['title'].encode('utf-8')
            
            if 'description' in metadata:
                exif_dict["0th"][piexif.ImageIFD.XPComment] = metadata['description'].encode('utf-8')
            
            if 'author' in metadata:
                exif_dict["0th"][piexif.ImageIFD.Artist] = metadata['author'].encode('utf-8')
            
            if 'copyright' in metadata:
                exif_dict["0th"][piexif.ImageIFD.Copyright] = metadata['copyright'].encode('utf-8')
            
            # Convert EXIF dict to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Create a new image with EXIF data
            output = io.BytesIO()
            piexif.insert(exif_bytes, buffer.getvalue(), output)
            output.seek(0)
            
            return Image.open(output)
        except Exception as e:
            print(f"Error embedding metadata: {e}")
            return image
    
    def optimize_image(self, image, quality=85):
        """
        Optimize image for web (reduce file size while maintaining quality)
        
        Args:
            image (PIL.Image): Image to optimize
            quality (int, optional): JPEG quality (0-100)
            
        Returns:
            PIL.Image: Optimized image
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image dimensions
            width, height = image.size
            
            # Resize if too large (max 2000px on longest side)
            max_dimension = 2000
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Save with optimized quality
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            output.seek(0)
            
            return Image.open(output)
        
        except Exception as e:
            print(f"Error optimizing image: {e}")
            return image
    
    def generate_metadata(self, quote, keywords=None):
        """
        Generate metadata for image based on quote and keywords
        
        Args:
            quote (dict): Quote dictionary
            keywords (list, optional): List of SEO keywords
            
        Returns:
            dict: Metadata dictionary
        """
        # Extract quote text and author
        quote_text = quote['text']
        author = quote.get('author', '')
        
        # Create title
        title = quote_text[:100]  # Truncate if too long
        
        # Create description
        description = quote_text
        if author:
            description += f" - {author}"
        
        # Create copyright
        current_year = datetime.now().year
        copyright_text = f"© {current_year} Quote Image Generator"
        
        # Create metadata dictionary
        metadata = {
            "title": title,
            "description": description,
            "author": author,
            "copyright": copyright_text
        }
        
        # Add keywords if provided
        if keywords:
            metadata["keywords"] = ", ".join(keywords)
        
        return metadata
    
    def create_sitemap(self, urls, output_path):
        """
        Create XML sitemap for SEO
        
        Args:
            urls (list): List of URL dictionaries with 'loc', 'lastmod', 'changefreq', and 'priority'
            output_path (str): Path to save sitemap
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create XML header
            xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
            xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            
            # Add URLs
            for url in urls:
                xml += '  <url>\n'
                xml += f'    <loc>{url["loc"]}</loc>\n'
                
                if "lastmod" in url:
                    xml += f'    <lastmod>{url["lastmod"]}</lastmod>\n'
                
                if "changefreq" in url:
                    xml += f'    <changefreq>{url["changefreq"]}</changefreq>\n'
                
                if "priority" in url:
                    xml += f'    <priority>{url["priority"]}</priority>\n'
                
                xml += '  </url>\n'
            
            # Close XML
            xml += '</urlset>'
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(xml)
            
            return True
        
        except Exception as e:
            print(f"Error creating sitemap: {e}")
            return False
