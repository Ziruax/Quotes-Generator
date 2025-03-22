import os
import requests
import time
import random
import re
from bs4 import BeautifulSoup
import cv2
import numpy as np
import io
from PIL import Image
from tqdm import tqdm
import threading
import queue
import crawl4ai

class ImageScraper:
    """
    Class for scraping high-quality images from various sources without using APIs.
    Supports Unsplash, Pexels, and Pixabay with intelligent filtering.
    Uses crawl4ai as a fallback if BeautifulSoup4 fails.
    """
    
    def __init__(self, cache_dir):
        """
        Initialize the image scraper with cache directory
        
        Args:
            cache_dir (str): Directory to cache downloaded images
        """
        self.cache_dir = cache_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create source-specific cache directories
        for source in ['unsplash', 'pexels', 'pixabay']:
            os.makedirs(os.path.join(self.cache_dir, source), exist_ok=True)
            
        # Define aspect ratio dimensions
        self.aspect_ratios = {
            'landscape': (16, 9),
            'portrait': (9, 16),
            'square': (1, 1)
        }
        
        # Minimum resolution threshold (width in pixels)
        self.min_resolution = 1000
        
        # Initialize crawl4ai for fallback scraping
        self.crawler = crawl4ai.Crawler()
        
    def search_images(self, keyword, source, aspect_ratio='landscape', num_images=5):
        """
        Search for images based on keyword from specified source
        
        Args:
            keyword (str): Search keyword
            source (str): Source website ('unsplash', 'pexels', or 'pixabay')
            aspect_ratio (str): Desired aspect ratio ('landscape', 'portrait', or 'square')
            num_images (int): Number of images to retrieve
            
        Returns:
            list: List of image paths (local cached paths)
        """
        # Check cache first
        cached_images = self._check_cache(keyword, source, aspect_ratio)
        if len(cached_images) >= num_images:
            return cached_images[:num_images]
            
        # If not enough cached images, try scraping with BeautifulSoup
        try:
            scraped_images = self._scrape_with_bs4(keyword, source, aspect_ratio, num_images - len(cached_images))
            if scraped_images:
                # Combine with cached images and return requested number
                all_images = cached_images + scraped_images
                return all_images[:num_images]
        except Exception as e:
            print(f"BeautifulSoup scraping failed: {e}")
            
        # If BeautifulSoup fails, try with crawl4ai
        try:
            crawl4ai_images = self._scrape_with_crawl4ai(keyword, source, aspect_ratio, num_images - len(cached_images))
            if crawl4ai_images:
                # Combine with cached images and return requested number
                all_images = cached_images + crawl4ai_images
                return all_images[:num_images]
        except Exception as e:
            print(f"crawl4ai scraping failed: {e}")
        
        # If all scraping methods fail, use sample images
        sample_images = self._get_sample_images(num_images - len(cached_images))
        
        # Combine with cached images and return requested number
        all_images = cached_images + sample_images
        return all_images[:num_images]
    
    def _check_cache(self, keyword, source, aspect_ratio):
        """
        Check if images are already cached
        
        Args:
            keyword (str): Search keyword
            source (str): Source website
            aspect_ratio (str): Desired aspect ratio
            
        Returns:
            list: List of cached image paths
        """
        cache_path = os.path.join(self.cache_dir, source)
        cached_files = []
        
        # Create regex pattern to match keyword and aspect ratio in filename
        pattern = re.compile(f"{keyword.lower().replace(' ', '-')}.*{aspect_ratio}", re.IGNORECASE)
        
        if os.path.exists(cache_path):
            for filename in os.listdir(cache_path):
                if pattern.search(filename):
                    cached_files.append(os.path.join(cache_path, filename))
                
        return cached_files
    
    def _scrape_with_bs4(self, keyword, source, aspect_ratio, num_images):
        """
        Scrape images using BeautifulSoup
        
        Args:
            keyword (str): Search keyword
            source (str): Source website
            aspect_ratio (str): Desired aspect ratio
            num_images (int): Number of images to retrieve
            
        Returns:
            list: List of image paths
        """
        # Build URL based on source and keyword
        if source == 'unsplash':
            url = f"https://unsplash.com/s/photos/{keyword.replace(' ', '-')}"
        elif source == 'pexels':
            url = f"https://www.pexels.com/search/{keyword.replace(' ', '%20')}/"
        elif source == 'pixabay':
            url = f"https://pixabay.com/images/search/{keyword.replace(' ', '%20')}/"
        else:
            raise ValueError(f"Unsupported source: {source}")
            
        # Send request
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch {url}: {response.status_code}")
            
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract image URLs based on source
        image_urls = []
        if source == 'unsplash':
            # Find image elements
            img_elements = soup.select('figure a img')
            for img in img_elements:
                if 'src' in img.attrs:
                    img_url = img['src']
                    if 'images.unsplash.com' in img_url:
                        # Get high-resolution version
                        img_url = img_url.split('?')[0] + '?w=1600&q=80'
                        image_urls.append(img_url)
        elif source == 'pexels':
            # Find image elements
            img_elements = soup.select('article a img')
            for img in img_elements:
                if 'src' in img.attrs:
                    img_url = img['src']
                    if 'images.pexels.com' in img_url:
                        # Get high-resolution version
                        img_url = img_url.split('?')[0]
                        image_urls.append(img_url)
        elif source == 'pixabay':
            # Find image elements
            img_elements = soup.select('.container img')
            for img in img_elements:
                if 'src' in img.attrs:
                    img_url = img['src']
                    if 'pixabay.com' in img_url:
                        # Get high-resolution version
                        img_url = img_url.split('?')[0]
                        image_urls.append(img_url)
        
        # Limit to requested number
        image_urls = image_urls[:num_images]
        
        # Download images
        downloaded_images = []
        for i, img_url in enumerate(image_urls):
            try:
                # Generate filename
                filename = f"{keyword.replace(' ', '-')}_{source}_{aspect_ratio}_{i+1}.jpg"
                filepath = os.path.join(self.cache_dir, source, filename)
                
                # Download image
                img_response = requests.get(img_url, headers=self.headers)
                if img_response.status_code == 200:
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    # Check image quality
                    img = Image.open(filepath)
                    width, height = img.size
                    
                    # Filter by resolution
                    if width >= self.min_resolution:
                        # Check for watermarks
                        if not self._has_watermark(img):
                            # Check quality
                            if self._has_good_quality(img):
                                downloaded_images.append(filepath)
            except Exception as e:
                print(f"Error downloading image {img_url}: {e}")
        
        return downloaded_images
    
    def _scrape_with_crawl4ai(self, keyword, source, aspect_ratio, num_images):
        """
        Scrape images using crawl4ai as a fallback
        
        Args:
            keyword (str): Search keyword
            source (str): Source website
            aspect_ratio (str): Desired aspect ratio
            num_images (int): Number of images to retrieve
            
        Returns:
            list: List of image paths
        """
        # Build URL based on source and keyword
        if source == 'unsplash':
            url = f"https://unsplash.com/s/photos/{keyword.replace(' ', '-')}"
        elif source == 'pexels':
            url = f"https://www.pexels.com/search/{keyword.replace(' ', '%20')}/"
        elif source == 'pixabay':
            url = f"https://pixabay.com/images/search/{keyword.replace(' ', '%20')}/"
        else:
            raise ValueError(f"Unsupported source: {source}")
        
        # Use crawl4ai to get the page content
        page = self.crawler.get(url)
        
        # Extract image URLs based on source
        image_urls = []
        
        if source == 'unsplash':
            # Extract image URLs using crawl4ai's image extraction
            images = page.images
            for img in images:
                if 'images.unsplash.com' in img.url:
                    # Get high-resolution version
                    img_url = img.url.split('?')[0] + '?w=1600&q=80'
                    image_urls.append(img_url)
        elif source == 'pexels':
            # Extract image URLs using crawl4ai's image extraction
            images = page.images
            for img in images:
                if 'images.pexels.com' in img.url:
                    # Get high-resolution version
                    img_url = img.url.split('?')[0]
                    image_urls.append(img_url)
        elif source == 'pixabay':
            # Extract image URLs using crawl4ai's image extraction
            images = page.images
            for img in images:
                if 'pixabay.com' in img.url:
                    # Get high-resolution version
                    img_url = img.url.split('?')[0]
                    image_urls.append(img_url)
        
        # Limit to requested number
        image_urls = image_urls[:num_images]
        
        # Download images
        downloaded_images = []
        for i, img_url in enumerate(image_urls):
            try:
                # Generate filename
                filename = f"{keyword.replace(' ', '-')}_{source}_{aspect_ratio}_{i+1}.jpg"
                filepath = os.path.join(self.cache_dir, source, filename)
                
                # Download image using crawl4ai
                img_response = self.crawler.get(img_url)
                if img_response.status_code == 200:
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    # Check image quality
                    img = Image.open(filepath)
                    width, height = img.size
                    
                    # Filter by resolution
                    if width >= self.min_resolution:
                        # Check for watermarks
                        if not self._has_watermark(img):
                            # Check quality
                            if self._has_good_quality(img):
                                downloaded_images.append(filepath)
            except Exception as e:
                print(f"Error downloading image {img_url}: {e}")
        
        return downloaded_images
    
    def _get_sample_images(self, num_images):
        """
        Get sample images for Streamlit Cloud deployment
        
        Args:
            num_images (int): Number of images to retrieve
            
        Returns:
            list: List of sample image paths
        """
        # Create sample images directory if it doesn't exist
        sample_dir = os.path.join(self.cache_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create sample images if they don't exist
        sample_images = []
        for i in range(1, 6):  # Create 5 sample images
            sample_path = os.path.join(sample_dir, f"sample_{i}.jpg")
            if not os.path.exists(sample_path):
                # Create a simple gradient image
                img = self._create_sample_image(800, 600, i)
                img.save(sample_path, "JPEG", quality=95)
            
            sample_images.append(sample_path)
        
        # Return requested number of sample images
        return sample_images[:num_images]
    
    def _create_sample_image(self, width, height, index):
        """
        Create a sample image with gradient
        
        Args:
            width (int): Image width
            height (int): Image height
            index (int): Image index for color variation
            
        Returns:
            PIL.Image: Sample image
        """
        # Create gradient colors based on index
        if index % 5 == 0:
            start_color = (66, 133, 244)  # Blue
            end_color = (15, 76, 129)
        elif index % 5 == 1:
            start_color = (219, 68, 55)  # Red
            end_color = (128, 25, 18)
        elif index % 5 == 2:
            start_color = (15, 157, 88)  # Green
            end_color = (8, 91, 51)
        elif index % 5 == 3:
            start_color = (244, 180, 0)  # Yellow
            end_color = (183, 129, 3)
        else:
            start_color = (98, 0, 238)  # Purple
            end_color = (55, 0, 121)
        
        # Create gradient image
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for y in range(height):
            for x in range(width):
                # Calculate gradient factor
                factor = y / height
                
                # Calculate color
                r = int(start_color[0] * (1 - factor) + end_color[0] * factor)
                g = int(start_color[1] * (1 - factor) + end_color[1] * factor)
                b = int(start_color[2] * (1 - factor) + end_color[2] * factor)
                
                # Set pixel color
                pixels[x, y] = (r, g, b)
        
        return img
    
    def _has_watermark(self, img):
        """
        Check if image has a watermark using OpenCV
        
        Args:
            img (PIL.Image): Image to check
            
        Returns:
            bool: True if watermark detected, False otherwise
        """
        try:
            # Convert PIL image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to detect text-like regions
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and aspect ratio
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Check if contour has characteristics of a watermark
                if 100 < w < 400 and 20 < h < 100 and 2 < aspect_ratio < 10:
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking for watermark: {e}")
            return False
    
    def _has_good_quality(self, img):
        """
        Check if image has good quality (contrast, brightness)
        
        Args:
            img (PIL.Image): Image to check
            
        Returns:
            bool: True if good quality, False otherwise
        """
        try:
            # Convert PIL image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Calculate standard deviation (contrast)
            std_dev = np.std(gray)
            
            # Calculate mean (brightness)
            mean = np.mean(gray)
            
            # Check if contrast and brightness are within good ranges
            if std_dev > 40 and 50 < mean < 200:
                return True
            
            return False
        except Exception as e:
            print(f"Error checking image quality: {e}")
            return True  # Default to True in case of error
