# Quote Image Generator - Streamlit Cloud Version

This is a modified version of the Quote Image Generator application specifically optimized for Streamlit Cloud deployment. The application has been restructured to address common deployment issues on Streamlit Cloud.

## Features

- **Image Handling**: Uses sample images instead of web scraping to avoid potential issues with external dependencies
- **Multi-language Support**: Supports English, German, Urdu, Hindi, and Arabic with appropriate font selection
- **Intelligent Text Placement**: Uses AI algorithms to position text for maximum visual appeal
- **Watermarking**: Add custom watermarks to your images with adjustable opacity, size, and position
- **SEO Optimization**: Generate SEO-friendly filenames and embed metadata for better discoverability
- **Batch Processing**: Process multiple quotes and images at once
- **Dark Mode**: Toggle between light and dark UI themes

## Deployment Instructions

1. **Clone this repository to your GitHub account**

2. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select this repository
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Enjoy your deployed application!**

## Local Development

To run this application locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

- `app.py`: Main application file
- `image_scraper.py`: Image handling module
- `quote_processor.py`: Quote processing module
- `text_overlay.py`: Text overlay module
- `watermark.py`: Watermarking module
- `seo_optimizer.py`: SEO optimization module
- `streamlit_ui.py`: Streamlit UI module
- `fonts/`: Directory containing default fonts
- `data/`: Directory for storing data
- `images/`: Directory for storing images
- `temp/`: Directory for temporary files

## Troubleshooting

If you encounter any issues with the deployment:

1. Check the Streamlit Cloud logs for error messages
2. Ensure all dependencies are correctly specified in `requirements.txt`
3. Verify that the application has the necessary permissions to create directories

## License

This project is licensed under the MIT License - see the LICENSE file for details.
