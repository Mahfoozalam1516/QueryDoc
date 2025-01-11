# Document Analysis Platform

A comprehensive document analysis platform built with Streamlit that provides text summarization, question answering, sentiment analysis, and document comparison capabilities.

## Features

- Document Processing:

  - Support for PDF, DOCX, Images (OCR), and TXT files
  - Text extraction and analysis
  - Document comparison
  - History tracking

- Analysis Capabilities:

  - Text summarization
  - Question answering
  - Sentiment analysis
  - Language detection
  - Word frequency analysis
  - Text statistics
  - Visualizations (word clouds and charts)

- Export Options:
  - PDF report generation
  - Analysis results download

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/document-analyzer.git
cd document-analyzer
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:

- Windows: Download and install from [Github](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

## Configuration

1. Create a `.streamlit` folder in your project directory
2. Create a `secrets.toml` file inside `.streamlit` folder:

```toml
HUGGINGFACE_API_TOKEN = "your-api-token-here"
```

3. Get your Hugging Face API token from [Hugging Face](https://huggingface.co/settings/tokens)

## Running the Application

```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your repository
4. Add your Hugging Face API token in the secrets management section
5. Deploy!

## Project Structure

```
document-analyzer/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── .streamlit/           # Streamlit configuration
│   └── secrets.toml      # API keys and secrets
└── nltk_data/            # NLTK data directory
    └── tokenizers/       # NLTK tokenizer data
```

## Troubleshooting

1. If you encounter NLTK data errors:

   - The application will automatically download required NLTK data
   - Check if the `nltk_data` directory exists in your project folder

2. If you get Tesseract OCR errors:

   - Ensure Tesseract is properly installed
   - Add Tesseract to your system PATH

3. If you get API errors:
   - Verify your Hugging Face API token
   - Check your internet connection
   - Ensure you're not exceeding API rate limits
