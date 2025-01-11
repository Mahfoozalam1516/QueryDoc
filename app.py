import streamlit as st
import PyPDF2
import docx
import pytesseract
from PIL import Image
import requests
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import base64
import hashlib
from langdetect import detect, DetectorFactory
import difflib
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import pickle

# NLTK data path configuration
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)

# Download and store NLTK data locally
def download_nltk_data():
    """Download required NLTK data to local directory"""
    try:
        # Check if punkt is already downloaded
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # Download punkt to custom directory
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)

# Call the download function
download_nltk_data()

# Set random seed for language detection
DetectorFactory.seed = 0

# Update the API_ENDPOINTS dictionary to use a different sentiment model
HUGGINGFACE_API_TOKEN = st.secrets.get("HUGGINGFACE_API_TOKEN")
API_ENDPOINTS = {
    "summarization": "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
    "qa": "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2",
    "sentiment": "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english",
    "zero_shot": "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
}

class DocumentAnalyzer:
    def __init__(self, api_token):
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.setup_session_state()

    @staticmethod
    def setup_session_state():
        """Initialize session state variables"""
        if 'file_history' not in st.session_state:
            st.session_state.file_history = []
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = {}
        if 'current_text' not in st.session_state:
            st.session_state.current_text = ""
            
    def query_huggingface_api(self, api_url: str, payload: dict) -> dict:
        """Make a request to the Hugging Face API with error handling"""
        try:
            response = requests.post(api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None

    def extract_text(self, file) -> str:
        """Extract text from various file formats"""
        file_type = file.type
        try:
            if 'pdf' in file_type:
                return self.extract_text_from_pdf(file)
            elif 'docx' in file_type:
                return self.extract_text_from_docx(file)
            elif 'image' in file_type:
                return self.extract_text_from_image(file)
            elif 'text/plain' in file_type:
                return file.getvalue().decode('utf-8')
            else:
                raise ValueError("Unsupported file type")
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_pdf(file) -> str:
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)

    @staticmethod
    def extract_text_from_docx(file) -> str:
        doc = docx.Document(file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    @staticmethod
    def extract_text_from_image(file) -> str:
        image = Image.open(file)
        return pytesseract.image_to_string(image)

    def get_summary(self, text: str) -> str:
        """Get text summary using chunking for long texts"""
        max_chunk_length = 1024
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks:
            payload = {
                "inputs": chunk,
                "parameters": {"max_length": 130, "min_length": 30, "do_sample": False}
            }
            response = self.query_huggingface_api(API_ENDPOINTS["summarization"], payload)
            if response and isinstance(response, list):
                summaries.append(response[0]["summary_text"])
        
        return " ".join(summaries)

    def get_answer(self, question: str, context: str) -> dict:
        """Get answer to question"""
        payload = {"inputs": {"question": question, "context": context}}
        return self.query_huggingface_api(API_ENDPOINTS["qa"], payload)

    def get_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment of text with improved handling of long texts and error cases
        """
        try:
            # Split long text into chunks of max 512 tokens (approximate by characters)
            max_chunk_length = 500
            chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            results = []
            for chunk in chunks:
                payload = {
                    "inputs": chunk,
                    "options": {"wait_for_model": True}
                }
                response = self.query_huggingface_api(API_ENDPOINTS["sentiment"], payload)
                
                if response and isinstance(response, list) and len(response) > 0:
                    chunk_sentiment = response[0]
                    if isinstance(chunk_sentiment, list):
                        # Get the highest scoring sentiment
                        chunk_sentiment = max(chunk_sentiment, key=lambda x: x['score'])
                    results.append(chunk_sentiment)
            
            if not results:
                return {"error": "No sentiment results available"}
            
            # Aggregate results from all chunks
            overall_sentiment = self.aggregate_sentiments(results)
            return overall_sentiment
            
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return {"error": f"Sentiment analysis failed: {str(e)}"}

    def aggregate_sentiments(self, results: list) -> dict:
        """
        Aggregate sentiment results from multiple chunks
        """
        if not results:
            return {"label": "NEUTRAL", "score": 0.0}
        
        # Calculate weighted average of sentiment scores
        total_score = 0
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        
        for result in results:
            label = result.get('label', 'NEUTRAL')
            score = result.get('score', 0.0)
            
            sentiment_counts[label] += 1
            total_score += score
        
        # Determine dominant sentiment
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        average_score = total_score / len(results)
        
        return {
            "label": dominant_sentiment,
            "score": average_score,
            "details": {
                "chunk_count": len(results),
                "sentiment_distribution": sentiment_counts
            }
        }


    def analyze_text(self, text: str) -> dict:
        """Perform comprehensive text analysis"""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Basic statistics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Word frequency
        word_freq = Counter(words)
        
        # Language detection
        try:
            language = detect(text)
        except:
            language = "unknown"
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "word_frequency": dict(word_freq.most_common(10)),
            "language": language
        }

    def generate_visualizations(self, analysis: dict):
        """Generate visualization for text analysis"""
        # Word frequency bar chart
        fig_word_freq = px.bar(
            x=list(analysis["word_frequency"].keys()),
            y=list(analysis["word_frequency"].values()),
            title="Top 10 Word Frequencies"
        )
        st.plotly_chart(fig_word_freq)

        # Word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(analysis["word_frequency"])
        fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wordcloud)

    def compare_documents(self, text1: str, text2: str) -> dict:
        """Compare two documents and return similarity metrics"""
        # Calculate similarity ratio
        similarity_ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # Find common phrases
        sentences1 = set(sent_tokenize(text1))
        sentences2 = set(sent_tokenize(text2))
        common_sentences = sentences1.intersection(sentences2)
        
        return {
            "similarity_ratio": similarity_ratio,
            "common_sentences": list(common_sentences)
        }

    def export_analysis(self, text: str, analysis: dict, summary: str = None) -> BytesIO:
        """Export analysis results to PDF"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        story.append(Paragraph("Document Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Add summary if available
        if summary:
            story.append(Paragraph("Summary:", styles['Heading1']))
            story.append(Paragraph(summary, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Add analysis results
        story.append(Paragraph("Text Statistics:", styles['Heading1']))
        story.append(Paragraph(f"Word Count: {analysis['word_count']}", styles['Normal']))
        story.append(Paragraph(f"Sentence Count: {analysis['sentence_count']}", styles['Normal']))
        story.append(Paragraph(f"Average Sentence Length: {analysis['avg_sentence_length']:.2f}", styles['Normal']))
        story.append(Paragraph(f"Detected Language: {analysis['language']}", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

def main():
    st.set_page_config(page_title="Advanced Document Analysis Platform", layout="wide")
    
    # Initialize the DocumentAnalyzer
    if HUGGINGFACE_API_TOKEN:
        analyzer = DocumentAnalyzer(HUGGINGFACE_API_TOKEN)
    else:
        st.error("Please set up your Hugging Face API token in the secrets.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Document Analysis", "Document Comparison", "History"])

    if page == "Document Analysis":
        st.title("📄 Document Analysis Platform")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, DOCX, Image, or TXT)", 
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'txt']
        )
        
        if uploaded_file:
            # Extract and display text
            with st.spinner("Processing document..."):
                text = analyzer.extract_text(uploaded_file)
                st.session_state.current_text = text
                
                # Add to history
                file_hash = hashlib.md5(text.encode()).hexdigest()
                if file_hash not in [h['hash'] for h in st.session_state.file_history]:
                    st.session_state.file_history.append({
                        'name': uploaded_file.name,
                        'hash': file_hash,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'text': text
                    })
            
            # Analysis options
            st.subheader("Analysis Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = analyzer.get_summary(text)
                        st.subheader("Summary")
                        st.write(summary)
            
            with col2:
                if st.button("Analyze Text"):
                    with st.spinner("Analyzing text..."):
                        analysis = analyzer.analyze_text(text)
                        st.subheader("Text Analysis")
                        st.write(analysis)
                        analyzer.generate_visualizations(analysis)
            
            with col3:
                if st.button("Analyze Sentiment"):
                    with st.spinner("Analyzing sentiment..."):
                        sentiment_result = analyzer.get_sentiment(text)
                        
                        if "error" in sentiment_result:
                            st.error(sentiment_result["error"])
                        else:
                            st.subheader("Sentiment Analysis")
                            
                            # Display overall sentiment with appropriate color
                            sentiment_color = {
                                "POSITIVE": "green",
                                "NEGATIVE": "red",
                                "NEUTRAL": "blue"
                            }
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                    <div style='padding: 10px; border-radius: 5px; background-color: {sentiment_color.get(sentiment_result['label'], 'gray')}25;'>
                                        <h3 style='color: {sentiment_color.get(sentiment_result['label'], 'gray')}; margin: 0;'>
                                            {sentiment_result['label']}
                                        </h3>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.metric(
                                    label="Confidence Score",
                                    value=f"{sentiment_result['score']:.2%}"
                                )
                            
                            # Display detailed analysis if available
                            if "details" in sentiment_result:
                                with st.expander("See detailed analysis"):
                                    st.write("Analysis Details:")
                                    st.write(f"- Number of text chunks analyzed: {sentiment_result['details']['chunk_count']}")
                                    st.write("- Sentiment distribution:")
                                    for sentiment, count in sentiment_result['details']['sentiment_distribution'].items():
                                        st.write(f"  • {sentiment}: {count} chunks")
            
            # Question Answering
            st.subheader("Ask Questions")
            question = st.text_input("Enter your question about the document:")
            if question:
                with st.spinner("Finding answer..."):
                    answer = analyzer.get_answer(question, text)
                    if answer:
                        st.write("Answer:", answer.get('answer', 'No answer found'))
                        st.write("Confidence Score:", f"{answer.get('score', 0):.2%}")
            
            # Export options
            st.subheader("Export Options")
            if st.button("Export Analysis Report"):
                analysis = analyzer.analyze_text(text)
                summary = analyzer.get_summary(text)
                pdf_buffer = analyzer.export_analysis(text, analysis, summary)
                st.download_button(
                    "Download Report",
                    pdf_buffer,
                    file_name="analysis_report.pdf",
                    mime="application/pdf"
                )

    elif page == "Document Comparison":
        st.title("📊 Document Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Document 1")
            doc1 = st.selectbox(
                "Select first document",
                options=[doc['name'] for doc in st.session_state.file_history],
                key="doc1"
            )
        
        with col2:
            st.subheader("Document 2")
            doc2 = st.selectbox(
                "Select second document",
                options=[doc['name'] for doc in st.session_state.file_history],
                key="doc2"
            )
        
        if st.button("Compare Documents"):
            if doc1 and doc2:
                text1 = next(doc['text'] for doc in st.session_state.file_history if doc['name'] == doc1)
                text2 = next(doc['text'] for doc in st.session_state.file_history if doc['name'] == doc2)
                
                comparison = analyzer.compare_documents(text1, text2)
                
                st.subheader("Comparison Results")
                st.write(f"Similarity Score: {comparison['similarity_ratio']:.2%}")
                
                if comparison['common_sentences']:
                    st.write("Common Sentences:")
                    for sentence in comparison['common_sentences']:
                        st.write(f"- {sentence}")

    else:  # History page
        st.title("📚 Document History")
        
        if st.session_state.file_history:
            for doc in st.session_state.file_history:
                with st.expander(f"{doc['name']} - {doc['timestamp']}"):
                    st.write("Document Preview:")
                    st.write(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
        else:
            st.write("No documents in history")

if __name__ == "__main__":
    main()