"""
Text Normalization Pipeline
----------------------------
Implements the preprocessing steps described in Chapter 3.1.3 (Data) of the
thesis. Before any text is fed into the Sentence-BERT (SBERT) model, it is
cleaned using three computational steps:

    1. Tokenization      - split continuous text into individual word tokens
    2. Stop-word removal - filter out common, non-semantic words
                            (e.g. "is", "the", "at")
    3. Lemmatization      - reduce words to their base dictionary form
                            (e.g. "grading", "graded", "grades" -> "grade")

This keeps the semantic comparison focused on the core meaning of a
response rather than surface-level wording differences.
"""

import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set NLTK data path to local folder - use absolute path
nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')
print(f"📁 Looking for NLTK data at: {nltk_data_path}")

if os.path.exists(nltk_data_path):
    nltk.data.path.insert(0, nltk_data_path)
    print(f"✅ NLTK data path set to: {nltk_data_path}")
else:
    print(f"⚠️ NLTK data folder not found at: {nltk_data_path}")
    # Try current directory
    alt_path = os.path.join(os.getcwd(), 'nltk_data')
    if os.path.exists(alt_path):
        nltk.data.path.insert(0, alt_path)
        print(f"✅ Using alternate path: {alt_path}")

# Print all NLTK paths for debugging
print("📂 NLTK search paths:")
for path in nltk.data.path:
    print(f"  - {path}")

# Initialize lemmatizer and stopwords
try:
    _lemmatizer = WordNetLemmatizer()
    _stop_words = set(stopwords.words("english"))
    print(f"✅ NLTK resources loaded successfully! Found {len(_stop_words)} stopwords.")
except LookupError as e:
    print(f"❌ Error loading NLTK resources: {e}")
    print("Attempting to fix by downloading...")
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._default_http_context = _create_unverified_https_context
        
        # Try to download if not found
        nltk.download('stopwords', quiet=True, download_dir=nltk_data_path)
        nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
        nltk.download('wordnet', quiet=True, download_dir=nltk_data_path)
        
        # Try again
        _lemmatizer = WordNetLemmatizer()
        _stop_words = set(stopwords.words("english"))
        print(f"✅ NLTK resources loaded after download!")
    except Exception as e2:
        print(f"❌ Still having issues: {e2}")
        # Fallback: use predefined stopwords
        _lemmatizer = WordNetLemmatizer()
        _stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'to', 'of', 'for', 'with', 'on', 'at', 'from', 'by', 'in', 'as',
                      'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'than',
                      'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                      'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                      'her', 'our', 'their', 'own', 'same', 'so', 'than', 'too', 'very'}


def normalize_text(text: str) -> str:
    """
    Runs the full Text Normalization Pipeline on a raw response string
    and returns the cleaned text, ready for SBERT embedding.
    """
    if not text or not text.strip():
        return ""

    # Lowercase and strip punctuation/symbols before tokenizing
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())

    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback: simple split if tokenizer not found
        tokens = text.split()
    
    tokens = [t for t in tokens if t.strip() and t.lower() not in _stop_words]
    lemmas = [_lemmatizer.lemmatize(t.lower()) for t in tokens]

    return " ".join(lemmas)
