import re
import string
import nltk
import itertools
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)

def clean_text(text):
    """
    Cleaning pipeline matching your notebook:
    1. Remove URLs
    2. Lowercase
    3. Remove punctuation
    4. Handle numbers/dates
    5. Remove stopwords and lemmatize
    """
    if not isinstance(text, str):
        return [] # Handle NaN/Empty values gracefully
    
    # 1. Basic Cleaning
    text = re.sub(r"http\S+", "", text)
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    text = text.lower()
    text = text.translate(punctuation_table)
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
    
    # 2. Tokenization
    raw_tokens = text.split()
    
    # 3. Number/Date Logic (From your Notebook)
    processed_tokens = []
    for token in raw_tokens:
        if token.isdigit():
            # Heuristic for years
            if len(token) == 4 and 1000 <= int(token) <= 2100:
                processed_tokens.append("date")
            else:
                processed_tokens.append("number")
        else:
            processed_tokens.append(token)
            
    # 4. Stopword Removal & Lemmatization
    final_tokens = [
        lemmatizer.lemmatize(t) 
        for t in processed_tokens 
        if t not in stop_words and len(t) > 1
    ]
    
    return final_tokens

def replace_unknowns(tokens, valid_vocab):
    """
    Replaces tokens not in valid_vocab with <UNK>.
    """
    if not tokens: return []
    return [token if token in valid_vocab else '<UNK>' for token in tokens]