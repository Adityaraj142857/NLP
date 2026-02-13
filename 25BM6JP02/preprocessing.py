import re
import string
import nltk
import itertools
from collections import Counter
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
    Cleaning pipeline: Lowercase, remove URLs/Punctuation, handle numbers, lemmatize.
    """
    if not isinstance(text, str):
        return [] 
    
    text = re.sub(r"http\S+", "", text)
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    text = text.lower()
    text = text.translate(punctuation_table)
    text = re.sub(r'\s+', ' ', text)
    
    raw_tokens = text.split()
    processed_tokens = []
    
    for token in raw_tokens:
        if token.isdigit():
            if len(token) == 4 and 1000 <= int(token) <= 2100:
                processed_tokens.append("date")
            else:
                processed_tokens.append("number")
        else:
            processed_tokens.append(token)
            
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

def get_filtered_ngrams(corpus_tokens, n, threshold_pct=0.01):
    """
    [cite: 88] Returns n-grams that appear in at least threshold_pct of articles.
    
    Args:
        corpus_tokens: List of Lists of tokens (e.g., [["apple", "eat"], ["banana", ...]])
        n: The 'n' in n-gram (1, 2, 3)
        threshold_pct: The percentage threshold (default 0.01 for 1%)
    """
    article_counts = Counter()
    total_articles = len(corpus_tokens)
    min_articles = total_articles * threshold_pct

    for tokens in corpus_tokens:
        # Create unique n-grams for this specific article
        if n == 1:
            unique_ngrams = set(tokens)
        else:
            # Generate n-grams list then convert to set for uniqueness in this article
            if len(tokens) < n:
                unique_ngrams = set()
            else:
                ngrams_list = zip(*[tokens[i:] for i in range(n)])
                unique_ngrams = set(ngrams_list)

        # Update how many articles each n-gram appears in
        for gram in unique_ngrams:
            article_counts[gram] += 1

    # Filter by the 1% threshold
    valid_ngrams = {gram for gram, count in article_counts.items() if count >= min_articles}
    return valid_ngrams