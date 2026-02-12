import math
import numpy as np
import pandas as pd
from models import get_laplace_prob, get_good_turing_prob
from preprocessing import replace_unknowns, clean_text

def calculate_perplexity(text_tokens, trainer, smoothing, V):
    """
    Calculates perplexity for a single document list of tokens.
    """
    n = trainer.n
    if len(text_tokens) < n: return float('inf')
    
    # Generate N-grams
    if n == 1: ngrams = text_tokens
    else: ngrams = list(zip(*[text_tokens[i:] for i in range(n)]))
    
    log_prob_sum = 0
    T = len(ngrams)
    
    for gram in ngrams:
        if smoothing == 'laplace':
            prob = get_laplace_prob(gram, trainer, V)
        else:
            prob = get_good_turing_prob(gram, trainer)
            
        if prob > 0:
            log_prob_sum += math.log(prob)
        else:
            return float('inf') # Infinite perplexity if prob is 0
            
    return math.exp(-1/T * log_prob_sum)

def evaluate_models(test_datasets, models, V):
    results = []
    
    for domain, df in test_datasets.items():
        print(f"Evaluating on {domain} ({len(df)} articles)...")
        
        # Pre-process test set with <UNK> once to save time
        processed_docs = []
        # ERROR FIX: We use the 'text' column which is guaranteed by main.py
        for text in df['text']:
            tokens = clean_text(text)
            processed_docs.append(replace_unknowns(tokens, models[0].vocab))
            
        for model in models:
            for smooth in ['laplace', 'good_turing']:
                pp_values = []
                for tokens in processed_docs:
                    pp = calculate_perplexity(tokens, model, smooth, V)
                    if pp != float('inf'):
                        pp_values.append(pp)
                
                avg_pp = np.mean(pp_values) if pp_values else float('inf')
                
                results.append({
                    'Domain': domain,
                    'Model': f"{model.n}-gram",
                    'Smoothing': smooth,
                    'Perplexity': avg_pp
                })
    return pd.DataFrame(results)