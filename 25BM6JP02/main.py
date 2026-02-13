import pandas as pd
import time
import os
from collections import Counter
from preprocessing import clean_text, replace_unknowns, get_filtered_ngrams
from models import NgramTrainer
from evaluation import evaluate_models

# --- CONFIGURATION ---
TRAIN_FILE = 'data/wiki_train.csv' 
TEST_FILES = {
    'Wikipedia': 'data/wiki_test.csv',
    'ArXiv': 'data/arxiv_test.csv',
    'Financial': 'data/financial_test.csv'
}
UNK_THRESHOLD = 0.01  # 1% rule

def main():
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("Loading Training Data...")
    if not os.path.exists(TRAIN_FILE):
        print(f"CRITICAL ERROR: {TRAIN_FILE} not found. Please check 'data' folder.")
        return

    df_train = pd.read_csv(TRAIN_FILE)
    
    # Split Validation (Task 1)
    df_val = df_train.sample(n=100, random_state=42)
    df_train = df_train.drop(df_val.index)
    
    # ---------------------------------------------------------
    # 2. Pre-process & Build Unigram Vocabulary
    # ---------------------------------------------------------
    print("Cleaning text and building Unigram Vocabulary...")
    all_clean_tokens = []
    
    # Safety check for column name
    if 'text' not in df_train.columns:
        if 'title' in df_train.columns: 
             # Assuming single column file structure might vary
             df_train.rename(columns={df_train.columns[-1]: 'text'}, inplace=True)
        else:
             print("Error: Could not locate 'text' column.")
             return

    for text in df_train['text']:
        tokens = clean_text(text)
        all_clean_tokens.append(tokens)
            
    # Select Unigrams appearing in 1% of articles
    valid_unigrams = get_filtered_ngrams(all_clean_tokens, n=1, threshold_pct=UNK_THRESHOLD)
    
    # Add <UNK> to the valid vocabulary
    valid_vocab = set(valid_unigrams)
    valid_vocab.add('<UNK>')
    
    print(f"Unigram Vocabulary Size (V): {len(valid_vocab)}")
    
    # ---------------------------------------------------------
    # 3. Replace <UNK> (Training Data Preparation)
    # ---------------------------------------------------------
    print("Replacing <UNK> in training data...")
    # This creates the final list of lists for training
    training_data_final = [replace_unknowns(doc, valid_vocab) for doc in all_clean_tokens]
    
    # NOTE: We do NOT strictly filter bigrams/trigrams here anymore.
    # Filtering them destroys N1 (items seen once), which breaks Good-Turing.
    # We rely on the fact that we strictly filtered the *words* (unigrams) themselves.

    # ---------------------------------------------------------
    # 4. Train Models (Unigram, Bigram, Trigram)
    # ---------------------------------------------------------
    models = []
    for n in [1, 2, 3]:
        # We pass valid_vocab mainly for reference/unigram counting
        trainer = NgramTrainer(n, valid_vocab) 
        trainer.train(training_data_final)
        models.append(trainer)
        
    # ---------------------------------------------------------
    # 5. Load Test Data
    # ---------------------------------------------------------
    print("\nLoading Test Data...")
    test_dfs = {}
    for name, path in TEST_FILES.items():
        try:
            df = pd.read_csv(path)
            if 'text' not in df.columns:
                df.rename(columns={df.columns[-1]: 'text'}, inplace=True)
            test_dfs[name] = df
        except FileNotFoundError:
            print(f"Warning: {path} not found.")
            
    # ---------------------------------------------------------
    # 6. Evaluation
    # ---------------------------------------------------------
    print("\n--- Starting Evaluation ---")
    results_df = evaluate_models(test_dfs, models, len(valid_vocab))
    
    # ---------------------------------------------------------
    # 7. Final Report
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL PERPLEXITY REPORT")
    print("="*60)
    
    pivot = results_df.pivot_table(index=['Domain', 'Model'], columns='Smoothing', values='Perplexity')
    print(pivot.round(2))
    
    print("\n" + "="*60)
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()