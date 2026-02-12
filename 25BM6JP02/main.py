import pandas as pd
import time
from collections import Counter
from preprocessing import clean_text, replace_unknowns
from models import NgramTrainer
from evaluation import evaluate_models
import os

# --- CONFIGURATION ---
# Ensure these match your actual folder structure
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
    # 2. Build Vocabulary (The 1% Rule)
    # ---------------------------------------------------------
    print("Building Vocabulary (1% Rule)...")
    article_counts = Counter()
    all_clean_tokens = []
    
    # First pass: Clean and count document frequency
    # Note: 'text' column assumed in train file based on your notebook
    for text in df_train['text']:
        tokens = clean_text(text)
        all_clean_tokens.append(tokens)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            article_counts[t] += 1
            
    min_articles = len(df_train) * UNK_THRESHOLD
    valid_vocab = {word for word, count in article_counts.items() if count >= min_articles}
    valid_vocab.add('<UNK>') 
    
    print(f"Vocabulary Size (V): {len(valid_vocab)}")
    
    # ---------------------------------------------------------
    # 3. Replace <UNK> in Training Data
    # ---------------------------------------------------------
    print("Processing Training Data with <UNK>...")
    # This creates the final list of lists for training
    training_data_final = [replace_unknowns(doc, valid_vocab) for doc in all_clean_tokens]
    
    # ---------------------------------------------------------
    # 4. Train Models (Unigram, Bigram, Trigram)
    # ---------------------------------------------------------
    models = []
    for n in [1, 2, 3]:
        trainer = NgramTrainer(n, valid_vocab)
        trainer.train(training_data_final)
        models.append(trainer)
        
    # ---------------------------------------------------------
    # 5. Load Test Data (WITH ERROR FIX)
    # ---------------------------------------------------------
    print("\nLoading Test Data...")
    test_dfs = {}
    for name, path in TEST_FILES.items():
        try:
            df = pd.read_csv(path)
            
            # --- FIX FOR KEYERROR: 'text' ---
            # If 'text' column is missing, rename the last column to 'text'
            if 'text' not in df.columns:
                last_col = df.columns[-1]
                df.rename(columns={last_col: 'text'}, inplace=True)
            # -------------------------------
                
            test_dfs[name] = df
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping {name}.")
            
    # ---------------------------------------------------------
    # 6. Evaluation
    # ---------------------------------------------------------
    print("\n--- Starting Evaluation ---")
    results_df = evaluate_models(test_dfs, models, len(valid_vocab))
    
    # ---------------------------------------------------------
    # 7. Final Report Formatting
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL PERPLEXITY REPORT (Lower is Better)")
    print("="*60)
    
    # Pivot for cleaner display
    pivot = results_df.pivot_table(index=['Domain', 'Model'], columns='Smoothing', values='Perplexity')
    print(pivot.round(2))
    
    print("\n" + "="*60)
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()