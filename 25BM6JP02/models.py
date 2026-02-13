from collections import Counter

class NgramTrainer:
    def __init__(self, n, valid_vocab):
        self.n = n
        self.vocab = valid_vocab
        self.counts = Counter()
        self.history_counts = Counter()
        self.total_N = 0
        self.nc_dict = Counter()
        
    def train(self, corpus_tokens_lists):
        """
        Trains the model by counting n-grams in the provided corpus.
        """
        print(f"Training {self.n}-gram model...")
        
        for tokens in corpus_tokens_lists:
            if len(tokens) < self.n: continue
            
            # Generate N-grams
            if self.n == 1:
                ngrams = tokens
            else:
                ngrams = list(zip(*[tokens[i:] for i in range(self.n)]))
            
            # Update Counts
            for gram in ngrams:
                self.counts[gram] += 1
                
                if self.n > 1:
                    history = gram[:-1] if self.n > 2 else gram[0]
                    self.history_counts[history] += 1
        
        # Pre-calculate constants
        self.total_N = sum(self.counts.values())
        self.nc_dict = Counter(self.counts.values())

def get_laplace_prob(ngram, trainer, V, k=1):
    """
    P = (Count + k) / (History_Count + k*V)
    """
    count = trainer.counts.get(ngram, 0)
    
    if trainer.n == 1:
        return (count + k) / (trainer.total_N + k * V)
    else:
        history = ngram[:-1] if trainer.n > 2 else ngram[0]
        h_count = trainer.history_counts.get(history, 0)
        return (count + k) / (h_count + k * V)

def get_good_turing_prob(ngram, trainer):
    """
    Calculates Good-Turing probability with P(unseen) = N1 / N
    """
    c = trainer.counts.get(ngram, 0)
    
    # Case 1: Unseen (c=0)
    if c == 0:
        n1 = trainer.nc_dict.get(1, 0)
        # Avoid division by zero
        if trainer.total_N == 0: return 0
        return n1 / trainer.total_N
    
    # Case 2: Seen (c > 0)
    nc = trainer.nc_dict.get(c, 0)
    nc_next = trainer.nc_dict.get(c + 1, 0)
    
    # Standard fallback if bins are empty
    if nc == 0 or nc_next == 0: 
        c_star = c
    else: 
        c_star = (c + 1) * (nc_next / nc)
    
    # Probability P = c* / N(h)
    if trainer.n == 1:
        return c_star / trainer.total_N
    else:
        history = ngram[:-1] if trainer.n > 2 else ngram[0]
        h_count = trainer.history_counts.get(history, 0)
        # Safety for unseen history
        if h_count == 0: return 0 
        return c_star / h_count