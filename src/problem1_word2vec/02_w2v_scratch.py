import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from collections import Counter
import random
import pickle

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. DATA PREPARATION ---
def load_and_prepare_data(filepath, min_freq=2):
    """Reads the cleaned corpus, builds the vocabulary, and filters rare words."""
    print(f"Loading corpus from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().split()

    # Count word frequencies and filter out very rare words (noise)
    word_counts = Counter(text)
    vocab_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create word-to-index and index-to-word mappings
    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Convert the text into a list of indices
    data_indices = [word2idx[word] for word in text if word in word2idx]
    
    print(f"Original Corpus Size: {len(text)} words")
    print(f"Filtered Vocab Size: {len(vocab_words)} unique words")
    
    return data_indices, word2idx, idx2word

# --- 2. CONTEXT PAIR GENERATION ---
def generate_training_data(data_indices, window_size=2):
    """Generates training pairs for CBOW and Skip-gram."""
    cbow_data = []
    skipgram_data = []
    
    print("Generating context pairs...")
    for i in range(window_size, len(data_indices) - window_size):
        # Context is words before and after the center word
        context = data_indices[i - window_size : i] + data_indices[i + 1 : i + window_size + 1]
        center = data_indices[i]
        
        cbow_data.append((context, center))
        for ctx_word in context:
            skipgram_data.append((center, ctx_word))
            
    return cbow_data, skipgram_data

# --- PyTorch Dataset Helper ---
class W2VDataset(Dataset):
    """Helper class to load our pairs in batches during training."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# --- 3. PYTORCH MODEL ARCHITECTURES ---
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        # Target word embedding (Center word)
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Context word embedding
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, target_idx, context_idx, negative_indices):
        """Calculates loss using Negative Sampling."""
        # Positive score (Target dot Context)
        v_target = self.target_embeddings(target_idx) # [batch, embed_dim]
        v_context = self.context_embeddings(context_idx) # [batch, embed_dim]
        pos_score = torch.sum(v_target * v_context, dim=1)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)
        
        # Negative score (Target dot Negative Samples)
        v_negatives = self.context_embeddings(negative_indices) # [batch, num_neg, embed_dim]
        neg_score = torch.bmm(v_negatives, v_target.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(torch.nn.functional.logsigmoid(-neg_score), dim=1)
        
        return (pos_loss + neg_loss).mean()

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size) # Predicts the exact center word
        
    def forward(self, context_indices):
        # Get embeddings for all context words and average them
        embeds = self.embeddings(context_indices) 
        avg_embeds = torch.mean(embeds, dim=1)    
        out = self.linear(avg_embeds)             
        return out

# --- 4. TRAINING LOOPS ---
def train_cbow(model, data, epochs, lr, batch_size):
    print("\n--- Training CBOW Model ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(W2VDataset(data), batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for contexts, targets in dataloader:
            optimizer.zero_grad()
            out = model(contexts)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    return model

def train_skipgram(model, data, vocab_size, num_neg_samples, epochs, lr, batch_size):
    print("\n--- Training Skip-gram Model (with Negative Sampling) ---")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(W2VDataset(data), batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for centers, contexts in dataloader:
            batch_size_actual = centers.size(0)
            # Generate random negative samples
            negatives = torch.randint(0, vocab_size, (batch_size_actual, num_neg_samples))
            
            optimizer.zero_grad()
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    return model

# --- 5. MAIN EXECUTION ---
def main():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    corpus_path = os.path.join(project_root, "data", "processed", "cleaned_corpus.txt")
    models_dir = os.path.join(project_root, "outputs", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Hyperparameters (Assignment requires experimenting with these)
    EMBED_DIM = 50
    WINDOW_SIZE = 2
    NUM_NEG_SAMPLES = 5
    EPOCHS = 10
    LEARNING_RATE = 0.01
    BATCH_SIZE = 256
    
    # 1. Load data
    data_indices, word2idx, idx2word = load_and_prepare_data(corpus_path)
    vocab_size = len(word2idx)
    
    # 2. Generate Pairs
    cbow_data, skipgram_data = generate_training_data(data_indices, window_size=WINDOW_SIZE)
    
    # 3. Train CBOW
    cbow_model = CBOWModel(vocab_size, EMBED_DIM)
    cbow_model = train_cbow(cbow_model, cbow_data, EPOCHS, LEARNING_RATE, BATCH_SIZE)
    
    # 4. Train Skip-gram
    sg_model = SkipGramModel(vocab_size, EMBED_DIM)
    sg_model = train_skipgram(sg_model, skipgram_data, vocab_size, NUM_NEG_SAMPLES, EPOCHS, LEARNING_RATE, BATCH_SIZE)
    
    # 5. Save the learned embeddings securely
    print("\nSaving Embeddings...")
    cbow_save_data = {
        'embeddings': cbow_model.embeddings.weight.data.numpy(),
        'word2idx': word2idx,
        'idx2word': idx2word
    }
    sg_save_data = {
        'embeddings': sg_model.target_embeddings.weight.data.numpy(),
        'word2idx': word2idx,
        'idx2word': idx2word
    }
    
    torch.save(cbow_save_data, os.path.join(models_dir, "cbow_embeddings.pt"))
    torch.save(sg_save_data, os.path.join(models_dir, "skipgram_embeddings.pt"))
    print(f"Successfully saved to {models_dir}")

if __name__ == "__main__":
    main()