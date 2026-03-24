import torch
from torch.utils.data import Dataset
import os

class NameDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading names from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read names, lowercase them, and remove empty lines
            self.names = [line.strip().lower() for line in f if line.strip()]
            
        # Create character vocabulary
        # We add special tokens: <PAD> (for batching), <SOS> (Start), <EOS> (End)
        self.chars = ['<PAD>', '<SOS>', '<EOS>'] + sorted(list(set(''.join(self.names))))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Find the longest name so we can pad all names to the same length for batching
        self.max_len = max(len(name) for name in self.names) + 2 # +2 for SOS and EOS
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        
        # Convert name characters to numbers, wrapped in SOS and EOS tokens
        indices = [self.char2idx['<SOS>']] + [self.char2idx[ch] for ch in name] + [self.char2idx['<EOS>']]
        
        # Pad with <PAD> tokens so all tensors are the exact same size
        padded_indices = indices + [self.char2idx['<PAD>']] * (self.max_len - len(indices))
        
        tensor = torch.tensor(padded_indices, dtype=torch.long)
        
        # Input (x) is the sequence without the last character
        # Target (y) is the sequence shifted by one (predicting the next character)
        x = tensor[:-1]
        y = tensor[1:]
        
        return x, y
