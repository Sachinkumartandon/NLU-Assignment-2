import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. VANILLA RNN ---
class VanillaRNN(nn.Module):
    """
    A standard Recurrent Neural Network.
    Takes the current character and the previous hidden state to predict the next character.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # batch_first=True means the input tensor shape is (batch, seq, feature)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        
        # Pass the RNN output through a linear layer to get probabilities for the next char
        # out shape: (batch_size, seq_len, vocab_size)
        logits = self.fc(out) 
        return logits, hidden

# --- 2. BIDIRECTIONAL LSTM (BLSTM) ---
class CharBLSTM(nn.Module):
    """
    Bidirectional Long Short-Term Memory.
    Uses gates (Forget, Input, Output) to remember long-term dependencies in the names.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(CharBLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # bidirectional=True makes it a BLSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Multiply hidden_size by 2 because bidirectional outputs a forward AND backward state
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        logits = self.fc(out)
        return logits, hidden

# --- 3. RNN WITH BASIC ATTENTION ---
class AttentionRNN(nn.Module):
    """
    An RNN that calculates "Attention Scores" to decide which past characters 
    are the most important to focus on when predicting the next character.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Attention mechanism layer
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        
        # Calculate attention weights
        # out shape: (batch_size, seq_len, hidden_size)
        attn_weights = F.softmax(self.attention(out), dim=1) # (batch_size, seq_len, 1)
        
        # Apply attention weights to the RNN outputs
        context_vector = out * attn_weights
        
        logits = self.fc(context_vector)
        return logits, hidden
