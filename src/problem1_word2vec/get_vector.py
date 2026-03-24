
import torch
import os

# Load your trained model
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
model_path = os.path.join(project_root, "outputs", "models", "cbow_embeddings.pt")

data = torch.load(model_path, weights_only=False)

# Choose a word that we know is in your vocabulary
target_word = "research"

if target_word in data['word2idx']:
    # Get the vector
    idx = data['word2idx'][target_word]
    vector = data['embeddings'][idx]
    
    # Format it as a comma-separated string rounded to 4 decimals
    formatted_vector = ", ".join([f"{val:.4f}" for val in vector])
    
    print(f"\nHere is your answer to copy and paste:")
    print(f"{target_word} - {formatted_vector}")
else:
    print("Word not found in vocabulary.")