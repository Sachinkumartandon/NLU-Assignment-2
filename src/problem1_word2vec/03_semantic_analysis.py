import torch
import torch.nn.functional as F
import os

def load_model(filepath):
    """Loads the saved PyTorch dictionary containing embeddings and vocab."""
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return None
    return torch.load(filepath, weights_only=False)

def get_nearest_neighbors(word, embeddings, word2idx, idx2word, top_k=5):
    """
    Finds the top_k most similar words using Cosine Similarity.
    Formula: CosineSim(A, B) = (A dot B) / (||A|| * ||B||)
    """
    if word not in word2idx:
        return [f"'{word}' not in vocabulary."]
    
    # Get the index and vector for the target word
    word_idx = word2idx[word]
    word_vec = torch.tensor(embeddings[word_idx]).unsqueeze(0) # [1, embed_dim]
    
    # Convert entire embedding matrix to tensor
    all_vecs = torch.tensor(embeddings) # [vocab_size, embed_dim]
    
    # Calculate cosine similarity between the target word and ALL other words
    # F.cosine_similarity automatically handles the vector normalization
    similarities = F.cosine_similarity(word_vec, all_vecs)
    
    # Sort the similarities in descending order
    # Skip the first result (index 0) because it will be the word itself (similarity 1.0)
    top_indices = torch.argsort(similarities, descending=True)[1:top_k+1]
    
    # Map indices back to words and format the output
    results = []
    for idx in top_indices:
        sim_score = similarities[idx].item()
        results.append(f"{idx2word[idx.item()]} ({sim_score:.4f})")
        
    return results

def get_analogy(word_a, word_b, word_c, embeddings, word2idx, idx2word, top_k=3):
    """
    Solves analogies: A is to B as C is to ?
    Mathematically: Vector_B - Vector_A + Vector_C = Target Vector
    Example: UG : BTech :: PG : ? -> Vector(BTech) - Vector(UG) + Vector(PG)
    """
    # Check if all words exist in our vocabulary
    for w in [word_a, word_b, word_c]:
        if w not in word2idx:
            return f"Error: '{w}' not in vocabulary."
            
    # Extract vectors
    vec_a = torch.tensor(embeddings[word2idx[word_a]])
    vec_b = torch.tensor(embeddings[word2idx[word_b]])
    vec_c = torch.tensor(embeddings[word2idx[word_c]])
    
    # Perform vector arithmetic
    target_vec = (vec_b - vec_a + vec_c).unsqueeze(0)
    all_vecs = torch.tensor(embeddings)
    
    # Calculate cosine similarities against the computed target vector
    similarities = F.cosine_similarity(target_vec, all_vecs)
    
    # Sort and return top results (excluding the input words to avoid trivial answers)
    top_indices = torch.argsort(similarities, descending=True)
    
    results = []
    for idx in top_indices:
        word = idx2word[idx.item()]
        if word not in [word_a, word_b, word_c]: # Prevent input words from being the answer
            sim_score = similarities[idx].item()
            results.append(f"{word} ({sim_score:.4f})")
            if len(results) == top_k:
                break
                
    return results

def run_analysis(model_name, filepath):
    """Main function to run the required assignments tests on a specific model."""
    print(f"\n" + "="*50)
    print(f"   ANALYSIS FOR: {model_name}")
    print("="*50)
    
    data = load_model(filepath)
    if not data: return
    
    embeddings = data['embeddings']
    word2idx = data['word2idx']
    idx2word = data['idx2word']
    
    # --- 1. NEAREST NEIGHBORS (Mandatory words) ---
    print("\n--- 1. Top 5 Nearest Neighbors ---")
    # Note: Using 'examination' as a fallback since 'exam' might be filtered out in formal text
    target_words = ['research', 'student', 'phd', 'examination'] 
    
    for word in target_words:
        neighbors = get_nearest_neighbors(word, embeddings, word2idx, idx2word)
        print(f"Target: '{word}' -> {', '.join(neighbors)}")
        
    # --- 2. ANALOGY EXPERIMENTS ---
    print("\n--- 2. Analogy Experiments ---")
    # Format: (Word A, Word B, Word C) -> A is to B as C is to ?
    analogies = [
        ('ug', 'btech', 'pg'),        # Mandatory Assignment Analogy
        ('student', 'study', 'faculty'), # Custom Analogy 1
        ('btech', 'engineering', 'phd')  # Custom Analogy 2
    ]
    
    for a, b, c in analogies:
        result = get_analogy(a, b, c, embeddings, word2idx, idx2word)
        if isinstance(result, str): # Error message
            print(f"[{a} : {b} :: {c} : ?] -> {result}")
        else:
            print(f"[{a} : {b} :: {c} : ?] -> {result[0]} (Top match)")

def main():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    models_dir = os.path.join(project_root, "outputs", "models")
    
    cbow_path = os.path.join(models_dir, "cbow_embeddings.pt")
    sg_path = os.path.join(models_dir, "skipgram_embeddings.pt")
    
    # Run analysis on both models to compare them for the report
    run_analysis("CONTINUOUS BAG OF WORDS (CBOW)", cbow_path)
    run_analysis("SKIP-GRAM", sg_path)
    
    print("\nNote for Report: The assignment asks you to discuss if these results are semantically meaningful.")

if __name__ == "__main__":
    main()