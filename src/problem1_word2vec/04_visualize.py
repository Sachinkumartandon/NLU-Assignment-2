import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def plot_embeddings(embeddings, word2idx, idx2word, words_to_plot, title, filename):
    """Reduces dimensionality using PCA and plots the words."""
    # Filter words that actually exist in our vocabulary
    valid_words = [w for w in words_to_plot if w in word2idx]
    
    if not valid_words:
        print(f"None of the target words found for {title}.")
        return

    # Extract the vectors for these specific words
    indices = [word2idx[w] for w in valid_words]
    vectors = np.array([embeddings[idx] for idx in indices])
    
    # Use PCA to reduce the 50-dimensional vectors down to 2 dimensions (X and Y coordinates)
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(vectors)
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='teal', edgecolors='k', s=50)
    
    # Annotate each point with its corresponding word
    for i, word in enumerate(valid_words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                     xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=10)
        
    plt.title(title, fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()

def main():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    models_dir = os.path.join(project_root, "outputs", "models")
    plots_dir = os.path.join(project_root, "outputs", "plots")
    
    cbow_path = os.path.join(models_dir, "cbow_embeddings.pt")
    sg_path = os.path.join(models_dir, "skipgram_embeddings.pt")
    
    # Load models
    cbow_data = torch.load(cbow_path, weights_only=False)
    sg_data = torch.load(sg_path, weights_only=False)
    
    # Select words that form clear semantic clusters (Academics, Degrees, Roles)
    target_words = [
        'btech', 'mtech', 'phd', 'ug', 'pg', 'degree',
        'student', 'faculty', 'instructor', 'supervisor', 'candidate',
        'research', 'thesis', 'project', 'examination', 'evaluation',
        'computer', 'science', 'biology', 'engineering', 'artificial', 'intelligence'
    ]
    
    print("\n--- Generating PCA Visualizations ---")
    plot_embeddings(cbow_data['embeddings'], cbow_data['word2idx'], cbow_data['idx2word'], 
                    target_words, "CBOW Word Embeddings (PCA)", 
                    os.path.join(plots_dir, "cbow_clusters.png"))
                    
    plot_embeddings(sg_data['embeddings'], sg_data['word2idx'], sg_data['idx2word'], 
                    target_words, "Skip-Gram Word Embeddings (PCA)", 
                    os.path.join(plots_dir, "skipgram_clusters.png"))
                    
    print("\nSUCCESS: Task 4 is complete! Check your 'outputs/plots' folder.")

if __name__ == "__main__":
    main()