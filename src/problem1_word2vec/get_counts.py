
import os
from collections import Counter

# Find your cleaned corpus file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
corpus_path = os.path.join(project_root, "data", "processed", "corpus.txt")

if not os.path.exists(corpus_path):
    print(f"Error: Could not find {corpus_path}")
else:
    # Read the entire text
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Split the text into a list of words
    words = text.split()
    
    # Count the frequencies of all words
    word_counts = Counter(words)
    
    # Get the top 10 most common words
    top_10 = word_counts.most_common(10)
    
    # Format it exactly as your assignment requested: word1, freq1, word2, freq2, ...
    formatted_result = ", ".join([f"{word}, {freq}" for word, freq in top_10])
    
    print("\nHere is your answer to copy and paste:")
    print(formatted_result)