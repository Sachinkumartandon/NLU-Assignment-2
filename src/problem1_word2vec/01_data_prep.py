import requests
from bs4 import BeautifulSoup
import re
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import PyPDF2
import urllib3

# Disable SSL warnings for cleaner terminal output when scraping
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Download necessary NLTK tokenization datasets (runs quietly)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def scrape_webpage(url):
    """
    Fetches a webpage and extracts all readable text.
    Uses headers and verify=False to bypass strict academic firewalls.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all readable text from the HTML, replacing tags with spaces
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"  -> Error scraping {url}: {e}")
        return ""

def preprocess_text(text):
    """
    Cleans the extracted text according to assignment requirements:
    1. Lowercasing
    2. Removal of formatting artifacts and excessive punctuation
    3. Tokenization
    """
    # Convert all text to lowercase to maintain uniformity
    text = text.lower()
    
    # Regex [^a-z\s] replaces anything that is NOT a lowercase letter or space with a space.
    # This automatically removes numbers, punctuation, and weird formatting symbols.
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace created by the previous regex step
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize the cleaned string into a list of individual words using NLTK
    tokens = nltk.word_tokenize(text)
    
    return tokens

def main():
    # --- DYNAMIC PATH SETUP ---
    # This ensures paths work perfectly regardless of where you run the script from in VS Code
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    
    data_raw_dir = os.path.join(project_root, "data", "raw")
    data_processed_dir = os.path.join(project_root, "data", "processed")
    outputs_plot_dir = os.path.join(project_root, "outputs", "plots")
    
    # Create necessary directories if they don't already exist
    os.makedirs(data_processed_dir, exist_ok=True)
    os.makedirs(outputs_plot_dir, exist_ok=True)

    raw_texts = [] # List to hold huge strings of text from each source

    # --- 1. DATA COLLECTION (WEBPAGES) ---
    urls = [
        "https://iitj.ac.in/academics/index.php?id=btech",
        "https://iitj.ac.in/department/index.php?id=cse",
        "https://iitj.ac.in/research/index.php"
    ]
    
    print("--- Starting Web Scraping ---")
    for url in urls:
        print(f"Scraping: {url}")
        scraped_text = scrape_webpage(url)
        if scraped_text:
            raw_texts.append(scraped_text)
            print(f"  -> Success! Extracted {len(scraped_text)} characters.")

    # --- 2. DATA COLLECTION (ALL PDFs IN RAW FOLDER) ---
    print("\n--- Looking for PDFs in: data/raw/ ---")
    
    # Loop through every file in the raw directory dynamically
    for filename in os.listdir(data_raw_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_raw_dir, filename)
            print(f"Extracting text from: {filename}")
            
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    pdf_text = ""
                    # Loop through all pages in the current PDF
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            pdf_text += extracted + " "
                    
                    if pdf_text.strip():
                        raw_texts.append(pdf_text)
                        print(f"  -> Success! Extracted {len(pdf_text)} characters.")
                    else:
                        print("  -> Warning: PDF read, but no text found.")
                        
            except Exception as e:
                print(f"  -> Error reading {filename}: {e}")

    # --- SAFETY CHECK ---
    if not raw_texts:
        print("\nCRITICAL ERROR: No text was collected from either the websites or the PDFs.")
        return

    # --- 3. PREPROCESSING ---
    print("\n--- Preprocessing collected text ---")
    all_tokens = []
    for doc_text in raw_texts:
        tokens = preprocess_text(doc_text)
        all_tokens.extend(tokens)

    if not all_tokens:
        print("Error: No words remained after preprocessing.")
        return

    # Save the final cleaned corpus to a single text file (Deliverable Requirement)
    corpus_path = os.path.join(data_processed_dir, 'cleaned_corpus.txt')
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(" ".join(all_tokens))
    print(f"SUCCESS: Cleaned corpus perfectly saved to {corpus_path}")

    # --- 4. DATASET STATISTICS ---
    # Calculate stats required by the assignment prompt
    total_documents = len(raw_texts)
    total_tokens = len(all_tokens)
    vocab_size = len(set(all_tokens))

    print("\n" + "="*30)
    print("      DATASET STATISTICS")
    print("="*30)
    print(f"Total Documents Processed : {total_documents}")
    print(f"Total Tokens (Words)      : {total_tokens}")
    print(f"Vocabulary Size (Unique)  : {vocab_size}")
    print("="*30 + "\n")

    # --- 5. WORD CLOUD VISUALIZATION ---
    print("--- Generating Word Cloud ---")
    text_for_cloud = " ".join(all_tokens)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_for_cloud)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("IIT Jodhpur Corpus Word Cloud")
    
    # Save the plot securely to the outputs folder
    plot_path = os.path.join(outputs_plot_dir, 'wordcloud.png')
    plt.savefig(plot_path)
    print(f"Word Cloud saved to {plot_path}")
    
    # Display the plot on the screen
    plt.show()

if __name__ == "__main__":
    main()