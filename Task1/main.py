import pandas as pd
import sys
import os
import argparse

# Add current directory to path to import BertWSD
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wsd_alert_system import BertWSD

def main():
    parser = argparse.ArgumentParser(description="BERT-based Word Sense Disambiguation Alert System")
    parser.add_argument("--word", type=str, default="cell", help="Target word to search for")
    parser.add_argument("--definition", type=str, default="A small room in which a prisoner is locked up.", help="Definition of the target word")
    parser.add_argument("--limit", type=int, default=500, help="Number of movie plots to scan")
    parser.add_argument("--threshold", type=float, default=0.50, help="Similarity threshold (0.0 to 1.0)")
    parser.add_argument("--window", type=int, default=10, help="Context window size (number of tokens)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset file")

    args = parser.parse_args()

    # Initialize the system
    wsd_system = BertWSD()

    # 1. Define target word and definition
    target_word = args.word
    definition = args.definition
    print(f"Target Word: {target_word}")
    print(f"Definition: {definition}")

    # 2. Generate (word, sentence) pairs to test the approach
    # Only run default tests if using the default word "cell"
    if target_word.lower() == "cell":
        print("\n--- Testing with generated pairs ---")
        test_sentences = [
            "The prisoner was taken back to his cell after the interrogation.", # Match
            "The guard locked the cell door with a loud clang.", # Match
            "Red blood cells carry oxygen to the body's tissues.", # Mismatch (Biological)
            "She checked her cell for any missed calls.", # Mismatch (Phone)
            "The terrorist cell was planning an attack.", # Mismatch (Group)
            "He spent ten years in a prison cell.", # Match
            "The monk retreated to his small cell for prayer." # Match (Room)
        ]

        alerts = wsd_system.scan_dataset(
            test_sentences, 
            target_word, 
            definition, 
            window_size=args.window, 
            threshold=args.threshold
        )

        print(f"\nFound {len(alerts)} alerts in test sentences:")
        for alert in alerts:
            print(f"  - Score: {alert['similarity']:.4f} | Context: {alert['context']}")
    else:
        print(f"\nSkipping default test sentences as target word is '{target_word}' (not 'cell').")

    # 3. Use the dataset attached in main dir
    print("\n--- Scanning Movie Plots Dataset ---")
    try:
        if args.dataset:
            dataset_path = args.dataset
        else:
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'movie_plots.csv.xz')
            if not os.path.exists(dataset_path):
                 dataset_path = 'movie_plots.csv.xz' 
        
        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} rows.")
        
        subset_size = args.limit
        print(f"Scanning first {subset_size} plots...")
        plots = df['Plot'].dropna().tolist()[:subset_size]
        
        movie_alerts = wsd_system.scan_dataset(
            plots, 
            target_word, 
            definition, 
            window_size=args.window, 
            threshold=args.threshold
        )
        
        print(f"\nFound {len(movie_alerts)} alerts in movie plots:")
        for alert in movie_alerts:
            doc_id = alert['doc_id']
            title = df.iloc[doc_id]['Title']
            print(f"  - [{title}] Score: {alert['similarity']:.4f} | Context: ...{alert['context']}...")

    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    main()
