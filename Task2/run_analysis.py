import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sentiment_analyzer import SentimentAnalyzer

def plot_sentiment_distribution(results, target_word, output_file):
    sentiments = [r['sentiment'] for r in results]
    
    # Count sentiments
    counts = pd.Series(sentiments).value_counts()
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['green' if 'POSITIVE' in idx.upper() else 'red' for idx in counts.index]
    
    counts.plot(kind='bar', color=colors, alpha=0.7)
    plt.title(f"Sentiment Distribution for '{target_word}' Matches")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha='center')
        
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis on WSD Matches")
    parser.add_argument("--word", type=str, default="cell", help="Target word")
    parser.add_argument("--definition", type=str, default="A small room in which a prisoner is locked up.", help="Definition")
    parser.add_argument("--limit", type=int, default=1000, help="Number of plots to scan")
    parser.add_argument("--threshold", type=float, default=0.50, help="WSD Threshold")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset")
    
    args = parser.parse_args()
    
    # Load Dataset
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'movie_plots.csv.xz')
        
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
        plots = df['Plot'].dropna().tolist()[:args.limit]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Run Analysis
    analyzer = SentimentAnalyzer()
    results = analyzer.process_dataset(plots, args.word, args.definition, threshold=args.threshold)
    
    if not results:
        print("No matches found.")
        return
        
    # Convert to DataFrame for easier viewing
    res_df = pd.DataFrame(results)
    print("\nAnalysis Results (First 10):")
    print(res_df[['sentiment', 'confidence', 'context']].head(10))
    
    # Save results
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sentiment_results_{args.word}.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Plot
    output_plot = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sentiment_plot_{args.word}.png")
    plot_sentiment_distribution(results, args.word, output_plot)
    
    # Summary
    print("\nSummary:")
    print(res_df['sentiment'].value_counts())
    print("\nAverage Confidence:", res_df['confidence'].mean())

if __name__ == "__main__":
    main()
