import sys
import os
import torch
from transformers import pipeline

# Add Task1 to path to import BertWSD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Task1'))
from wsd_alert_system import BertWSD

class SentimentAnalyzer:
    def __init__(self, wsd_model_name='bert-base-uncased', sentiment_model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        print("Initializing WSD System...")
        self.wsd = BertWSD(model_name=wsd_model_name)
        
        print(f"Loading Sentiment Model ({sentiment_model_name})...")
        device = 0 if torch.cuda.is_available() else -1 #It fine Bert is small
        if torch.backends.mps.is_available():
            device = 0 #trying out Mac for fun
            
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model_name, device=device)
        print("Sentiment Model loaded.")

    def analyze_sentiment(self, text):
        # Truncate text if it's too long for the model (limit is 512 tokens)
        result = self.sentiment_pipeline(text, truncation=True, max_length=512)[0]
        return result['label'], result['score']

    def process_dataset(self, plots, target_word, definition, threshold=0.50, window_size=10):
        """
        1. Finds matches of target_word using WSD.
        2. Analyzes sentiment of the context containing the match.
        """
        print(f"Scanning {len(plots)} documents for '{target_word}'...")
        
        # Get WSD matches
        alerts = self.wsd.scan_dataset(plots, target_word, definition, window_size=window_size, threshold=threshold)
        print(f"Found {len(alerts)} matches. Analyzing sentiment...")
        
        results = []
        for alert in alerts:
            context = alert['context']
            # We analyze the sentiment of the context sentences
            label, score = self.analyze_sentiment(context)
            
            results.append({
                'doc_id': alert['doc_id'],
                'similarity': alert['similarity'],
                'context': context,
                'sentiment': label,
                'confidence': score
            })
            
        return results
