import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsd_alert_system import BertWSD

def main():
    wsd_system = BertWSD()
    
    dataset = [
        "The river bank was muddy after the rain.",                 # River
        "The bank approved my loan application yesterday.",         # Financial
        "She works as a teller at the local bank."                  # Financial
    ]
    target_word = "bank"
    definition = "A financial institution that accepts deposits and channels the money into lending activities."
    
    print(f"Target: {target_word}")
    print(f"Definition: {definition}")
    
    alerts = wsd_system.scan_dataset(
        dataset, 
        target_word, 
        definition, 
        window_size=10, 
        threshold=0.0 # Get all scores
    )
    
    for alert in alerts:
        print(f"Context: {alert['context']}")
        print(f"Score: {alert['similarity']:.4f}")
        print("-" * 20)

if __name__ == "__main__":
    main()
