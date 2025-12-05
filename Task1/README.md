# BERT Word Sense Disambiguation Alert System

This project implements a context-aware alert system using BERT. It scans text for a specific word and triggers an alert only if the word's usage matches a provided definition.

## Approach

1.  **Contextual Embeddings**: We use a pre-trained BERT model (`bert-base-uncased`) to generate embeddings. We **sum the last 4 hidden layers** to capture rich semantic information while avoiding the bias of the final output layer.
2.  **Subword Handling**: We correctly handle BERT's WordPiece tokenization by identifying all subword tokens for a target word (e.g., "unfriendly" -> "un", "##friend", "##ly") and **averaging their embeddings** to create a single, robust vector.
3.  **Constructed Context Comparison**: Instead of comparing a word to a generic sentence embedding, we construct a "definition sentence" (e.g., `"{word} means {definition}"`) and extract the embedding of the target word from that specific context.
4.  **Similarity Matching**: We calculate the Cosine Similarity between the target word's vector in the query text and the target word's vector in the definition context.
5.  **Alerting**: If the similarity score exceeds a dynamically optimized threshold (default: 0.50), we flag the instance as a match.

## Directory Structure

```
Task1/
├── wsd_alert_system.py    # Core WSD logic using BERT
├── main.py                # CLI entry point for scanning datasets
├── elboSearch.py          # Threshold optimization script (Elbow Method)
├── requirements.txt       # Project dependencies
├── tests/                 # Unit tests
├── threshold_results.txt  # Output of threshold optimization
└── threshold_elbow_plot.png # Visualization of optimal threshold
```

## Setup

### Local Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Docker Setup
You can run the system in a Docker container. From the project root:
```bash
docker build -t bert-wsd-sentiment .
docker run -it bert-wsd-sentiment
```

## Usage

### 1. Run the Alert System (CLI)

You can use `main.py` to scan a dataset (like movie plots) for a specific word sense.

**Example: Searching for "bank" (Financial Institution)**
```bash
python main.py --word "bank" --definition "A financial institution that accepts deposits" --limit 100
```

**Arguments:**
- `--word`: The target word to search for (default: "cell").
- `--definition`: The definition of the target sense.
- `--limit`: Number of documents to scan (default: 500).
- `--threshold`: Similarity threshold (default: 0.50).
- `--dataset`: Path to a CSV dataset (optional).

### 2. Optimize Thresholds (Elbow Method)

To find the optimal similarity threshold, run `elboSearch.py`. This script evaluates the system against a labeled dataset of 5 ambiguous words ("bank", "cell", "bat", "apple", "date") and calculates the F1 score at various thresholds.

```bash
python elboSearch.py
```

**Outputs:**
- `threshold_results.txt`: Detailed performance metrics (Precision, Recall, F1) for each word and the global average.
- `threshold_elbow_plot.png`: A graph visualizing the F1 scores to help identify the optimal "elbow" point.

### 3. Python Module Usage

You can import the core class into your own scripts:

```python
from wsd_alert_system import BertWSD

# Initialize
system = BertWSD()

# Data
text = "I went to the bank to deposit money."
target = "bank"
definition = "A financial institution."

# Scan
alerts = system.scan_dataset([text], target, definition)
print(alerts)
```

## Performance

Based on our global optimization analysis (see `threshold_results.txt`), the optimal threshold is **0.50**, achieving a global average F1 score of **~0.84**.

- **Bank**: 0.86 F1
- **Cell**: 0.90 F1
- **Date**: 0.95 F1

