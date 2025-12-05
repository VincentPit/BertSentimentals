# Task 2: Sentiment Analysis on WSD Matches

This task extends the Word Sense Disambiguation (WSD) system to analyze the sentiment of the contexts where a specific word sense appears.

## Methodology

1.  **Word Sense Disambiguation (WSD)**:
    *   We reuse the `BertWSD` system from Task 1.
    *   The system scans documents (movie plots) and identifies instances where the target word matches the provided definition (using BERT embeddings and Cosine Similarity).
    *   This ensures we only analyze the sentiment of the *relevant* sense (e.g., "bank" as a financial institution, ignoring "river bank").

2.  **Sentiment Analysis**:
    *   For every confirmed match, we extract the **context window** (surrounding tokens).
    *   We use a pre-trained Transformer model: `distilbert-base-uncased-finetuned-sst-2-english`.
    *   This model is fine-tuned on the SST-2 dataset (movie reviews) and outputs `POSITIVE` or `NEGATIVE` labels with a confidence score.

3.  **Visualization**:
    *   We aggregate the sentiment labels and generate a bar chart showing the distribution of sentiment for the target word sense.

## Usage

### Local Execution
Run the analysis script from the root directory:

```bash
python Task2/run_analysis.py --word "bank" --definition "A financial institution" --limit 1000
```

### Docker Execution
You can also run this task inside the Docker container. From the project root:
```bash
docker build -t bert-wsd-sentiment .
docker run -it bert-wsd-sentiment python Task2/run_analysis.py --word "bank" --definition "A financial institution" --limit 1000
```

**Arguments:**
- `--word`: Target word.
- `--definition`: Definition of the target sense.
- `--limit`: Number of plots to scan.
- `--threshold`: WSD similarity threshold (default: 0.50).

## Observations

### Case Study: "Bank" (Financial Institution)
*   **Distribution**: In a scan of 1000 movie plots, we found a skew towards **NEGATIVE** sentiment (approx. 55% Negative vs 45% Positive).
*   **Context Analysis**:
    *   **Negative**: Frequently associated with keywords like "robbery", "heist", "overdrawn", "debt", and "stakeout". This reflects the common trope of banks being targets of crime or sources of conflict in movies.
    *   **Positive**: Associated with neutral or constructive actions like "clerk", "performed", "care for", though often these were descriptive rather than truly "happy" sentiments.

### Case Study: "Cell" (Prison)
*   **Distribution**: Mixed sentiment.
*   **Context Analysis**:
    *   **Negative**: Phrases like "locked in", "in vain", "condemned".
    *   **Positive**: Surprisingly, contexts involving "escape" or "found" were sometimes labeled positive. This highlights a nuance: "escaping a cell" is a positive outcome for the character, even if the cell itself is negative.

## Limitations & Failure Modes

1.  **Domain Mismatch**:
    *   The sentiment model is trained on **Movie Reviews** (opinions), but we are applying it to **Movie Plots** (descriptions).
    *   *Failure Mode*: A sentence like "The bank robber shot the guard" might be classified based on the tone of the words rather than the moral polarity of the event. "Shot" and "robber" are negative, so it works, but subtle plot descriptions might be misclassified.

2.  **Context Window**:
    *   We analyze the immediate context (sentence/window). Sometimes the sentiment is determined by the broader narrative which is lost in the window.

3.  **WSD Dependency**:
    *   The quality of this analysis depends entirely on the accuracy of Task 1. If `BertWSD` incorrectly identifies a "river bank" as a "financial bank", the sentiment analysis will be applied to the wrong concept.
