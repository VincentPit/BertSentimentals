# Context-Aware Word Sense Disambiguation & Sentiment Analysis

This project implements a system to detect specific word senses in text using BERT embeddings and analyze the sentiment of the surrounding context.

## Task 1: WSD Alert System
A BERT-based system that disambiguates word senses by comparing the contextual embedding of a target word against a "definition embedding".
- **Method**: Cosine similarity between target word vectors.
- **Optimization**: Uses the Elbow Method to find the optimal similarity threshold (0.50).

## Task 2: Sentiment Analysis
Analyzes the sentiment (Positive/Negative) of the contexts where the target word sense appears.
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`.
- **Output**: Sentiment distribution plots and CSV reports.

## Project Artifacts & Documentation

### Documentation
*   [Task 1 README (WSD System)](Task1/README.md)
*   [Task 2 README (Sentiment Analysis)](Task2/README.md)

### Task 1: Optimization Results
*   [Threshold Optimization Plot (Elbow Method)](Task1/threshold_elbow_plot.png)
*   [Detailed Threshold Results (TXT)](Task1/threshold_results.txt)

### Task 2: Analysis Results
**Target: "Bank" (Financial)**
*   [Sentiment Results (CSV)](Task2/sentiment_results_bank.csv)
*   [Sentiment Distribution Plot](Task2/sentiment_plot_bank.png)

**Target: "Cell" (Prison)**
*   [Sentiment Results (CSV)](Task2/sentiment_results_cell.csv)
*   [Sentiment Distribution Plot](Task2/sentiment_plot_cell.png)

## Limitations
1. **Threshold Sensitivity**: The system relies on a fixed cosine similarity threshold (0.50). While optimized stochastically, it may not be perfect for every word.
2. **Context Window**: The sentiment analysis is limited to a fixed window around the target word. Important sentiment-bearing words outside this window might be missed.
3. **Domain Mismatch**: The sentiment model is trained on movie reviews (SST-2), while the dataset consists of movie plots. Descriptive text in plots might be misclassified as "negative" due to words like "crime" or "kill", even if the narrative tone is neutral.

## Possible Improvements
1. **Fine-Tuning** (THE MOST EFFECTIVE WAY): Fine-tune the BERT model on a specific WSD dataset to improve embedding separation between senses.
2. **Dynamic Windowing**: Implement a sliding window or dependency parsing to capture the full relevant context for sentiment analysis, rather than a fixed token count.
3. **Domain Adaptation**: Retrain or fine-tune the sentiment model on a dataset of plot summaries to better distinguish between "negative events" and "negative sentiment".

## References
*   **BERT Word Embeddings Tutorial**: [McCormickML Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#21-special-tokens) - Used as a reference for extracting and summing hidden states from BERT.
*   **Sentiment Analysis Model**: [DistilBERT Finetuned SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) - The Hugging Face model used for sentiment classification.
