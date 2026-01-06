# Sentiment Analysis using BERT

This project performs sentiment analysis on text reviews using a pre-trained BERT model from Hugging Face Transformers. It predicts sentiment on a 1–5 scale, where 1 indicates very negative sentiment and 5 indicates very positive sentiment.

---

## Features

- Uses BERT (nlptown/bert-base-multilingual-uncased-sentiment)
- Predicts sentiment scores from 1 to 5
- Supports custom text input
- Scrapes online reviews using BeautifulSoup
- Stores results in a Pandas DataFrame
- Runs locally in VS Code

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- BeautifulSoup
- Requests
- Pandas
- NumPy

---

## Project Structure

sentiment-analysis/
│
├── sentiment_analysis.py
├── README.md

---

## Installation & Setup

1. Install Python (3.9 or above)

2. Install required libraries:
   python -m pip install torch transformers requests beautifulsoup4 pandas numpy

---

## How to Run

python sentiment_analysis.py

Note: The BERT model will be downloaded automatically during the first run.

---

## Output

Example console output:

Sentiment score: 4

Example DataFrame output:

Review: "Great coffee and friendly staff"
Sentiment: 5

Review: "Service was slow"
Sentiment: 2

Sentiment Scale:
1 - Very Negative  
2 - Negative  
3 - Neutral  
4 - Positive  
5 - Very Positive  

---

## How It Works

1. Input text is tokenized using a BERT tokenizer
2. Tokens are passed to a pre-trained BERT model
3. The model produces sentiment logits
4. The highest logit determines the sentiment score

---

## Notes

- Some websites may block web scraping
- Maximum input length is limited to 512 tokens
- CPU inference may take more time

---

## Resume Description

Developed a sentiment analysis application using a pre-trained BERT model to classify text reviews on a 1–5 sentiment scale. Implemented web scraping using BeautifulSoup and performed data processing with Pandas and PyTorch.

---
