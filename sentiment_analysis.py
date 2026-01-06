from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

# Test sentence
text = input("Enter a sentence for sentiment analysis: ")
tokens = tokenizer.encode(text, return_tensors="pt")
result = model(tokens)

print("Sentiment score:", int(torch.argmax(result.logits)) + 1)

# Scrape Yelp reviews
url = "https://www.yelp.com/biz/social-brew-cafe-pyrmont"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(r.text, "html.parser")

regex = re.compile(".*comment.*")
results = soup.find_all("p", {"class": regex})
reviews = [res.text for res in results]

# Create DataFrame
df = pd.DataFrame(reviews, columns=["review"])

# Sentiment function
def sentiment_score(review):
    tokens = tokenizer.encode(review[:512], return_tensors="pt")
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

df["sentiment"] = df["review"].apply(sentiment_score)

print(df.head())
