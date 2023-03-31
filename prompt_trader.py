import os
import string
import nltk
import numpy as np
import tensorflow_hub as hub
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from termcolor import colored, cprint
from sklearn.metrics import pairwise_distances
from collections import Counter
import heapq
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


os.system("colored")
os.system("cls||clear")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

#text preprossing: stop words, punctualtion, Lemmatization, Remove non-alphabetic tokens
def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Remove non-alphabetic tokens and tokens with length <= 1
    alphabetic_tokens = [word for word in lemmatized_tokens if word.isalpha() and len(word) > 1]


    return " ".join(alphabetic_tokens)

#define elbow - what is the K-value
def find_elbow_point(sse_values):
    slopes = [sse_values[i] - sse_values[i - 1] for i in range(1, len(sse_values))]
    elbow_point = slopes.index(max(slopes)) + 1
    return elbow_point

#sentiment and value for sentiment
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    global compound_score
    compound_score = sentiment_scores['compound']

    if compound_score > 0.55:
        return 'positive'
    elif compound_score < -0.55:
        return 'negative'
    else:
        return 'neutral'

#input test to analyze
with open("c3.csv", "r", encoding="utf-8") as file:
    lines = file.readlines()

#lists are initialized: "preprocessed_lines": This list will store the preprocessed version of each line from the input file. "original_non_blank_lines": This list will store the original lines without any leading/trailing whitespace, but only if they are not empty or just whitespace. "original_line_indices": This list will store the index of each non-blank line in the input file.
preprocessed_lines = []
original_non_blank_lines = []
original_line_indices = []

#iterates through each line in the input file, along with its index
for index, line in enumerate(lines):
    if line.strip():
        preprocessed_lines.append(preprocess_text(line))
        original_non_blank_lines.append(line.strip())
        original_line_indices.append(index)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed(preprocessed_lines).numpy()

sse = []
#To choose the most suitable range for your dataset, you can experiment with different k ranges, such as 1-15 or 1-20, and compare the results. Keep in mind that as you increase the maximum k value, the computation time for the elbow method will also increase.
k_values = range(1, 16)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    sse.append(kmeans.inertia_)

#plot the elbow
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared errors (SSE)')
plt.title('Elbow method for determining k')
plt.show()

# Find the optimal k value
k_optimal = find_elbow_point(sse)

kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(embeddings)

# Create lists for each cluster
cluster_lists = [[] for _ in range(k_optimal)]
cluster_embeddings = [[] for _ in range(k_optimal)]

# Assign each original line and its embedding to its corresponding cluster list
for index, original_line in enumerate(original_non_blank_lines):
    cluster_label = kmeans.labels_[index]
    cluster_lists[cluster_label].append(original_line)
    cluster_embeddings[cluster_label].append(embeddings[index])

# Calculate the average similarity and top words within each cluster
average_similarities = []
top_words_per_cluster = []

for cluster_label, cluster_list in enumerate(cluster_lists):
    # Calculate average similarity
    if len(cluster_embeddings[cluster_label]) > 1:
        similarity_matrix = 1 - pairwise_distances(cluster_embeddings[cluster_label], metric='cosine')
        avg_similarity = np.mean(similarity_matrix)
    else:
        avg_similarity = 1
    average_similarities.append(avg_similarity)

    # Find top words
    words_counter = Counter()
    for line in cluster_list:
        preprocessed_line = preprocess_text(line)
        words_counter.update(preprocessed_line.split())
    top_words = heapq.nlargest(5, words_counter, key=words_counter.get)
    top_words_per_cluster.append(top_words)


#print cluster data in DESC order based on Cluster Similarity
cluster_data = []

for cluster_label, cluster_list in enumerate(cluster_lists):
    cluster_data.append({
        'label': cluster_label,
        'similarity': average_similarities[cluster_label],
        'top_words': top_words_per_cluster[cluster_label],
        'lines': cluster_list
    })

# Sort cluster_data by the number of lines (descending) and similarity (descending)
sorted_cluster_data = sorted(cluster_data, key=lambda x: (-len(x['lines']), x['similarity']))

cutoff = 0.270

# Print clusters with similarity > 0.299 first
print(colored("\n\n[+] HIGH Priority News Clusters:\n", "magenta"))

for cluster in sorted_cluster_data:
    if cluster['similarity'] > cutoff:
        sentiment = analyze_sentiment(' '.join(cluster['top_words']))
        sentiment_color = 'green' if sentiment == 'positive' else 'red' if sentiment == 'negative' else 'yellow'

        print(f"\nCluster {cluster['label']} (Cluster Similarity: {cluster['similarity']:.4f}):")
        print(colored(f"Top words: {', '.join(cluster['top_words'])}", sentiment_color), compound_score)
        for line in cluster['lines']:
            print(colored(f"  {line}", "cyan"))
        print()

# Print remaining clusters
print(colored("\n\n[-] LOW Priority News Clusters:", "magenta"))
for cluster in sorted_cluster_data:
    if cluster['similarity'] <= cutoff:
        sentiment = analyze_sentiment(' '.join(cluster['top_words']))
        sentiment_color = 'green' if sentiment == 'positive' else 'red' if sentiment == 'negative' else 'yellow'

        print(f"\nCluster {cluster['label']} (Cluster Similarity: {cluster['similarity']:.4f}):")
        print(colored(f"Top words: {', '.join(cluster['top_words'])}", sentiment_color), compound_score)
        for line in cluster['lines']:
            print(colored(f"  {line}", "cyan"))
        print()

#create prompt (just for copy & paste into GPT / should be API integration if you have the tokens :) )
cluster = sorted_cluster_data[0]
print(colored("\n\n[+] Generating prompt for cluster: ", "magenta"), cluster['label'])

sentiment = analyze_sentiment(' '.join(cluster['top_words']))
sentiment_decision = 'positive' if sentiment == 'positive' else 'negative' if sentiment == 'negative' else 'negative'
top_words = ' '.join(cluster['top_words'])
news = [{'headline': line} for line in cluster['lines']]


print(f"\nHere are the headlines (each headline starts with headline:) that are important for this analysis: {news}\n\nI want to trade the market today and I want you to analyze some of the current news headlines I have harvested to make a decision on how to trade. Only consider headlines which have the same underlying meaning with these themse: \"{top_words}\", make sure to include all headlines which match this critera in your analysis and discard the other headlines from your analysis that don't. You can choose to long buy(a specific financial instrument which aligns with your trading analysis) or short sell(a specific financial instrument which aligns with your trading analysis). Overall, the sentiment appears it could be {sentiment_decision} based on my research of the news headlines, but I may be wrong. Pick out specific details from the the headlines and lets think about this step by step to create a trading strategy:\n\n")



