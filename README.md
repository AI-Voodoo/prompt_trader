# prompt_trader
Source code for a personal project to Prompt Engineer Generative AI Trading Decisions using news headlines as input data. The project processes the data, clusters the headlines based on similarity (finding the most mentioned news themes), analyzes sentiment, and generates a prompt for LLM-based trading suggestions, using these popular news themes and sentiment.

First, it imports necessary libraries such as os, string, nltk, numpy, TensorFlow Hub, and scikit-learn. The code then defines a text preprocessing function that removes stopwords, punctuation, lemmatizes words, and filters non-alphabetic tokens.

The 'find_elbow_point' function is defined to find the optimal number of clusters for KMeans clustering. Sentiment analysis is performed using the 'analyze_sentiment' function, which categorizes the sentiment as positive, negative, or neutral based on a threshold.

The input headlines are read from a file, preprocessed, and embedded using the Universal Sentence Encoder. KMeans clustering is applied to the embeddings, and the elbow method is used to determine the optimal number of clusters (k_optimal). Clusters are then created, and the average similarity and top words within each cluster are calculated.

The code sorts the clusters by the number of lines and similarity, and prints high-priority news clusters (with similarity > 0.270) and low-priority news clusters (similarity <= 0.270). It then generates a prompt for the AI using the most important cluster's top words and sentiment analysis results.
