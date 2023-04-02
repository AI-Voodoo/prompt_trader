# Prompt Trader
This Python script is designed to perform prompt engineering by leveraging natural language processing (NLP) outputs to create high-quality input prompts for generative AI models. The main goal is to generate trading prompts based on the most prevalent themes in 100 recent news headlines, which can then be fed into an AI model for generating detailed trading strategies.

By preprocessing, clustering, and analyzing the sentiment of headlines, the script identifies the most relevant themes and their associated sentiment. These themes, along with the sentiment and specific details from the headlines, are used to construct well-formed prompts for generative AI models.

The prompts are designed in a way that provides clear instructions to the AI model, allowing it to focus on the specific themes and sentiment found in the headlines. This approach helps the generative AI model to produce more relevant and context-aware trading strategies based on the input prompts and actually return specific trading instruments it deems useful to achieve the strategy objectives.


## The script follows these main steps:

1. Import the necessary libraries and modules for text processing, clustering, and visualization. Download the required NLTK resources, such as tokenizers, stopwords, and sentiment lexicons.
2. Define the preprocess_text function to remove punctuation, tokenize, filter stopwords, lemmatize, and keep only alphabetic tokens with length greater than
3. Define the find_elbow_point function to find the optimal k value for clustering using the elbow method.
4. Define the analyze_sentiment function to compute the sentiment of a given text using the SentimentIntensityAnalyzer from NLTK.
5. Read the "100-headlines.txt" file and preprocess each headline, storing them in separate lists for further processing.
6. Load the Universal Sentence Encoder from TensorFlow Hub and compute embeddings for the preprocessed headlines.
7. Perform K-means clustering on the embeddings using a range of k values, and plot the elbow method graph to show the optimal k value. Fit the K-means model with the optimal k value and assign each headline to its corresponding cluster.
8. Calculate the average similarity and top words within each cluster.
9. Filter the clusters based on a similarity threshold (cutoff) and sort them in descending order based on the number of headlines and similarity.
10 Print the high-priority news clusters with their top words, sentiment, and headlines.
11. For the top three clusters, generate trading prompts that include the cluster's top words, sentiment, and headlines. The prompts ask the LLM to create a trading strategy based on the given information.


## Workflow & Results:
Perform K-means clustering on the embeddings and assign each headline to its corresponding cluster:
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/clustering.png?raw=true)

Plot the elbow method graph to show the optimal k value:
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/Figure_1.png?raw=true)

#### Top prompt Q & A with LLM + Market Validation

Top Prompt:
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/top-prompt.png?raw=true)

LLM Answer:
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/p1-answer.png?raw=true)

Market Validation: EUR/USD
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/EUR-USD.png?raw=true)

#### 2nd prompt Q & A with LLM + Market Validation

2nd Prompt:
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/2nd-prompt.png?raw=true)

LLM Answer:
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/2nd-answer.png?raw=true)

Market Validation: PDR S&P 500 ETF (SPY)
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/SPDR%20S&P%20500%20ETF.png?raw=true)

#### 3rd prompt Q & A with LLM + Market Validation

3rd Prompt:
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/3rd-prompt.png?raw=true)

LLM Answer:
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/3rd-answer.png?raw=true)

Market Validation: Digital World Acquisition Corp. (DWAC) Trump exposure
![Alt text describing the image](https://github.com/AI-Voodoo/prompt_trader/blob/main/assets/DWAC.png?raw=true)
