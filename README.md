# Word2Vec Skip-gram with Negative Sampling (SGNS)

An educational, from-scratch implementation of the Word2Vec Skip-gram model using Python and NumPy. This project demonstrates how to learn dense word embeddings by training a neural network to predict context words from a center word.

## 🚀 Project Overview
The goal of this implementation is to map words into a high-dimensional vector space where semantically similar words are positioned close to one another. Unlike basic one-hot encoding, these embeddings capture the "meaning" of words based on their distribution in a corpus.

### Key Implementation Features
* **Pure NumPy Logic**: Manual implementation of forward passes, backpropagation, and weight updates.
* **Negative Sampling**: Instead of a costly Softmax over the entire vocabulary, the model optimizes against $K$ noise samples.
* **Subsampling**: Implements frequency-based discarding of common words (like "the", "is") to focus on informative tokens.
* **Dynamic Context Window**: Samples window sizes randomly to weight closer words more heavily.

## 🛠 Model Architecture
The architecture consists of two embedding matrices:
1.  **Input Matrix ($W_{in}$)**: Stores the vectors for words when they act as the "Center" word.
2.  **Output Matrix ($W_{out}$)**: Stores the vectors for words when they act as the "Context" word.



### Mathematical Objective
The model minimizes the Negative Log-Likelihood of the center word ($v_c$) appearing with its context word ($u_o$) and not appearing with $K$ negative samples ($u_{n_i}$):

$$\mathcal{L} = -\log \sigma(u_o^\top v_c) - \sum_{i=1}^K \log \sigma(-u_{n_i}^\top v_c)$$

## 📂 Structure
* `Tokenizer`: Handles text cleaning, vocabulary mapping, and frequency calculation.
* `SkipGramDataset`: Manages the sliding window and generates negative samples using the $P(w)^{0.75}$ distribution.
* `EmbeddingLayer`: A sparse update layer that accumulates gradients and applies updates via SGD.
* `Inference`: Uses Cosine Similarity to find the nearest neighbors of a query word.

## 🚦 Getting Started
1.  **Data**: The code automatically fetches *Frankenstein* from Project Gutenberg.
2.  **Run**: Execute the notebook cells to train the model. 
3.  **Evaluate**: Use the `get_similar(word)` function to test the learned relationships.
