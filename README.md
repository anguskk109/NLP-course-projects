# NLP Project Portfolio

This repository contains my implementations for three core assignments from the University of Toronto's NLP course.  
Each project tackles a different, fundamental problem in modern NLP and machine learning, providing a solid foundation in both theoretical concepts and practical coding.

The projects are organized into three main folders, corresponding to the assignments:
1.  **`financial-sentiment-llm-benchmark`**: Financial Sentiment Analysis & LLM Benchmarking.
2.  **`speech-sequence-classification`**: Speech Sequence Classification using GMMs and RNNs.
3.  **`transformer-nmt-from-scratch`**: Building a Neural Machine Translation Transformer from Scratch.

---

## 📁 `financial-sentiment-llm-benchmark`

**Financial Sentiment Analysis**

This project focuses on classifying the sentiment of financial news texts and comparing traditional ML methods with large language models (LLMs).

### Key Tasks & Features
*   **Data Preprocessing**: Cleaned text data from the Financial Phrase Bank (FPB) and Wall Street Journal (WSJ) datasets using spaCy for lemmatization and noise removal.
*   **Feature Engineering**: Implemented three vectorization techniques:
    *   Count Vectorizer (unigrams, bigrams, trigrams)
    *   TF-IDF Vectorizer
    *   MPNet Embeddings (using `sentence-transformers/all-mpnet-base-v2`)
*   **Model Training & Evaluation**: Built and evaluated classifiers (`GaussianNB`, `MLPClassifier`) using scikit-learn. Performed feature selection (`SelectKBest`) and 5-fold cross-validation.
*   **LLM Benchmarking**: Integrated with a deployed Llama 3.1 8B Instruct model via an API endpoint to perform zero-shot sentiment classification and compare its performance against the trained models.

---

## 📁 `speech-sequence-classification`

**Sequence Classification for Speakers and Lies**

This project explores acoustic-based classification using Gaussian Mixture Models (GMMs) and Recurrent Neural Networks (RNNs) on speech data.

### Key Tasks & Features
*   **Gaussian Mixture Models (GMMs)**:
    *   Implemented core GMM utility functions (`log_b_m_x`, `log_p_m_x`, `logLik`) for calculating observation probabilities and likelihoods.
    *   Built a training algorithm to learn GMM parameters (means, covariances, weights) for speaker identification.
    *   Developed a classification system to identify speakers based on maximizing log-likelihood.
    *   Conducted experiments to analyze the impact of the number of mixture components (`M`) on classification accuracy.
*   **GRU for Deception Detection**:
    *   Designed and trained a simple unidirectional GRU model using PyTorch to classify utterances as truthful or deceitful based on MFCC features.
    *   Experimented with different hidden layer sizes (5, 10, 50) to observe their effect on detection performance.
*   **Dynamic Programming Applications**:
    *   Implemented the Levenshtein algorithm to calculate Word Error Rate (WER) for evaluating ASR systems (Kaldi vs. Google).
    *   Implemented Dynamic Time Warping (DTW) for speaker verification, aligning MFCC sequences to measure similarity.

---

## 📁 `transformer-nmt-from-scratch`

**Neural Machine Translation using Transformers**

This project involves building a complete Transformer encoder-decoder model from scratch for machine translation, using the Canadian Hansards dataset (French-to-English).

### Key Tasks & Features
*   **Transformer Architecture Implementation**:
    *   Built foundational blocks: `LayerNorm`, `FeedForwardLayer`, and `MultiHeadAttention`.
    *   Constructed the full `TransformerEncoderLayer` and `TransformerDecoderLayer` with both pre- and post-layer normalization options.
    *   Assembled the complete `TransformerEncoder`, `TransformerDecoder`, and `TransformerEncoderDecoder` models.
*   **Inference Algorithms**:
    *   Implemented a greedy decoding algorithm.
    *   Implemented a more sophisticated beam search decoder for generating higher-quality translations.
*   **Training & Evaluation Pipeline**:
    *   Developed a training loop with gradient accumulation support.
    *   Implemented BLEU score calculation from scratch for evaluation.
    *   Trained the model on the Hansards dataset and evaluated its performance.
*   **Translation & Analysis**:
    *   Created an interactive mode to translate sentences.
    *   Compared the performance of the custom-built model against fine-tuned pre-trained models (e.g., T5, BART) and commercial services (e.g., Google Translate, ChatGPT).

---


> **Note**: The original datasets are hosted on the UofT servers and are not included in this public repository for copyright reasons.

