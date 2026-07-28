# IMDb Sentiment Analysis Using LSTM and GRU

## Project Overview

This project demonstrates how to build and compare two popular Recurrent Neural Network (RNN) architectures—**Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)**—for sentiment classification on the IMDb movie reviews dataset using TensorFlow and Keras.

The models are trained to classify movie reviews as either **positive** or **negative**, and their performances are compared based on test accuracy and loss.

---

## Objective

- Load and preprocess the IMDb movie reviews dataset.
- Build an LSTM-based sentiment analysis model.
- Build a GRU-based sentiment analysis model.
- Train both models using identical settings.
- Evaluate and compare their performance.

---

## Dataset

**Dataset:** IMDb Movie Reviews

- **Source:** TensorFlow/Keras Datasets
- **Training Samples:** 25,000
- **Testing Samples:** 25,000
- **Vocabulary Size:** 10,000 most frequent words
- **Task:** Binary Sentiment Classification

Each review is represented as a sequence of integer-encoded words.

---

## Technologies Used

- Python
- TensorFlow
- Keras

---

## Project Workflow

### 1. Import Libraries

The required TensorFlow and Keras modules are imported for:

- Loading the dataset
- Padding sequences
- Building neural networks
- Training and evaluation

---

### 2. Load the Dataset

The IMDb dataset is loaded using:

```python
imdb.load_data(num_words=10000)
```

Only the 10,000 most frequent words are retained to reduce model complexity.

---

### 3. Preprocess the Data

Movie reviews have varying lengths.

To make them uniform, each review is padded to a fixed length of **200 words**.

```python
pad_sequences(..., maxlen=200, padding="post")
```

This ensures every input has the same dimensions.

---

## LSTM Model Architecture

| Layer | Description |
|--------|-------------|
| Embedding | Converts word indices into dense 128-dimensional vectors |
| LSTM | 128 memory units for learning long-term dependencies |
| Dense | Sigmoid activation for binary classification |

### Model Summary

```
Embedding
      ↓
LSTM (128)
      ↓
Dense (1, Sigmoid)
```

---

## GRU Model Architecture

| Layer | Description |
|--------|-------------|
| Embedding | Converts words into 128-dimensional vectors |
| GRU | 128 GRU units for sequence learning |
| Dense | Sigmoid activation for binary classification |

### Model Summary

```
Embedding
      ↓
GRU (128)
      ↓
Dense (1, Sigmoid)
```

---

## Model Compilation

Both models use identical training settings:

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy

```python
optimizer='adam'
loss='binary_crossentropy'
metrics=['accuracy']
```

---

## Training Configuration

- Epochs: 5
- Batch Size: 32
- Validation Split: 20%

Each model is trained independently using the same hyperparameters to ensure a fair comparison.

---

## Model Evaluation

After training, both models are evaluated on the unseen IMDb test dataset.

The following metrics are reported:

- Test Loss
- Test Accuracy

Example:

```
LSTM Test Accuracy: 0.88

GRU Test Accuracy: 0.89
```

*(Actual values may vary depending on random initialization and training conditions.)*

---

## LSTM vs GRU

| Feature | LSTM | GRU |
|----------|------|-----|
| Gates | Input, Forget, Output | Update, Reset |
| Complexity | Higher | Lower |
| Parameters | More | Fewer |
| Training Speed | Slower | Faster |
| Memory Usage | Higher | Lower |
| Long-Term Dependency Learning | Excellent | Very Good |

---

## Advantages of LSTM

- Captures long-term dependencies effectively.
- Performs well on complex sequential data.
- Widely used in natural language processing tasks.

### Limitations

- More computationally expensive.
- Longer training time.
- Larger number of parameters.

---

## Advantages of GRU

- Simpler architecture.
- Faster training.
- Requires fewer parameters.
- Often achieves performance comparable to LSTM.

### Limitations

- Slightly less expressive than LSTM for very long sequences.
- May underperform LSTM on tasks requiring extensive long-term memory.

---

## Expected Results

Typically:

- Both models achieve high sentiment classification accuracy (around 85–90%).
- GRU often trains faster due to its simpler architecture.
- LSTM may slightly outperform GRU on tasks requiring longer-term context, although their performance is often very similar on the IMDb dataset.

---

## Learning Outcomes

By completing this project, you will learn how to:

- Load and preprocess text datasets.
- Represent text using embedding layers.
- Build LSTM and GRU models in TensorFlow/Keras.
- Train recurrent neural networks for sentiment analysis.
- Evaluate binary classification models.
- Compare different recurrent architectures for sequence modelling.

---

## Future Improvements

- Add dropout for regularization.
- Experiment with bidirectional LSTM and GRU layers.
- Tune hyperparameters such as learning rate, embedding size, and sequence length.
- Use pre-trained word embeddings (e.g., GloVe or Word2Vec).
- Visualize training and validation accuracy/loss.
- Evaluate using precision, recall, F1-score, and confusion matrix.

---

## Conclusion

This project demonstrates the implementation and comparison of **LSTM** and **GRU** models for sentiment analysis using the IMDb movie reviews dataset. Both architectures effectively classify reviews as positive or negative, with GRU generally offering faster training and fewer parameters, while LSTM may provide a slight advantage in modelling long-term dependencies. The comparison highlights the strengths and trade-offs of these recurrent neural network architectures for natural language processing tasks.
