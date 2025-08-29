# Project 04: Text Classification using RNN

## 🎯 Project Overview

This project introduces **Natural Language Processing (NLP)** with **Recurrent Neural Networks (RNNs)**. You'll learn to process text data, build word embeddings, and classify movie reviews using LSTM and GRU networks.

## 📚 Learning Objectives

By completing this project, you will:
- Understand text preprocessing and tokenization
- Learn word embeddings (Word2Vec, GloVe)
- Implement RNN, LSTM, and GRU architectures
- Master sequence modeling for text classification
- Achieve >90% accuracy on IMDB movie reviews
- Visualize word embeddings and attention patterns

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **NLTK** - Natural language processing
- **Scikit-learn** - Text preprocessing
- **Matplotlib/Seaborn** - Visualization
- **WordCloud** - Text visualization

## 📊 Dataset: IMDB Movie Reviews

- **Classes**: 2 (Positive, Negative sentiment)
- **Reviews**: 50,000 (25,000 training + 25,000 test)
- **Challenge**: Variable length sequences, sentiment analysis
- **Vocabulary**: ~88,000 unique words

## 🏗️ RNN Architectures Implemented

### 1. Simple RNN
- Basic recurrent layer
- Word embeddings
- **Expected Accuracy**: ~85%

### 2. LSTM Network
- Long Short-Term Memory
- Bidirectional processing
- **Expected Accuracy**: ~88%

### 3. GRU with Attention
- Gated Recurrent Unit
- Attention mechanism
- **Expected Accuracy**: >90%

## 🎯 Key Concepts Covered

### Text Processing
- **Tokenization**: Converting text to sequences
- **Padding**: Handling variable lengths
- **Embeddings**: Dense word representations
- **Vocabulary**: Building word dictionaries

### RNN Components
- **Recurrent Layers**: Processing sequences
- **LSTM/GRU**: Handling long dependencies
- **Bidirectional**: Forward and backward processing
- **Attention**: Focusing on important words

## 📈 Expected Results

| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| Simple RNN | ~85% | ~200K | 10 min |
| LSTM | ~88% | ~500K | 15 min |
| GRU + Attention | >90% | ~800K | 20 min |

## 🔍 Visualizations

The notebook includes:
- Text preprocessing pipeline
- Word embedding visualizations (t-SNE)
- Training curves comparison
- Attention weight heatmaps
- Confusion matrix analysis
- Sample predictions with explanations

## 🚀 Advanced Experiments

Try these extensions:
1. **Pre-trained Embeddings**: GloVe, FastText
2. **Transformer Models**: BERT, RoBERTa
3. **Multi-class Classification**: News categorization
4. **Sequence-to-Sequence**: Text summarization

## 📚 Additional Resources

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

**Ready to teach machines to understand language? Let's dive into NLP! 📝**
