import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Import data generator from level_1_ml to keep data identical
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from level_1_ml.train import generate_synthetic_data, TextPreprocessor

# 1. SEQUENTIAL TEXT REPRESENTATION
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.num_words = 2
        
    def fit(self, texts):
        for text in texts:
            for word in word_tokenize(text.lower()):
                if word not in self.word2idx:
                    self.word2idx[word] = self.num_words
                    self.idx2word[self.num_words] = word
                    self.num_words += 1
                    
    def transform(self, texts, max_len=20):
        sequences = []
        for text in texts:
            seq = []
            for word in word_tokenize(text.lower()):
                seq.append(self.word2idx.get(word, 1)) # Default to <UNK>
            # Pad or truncate
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            sequences.append(seq)
        return np.array(sequences)
        
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.word2idx, f)
            
    @classmethod
    def load(cls, filepath):
        vocab = cls()
        with open(filepath, 'r') as f:
            vocab.word2idx = json.load(f)
            vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
            vocab.num_words = len(vocab.word2idx)
        return vocab

# 2. LSTM NEURAL NETWORK ARCHITECTURE
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, bidirectional=False):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        # lstm output shape: [batch_size, seq_len, hidden_dim * num_directions]
        # hn shape: [num_layers * num_directions, batch_size, hidden_dim]
        out, (hn, cn) = self.lstm(embedded)
        
        # Take final hidden state of LSTM for classification
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hn_last = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        else:
            hn_last = hn[-1,:,:]
            
        return self.fc(hn_last)

def main():
    print("Generating synthetic dataset for Deep Learning training...")
    df = generate_synthetic_data(1500) # Slightly larger dataset for DL
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    df['cleaned_headline'] = df['headline'].apply(preprocessor.preprocess)
    
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = df['sentiment'].map(sentiment_map).values
    
    # Chronological Split
    split_idx = int(len(df) * 0.8)
    train_texts = df['cleaned_headline'].iloc[:split_idx].values
    test_texts = df['cleaned_headline'].iloc[split_idx:].values
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.fit(train_texts)
    print(f"Vocabulary size: {vocab.num_words} words.")
    
    # Convert text to sequences
    max_len = 20
    X_train = vocab.transform(train_texts, max_len=max_len)
    X_test = vocab.transform(test_texts, max_len=max_len)
    
    # Save vocabulary
    os.makedirs(MODELS_DIR, exist_ok=True)
    vocab_path = os.path.join(MODELS_DIR, 'vocabulary.json')
    vocab.save(vocab_path)
    
    # PyTorch DataLoaders
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Instantiating the LSTM model
    model = SentimentLSTM(
        vocab_size=vocab.num_words,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=3,
        n_layers=1,
        bidirectional=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    # Training Loop
    epochs = 12
    print(f"Training Bidirectional LSTM model for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss /= len(train_loader.dataset)
        acc = correct / total
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Train Acc: {acc:.2%}")
        
    # Quick Test evaluation
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_acc = correct / total
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")
    
    # Save the PyTorch Model weights
    model_save_path = os.path.join(MODELS_DIR, 'lstm_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved LSTM model weights to '{model_save_path}'")

if __name__ == '__main__':
    main()
