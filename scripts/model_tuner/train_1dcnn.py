import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import os
import csv

class ObjectivityDataset(Dataset):
    def __init__(self, data_path, vocab=None, max_vocab_size=20000):
        self.data_path = data_path
        self.texts = []
        self.labels = []
        
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract features and targets
        for item in data:
            if 'text' in item and 'predicted_score' in item and item['predicted_score'] is not None:
                self.texts.append(self.tokenize(item['text']))
                # Use predicted_score as label
                self.labels.append(float(item['predicted_score']))
                
        # Build or use existing vocabulary
        if vocab is None:
            self.vocab = self.build_vocab(self.texts, max_vocab_size)
        else:
            self.vocab = vocab
            
        # Encode all texts into integer sequences
        self.encoded_texts = [self.encode(text) for text in self.texts]
        
    def tokenize(self, text):
        # Extremely simplified tokenization - just words/numbers
        return re.findall(r'\b\S+\b', text.lower())
    
    def build_vocab(self, texts, max_size):
        counter = Counter()
        for text in texts:
            counter.update(text)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in counter.most_common(max_size - 2):
            vocab[word] = len(vocab)
        return vocab
        
    def encode(self, text):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in text]
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

def generate_collate_fn(min_len):
    """
    Collate function to ensure variable length sequences are padded correctly.
    min_len ensures that sequences are at least as long as our largest Conv1D kernel to avoid errors.
    """
    def collate_fn(batch):
        texts, labels = zip(*batch)
        lengths = [len(t) for t in texts]
        
        # Batch size is padded to max length in the batch, but at least min_len
        max_len = max(max(lengths), min_len)
        
        padded_texts = []
        for t in texts:
            pad_len = max_len - len(t)
            if pad_len > 0:
                padded_texts.append(torch.cat([t, torch.zeros(pad_len, dtype=torch.long)]))
            else:
                padded_texts.append(t)
                
        return torch.stack(padded_texts), torch.stack(labels)
    return collate_fn

class Text1DCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(Text1DCNN, self).__init__()
        
        # Embeddings map integers to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1D Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        embedded = self.embedding(text) 
        # embedded shape: [batch_size, seq_len, emb_dim]
        
        # Conv1d expects [batch_size, channels(emb_dim), seq_len]
        embedded = embedded.permute(0, 2, 1) 
        
        # Apply conv layers and ReLU
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        # conved_n shape: [batch_size, num_filters, seq_len - filter_sizes[n] + 1]
        
        # Global Max Pooling (adaptive max pooling to 1 handles variable sequence lengths gracefully)
        pooled = [torch.nn.functional.adaptive_max_pool1d(conv, 1).squeeze(2) for conv in conved]
        # pooled_n shape: [batch_size, num_filters]
        
        # Concatenate features from all filter sizes
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat shape: [batch_size, num_filters * len(filter_sizes)]
        
        # Linear layer output
        output = self.fc(cat).squeeze(1)
        
        # Apply Tanh to bound the output to [-1, 1], but scale pre-activation 
        # to prevent vanishing gradients if the linear layer outputs large values initially
        return torch.tanh(output / 2.0)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for texts, labels in iterator:
        optimizer.zero_grad()
        
        predictions = model(texts)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for texts, labels in iterator:
            predictions = model(texts)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

if __name__ == '__main__':
    # Hyperparameters
    DATA_PATH = r"c:\Users\ZULUL\SentimentDetector\datasets\average_review_no_score_judged_deepseek.json"
    VOCAB_SIZE = 25000
    EMBEDDING_DIM = 100
    NUM_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = 1 # Regression Output
    DROPOUT = 0.5
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = ObjectivityDataset(DATA_PATH, max_vocab_size=VOCAB_SIZE)
    
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Loaded {len(dataset)} examples.")
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    min_sequence_length = max(FILTER_SIZES)
    collate = generate_collate_fn(min_len=min_sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    
    print("Building model...")
    model = Text1DCNN(
        vocab_size=len(dataset.vocab),
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT
    )
    
    # Optimize using Adam and MSELoss for continuous score predicting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Using L1Loss (MAE) or MSELoss; MSELoss is more typical for regression.
    criterion = nn.MSELoss()
    
    print("Beginning training loop...")
    
    best_val_loss = float('inf')
    
    # Open CSV file to log losses
    os.makedirs('logs', exist_ok=True)
    csv_file_path = 'logs/training_loss.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
        
        for epoch in range(EPOCHS):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss = evaluate(model, val_loader, criterion)
            
            print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
            # Log train and val loss
            writer.writerow([epoch + 1, train_loss, val_loss])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'vocab': dataset.vocab,
                    'model_state_dict': model.state_dict(),
                    'embedding_dim': EMBEDDING_DIM,
                    'num_filters': NUM_FILTERS,
                    'filter_sizes': FILTER_SIZES
                }, 'models/1dcnn_objectivity_model.pt')
                print("  [Saved best model]")
            
    print(f"Training complete. The best model has been saved to 'models/1dcnn_objectivity_model.pt'.")
    print(f"Training loss data saved to '{csv_file_path}'.")
