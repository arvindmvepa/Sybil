import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, classification_report
import argparse
from sybil.models.pooling_layer import MultiAttentionPool


class BinaryClassificationDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load embedding
        embedding_data = load_file(item['embedding_path'])
        embedding = embedding_data['embeddings']  # Shape should be [512, 25, 16, 16]
        
        # Flatten the embedding for the classifier head
        embedding = embedding.flatten()
        
        # Extract labels (only first entry of each list)
        content_info = item['content_info']
        
        # Binary classification task
        cancer = content_info['cancer']
        
        # Create labels tensor
        classification_labels = torch.tensor(cancer, dtype=torch.long)
    
        return {
            'embedding': embedding,
            'classification_labels': classification_labels,
            'pid': item['pid']
        }


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.pool = MultiAttentionPool()
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.25)
        self.linear = nn.Linear(self.hidden_dim, 2)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embedding'].to(device)
            labels = batch['classification_labels'].to(device)
            
            outputs = model(embeddings)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Collect predictions for metrics
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    avg_loss = total_loss / len(dataloader)
    metrics['loss'] = avg_loss
    
    # Classification metrics
    if len(all_preds) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        metrics['accuracy'] = accuracy
        auc = roc_auc_score(all_labels, all_preds)
        metrics['auc'] = auc
        
        # Calculate per-class metrics
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        metrics['precision'] = report['macro avg']['precision']
        metrics['recall'] = report['macro avg']['recall']
        metrics['f1'] = report['macro avg']['f1-score']
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train multi-task head for NLST embeddings')
    parser.add_argument('--train_file', required=False, default="/home/avepa/MedTrinity-25M/nlst_cancer_train_aux_vqa_delta2True_v0.json", help='Path to training data JSON file')
    parser.add_argument('--val_file', required=False, default="/home/avepa/MedTrinity-25M/nlst_cancer_val_aux_vqa_delta2True_v0.json", help='Path to validation data JSON file') 
    parser.add_argument('--test_file', required=False, default="/home/avepa/MedTrinity-25M/nlst_cancer_test_aux_vqa_delta2True_v0.json", help='Path to test data JSON file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', default='./cancer1_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets and dataloaders
    train_dataset = BinaryClassificationDataset(args.train_file)
    val_dataset = BinaryClassificationDataset(args.val_file)
    test_dataset = BinaryClassificationDataset(args.test_file)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    embedding_dim = 512 * 25 * 16 * 16  # Flattened embedding dimension
    model = ClassificationHead(
        input_dim=embedding_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = 20
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            embeddings = batch['embedding'].to(device)
            labels = batch['classification_labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(embeddings)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Print validation metrics
        for metric_name, value in val_metrics.items():
            if metric_name != 'loss':
                print(f"Val {metric_name}: {value:.4f}")
        print("-" * 50)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"New best model saved with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
            break
    
    print(f"Training completed. Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.4f}")
    
    # Load best model for testing
    print("Loading best model for testing...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print("\n=== TEST RESULTS ===")
    for metric_name, value in test_metrics.items():
        print(f"Test {metric_name}: {value:.4f}")
    
    # Save test results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"Test results saved to {os.path.join(args.save_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()