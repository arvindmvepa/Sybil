import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import argparse


class MultiTaskDataset(Dataset):
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
        
        # Multi-class tasks (first 5)
        location = content_info['location'][0]
        interval_change = content_info['interval_change'][0]
        interval_growth = content_info['interval_growth'][0]
        margins = content_info['margins'][0]
        predominant_attenuation = content_info['predominant_attenuation'][0]
        
        # Regression tasks (last 2)
        longest_diameter = content_info['longest_diameter'][0]
        longest_perpendicular_diameter = content_info['longest_perpendicular_diameter'][0]
        
        # Create labels tensor
        classification_labels = torch.tensor([
            location, interval_change, interval_growth, margins, predominant_attenuation
        ], dtype=torch.long)
        
        regression_labels = torch.tensor([
            longest_diameter, longest_perpendicular_diameter
        ], dtype=torch.float32)
        
        # Create masks for valid labels (not -1)
        classification_mask = (classification_labels != -1)
        regression_mask = (regression_labels != -1.0)
        
        # Debug: Print first few samples to understand the data
        if idx < 3:  # Only print for first 3 samples
            print(f"Sample {idx}: classification_labels = {classification_labels}")
            print(f"Sample {idx}: classification_mask = {classification_mask}")
        
        return {
            'embedding': embedding,
            'classification_labels': classification_labels,
            'regression_labels': regression_labels,
            'classification_mask': classification_mask,
            'regression_mask': regression_mask,
            'pid': item['pid']
        }


class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, num_classes_per_task, num_regression_tasks, hidden_dim=512):
        super(MultiTaskHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classification_tasks = len(num_classes_per_task)
        self.num_regression_tasks = num_regression_tasks
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Classification heads
        self.classification_heads = nn.ModuleList()
        for num_classes in num_classes_per_task:
            self.classification_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
            )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_regression_tasks)
        )
    
    def forward(self, x):
        # Shared features
        shared_features = self.shared_layers(x)
        
        # Classification outputs
        classification_outputs = []
        for head in self.classification_heads:
            classification_outputs.append(head(shared_features))
        
        # Regression output
        regression_output = self.regression_head(shared_features)
        
        return classification_outputs, regression_output


class MultiTaskLoss(nn.Module):
    def __init__(self, num_classification_tasks, num_regression_tasks):
        super(MultiTaskLoss, self).__init__()
        self.num_classification_tasks = num_classification_tasks
        self.num_regression_tasks = num_regression_tasks
        
        self.classification_criterion = nn.CrossEntropyLoss(reduction='none')
        self.regression_criterion = nn.MSELoss(reduction='none')
    
    def forward(self, classification_outputs, regression_output, 
                classification_labels, regression_labels,
                classification_mask, regression_mask):
        
        total_loss = 0.0
        losses = {}
        
        # Classification losses
        for i, output in enumerate(classification_outputs):
            task_mask = classification_mask[:, i]
            print(f"Task {i}: mask sum = {task_mask.sum()}, original labels = {classification_labels[:, i][:10]}")
            if task_mask.sum() > 0:  # Only compute loss if there are valid labels
                valid_labels = classification_labels[:, i][task_mask]
                valid_outputs = output[task_mask]
                print(f"Task {i}: valid_labels min={valid_labels.min()}, max={valid_labels.max()}, unique={torch.unique(valid_labels)}")
                print(f"Task {i}: output shape = {valid_outputs.shape}, expected classes = {valid_outputs.shape[1] if len(valid_outputs.shape) > 1 else 'N/A'}")
                task_loss = self.classification_criterion(valid_outputs, valid_labels).mean()
                total_loss += task_loss
                losses[f'classification_task_{i}'] = task_loss.item()
            else:
                print(f"Task {i}: No valid labels, skipping")
        
        # Regression losses
        for i in range(self.num_regression_tasks):
            task_mask = regression_mask[:, i]
            if task_mask.sum() > 0:  # Only compute loss if there are valid labels
                valid_labels = regression_labels[:, i][task_mask]
                valid_outputs = regression_output[:, i][task_mask]
                task_loss = self.regression_criterion(valid_outputs, valid_labels).mean()
                total_loss += task_loss
                losses[f'regression_task_{i}'] = task_loss.item()
        
        losses['total'] = total_loss.item()
        return total_loss, losses


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_classification_preds = [[] for _ in range(5)]
    all_classification_labels = [[] for _ in range(5)]
    all_regression_preds = [[] for _ in range(2)]
    all_regression_labels = [[] for _ in range(2)]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embedding'].to(device)
            classification_labels = batch['classification_labels'].to(device)
            regression_labels = batch['regression_labels'].to(device)
            classification_mask = batch['classification_mask'].to(device)
            regression_mask = batch['regression_mask'].to(device)
            
            classification_outputs, regression_output = model(embeddings)
            
            loss, _ = criterion(classification_outputs, regression_output,
                              classification_labels, regression_labels,
                              classification_mask, regression_mask)
            total_loss += loss.item()
            
            # Collect predictions for metrics
            for i, output in enumerate(classification_outputs):
                task_mask = classification_mask[:, i]
                if task_mask.sum() > 0:
                    valid_labels = classification_labels[:, i][task_mask]
                    valid_preds = torch.argmax(output[task_mask], dim=1)
                    all_classification_preds[i].extend(valid_preds.cpu().numpy())
                    all_classification_labels[i].extend(valid_labels.cpu().numpy())
            
            for i in range(2):
                task_mask = regression_mask[:, i]
                if task_mask.sum() > 0:
                    valid_labels = regression_labels[:, i][task_mask]
                    valid_preds = regression_output[:, i][task_mask]
                    all_regression_preds[i].extend(valid_preds.cpu().numpy())
                    all_regression_labels[i].extend(valid_labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    avg_loss = total_loss / len(dataloader)
    metrics['loss'] = avg_loss
    
    # Classification metrics
    classification_task_names = ['location', 'interval_change', 'interval_growth', 'margins', 'predominant_attenuation']
    for i, task_name in enumerate(classification_task_names):
        if len(all_classification_preds[i]) > 0:
            accuracy = accuracy_score(all_classification_labels[i], all_classification_preds[i])
            metrics[f'{task_name}_accuracy'] = accuracy
    
    # Regression metrics
    regression_task_names = ['longest_diameter', 'longest_perpendicular_diameter']
    for i, task_name in enumerate(regression_task_names):
        if len(all_regression_preds[i]) > 0:
            mse = mean_squared_error(all_regression_labels[i], all_regression_preds[i])
            r2 = r2_score(all_regression_labels[i], all_regression_preds[i])
            metrics[f'{task_name}_mse'] = mse
            metrics[f'{task_name}_r2'] = r2
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train multi-task head for NLST embeddings')
    parser.add_argument('--train_file', required=True, help='Path to training data JSON file')
    parser.add_argument('--val_file', required=True, help='Path to validation data JSON file') 
    parser.add_argument('--test_file', required=True, help='Path to test data JSON file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets and dataloaders
    train_dataset = MultiTaskDataset(args.train_file)
    val_dataset = MultiTaskDataset(args.val_file)
    test_dataset = MultiTaskDataset(args.test_file)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    num_classes_per_task = [8, 4, 7, 3, 3]
    print(f"Model expects number of classes per task: {num_classes_per_task}")
    
    # Model
    embedding_dim = 512 * 25 * 16 * 16  # Flattened embedding dimension
    model = MultiTaskHead(
        input_dim=embedding_dim,
        num_classes_per_task=num_classes_per_task,
        num_regression_tasks=2,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Loss and optimizer
    criterion = MultiTaskLoss(num_classification_tasks=5, num_regression_tasks=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            embeddings = batch['embedding'].to(device)
            classification_labels = batch['classification_labels'].to(device)
            regression_labels = batch['regression_labels'].to(device)
            classification_mask = batch['classification_mask'].to(device)
            regression_mask = batch['regression_mask'].to(device)
            
            optimizer.zero_grad()
            
            classification_outputs, regression_output = model(embeddings)
            
            # Debug: Print tensor shapes and label ranges for first batch
            if total_train_loss == 0:  # First batch only
                print("=== FIRST BATCH DEBUG INFO ===")
                for i, output in enumerate(classification_outputs):
                    print(f"Task {i}: output shape = {output.shape}, num_classes = {num_classes_per_task[i]}")
                print(f"Classification labels shape: {classification_labels.shape}")
                print(f"Classification mask shape: {classification_mask.shape}")
                print("=== END DEBUG INFO ===")
            
            loss, loss_dict = criterion(classification_outputs, regression_output,
                                      classification_labels, regression_labels,
                                      classification_mask, regression_mask)
            
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
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"New best model saved with val loss: {val_loss:.4f}")
    
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