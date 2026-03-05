from sybil import Serie, Sybil
import pandas as pd
from glob import glob
import os
import pyreadstat
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, classification_report


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


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embedding'].to(device)
            labels = batch['classification_labels'].to(device)
            
            outputs = model.partial_predict(embeddings)
            
            
            # Collect predictions for metrics
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    
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
    parser.add_argument('--eval_file', required=False, default="/home/avepa/MedTrinity-25M/nlst_cancer_train_aux_vqa_delta2True_v0.json", help='Path to training data JSON file')
    parser.add_argument('--save_dir', default='./results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets and dataloaders
    eval_dataset = BinaryClassificationDataset(args.eval_file)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load a trained model
    model = Sybil("sybil_ensemble")

    # Evaluation
    print("Evaluating on evaluation set...")
    eval_metrics = evaluate_model(model, eval_loader, device)
    
    print("\n=== EVAL RESULTS ===")
    for metric_name, value in eval_metrics.items():
        print(f"Eval {metric_name}: {value:.4f}")
    
    # Save evaluation results
    with open(os.path.join(args.save_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=2)

if __name__ == "__main__":
    main()