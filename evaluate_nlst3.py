from sybil import Serie, Sybil
import pandas as pd
import json
from glob import glob
import os
import pyreadstat
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, classification_report


class BinaryClassificationDataset(Dataset):
    def __init__(self, data_file, root_img_dir="/mii/data/lung/nlst/NLST_CT_raw/data"):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples from {data_file}")
        self.root_img_dir = root_img_dir
    
    def __len__(self):
        return len(self.data)

    def get_series(self, embedding_path):
        embedding_path = os.path.basename(embedding_path)
        pid_str, ts_str = embedding_path.replace(".st", "").split("_")
        pid_str = pid_str.replace("pid", "")
        embedding_path_time_index = int(ts_str.replace("ts", ""))
        patient_dir = os.path.join(self.root_img_dir, pid_str)
        time_points = sorted(glob(os.path.join(patient_dir, "*")))

        for time_index, time_point in enumerate(time_points):
            img_dirs = sorted(glob(os.path.join(time_point, "*")))
            min_img_files = None
            for img_dir_pos, img_dir in enumerate(img_dirs):
                img_files = glob(os.path.join(img_dir, "*"))
                # skip localizers with less than 20 slices
                if len(img_files) < 20:
                    continue
                if min_img_files is None or len(img_files) < len(min_img_files):
                    min_img_files = img_files
                    min_img_dir_pos = img_dir_pos
            if min_img_files is None or len(min_img_files) < 20:
                continue
            if time_index == embedding_path_time_index:
                serie = Serie(min_img_files)
                return serie
        raise ValueError(f"Method should return beforehand! PID: {pid_str}, TS: {embedding_path_time_index}")
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load series
        serie = self.get_series(item['embedding_path'])
        
        # Extract labels (only first entry of each list)
        content_info = item['content_info']
        
        # Binary classification task
        cancer = content_info['cancer']
        
        # Create labels tensor
        classification_labels = torch.tensor(cancer, dtype=torch.long)
    
        return {
            'serie': serie,
            'classification_labels': classification_labels,
            'pid': item['pid']
        }


def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['classification_labels'].to(device)
            
            outputs = model.predict(batch['serie'])
            preds = outputs.scores[0][-1]
            
            all_preds.append(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    
    # Classification metrics
    if len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds)
        metrics['auc'] = auc
        
        # Calculate per-class metrics
        all_pred_labels = [1 if p >= 0.5 else 0 for p in all_preds]
        accuracy = accuracy_score(all_labels, all_pred_labels)
        metrics['accuracy'] = accuracy
        report = classification_report(all_labels, all_pred_labels, output_dict=True, zero_division=0)
        metrics['precision'] = report['macro avg']['precision']
        metrics['recall'] = report['macro avg']['recall']
        metrics['f1'] = report['macro avg']['f1-score']
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Sybil')
    parser.add_argument('--eval_file', required=False, default="/home/avepa/MedTrinity-25M/nlst_cancer_train_aux_vqa_delta2True_v0.json", help='Path to training data JSON file')
    parser.add_argument('--save_dir', default='./eval3_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets and dataloaders
    eval_dataset = BinaryClassificationDataset(args.eval_file)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Load a trained model
    model = Sybil("sybil_ensemble", device=device)

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