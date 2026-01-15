import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class PrimeVulDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512, max_samples=None):
        self.data = []
        self.targets = []
        self.cwe = []
        
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    code = item.get('func', '')
                    target = int(item.get('target', 0))
                    if code and len(code) > 10:
                        self.data.append(code)
                        self.targets.append(target)
                        self.cwe.append(item.get('CWE', 'unknown'))
                except:
                    continue
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        code = self.data[idx]
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }


class ClassificationHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class VulnerabilityDetectionModel(nn.Module):
    def __init__(self, model_name='microsoft/graphcodebert-base'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classification_head = ClassificationHead(input_dim=768)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        cls_embedding = hidden_states[:, 0, :]
        
        logits = self.classification_head(cls_embedding)
        
        return logits, cls_embedding


def predict(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    return all_predictions, all_targets, all_probs


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
    
    test_dataset = PrimeVulDataset('primevul_test.jsonl', tokenizer, max_samples=3384)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Loaded test set: {len(test_dataset)} samples\n")
    
    model = VulnerabilityDetectionModel('microsoft/graphcodebert-base').to(device)
    model.load_state_dict(torch.load('advanced_best.pt'))
    
    predictions, targets, probs = predict(model, test_loader, device)
    
    tp = np.sum((predictions == 1) & (targets == 1))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    vd_score = 1 - (fnr * 0.5 + fpr * 0.5)
    
    print("Test Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"FNR: {fnr:.4f}")
    print(f"VD-S: {vd_score:.4f}")


if __name__ == '__main__':
    main()
