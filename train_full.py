import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import json
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class PrimeVulDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512, max_samples=None):
        self.data = []
        self.targets = []
        
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
                except:
                    continue
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        vuln_count = sum(self.targets)
        normal_count = len(self.targets) - vuln_count
        print(f"Loaded {len(self.data)} samples")
        print(f"Vulnerable: {vuln_count} ({100*vuln_count/len(self.targets):.1f}%)")
        print(f"Normal: {normal_count} ({100*normal_count/len(self.targets):.1f}%)")
    
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


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)


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
        self.projection_head = ProjectionHead(input_dim=768)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        cls_embedding = hidden_states[:, 0, :]
        
        logits = self.classification_head(cls_embedding)
        projections = self.projection_head(cls_embedding)
        
        return logits, cls_embedding, projections


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    batch_size = embeddings.shape[0]
    
    similarity_matrix = torch.matmul(embeddings, embeddings.t()) / temperature
    
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    similarity_matrix = similarity_matrix - logits_max.detach()
    
    mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    similarity_matrix.masked_fill_(mask, float('-inf'))
    
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) & ~mask
    
    pos_similarities = similarity_matrix[pos_mask].exp()
    neg_similarities = similarity_matrix[neg_mask].exp().sum()
    
    if pos_similarities.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    loss = -torch.log(pos_similarities.sum() / (pos_similarities.sum() + neg_similarities + 1e-8)).mean()
    return loss


def train_epoch(model, dataloader, optimizer, device, use_scl=False):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        logits, embeddings, projections = model(input_ids, attention_mask)
        
        ce_loss = F.cross_entropy(logits, targets, weight=torch.tensor([0.52, 16.5], device=device))
        
        if use_scl:
            scl = supervised_contrastive_loss(projections, targets, temperature=0.07)
            loss = 0.5 * ce_loss + 0.5 * scl
        else:
            loss = ce_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            logits, _, _ = model(input_ids, attention_mask)
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    predictions = torch.argmax(all_logits, dim=1).numpy()
    targets_np = all_targets.numpy()
    
    probabilities = F.softmax(all_logits, dim=1)[:, 1].numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, predictions, average='weighted', zero_division=0
    )
    
    auc = roc_auc_score(targets_np, probabilities)
    
    tn, fp, fn, tp = confusion_matrix(targets_np, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'fpr': fpr,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    model_name = 'microsoft/graphcodebert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading training data...")
    train_dataset = PrimeVulDataset('primevul_train.jsonl', tokenizer, max_samples=None)
    print("Loading validation data...")
    valid_dataset = PrimeVulDataset('primevul_valid.jsonl', tokenizer, max_samples=None)
    print("Loading test data...")
    test_dataset = PrimeVulDataset('primevul_test.jsonl', tokenizer, max_samples=None)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("\n=== Training without Contrastive Learning ===\n")
    
    model = VulnerabilityDetectionModel(model_name).to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    best_f1 = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, use_scl=False)
        print(f"Train Loss: {train_loss:.4f}")
        
        valid_metrics = evaluate(model, valid_loader, device)
        print(f"Precision: {valid_metrics['precision']:.4f}, Recall: {valid_metrics['recall']:.4f}, "
              f"F1: {valid_metrics['f1']:.4f}, AUC: {valid_metrics['auc']:.4f}\n")
        
        if valid_metrics['f1'] > best_f1:
            best_f1 = valid_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), 'baseline_best.pt')
            print(f"Saved best model (F1: {best_f1:.4f})\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}\n")
                break
        
        scheduler.step()
    
    print("\n=== Training with Supervised Contrastive Learning ===\n")
    
    model = VulnerabilityDetectionModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, use_scl=True)
        print(f"Train Loss: {train_loss:.4f}")
        
        valid_metrics = evaluate(model, valid_loader, device)
        print(f"Precision: {valid_metrics['precision']:.4f}, Recall: {valid_metrics['recall']:.4f}, "
              f"F1: {valid_metrics['f1']:.4f}, AUC: {valid_metrics['auc']:.4f}\n")
        
        if valid_metrics['f1'] > best_f1:
            best_f1 = valid_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), 'advanced_best.pt')
            print(f"Saved best model (F1: {best_f1:.4f})\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}\n")
                break
        
        scheduler.step()
    
    print("=== Final Test Evaluation ===\n")
    
    model.load_state_dict(torch.load('advanced_best.pt'))
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"Test Results:")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"FPR: {test_metrics['fpr']:.4f}")
    
    vd_score = 1 - (test_metrics['fn'] / (test_metrics['fn'] + test_metrics['tp']) * 0.5 +
                     test_metrics['fp'] / (test_metrics['fp'] + test_metrics['tn']) * 0.5)
    print(f"VD-S: {vd_score:.4f}")


if __name__ == '__main__':
    main()
