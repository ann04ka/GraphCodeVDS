import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
sns.set_palette("rocket")


class PrimeVulDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512, max_samples=5000):
        self.data = []
        self.targets = []
        self.projects = []
        
        print(f"Loading {jsonl_path}...")
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
                        self.projects.append(item.get('project', 'unknown'))
                except:
                    continue
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Vulnerable: {sum(self.targets)} ({100*sum(self.targets)/len(self.targets):.1f}%)")
        print(f"Normal: {len(self.targets)-sum(self.targets)} ({100*(len(self.targets)-sum(self.targets))/len(self.targets):.1f}%)")
    
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
            'target': torch.tensor(self.targets[idx], dtype=torch.long),
            'project': self.projects[idx]
        }


def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    targets = []
    projects = []
    
    print("\n Extracting embeddings...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
            
            embeddings.extend(cls_embeddings)
            targets.extend(batch['target'].cpu().numpy())
            projects.extend(batch['project'])
    
    embeddings = np.array(embeddings)
    targets = np.array(targets)
    
    print(f"Extracted {embeddings.shape[0]} embeddings of shape {embeddings.shape[1]}")
    
    return embeddings, targets, projects


def reduce_to_2d(embeddings, method='umap'):
    print(f"\n Reducing to 2D using {method.upper()}...")
    
    if method == 'umap':
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            n_jobs=-1
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        
    elif method == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        embeddings_2d = reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Reduced to shape {embeddings_2d.shape}")
    
    return embeddings_2d


def visualize_embeddings(embeddings_2d, targets, projects, title="Embedding Space"):
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')
    
    ax = axes[0]
    ax.set_facecolor('#0d1117')
    
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=targets,
        cmap='plasma',
        s=80,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.3
    )
    
    ax.set_xlabel('Dimension 1', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Dimension 2', fontsize=14, fontweight='bold', color='white')
    ax.set_title(f'{title}\n(Vulnerable vs Normal)', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.grid(True, alpha=0.15, color='white')
    ax.tick_params(colors='white')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Target Class', fontsize=12, color='white')
    cbar.ax.tick_params(colors='white')
    
    ax = axes[1]
    ax.set_facecolor('#0d1117')
    
    unique_projects = list(set(projects))
    project_counts = [(p, sum(1 for x in projects if x == p)) for p in unique_projects]
    top_projects = sorted(project_counts, key=lambda x: x[1], reverse=True)[:10]
    top_project_names = [p[0] for p in top_projects]
    
    cmap = plt.cm.tab10
    colors_map = {}
    for i, proj in enumerate(top_project_names):
        colors_map[proj] = cmap(i)
    
    for proj in top_project_names:
        mask = np.array([p == proj for p in projects])
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=f'{proj} ({mask.sum()})',
            s=80,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.3
        )
    
    other_mask = np.array([p not in top_project_names for p in projects])
    if other_mask.sum() > 0:
        ax.scatter(
            embeddings_2d[other_mask, 0],
            embeddings_2d[other_mask, 1],
            c='gray',
            s=30,
            alpha=0.3,
            label=f'Others ({other_mask.sum()})'
        )
    
    ax.set_xlabel('Dimension 1', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Dimension 2', fontsize=14, fontweight='bold', color='white')
    ax.set_title(f'{title}\n(Top 10 Projects)', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.grid(True, alpha=0.15, color='white')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    
    return fig


def visualize_by_vulnerability_density(embeddings_2d, targets):
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')
    
    ax = axes[0]
    ax.set_facecolor('#0d1117')
    
    normal_mask = targets == 0
    ax.scatter(
        embeddings_2d[normal_mask, 0],
        embeddings_2d[normal_mask, 1],
        c='#00ff88',
        s=60,
        alpha=0.6,
        label=f'Normal Code ({sum(normal_mask)})',
        edgecolors='white',
        linewidth=0.3
    )
    
    vulnerable_mask = targets == 1
    ax.scatter(
        embeddings_2d[vulnerable_mask, 0],
        embeddings_2d[vulnerable_mask, 1],
        c='#ff0055',
        s=120,
        alpha=0.85,
        label=f'Vulnerable Code ({sum(vulnerable_mask)})',
        edgecolors='yellow',
        linewidth=1.2,
        marker='*'
    )
    
    ax.set_xlabel('Dimension 1', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Dimension 2', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Embedding Space: Separated Classes', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.15, color='white')
    ax.tick_params(colors='white')
    
    ax = axes[1]
    ax.set_facecolor('#0d1117')
    
    from scipy.stats import gaussian_kde
    
    xy = np.vstack([embeddings_2d[:, 0], embeddings_2d[:, 1]])
    z = targets
    
    x = np.linspace(embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max(), 100)
    y = np.linspace(embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max(), 100)
    X, Y = np.meshgrid(x, y)
    
    if sum(vulnerable_mask) > 1:
        xy_vuln = np.vstack([
            embeddings_2d[vulnerable_mask, 0],
            embeddings_2d[vulnerable_mask, 1]
        ])
        
        kde = gaussian_kde(xy_vuln)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
        
        contourf = ax.contourf(X, Y, Z, levels=15, cmap='hot', alpha=0.8)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Vulnerability Density', fontsize=12, color='white')
        cbar.ax.tick_params(colors='white')
    
    ax.scatter(
        embeddings_2d[normal_mask, 0],
        embeddings_2d[normal_mask, 1],
        c='cyan',
        s=40,
        alpha=0.4,
        label='Normal Code'
    )
    
    ax.scatter(
        embeddings_2d[vulnerable_mask, 0],
        embeddings_2d[vulnerable_mask, 1],
        c='#ff0055',
        s=120,
        alpha=0.95,
        label='Vulnerable Code',
        marker='*',
        edgecolors='yellow',
        linewidth=1.2
    )
    
    ax.set_xlabel('Dimension 1', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Dimension 2', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Vulnerability Density Heatmap', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.15, color='white')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    
    return fig


def main():
    print("\n" + "="*70)
    print("EMBEDDING SPACE VISUALIZATION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model_name = 'microsoft/codebert-base'
    print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    dataset = PrimeVulDataset(
        'primevul_train.jsonl',
        tokenizer,
        max_samples=5000
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )
    
    embeddings, targets, projects = extract_embeddings(model, dataloader, device)
    
    embeddings_2d = reduce_to_2d(embeddings, method='umap')
    
    print("\nCreating visualizations...")
    
    fig1 = visualize_embeddings(
        embeddings_2d,
        targets,
        projects,
        title="CodeBERT Embedding Space"
    )
    plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print("Saved: embeddings_visualization.png")
    
    fig2 = visualize_by_vulnerability_density(embeddings_2d, targets)
    plt.savefig('vulnerability_density.png', dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print("Saved: vulnerability_density.png")
   
    print("\nStatistics:")
    print(f" Total samples: {len(targets)}")
    print(f" Vulnerable: {sum(targets)} ({100*sum(targets)/len(targets):.1f}%)")
    print(f" Normal: {len(targets)-sum(targets)} ({100*(len(targets)-sum(targets))/len(targets):.1f}%)")
    
    from sklearn.metrics.pairwise import euclidean_distances
    
    vulnerable_embeddings = embeddings[targets == 1]
    normal_embeddings = embeddings[targets == 0]
    
    if len(vulnerable_embeddings) > 1 and len(normal_embeddings) > 1:
        vuln_intra_dist = np.mean([
            euclidean_distances([vulnerable_embeddings[i]], vulnerable_embeddings).min()
            for i in range(min(100, len(vulnerable_embeddings)))
        ])
        
        normal_intra_dist = np.mean([
            euclidean_distances([normal_embeddings[i]], normal_embeddings).min()
            for i in range(min(100, len(normal_embeddings)))
        ])
        
        inter_dist = np.mean([
            euclidean_distances([vulnerable_embeddings[i]], normal_embeddings).min()
            for i in range(min(100, len(vulnerable_embeddings)))
        ])
        
        print(f"\n Embedding Space Metrics:")
        print(f"   Vulnerable intra-class distance: {vuln_intra_dist:.4f}")
        print(f"   Normal intra-class distance: {normal_intra_dist:.4f}")
        print(f"   Inter-class distance: {inter_dist:.4f}")
        print(f"   Separability ratio: {inter_dist / ((vuln_intra_dist + normal_intra_dist) / 2):.4f}")
        
        if inter_dist / ((vuln_intra_dist + normal_intra_dist) / 2) > 1.5:
            print("   Classes are well-separated!")
        else:
            print("   Classes are overlapping (model needs improvement)")
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
