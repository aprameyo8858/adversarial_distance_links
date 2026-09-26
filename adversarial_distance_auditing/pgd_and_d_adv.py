import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return idx, img, target

def get_resnet18(num_classes):
    net = models.resnet18(weights=None)
    # Standard CIFAR adaptation: replace 7x7 conv with 3x3 and remove maxpool
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

def compute_cw_adv_distance(model, x, y, max_eps=6.0, cw_bs_steps=5, cw_iters=200, cw_lr=0.01):
    model.eval()
    B = x.size(0)
    device = x.device
    min_d_adv = torch.full((B,), max_eps, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        initial_mis = (model(x).argmax(dim=1) != y)
        min_d_adv[initial_mis] = 0.0
        
    unresolved = ~initial_mis
    if not unresolved.any():
        return min_d_adv.cpu().numpy()

    # Optimization in arctanh space: x' = 0.5 * (tanh(w) + 1)
    x_clamped = torch.clamp(x, 0.0, 1.0)
    x_atanh = torch.atanh((x_clamped * 2.0) - 1.0 + 1e-6)
    
    const = torch.full((B,), 0.01, device=device)
    lower_bound = torch.zeros(B, device=device)
    upper_bound = torch.full((B,), 1e10, device=device)
    
    for _ in range(cw_bs_steps):
        w = x_atanh.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([w], lr=cw_lr)
        best_l2 = torch.full((B,), 1e10, device=device)
        success = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(cw_iters):
            optimizer.zero_grad()
            x_new = 0.5 * (torch.tanh(w) + 1.0)
            logits = model(x_new)
            
            l2_dist = torch.norm((x_new - x).view(B, -1), p=2, dim=1)
            real_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
            other_logits = logits.clone()
            other_logits[torch.arange(B), y] = -1e4
            max_other = other_logits.max(dim=1)[0]
            
            loss_f = torch.clamp(real_logits - max_other, min=0.0)
            loss = torch.sum(const * loss_f + l2_dist)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                misclassified = (logits.argmax(dim=1) != y)
                mask = misclassified & (l2_dist < best_l2) & unresolved
                best_l2 = torch.where(mask, l2_dist, best_l2)
                min_d_adv = torch.where(mask & (l2_dist < min_d_adv), l2_dist, min_d_adv)
                success |= misclassified
                
        with torch.no_grad():
            for i in range(B):
                if success[i]:
                    upper_bound[i] = min(upper_bound[i].item(), const[i].item())
                    if upper_bound[i] < 1e9:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2.0
                else:
                    lower_bound[i] = max(lower_bound[i].item(), const[i].item())
                    if upper_bound[i] < 1e9:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2.0
                    else:
                        const[i] *= 10.0
                        
    return min_d_adv.cpu().numpy()

def compute_pgd_binary_search(model, x, y, pgd_steps, max_eps=6.0, search_steps=25):
    """
    Untargeted L2 PGD bounded via a 25-step binary search over perturbation radius eps in [0, max_eps].
    """
    model.eval()
    B = x.size(0)
    device = x.device
    best_d = torch.full((B,), max_eps, device=device)
    
    with torch.no_grad():
        initial_mis = (model(x).argmax(dim=1) != y)
        best_d[initial_mis] = 0.0
        
    unresolved = ~initial_mis
    if not unresolved.any():
        return best_d.cpu().numpy()

    lower_bound = torch.zeros(B, device=device)
    upper_bound = torch.full((B,), max_eps, device=device)

    for _ in range(search_steps):
        eps = (lower_bound + upper_bound) / 2.0
        
        delta = torch.zeros_like(x).uniform_(-1e-4, 1e-4)
        delta.requires_grad = True
        
        alpha = (eps / float(pgd_steps)) * 2.5
        success = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(pgd_steps):
            adv_x = torch.clamp(x + delta, 0.0, 1.0)
            outputs = model(adv_x)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, y)
            loss.mean().backward()
            
            with torch.no_grad():
                grad = delta.grad
                grad_norm = torch.norm(grad.view(B, -1), p=2, dim=1).view(-1, 1, 1, 1) + 1e-12
                delta += alpha.view(-1, 1, 1, 1) * (grad / grad_norm)
                
                # Projection to L2 ball of radius eps
                delta_norm = torch.norm(delta.view(B, -1), p=2, dim=1).view(-1, 1, 1, 1)
                factor = torch.clamp(delta_norm / eps.view(-1, 1, 1, 1), min=1.0)
                delta.data = delta.data / factor
                delta.grad.zero_()
                
                # Check misclassification on current step
                current_preds = model(torch.clamp(x + delta, 0.0, 1.0)).argmax(dim=1)
                success |= (current_preds != y)
                
        upper_bound = torch.where(success, eps, upper_bound)
        lower_bound = torch.where(~success, eps, lower_bound)
        
        with torch.no_grad():
            final_dist = torch.norm(delta.view(B, -1), p=2, dim=1)
            improved = success & (final_dist < best_d) & unresolved
            best_d = torch.where(improved, final_dist, best_d)
            
    return best_d.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Appendix C.5 Zero-Overhead Adv Auditing (Full Horizon Only)")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--max-eps', type=float, default=6.0, help='Max epsilon perturbation for binary search')
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10 if args.dataset == 'cifar10' else 100
    L_u = np.log(num_classes) + 1e-3
    os.makedirs(args.out_dir, exist_ok=True)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    eval_transform = transforms.Compose([transforms.ToTensor()])
    
    ds_cls = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100
    raw_train = ds_cls(root='./data', train=True, download=True)
    
    train_ds = IndexedDataset([(train_transform(img), tgt) for img, tgt in raw_train])
    eval_ds = IndexedDataset([(eval_transform(img), tgt) for img, tgt in raw_train])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    N = len(raw_train)
    all_records = []
    
    for seed in args.seeds:
        print(f"\n[*] [Appendix C.5 Full Horizon] ResNet-18 | {args.dataset.upper()} | Seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = get_resnet18(num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
        )
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Accumulate CSL and CSG strictly over the full 200-epoch horizon
        csl = np.zeros(N, dtype=np.float64)
        csg = np.zeros(N, dtype=np.float64)
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            for idxs, x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x.requires_grad = True
                
                optimizer.zero_grad()
                out = model(x)
                losses = criterion(out, y)
                losses.mean().backward()
                
                with torch.no_grad():
                    B = x.size(0)
                    grad_norms = torch.norm(x.grad.view(B, -1), p=2, dim=1) * B
                    idx_arr = idxs.numpy()
                    csl[idx_arr] += losses.cpu().numpy()
                    csg[idx_arr] += grad_norms.cpu().numpy()

                optimizer.step()
            scheduler.step()
            
        print(f"[*] Attacking {N} training instances at terminal checkpoint (Seed {seed})...")
        d_cw = np.zeros(N, dtype=np.float64)
        d_pgd3 = np.zeros(N, dtype=np.float64)
        d_pgd5 = np.zeros(N, dtype=np.float64)
        d_pgd10 = np.zeros(N, dtype=np.float64)
        d_pgd20 = np.zeros(N, dtype=np.float64)
        
        for idxs, x, y in tqdm(eval_loader, desc=f"Evaluating Attacks (Seed {seed})", leave=False):
            x, y = x.to(device), y.to(device)
            idx_arr = idxs.numpy()
            
            d_cw[idx_arr] = compute_cw_adv_distance(model, x, y, max_eps=args.max_eps)
            d_pgd3[idx_arr] = compute_pgd_binary_search(model, x, y, pgd_steps=3, max_eps=args.max_eps, search_steps=25)
            d_pgd5[idx_arr] = compute_pgd_binary_search(model, x, y, pgd_steps=5, max_eps=args.max_eps, search_steps=25)
            d_pgd10[idx_arr] = compute_pgd_binary_search(model, x, y, pgd_steps=10, max_eps=args.max_eps, search_steps=25)
            d_pgd20[idx_arr] = compute_pgd_binary_search(model, x, y, pgd_steps=20, max_eps=args.max_eps, search_steps=25)
            
        proxy_full = 2.0 * ((L_u * args.epochs) - csl) / np.maximum(csg, 1e-9)
        
        for i in range(N):
            all_records.append({
                'dataset': args.dataset,
                'seed': seed,
                'sample_idx': i,
                'target': raw_train[i][1],
                'csl_full': csl[i],
                'csg_full': csg[i],
                'proxy_full': proxy_full[i],
                'd_cw': d_cw[i],
                'd_pgd3': d_pgd3[i],
                'd_pgd5': d_pgd5[i],
                'd_pgd10': d_pgd10[i],
                'd_pgd20': d_pgd20[i]
            })
            
    df = pd.DataFrame(all_records)
    out_csv = os.path.join(args.out_dir, f"c5_full_horizon_{args.dataset}_resnet18_3seeds.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Detailed per-sample records saved to: {out_csv}")
    
    # Calculate Spearman rank correlations (Full Horizon column of Table 19)
    print("\n" + "="*50)
    print(f"Table 19 Replication (Full Horizon only) - {args.dataset.upper()}")
    print("="*50)
    print(f"{'Target Attack':<16} | {'Full Horizon (rho)':<25}")
    print("-" * 50)
    
    attacks = [
        ('PGD-3', 'd_pgd3'),
        ('PGD-5', 'd_pgd5'),
        ('PGD-10', 'd_pgd10'),
        ('PGD-20', 'd_pgd20'),
        ('C&W', 'd_cw')
    ]
    
    for atk_name, atk_col in attacks:
        corrs = []
        for seed in args.seeds:
            sub = df[df['seed'] == seed]
            rho, _ = spearmanr(sub[atk_col], sub['proxy_full'])
            corrs.append(rho)
        mean_rho = np.mean(corrs)
        std_rho = np.std(corrs)
        print(f"{atk_name:<16} | {mean_rho:.4f} ± {std_rho:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
