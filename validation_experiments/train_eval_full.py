import os
import argparse
import numpy as np
import pandas as pd
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

def get_resnet(model_name, num_classes):
    factory = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50
    }
    net = factory[model_name](weights=None)
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

    # Bounded optimization in arctanh space: x' = 0.5 * (tanh(w) + 1)
    x_clamped = torch.clamp(x, 0.0, 1.0)
    x_atanh = torch.atanh((x_clamped * 2) - 1 + 1e-6)
    
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
            x_new = 0.5 * (torch.tanh(w) + 1)
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

def main():
    parser = argparse.ArgumentParser(description="Full Dataset Training & C&W Boundary Evaluation")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10 if args.dataset == 'cifar10' else 100
    L_u = np.log(num_classes) + 1e-3

    ckpt_base = os.path.join(args.out_dir, "checkpoints_full", f"{args.dataset}_{args.model}")
    os.makedirs(ckpt_base, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_eval = transforms.Compose([transforms.ToTensor()])
    
    ds_cls = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100
    raw_train = ds_cls(root='./data', train=True, download=True)
    
    train_ds = IndexedDataset([(transform_train(img), tgt) for img, tgt in raw_train])
    eval_ds = IndexedDataset([(transform_eval(img), tgt) for img, tgt in raw_train])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    all_records = []
    
    for seed in args.seeds:
        print(f"\n[*] [Full Dataset 50k] Training Model: {args.model} | Dataset: {args.dataset} | Seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = get_resnet(args.model, num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
        )
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        csl = np.zeros(len(raw_train), dtype=np.float64)
        csg = np.zeros(len(raw_train), dtype=np.float64)
        
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
                    csl[idx_arr] += losses.detach().cpu().numpy()
                    csg[idx_arr] += grad_norms.detach().cpu().numpy()
                    
                optimizer.step()
            scheduler.step()
            
            # Checkpoint target model for every epoch
            ckpt_path = os.path.join(ckpt_base, f"target_seed{seed}_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'csl': csl,
                'csg': csg
            }, ckpt_path)
            
        print(f"[*] Running C&W attack evaluation across all {len(raw_train)} samples (Seed {seed})...")
        d_adv = np.zeros(len(raw_train), dtype=np.float64)
        for idxs, x, y in tqdm(eval_loader, desc=f"C&W Eval Seed {seed}", leave=False):
            x, y = x.to(device), y.to(device)
            batch_d = compute_cw_adv_distance(model, x, y)
            d_adv[idxs.numpy()] = batch_d

        d_adv_proxy = 2.0 * ((L_u * args.epochs) - csl) / np.maximum(csg, 1e-9)
        
        for i in range(len(raw_train)):
            all_records.append({
                'model': args.model,
                'dataset': args.dataset,
                'seed': seed,
                'global_idx': i,
                'target': raw_train[i][1],
                'csl': csl[i],
                'csg': csg[i],
                'd_adv': d_adv[i],
                'd_adv_proxy': d_adv_proxy[i]
            })

    out_csv = os.path.join(args.out_dir, f"full_{args.dataset}_{args.model}_3seeds.csv")
    pd.DataFrame(all_records).to_csv(out_csv, index=False)
    print(f"[+] Successfully saved full evaluation metrics to {out_csv}")

if __name__ == '__main__':
    main()
