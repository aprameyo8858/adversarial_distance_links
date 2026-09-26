import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import expit
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
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
    parser = argparse.ArgumentParser(description="MIA Half-Dataset Target & Shadow Training with C&W Eval")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--num-shadows', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10 if args.dataset == 'cifar10' else 100

    ckpt_target_dir = os.path.join(args.out_dir, "checkpoints_target_mia", f"{args.dataset}_{args.model}")
    ckpt_shadow_dir = os.path.join(args.out_dir, "checkpoints_shadows", f"{args.dataset}_{args.model}")
    os.makedirs(ckpt_target_dir, exist_ok=True)
    os.makedirs(ckpt_shadow_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_eval = transforms.Compose([transforms.ToTensor()])

    ds_cls = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100
    raw_train = ds_cls(root='./data', train=True, download=True)
    total_samples = len(raw_train)
    half_size = total_samples // 2

    train_ds = IndexedDataset([(transform_train(img), tgt) for img, tgt in raw_train])
    eval_ds = IndexedDataset([(transform_eval(img), tgt) for img, tgt in raw_train])

    # Fixed member split for the targets, and randomized 50% splits for the 64 shadows
    split_index_file = os.path.join(args.out_dir, f"splits_{args.dataset}_{args.model}.pt")
    if os.path.exists(split_index_file):
        splits = torch.load(split_index_file, weights_only=False)
        in_member_split = splits['in_member_split']
        shadow_splits = splits['shadow_splits']
    else:
        np.random.seed(42)
        in_member_split = np.random.choice(total_samples, half_size, replace=False)
        shadow_splits = [np.random.choice(total_samples, half_size, replace=False) for _ in range(args.num_shadows)]
        torch.save({'in_member_split': in_member_split, 'shadow_splits': shadow_splits}, split_index_file)

    target_eval_loader = DataLoader(Subset(eval_ds, in_member_split), batch_size=args.batch_size, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Step 1: Train 3 Target Seeds on the in-member half split, save every epoch, track CSL/CSG, and run C&W
    target_models = []
    target_stats = []

    for seed in args.seeds:
        print(f"\n[*] [Half Dataset 25k Target] Training Model: {args.model} | Seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = get_resnet(args.model, num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
        )
        
        train_loader = DataLoader(Subset(train_ds, in_member_split), batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        csl = np.zeros(total_samples, dtype=np.float64)
        csg = np.zeros(total_samples, dtype=np.float64)
        
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

            # Save checkpoint every epoch
            ckpt_path = os.path.join(ckpt_target_dir, f"target_seed_{seed}_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'csl': csl[in_member_split],
                'csg': csg[in_member_split]
            }, ckpt_path)

        print(f"[*] Computing C&W adversarial distance on all 25,000 in-members for Seed {seed}...")
        d_adv = np.zeros(half_size, dtype=np.float64)
        global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(in_member_split)}
        
        for idxs, x, y in tqdm(target_eval_loader, desc=f"C&W Target Seed {seed}", leave=False):
            x, y = x.to(device), y.to(device)
            batch_d = compute_cw_adv_distance(model, x, y)
            for i, g_idx in enumerate(idxs.numpy()):
                d_adv[global_to_local[g_idx]] = batch_d[i]

        target_stats.append({
            'seed': seed,
            'csl': csl[in_member_split],
            'csg': csg[in_member_split],
            'd_adv': d_adv
        })
        target_models.append(model)

    # Step 2: Train 64 Shadow Models
    print(f"\n[*] Training/Evaluating {args.num_shadows} Shadow Models on 50% Random Subsets...")
    shadow_losses = np.zeros((args.num_shadows, half_size), dtype=np.float32)
    shadow_in_mask = np.zeros((args.num_shadows, half_size), dtype=bool)

    for k in tqdm(range(args.num_shadows), desc="Shadow Ensembles"):
        shadow_ckpt = os.path.join(ckpt_shadow_dir, f"shadow_{k+1}_final.pth")
        s_model = get_resnet(args.model, num_classes).to(device)
        
        if os.path.exists(shadow_ckpt):
            s_model.load_state_dict(torch.load(shadow_ckpt, map_location=device, weights_only=False))
        else:
            s_loader = DataLoader(Subset(train_ds, shadow_splits[k]), batch_size=args.batch_size, shuffle=True, num_workers=4)
            s_opt = optim.SGD(s_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
            s_sched = optim.lr_scheduler.MultiStepLR(s_opt, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1)
            
            s_model.train()
            for _ in range(args.epochs):
                for _, x, y in s_loader:
                    x, y = x.to(device), y.to(device)
                    s_opt.zero_grad()
                    criterion(s_model(x), y).mean().backward()
                    s_opt.step()
                s_sched.step()
            torch.save(s_model.state_dict(), shadow_ckpt)

        s_model.eval()
        with torch.no_grad():
            for idxs, x, y in target_eval_loader:
                x, y = x.to(device), y.to(device)
                losses = criterion(s_model(x), y).cpu().numpy()
                for i, g_idx in enumerate(idxs.numpy()):
                    l_idx = global_to_local[g_idx]
                    shadow_losses[k, l_idx] = losses[i]
                    shadow_in_mask[k, l_idx] = (g_idx in shadow_splits[k])

    # Step 3: Compute LiRA and Yeom Loss-Threshold Advantages for each member sample
    print("[*] Computing LiRA and Yeom MIA Advantages...")
    yeom_adv = np.zeros(half_size, dtype=np.float64)
    lira_adv = np.zeros(half_size, dtype=np.float64)

    for l_idx in range(half_size):
        in_l = shadow_losses[:, l_idx][shadow_in_mask[:, l_idx]]
        out_l = shadow_losses[:, l_idx][~shadow_in_mask[:, l_idx]]
        if len(in_l) == 0 or len(out_l) == 0:
            continue

        yeom_adv[l_idx] = np.mean(np.exp(-in_l)) - np.mean(np.exp(-out_l))

        log_in = np.log(in_l + 1e-10)
        log_out = np.log(out_l + 1e-10)
        mu_in, std_in = np.mean(log_in), np.std(log_in) + 1e-10
        mu_out, std_out = np.mean(log_out), np.std(log_out) + 1e-10

        lira_ratio_in = stats.norm.logpdf(log_in, loc=mu_in, scale=std_in) - stats.norm.logpdf(log_in, loc=mu_out, scale=std_out)
        lira_ratio_out = stats.norm.logpdf(log_out, loc=mu_in, scale=std_in) - stats.norm.logpdf(log_out, loc=mu_out, scale=std_out)
        lira_adv[l_idx] = np.mean(expit(lira_ratio_in)) - np.mean(expit(lira_ratio_out))

    # Step 4: Save aggregated results
    all_mia_records = []
    for stat in target_stats:
        seed = stat['seed']
        csl = stat['csl']
        csg = stat['csg']
        d_adv = stat['d_adv']
        
        for l_idx, g_idx in enumerate(in_member_split):
            all_mia_records.append({
                'model': args.model,
                'dataset': args.dataset,
                'seed': seed,
                'global_idx': int(g_idx),
                'in_member_local_idx': l_idx,
                'target': raw_train[g_idx][1],
                'csl': csl[l_idx],
                'csg': csg[l_idx],
                'd_adv': d_adv[l_idx],
                'yeom_adv': yeom_adv[l_idx],
                'lira_adv': lira_adv[l_idx]
            })

    out_csv = os.path.join(args.out_dir, f"mia_{args.dataset}_{args.model}_3seeds.csv")
    pd.DataFrame(all_mia_records).to_csv(out_csv, index=False)
    print(f"[+] Successfully saved MIA evaluation metrics to {out_csv}")

if __name__ == '__main__':
    main()
