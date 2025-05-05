#!/usr/bin/env python3
"""
Chain-of-responsibility ensemble for Speech Commands

model-1 : silence vs command    (task-2 winner)
model-2 : known vs unknown      (task-3 winner)
model-3 : 10-class command      (task-4 winner)

During inference on TASK-1:
    • if model-1 ⇒ “silence”            → final label = silence
    • else model-2 ⇒ “unknown”          → final label = unknown
    • else model-3 ⇒ one of 10 commands → final label = that command
"""

import copy, json, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from helpers.model_runner_utils import (
    set_seed, get_device, train_epoch, validate, test_model
)
from helpers.speech_datasets import (
    get_task1_dataloaders, get_task2_dataloaders,
    get_task3_dataloaders, get_task4_dataloaders
)
import helpers.models as models

# ────────────────────────────────────────────────────────────────────────────
# Fixed hyper-parameter sets (your best configs)
# ────────────────────────────────────────────────────────────────────────────
BEST_CFG_TASK2 = {
    'n_mfcc': 40, 'd_model': 64,  'num_heads': 2,  'num_layers': 2,
    'kernel_size': 31, 'expansion_factor': 2, 'dropout': 0.1, 'pool': 'mean',
    'optimizer': 'adamw', 'learning_rate': 5e-4, 'weight_decay': 0.01,
    'epochs': 50, 'scheduler': 'cosine', 'scheduler_step_size': 10,
    'scheduler_gamma': 0.5
}
BEST_CFG_TASK3 = {
    'n_mfcc': 40, 'd_model': 192, 'num_heads': 6,  'num_layers': 6,
    'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean',
    'optimizer': 'adamw', 'learning_rate': 5e-4, 'weight_decay': 0.01,
    'epochs': 50, 'scheduler': 'cosine', 'scheduler_step_size': 10,
    'scheduler_gamma': 0.5
}
BEST_CFG_TASK4 = {
    'n_mfcc': 40, 'd_model': 256, 'num_heads': 8,  'num_layers': 6,
    'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean',
    'optimizer': 'adamw', 'learning_rate': 5e-4, 'weight_decay': 0.01,
    'epochs': 50, 'scheduler': 'cosine', 'scheduler_step_size': 10,
    'scheduler_gamma': 0.5
}

RESULTS_DIR = Path("results/conformer_chain_task1")

# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────
def build_model(cfg, n_classes):
    return models.ConformerClassifier(
        n_classes=n_classes,
        n_mfcc=cfg['n_mfcc'],
        d_model=cfg['d_model'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        kernel_size=cfg['kernel_size'],
        expansion_factor=cfg['expansion_factor'],
        dropout=cfg['dropout'],
        pool=cfg['pool'],
    )

def build_optimizer(model, cfg):
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['epochs'], eta_min=1e-6)
    return opt, sched

def train_model(cfg, loaders, device, early_stop=5):
    train_loader, val_loader, test_loader, id2label = loaders
    n_classes = len(id2label)

    model = build_model(cfg, n_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optim, sched = build_optimizer(model, cfg)

    best_val, patience, best_weights = 0.0, 0, None
    for epoch in range(1, cfg['epochs']+1):
        t_loss, t_acc = train_epoch(model, device, train_loader, criterion, optim)
        v_loss, v_acc = validate(model, device, val_loader, criterion)
        sched.step()
        if v_acc > best_val:
            best_val, best_weights, patience = v_acc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= early_stop: break
    model.load_state_dict(best_weights)
    test_acc, cm = test_model(model, device, test_loader, id2label)
    return model, test_acc, cm, id2label

def chain_predict(m1, m2, m3, loader, device,
                  id2label_task2, id2label_task3, id2label_task4, id2label_task1):
    """run chain, return accuracy & confusion matrix on task-1 test-set"""
    # Get numeric indices for key labels
    idx_silence = list(id2label_task2.keys())[list(id2label_task2.values()).index('silence')]
    idx_command = 1-idx_silence  # the other class
    idx_known   = list(id2label_task3.keys())[list(id2label_task3.values()).index('known')]

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # stage 1
            p1 = torch.softmax(m1(x), 1)
            c1 = p1.argmax(1)
            silence_mask = (c1 == idx_silence)
            command_mask = ~silence_mask

            # default prediction array
            pred = torch.empty_like(y)

            # silence predictions
            silence_label = list(id2label_task1.keys())[list(id2label_task1.values()).index('silence')]
            pred[silence_mask] = silence_label

            # stage 2 – run only where command
            if command_mask.any():
                x2 = x[command_mask]
                p2 = torch.softmax(m2(x2), 1)
                c2 = p2.argmax(1)
                known_mask = (c2 == idx_known)
                unknown_mask = ~known_mask

                # unknown predictions
                unknown_label = list(id2label_task1.keys())[list(id2label_task1.values()).index('unknown')]
                pred[command_mask.nonzero(as_tuple=True)[0][unknown_mask]] = unknown_label

                # stage 3 – only known commands
                if known_mask.any():
                    x3 = x2[known_mask]
                    p3 = torch.softmax(m3(x3), 1)
                    c3 = p3.argmax(1)   # indices inside 10-command space
                    # map 0-9 known->task1 label index
                    known_map = {v: k for k, v in id2label_task4.items()}  # command-name→idx
                    # Build list of final indices
                    final_labels = torch.tensor([ list(id2label_task1.keys())[list(id2label_task1.values()).index(id2label_task4[int(i)])] for i in c3 ], dtype=torch.long)
                    pred_idx = command_mask.nonzero(as_tuple=True)[0][known_mask]
                    pred[pred_idx] = final_labels

            all_preds.append(pred.cpu())
            all_targets.append(y)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    acc = (preds == targets).float().mean().item()
    cm = confusion_matrix(targets, preds, labels=list(id2label_task1.keys()))
    return acc, cm

# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/speech_commands_v0.01")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Data loaders for each task
    loaders_t1 = get_task1_dataloaders(
        data_dir=args.data_path,
        known_commands=['yes','no','up','down','left','right','on','off','stop','go'],
        batch_size=args.batch_size, seed=args.seed)
    loaders_t2 = get_task2_dataloaders(
        data_dir=args.data_path,
        batch_size=args.batch_size, seed=args.seed)
    loaders_t3 = get_task3_dataloaders(
        data_dir=args.data_path,
        known_commands=['yes','no','up','down','left','right','on','off','stop','go'],
        batch_size=args.batch_size, seed=args.seed)
    loaders_t4 = get_task4_dataloaders(
        data_dir=args.data_path,
        known_commands=['yes','no','up','down','left','right','on','off','stop','go'],
        batch_size=args.batch_size, seed=args.seed)

    # Train / load each specialist
    print("\n── Training specialist models ──")
    m2, acc2, cm2, id2l2 = train_model(BEST_CFG_TASK2, loaders_t2, device)
    print(f"task-2 model (silence vs command) test_acc = {acc2:.3%}")

    m3, acc3, cm3, id2l3 = train_model(BEST_CFG_TASK3, loaders_t3, device)
    print(f"task-3 model (known vs unknown)   test_acc = {acc3:.3%}")

    m4, acc4, cm4, id2l4 = train_model(BEST_CFG_TASK4, loaders_t4, device)
    print(f"task-4 model (10-class command)  test_acc = {acc4:.3%}")

    # Evaluate chain on task-1 test split
    _, _, test_loader_t1, id2l1 = loaders_t1
    chain_acc, chain_cm = chain_predict(
        m2, m3, m4, test_loader_t1, device,
        id2label_task2=id2l2, id2label_task3=id2l3,
        id2label_task4=id2l4, id2label_task1=id2l1)

    print("\n======================  CHAIN RESULT  ======================")
    print(f"Composite accuracy on Task-1 test-set: {chain_acc:.3%}")

    # Save all meta-data
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = RESULTS_DIR / f"chain_result_{stamp}.json"
    with out_json.open("w") as f:
        json.dump({
            "specialists": [
                {"task": 2, "config": BEST_CFG_TASK2, "test_acc": acc2},
                {"task": 3, "config": BEST_CFG_TASK3, "test_acc": acc3},
                {"task": 4, "config": BEST_CFG_TASK4, "test_acc": acc4},
            ],
            "chain_test_acc": chain_acc,
            "chain_confusion_matrix": chain_cm.tolist(),
        }, f, indent=2)
    print(f"✅  saved {out_json}")


if __name__ == "__main__":
    main()
