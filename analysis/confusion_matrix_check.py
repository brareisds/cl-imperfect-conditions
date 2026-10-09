"""
confusion_matrix_check.py

Sanity check requested: loads a trained checkpoint (task3.ckpt, the
final task, produced by run_sanity_savemodel.sh with --save-models)
and computes a real confusion matrix on the test set, task-agnostic
(all classes seen so far competing together, same setting as TAg
Acc) -- to check whether the reported accuracy is genuine
discrimination between classes, or an artifact (e.g. the model
collapsing onto the majority class(es) while still scoring well on
an imbalanced test set).

Must be run from inside LifeLonger_default/src, so relative imports
(networks, datasets, approach) resolve the same way main_incremental.py
uses them.

Usage:
    cd LifeLonger_default/src
    export SHUFFLE_CLASSES_FIX=1   # must match the flag used during training
    python confusion_matrix_check.py \
        --checkpoint-dir ../results/official_savemodel/organamnist_ewc/organamnist_ewc_fixd_0 \
        --dataset organamnist --num-tasks 4 --network resnet18 \
        --seed 0 \
        --output-csv ../analysis/organamnist_ewc_confusion.csv
"""

import argparse
import os
import sys
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, ".")
from networks.network import LLL_Net
import networks
from datasets.data_loader import get_loaders


def build_model(network_name, taskcla, keep_existing_head=False):
    # Mirrors main_incremental.py's model construction
    import torchvision.models as tv_models
    if network_name == "resnet18":
        init_model = tv_models.resnet18(pretrained=False, num_classes=taskcla[0][1])
        init_model.head_var = "fc"
    else:
        raise NotImplementedError(f"Only resnet18 wired up here, got {network_name}")

    net = LLL_Net(init_model, remove_existing_head=not keep_existing_head)
    for t, ncla in taskcla:
        net.add_head(ncla)
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True,
                         help="results-path used for the sanity-check run "
                              "(contains models/task<N>.ckpt)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--network", default="resnet18")
    parser.add_argument("--final-task", type=int, default=3,
                         help="index of the last task (0-indexed); "
                              "loads models/task<final-task>.ckpt")
    parser.add_argument("--output-csv", default="confusion_matrix.csv")
    parser.add_argument("--seed", type=int, required=True,
                         help="must match the seed used during training, "
                              "so class_order (if shuffled) is reproduced "
                              "identically -- required, not just recommended")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Must replicate main_incremental.py's exact call order: seed
    # fixed immediately before get_loaders(), since class shuffling
    # (when enabled) depends on the global RNG state at that point.
    import utils as project_utils
    project_utils.seed_everything(seed=args.seed)

    # Reload the same loaders main_incremental.py would build, to get
    # taskcla (classes per task) and the test sets in the same order
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(
        [args.dataset], args.num_tasks, nc_first_task=None,
        batch_size=32, num_workers=2, pin_memory=False
    )

    net = build_model(args.network, taskcla)
    ckpt_path = os.path.join(args.checkpoint_dir, "models",
                              f"task{args.final_task}.ckpt")
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    # Diagnostic mode: replicate incremental_learning.py's eval()/
    # calculate_metrics() EXACTLY, per task, to find where this
    # script's numbers diverge from the training-time log.
    print("\n=== Per-task diagnostic (replicating calculate_metrics exactly) ===")
    all_preds_tag, all_targets = [], []
    per_task_taw, per_task_tag = [], []
    with torch.no_grad():
        for u in range(args.final_task + 1):
            hits_taw_sum, hits_tag_sum, n_sum = 0, 0, 0
            shown = 0
            for images, targets in tst_loader[u]:
                images = images.to(device)
                targets_dev = targets.to(device)
                outputs = net(images)

                # Task-Aware (mirrors calculate_metrics exactly)
                pred_taw = torch.zeros_like(targets_dev)
                for m in range(len(pred_taw)):
                    this_task = (net.task_cls.cumsum(0) <= targets[m]).sum()
                    pred_taw[m] = outputs[this_task][m].argmax() + net.task_offset[this_task]
                hits_taw = (pred_taw == targets_dev).float()

                # Task-Agnostic (mirrors calculate_metrics exactly)
                pred_tag = torch.cat(outputs, dim=1).argmax(1)
                hits_tag = (pred_tag == targets_dev).float()

                if shown < 1:
                    print(f"  [task {u}] sample targets: {targets[:8].tolist()}")
                    print(f"  [task {u}] sample pred_taw: {pred_taw[:8].tolist()}")
                    print(f"  [task {u}] sample pred_tag: {pred_tag[:8].tolist()}")
                    shown += 1

                hits_taw_sum += hits_taw.sum().item()
                hits_tag_sum += hits_tag.sum().item()
                n_sum += len(targets)

                all_preds_tag.extend(pred_tag.cpu().tolist())
                all_targets.extend(targets.tolist())

            taw_acc = 100 * hits_taw_sum / n_sum
            tag_acc = 100 * hits_tag_sum / n_sum
            per_task_taw.append(taw_acc)
            per_task_tag.append(tag_acc)
            print(f"  Task {u}: n={n_sum}  TAw={taw_acc:.1f}%  TAg={tag_acc:.1f}%")

    print(f"\n  Mean TAw across tasks: {sum(per_task_taw)/len(per_task_taw):.1f}%")
    print(f"  Mean TAg across tasks: {sum(per_task_tag)/len(per_task_tag):.1f}%")
    print("  (compare these two lines directly to the log's last Avg.: "
          "line in each of the TAw Acc / TAg Acc blocks)")

    all_preds = np.array(all_preds_tag)
    all_targets = np.array(all_targets)

    n_classes = int(net.task_cls.sum().item())
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(n_classes)))

    print("\nConfusion matrix (rows=true class, cols=predicted class):")
    print(cm)

    print("\nPer-class report:")
    print(classification_report(all_targets, all_preds, zero_division=0))

    # Flag classes the model never predicts at all (column sum == 0)
    # and classes it always predicts as something else (diagonal == 0
    # but the class has samples) -- the two patterns that would
    # explain an inflated aggregate accuracy without real
    # discrimination.
    col_sums = cm.sum(axis=0)
    never_predicted = np.where(col_sums == 0)[0]
    if len(never_predicted) > 0:
        print(f"\n[flag] Classes NEVER predicted (collapsed): {never_predicted.tolist()}")
    else:
        print("\n[ok] Every class is predicted at least once.")

    np.savetxt(args.output_csv, cm, delimiter=",", fmt="%d")
    print(f"\nConfusion matrix saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
