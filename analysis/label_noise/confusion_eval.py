import argparse
import os

import numpy as np
import torch

import nme_eval as ne

OUT = os.path.join(ne.OUT_DIR, "confusions")
NATIVE = {"ewc": "logits", "icarl": "nme"}


def conf(y, pred, n):
    c = np.zeros((n, n), dtype=np.int64)
    np.add.at(c, (y, pred), 1)
    return c


def run_one(d, s, ckpt_dir, device):
    ne.project_utils.seed_everything(seed=s)
    _, _, tst, taskcla = ne.get_loaders([d], 4, nc_first_task=None, batch_size=32, num_workers=2, pin_memory=False)
    net = ne.build_model("resnet18", taskcla)
    net.load_state_dict(torch.load(os.path.join(ckpt_dir, "models", "task3.ckpt"), map_location=device))
    net.to(device).eval()
    n_cls = int(net.task_cls.sum().item())
    ex = torch.load(os.path.join(ckpt_dir, "exemplars", "task3.pt"), map_location="cpu", weights_only=False)
    means, classes = ne.class_means(net, tst[0].dataset, ex, device, 32)
    assert list(classes) == list(range(n_cls)), f"exemplar classes {list(classes)}"
    ys, pl, pn = [], [], []
    for ld in tst:
        o, f, y = ne.forward_all(net, ld, device)
        dist = torch.cdist(f / f.norm(dim=1, keepdim=True), means)
        ys.append(y.numpy())
        pl.append(o.argmax(1).numpy())
        pn.append(dist.argmin(1).numpy())
    y = np.concatenate(ys)
    return {"logits": conf(y, np.concatenate(pl), n_cls), "nme": conf(y, np.concatenate(pn), n_cls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["organamnist", "bloodmnist"])
    ap.add_argument("--methods", nargs="+", default=["icarl", "ewc"])
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--legacy-root", default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    assert float(os.environ.get("LABEL_NOISE", "0") or 0) == 0, "LABEL_NOISE must be unset when evaluating"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    bad = tot = 0
    for d in a.datasets:
        for m in a.methods:
            phase = ne.PHASES["intra"][m]
            for p in a.noise:
                for s in a.seeds:
                    cdir = ne.find_ckpt_dir(phase, d, m, p, s, a.legacy_root)
                    if cdir is None:
                        print(f"WARN no checkpoint+exemplars: {phase} {d} {m} p{p:g} seed{s}")
                        continue
                    res = run_one(d, s, cdir, device)
                    for clf, c in res.items():
                        name = m if clf == NATIVE[m] else f"{m}-{clf}"
                        path = os.path.join(a.out, f"{d}_{name}_p{p:g}_seed{s}_confusion.csv")
                        if os.path.exists(path):
                            tot += 1
                            same = np.array_equal(np.loadtxt(path, delimiter=",").astype(np.int64), c)
                            bad += int(not same)
                            print(f"check {os.path.basename(path)}: {'SAME' if same else 'DIFFERENT'} (existing file kept)")
                            continue
                        np.savetxt(path, c, fmt="%d", delimiter=",")
                        print(f"wrote {os.path.basename(path)} acc={100 * np.trace(c) / c.sum():.2f}")
    print(f"existing files compared: differ {bad}/{tot}")


if __name__ == "__main__":
    main()
