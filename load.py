import os, random, shutil
from pathlib import Path

random.seed(42)

RAW = Path("data/raw")
OUT = Path("data")

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

def all_videos(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT]

videos = all_videos(RAW)
print("Total videos found:", len(videos))

# Heuristic labeling by folder names in path
real = []
fake = []
for p in videos:
    s = str(p).lower()
    # common naming patterns in celeb-df / kaggle mirrors
    if "real" in s or "original" in s:
        real.append(p)
    elif "fake" in s or "synthesis" in s or "deepfake" in s:
        fake.append(p)

print("Real candidates:", len(real))
print("Fake candidates:", len(fake))

if len(real) == 0 or len(fake) == 0:
    print("\n[!] Could not auto-detect real/fake folders.")
    print("Open data/raw and tell me the folder names for REAL and FAKE.")
    raise SystemExit(1)

def split(lst, train=0.8, val=0.1):
    lst = lst[:]
    random.shuffle(lst)
    n = len(lst)
    n_tr = int(n*train)
    n_va = int(n*val)
    return lst[:n_tr], lst[n_tr:n_tr+n_va], lst[n_tr+n_va:]

real_tr, real_va, real_te = split(real)
fake_tr, fake_va, fake_te = split(fake)

def copy_subset(files, dst, limit=None):
    dst.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        files = files[:limit]
    for p in files:
        shutil.copy2(p, dst / p.name)

# ---- IMPORTANT: start small for first successful run ----
copy_subset(real_tr, OUT/"train"/"real", limit=200)
copy_subset(fake_tr, OUT/"train"/"fake", limit=200)
copy_subset(real_va, OUT/"val"/"real",   limit=50)
copy_subset(fake_va, OUT/"val"/"fake",   limit=50)
copy_subset(real_te, OUT/"test"/"real",  limit=50)
copy_subset(fake_te, OUT/"test"/"fake",  limit=50)

print("\nDone.")
print("Check: data/train, data/val, data/test")
