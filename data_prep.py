import random
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

class Video3DClips(Dataset):
    def __init__(self, root_dir, clip_len=8, size=224, stride=2, training=True):
        self.root = Path(root_dir)
        self.clip_len = clip_len
        self.size = size
        self.stride = stride
        self.training = training

        self.samples = []
        for label_name, label in [("real", 0), ("fake", 1)]:
            folder = self.root / label_name
            for p in folder.rglob("*"):
                if p.suffix.lower() in VIDEO_EXT:
                    self.samples.append((p, label))

        if not self.samples:
            raise ValueError(f"No videos found in {root_dir} (expected real/ and fake/)")

    def __len__(self):
        return len(self.samples)

    def _read_frames(self, path):
        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        indices = list(range(0, total_frames, max(1, self.stride)))

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)

            if len(frames) >= self.clip_len:
                break

        cap.release()
        return frames

    def _sample_clip(self, frames):
        if len(frames) == 0:
            return torch.zeros(3, self.clip_len, self.size, self.size)

        n = len(frames)

        if n >= self.clip_len:
            start = random.randint(0, n - self.clip_len) if self.training else (n - self.clip_len) // 2
            clip = frames[start:start + self.clip_len]
        else:
            clip = frames + [frames[-1]] * (self.clip_len - n)

        clip_t = []
       
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        for img in clip:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            t = (t - mean) / std
            clip_t.append(t)

        x = torch.stack(clip_t, dim=0).permute(1, 0, 2, 3).contiguous()
        return x

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        frames = self._read_frames(path)
        x = self._sample_clip(frames)

        if self.training and random.random() < 0.5:
            x = torch.flip(x, dims=[3])

        return x, y
