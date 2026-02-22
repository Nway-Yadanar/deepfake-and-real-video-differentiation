import torch
import cv2
import numpy as np
from model3d import VideoResNet2D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Load Model --------
def load_model(model_path):
    model = VideoResNet2D(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# -------- Preprocess Video --------
def preprocess_video(video_path, clip_len=8, size=224):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("No frames found.")
        return None

    indices = np.linspace(0, total_frames - 1, clip_len, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))

        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        frame = (frame - mean) / std

        frames.append(frame)

    cap.release()

    if len(frames) < clip_len:
        print("Not enough frames.")
        return None

    clip = torch.stack(frames)            # (T,C,H,W)
    clip = clip.permute(1, 0, 2, 3)       # (C,T,H,W)
    clip = clip.unsqueeze(0)              # (1,C,T,H,W)

    return clip.to(DEVICE)


# -------- Predict --------
def predict_video(model, video_path):
    clip = preprocess_video(video_path)

    if clip is None:
        return None, None

    with torch.no_grad():
        output = model(clip)
        probs = torch.softmax(output, dim=1)

    confidence, pred = torch.max(probs, 1)

    class_map = {0: "REAL", 1: "FAKE"}

    print("Raw logits:", output.cpu().numpy())
    print("Probabilities:", probs.cpu().numpy())
    print("Prediction:", class_map[pred.item()])
    print("Confidence:", confidence.item())

    return class_map[pred.item()], confidence.item()


# -------- Main --------
if __name__ == "__main__":
    model = load_model("best_model.pt")

    video_path = r"D:\deepfake-and-real-video-differentiation\\testingvids\\Fake\\120_118.mp4"


    predict_video(model, video_path)
