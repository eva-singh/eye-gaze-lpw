import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.frcnn import get_model
import time

# ── Dataset ────────────────────────────────────────────────────────────────────

class GazeDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms.ToTensor()

        all_images = sorted(os.listdir(images_dir))
        self.samples = []

        for img_file in all_images:
            if not img_file.endswith(".jpg"):
                continue
            label_file = img_file.replace(".jpg", ".txt")
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                self.samples.append(img_file)

        print(f"Dataset: {len(self.samples)} samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_file)
        label_path = os.path.join(self.labels_dir, img_file.replace(".jpg", ".txt"))

        # Load image
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        img_tensor = self.transforms(img)

        # Load label (YOLO format: class xc yc w h)
        with open(label_path, "r") as f:
            line = f.readline().strip().split()

        cls, xc, yc, bw, bh = map(float, line)
        xc, yc, bw, bh = xc * W, yc * H, bw * W, bh * H

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        boxes   = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels  = torch.tensor([1], dtype=torch.int64)  # 1 = gaze, 0 = background

        target = {"boxes": boxes, "labels": labels}
        return img_tensor, target


# ── Collate (handles variable-size targets) ────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


# ── Training Loop ──────────────────────────────────────────────────────────────

def train():
    IMAGES_DIR  = "dataset/images"
    LABELS_DIR  = "dataset/labels"
    OUTPUT_DIR  = "runs/frcnn"
    NUM_EPOCHS  = 10
    BATCH_SIZE  = 4
    LR          = 0.005
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # Dataset & Loader
    dataset    = GazeDataset(IMAGES_DIR, LABELS_DIR)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)

    # Model
    model = get_model()
    model.to(DEVICE)

    # Optimizer
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss     = 0
        start_time     = time.time()

        for batch_idx, (images, targets) in enumerate(loader):
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            # Log every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx+1}/{len(loader)}] "
                      f"Loss: {losses.item():.4f}")

        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss = total_loss / len(loader)

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Avg Loss: {avg_loss:.4f} "
              f"Time: {elapsed:.1f}s\n")

        # Save checkpoint every epoch
        ckpt_path = os.path.join(OUTPUT_DIR, f"frcnn_epoch{epoch+1}.pth")
        torch.save({
            "epoch":      epoch + 1,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "loss":       avg_loss,
        }, ckpt_path)
        print(f"Saved checkpoint → {ckpt_path}")

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "frcnn_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model → {final_path}")


if __name__ == "__main__":
    train()
