# train.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


## !! TO REMOVE AFTER TESTING !! ######################################################
#import sys
# Remove installed package from sys.path if present
#for p in list(sys.path):
#    if "site-packages" in p:
#        if "AutomaticRange" in p:
#            sys.path.remove(p)

# Add your local src directory to the front of sys.path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
## !! TO REMOVE AFTER TESTING !! ######################################################

from AutomaticRange.models import AutomaticRangeNet
from AutomaticRange.data import RangeAnnotationDataset


# Batch
BATCH = "training_set_processed_CD4_nsamp_20_ntile_5_07102025"
MODEL = "small_model_training_set_processed_CD4_nsamp_20_ntile_5_07102025"

# --- Config ---
config = {
    "data": {
        "annotations_dir": "annotations/" + BATCH + "/",
        "marker_dir": "data/" + BATCH + "/tiles_marker/",
        "dapi_dir": "data/" + BATCH + "/tiles_DAPI/"
    },
    "train_params": {
        "batch_size": 10,
        "epochs": 25,
        "learning_rate": 1e-3,  
        "val_split": 0.2,
        "shuffle": True,
        "augment": True
    },
    "log_dir": "logs"
}

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset ---
dataset = RangeAnnotationDataset(
    annotations_dir=config["data"]["annotations_dir"],
    tiles_marker_dir=config["data"]["marker_dir"],
    tiles_dapi_dir=config["data"]["dapi_dir"],
    augment=config["train_params"]["augment"]
)

print("Finished loarding dataset")

val_size = int(len(dataset) * config["train_params"]["val_split"])
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

print(f"Training set size: {len(train_set)}, Validation set size: {len(val_set)}")
print(f"Total dataset size: {len(dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=config["train_params"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["train_params"]["batch_size"])

# --- Model ---
model = AutomaticRangeNet().to(device)  # Move model to GPU
print(f"Model file: {model.eval()}")

#model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["train_params"]["learning_rate"])

writer = SummaryWriter(log_dir=config["log_dir"])

# --- Training loop ---
for epoch in range(config["train_params"]["epochs"]):
    total_loss = 0
    for imgs, targets in train_loader:
        # Move data to GPU
        imgs, targets = imgs.to(device), targets.to(device)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            # Move data to GPU
            imgs, targets = imgs.to(device), targets.to(device)

            preds = model(imgs)
            loss = criterion(preds, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    model.train()

    print(f"Epoch {epoch+1}/{config['train_params']['epochs']} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/" + MODEL + "_automatic_range.pt")
writer.close()