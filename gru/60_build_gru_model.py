import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. THE DATASET LOADER (CSV to Tensors)
# ==========================================
class HoopsDataset(Dataset):
    def __init__(self, csv_file, window_size=15):
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size
        
        # The exact 26 features the GRU will learn from. Ball_Detected (1.0
        # when YOLO/tracker found the ball this frame, 0.0 otherwise) lets
        # the GRU distinguish a real (0,0) ball position from a missed
        # detection that was filled with 0.
        self.feature_cols = [
            "Rel_Ball_X", "Rel_Ball_Y",
            "Rel_LeftElbow_X", "Rel_LeftElbow_Y", "LeftElbow_Vis",
            "Rel_RightElbow_X", "Rel_RightElbow_Y", "RightElbow_Vis",
            "Rel_LeftWrist_X", "Rel_LeftWrist_Y", "LeftWrist_Vis",
            "Rel_RightWrist_X", "Rel_RightWrist_Y", "RightWrist_Vis",
            "Rel_LeftAnkle_X", "Rel_LeftAnkle_Y", "LeftAnkle_Vis",
            "Rel_RightAnkle_X", "Rel_RightAnkle_Y", "RightAnkle_Vis",
            "Norm_Torso_Height",
            "Dist_Ball_L_Wrist",
            "Dist_Ball_R_Wrist",
            "Delta_Ball_Y",
            "Delta_Ball_X",
            "Ball_Detected",        # <--- NEW (index 25)
        ]
        
        # Fill missing data (NaN) with 0.0 so PyTorch doesn't crash
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0.0)
        
        self.samples = []
        
        # Group by Video_ID to prevent windows from crossing between two different videos
        grouped = self.df.groupby('Video_ID')
        
        for video_id, group in grouped:
            group = group.sort_values('Frame_ID').reset_index(drop=True)
            
            # Extract raw numpy arrays for speed
            features = group[self.feature_cols].values
            
            # Using the exact column names you provided
            actions = group[['Dribble', 'Crossover', 'Hand_Touch']].values
            ball_coords = group[['Rel_Ball_X', 'Rel_Ball_Y']].values
            
            # Slide the window across the video
            for i in range(len(group) - self.window_size):
                x_seq = features[i : i + self.window_size]
                y_actions = actions[i + self.window_size - 1]
                y_coords = ball_coords[i + self.window_size]
                
                self.samples.append((x_seq, y_actions, y_coords))
                
        print(f"-> Created {len(self.samples)} sliding window sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y_act, y_crd = self.samples[idx]
        return (
            torch.FloatTensor(x_seq),
            torch.FloatTensor(y_act),
            torch.FloatTensor(y_crd)
        )

# ==========================================
# 2. THE MULTI-HEAD GRU MODEL (With Regularization)
# ==========================================
class HoopsWorldModel(nn.Module):
    def __init__(self, input_size=26, hidden_size=64, num_layers=2):
        super(HoopsWorldModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 # Regularization 1: Turn off 20% of GRU neurons
        )
        
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3), # Regularization 2: Stop head from memorizing (bumped 0.2 -> 0.3 to tighten init variance)
            nn.Linear(32, 3), # 3 Labels: Dribble, Crossover, Hand_Touch
            nn.Sigmoid()
        )
        
        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularization 3: Stop head from memorizing
            nn.Linear(32, 2) # 2 Coordinates: Next_X, Next_Y
        )

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        final_summary = gru_out[:, -1, :]
        action_probs = self.classifier_head(final_summary)
        next_coords = self.predictor_head(final_summary)
        return action_probs, next_coords

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train_model():
    # Reproducibility — seed every RNG that affects training (init, dropout, shuffle).
    # Set BEFORE creating datasets, model, or DataLoaders so the seed actually applies.
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # cuDNN determinism — small perf cost, but matches the spirit of reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Setup File Paths (Matched to your terminal output)
    train_csv_path = "../data/bskt/current_datasets/training_dataset_smooth_tracker.csv"
    val_csv_path = "../data/bskt/current_datasets/validation_dataset_smooth_tracker.csv"
    
    print("--- PREPARING TRAINING DATA ---")
    train_dataset = HoopsDataset(csv_file=train_csv_path, window_size=15)
    
    print("\n--- PREPARING VALIDATION DATA ---")
    val_dataset = HoopsDataset(csv_file=val_csv_path, window_size=15)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. Initialize Model (input_size=26: original 25 features + Ball_Detected)
    model = HoopsWorldModel(input_size=26, hidden_size=64)
    
    # 3. Setup Loss Functions and Optimizer
    # reduction='none' so we can break the BCE down per action for reporting,
    # then take the mean to match the original training signal.
    action_criterion = nn.BCELoss(reduction='none')
    coord_criterion = nn.MSELoss()
    action_names = ["Dribble", "Crossover", "Hand_Touch"]
    
    # Regularization 4: Weight Decay added to the optimizer!
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 60

    # Trackers for early stopping and best-model selection
    best_val_loss = float('inf')
    best_val_f1 = -1.0
    best_loss_epoch = 0
    best_f1_epoch = 0
    patience = 8
    epochs_since_improvement = 0

    print("\nStarting Training...\n")
    # 4. The Main Loop
    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        running_train_coord = 0.0
        running_train_per_action = torch.zeros(3)
        train_tp = torch.zeros(3)
        train_fp = torch.zeros(3)
        train_fn = torch.zeros(3)
        # Apex F1 trackers: positive = label exactly 1.0 (peak frame),
        # negative = label exactly 0.0 (background). Shoulder frames excluded.
        train_apex_tp = torch.zeros(3)
        train_apex_fp = torch.zeros(3)
        train_apex_fn = torch.zeros(3)

        for batch_idx, (x_seq, y_actions, y_coords) in enumerate(train_loader):
            optimizer.zero_grad()

            pred_actions, pred_coords = model(x_seq)

            per_action_loss = action_criterion(pred_actions, y_actions).mean(dim=0)
            loss_action = per_action_loss.mean()
            loss_coord = coord_criterion(pred_coords, y_coords)
            total_train_loss = loss_action + (loss_coord * 10)

            total_train_loss.backward()
            optimizer.step()
            running_train_loss += total_train_loss.item()
            running_train_coord += loss_coord.item()
            running_train_per_action += per_action_loss.detach()

            # Per-class confusion counts. Predictions thresholded at 0.5.
            # Labels are soft (Gaussian-smoothed kernel like 0.25/0.75/1/0.75/0.25),
            # so threshold ground truth at >=0.5 — shoulder frames (0.75) count as
            # positive, frames at 0.25 count as negative. Avoids silently dropping
            # shoulder frames from the metric.
            # Apex F1: positive = label==1.0 (peak), negative = label==0.0 (bg);
            # shoulder frames excluded. Tells us how well the model handles peaks.
            with torch.no_grad():
                preds_bin = (pred_actions > 0.5).float()
                labels_bin = (y_actions >= 0.5).float()
                train_tp += ((preds_bin == 1) & (labels_bin == 1)).sum(dim=0)
                train_fp += ((preds_bin == 1) & (labels_bin == 0)).sum(dim=0)
                train_fn += ((preds_bin == 0) & (labels_bin == 1)).sum(dim=0)
                apex_pos = (y_actions == 1).float()
                apex_neg = (y_actions == 0).float()
                train_apex_tp += ((preds_bin == 1) & (apex_pos == 1)).sum(dim=0)
                train_apex_fp += ((preds_bin == 1) & (apex_neg == 1)).sum(dim=0)
                train_apex_fn += ((preds_bin == 0) & (apex_pos == 1)).sum(dim=0)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        running_val_coord = 0.0
        running_val_per_action = torch.zeros(3)
        val_tp = torch.zeros(3)
        val_fp = torch.zeros(3)
        val_fn = torch.zeros(3)
        val_apex_tp = torch.zeros(3)
        val_apex_fp = torch.zeros(3)
        val_apex_fn = torch.zeros(3)

        with torch.no_grad(): # Disable gradients for memory efficiency
            for x_seq, y_actions, y_coords in val_loader:
                pred_actions, pred_coords = model(x_seq)

                per_action_loss = action_criterion(pred_actions, y_actions).mean(dim=0)
                loss_action = per_action_loss.mean()
                loss_coord = coord_criterion(pred_coords, y_coords)
                total_val_loss = loss_action + (loss_coord * 10)

                running_val_loss += total_val_loss.item()
                running_val_coord += loss_coord.item()
                running_val_per_action += per_action_loss

                preds_bin = (pred_actions > 0.5).float()
                labels_bin = (y_actions >= 0.5).float()
                val_tp += ((preds_bin == 1) & (labels_bin == 1)).sum(dim=0)
                val_fp += ((preds_bin == 1) & (labels_bin == 0)).sum(dim=0)
                val_fn += ((preds_bin == 0) & (labels_bin == 1)).sum(dim=0)
                apex_pos = (y_actions == 1).float()
                apex_neg = (y_actions == 0).float()
                val_apex_tp += ((preds_bin == 1) & (apex_pos == 1)).sum(dim=0)
                val_apex_fp += ((preds_bin == 1) & (apex_neg == 1)).sum(dim=0)
                val_apex_fn += ((preds_bin == 0) & (apex_pos == 1)).sum(dim=0)

        # Calculate Averages
        n_train = len(train_loader)
        n_val = len(val_loader) if len(val_loader) > 0 else 1
        avg_train = running_train_loss / n_train
        avg_val = running_val_loss / n_val
        avg_train_coord = running_train_coord / n_train
        avg_val_coord = running_val_coord / n_val
        avg_train_per_action = (running_train_per_action / n_train).tolist()
        avg_val_per_action = (running_val_per_action / n_val).tolist()

        # Precision / Recall / F1 per class (eps avoids div-by-zero on empty preds)
        eps = 1e-9
        train_p = train_tp / (train_tp + train_fp + eps)
        train_r = train_tp / (train_tp + train_fn + eps)
        train_f1 = 2 * train_p * train_r / (train_p + train_r + eps)
        val_p = val_tp / (val_tp + val_fp + eps)
        val_r = val_tp / (val_tp + val_fn + eps)
        val_f1 = 2 * val_p * val_r / (val_p + val_r + eps)
        val_macro_f1 = val_f1.mean().item()

        # Apex P/R/F1 — peak-only diagnostic, more aligned with event counting
        train_apex_p = train_apex_tp / (train_apex_tp + train_apex_fp + eps)
        train_apex_r = train_apex_tp / (train_apex_tp + train_apex_fn + eps)
        train_apex_f1 = 2 * train_apex_p * train_apex_r / (train_apex_p + train_apex_r + eps)
        val_apex_p = val_apex_tp / (val_apex_tp + val_apex_fp + eps)
        val_apex_r = val_apex_tp / (val_apex_tp + val_apex_fn + eps)
        val_apex_f1 = 2 * val_apex_p * val_apex_r / (val_apex_p + val_apex_r + eps)
        val_apex_macro_f1 = val_apex_f1.mean().item()

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val macro F1: {val_macro_f1:.4f} | Apex: {val_apex_macro_f1:.4f}")
        print(f"  Coord  -> Train: {avg_train_coord:.4f} | Val: {avg_val_coord:.4f}")
        for i, name in enumerate(action_names):
            print(f"  {name:<10} -> Train: {avg_train_per_action[i]:.4f} | Val: {avg_val_per_action[i]:.4f}")
            print(f"    {'':<8}    Train P/R/F1: {train_p[i]:.2f}/{train_r[i]:.2f}/{train_f1[i]:.2f}  "
                  f"Val P/R/F1: {val_p[i]:.2f}/{val_r[i]:.2f}/{val_f1[i]:.2f}")
            print(f"    {'':<8}    Apex T/V F1: {train_apex_f1[i]:.2f}/{val_apex_f1[i]:.2f}")

        # --- BEST MODEL TRACKING (loss + F1, saved to separate files) ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_loss_epoch = epoch + 1
            torch.save(model.state_dict(), "hoops_world_model_best.pth")
            print(f"  --> New best val loss ({avg_val:.4f}) -> hoops_world_model_best.pth")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_f1_epoch = epoch + 1
            torch.save(model.state_dict(), "hoops_world_model_best_f1.pth")
            print(f"  --> New best macro F1 ({val_macro_f1:.4f}) -> hoops_world_model_best_f1.pth")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # --- EARLY STOPPING (patience tied to val macro F1 — the metric we care about) ---
        if epochs_since_improvement >= patience:
            print(f"\nEarly stopping: val macro F1 hasn't improved for {patience} epochs.")
            break

    print("\nTraining Complete!")
    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_loss_epoch} -> hoops_world_model_best.pth")
    print(f"  Best macro F1: {best_val_f1:.4f} at epoch {best_f1_epoch} -> hoops_world_model_best_f1.pth")

if __name__ == "__main__":
    train_model()