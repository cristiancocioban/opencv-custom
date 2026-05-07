import pandas as pd
import numpy as np

def inject_new_features(csv_path):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Calculate 2D Euclidean Distance from Ball to Wrists
    df['Dist_Ball_L_Wrist'] = np.sqrt(
        (df['Rel_Ball_X'] - df['Rel_LeftWrist_X'])**2 + 
        (df['Rel_Ball_Y'] - df['Rel_LeftWrist_Y'])**2
    )
    
    df['Dist_Ball_R_Wrist'] = np.sqrt(
        (df['Rel_Ball_X'] - df['Rel_RightWrist_X'])**2 + 
        (df['Rel_Ball_Y'] - df['Rel_RightWrist_Y'])**2
    )

    # 2. Calculate Y-Axis Velocity (Current Y - Previous Y)
    # We group by Video_ID to protect the boundaries between different videos
    df['Delta_Ball_Y'] = df.groupby('Video_ID')['Rel_Ball_Y'].diff()
    
    # The very first frame of every video won't have a "previous" frame, so we fill it with 0.0
    df['Delta_Ball_Y'] = df['Delta_Ball_Y'].fillna(0.0)

    # 3. Calculate X-Axis Velocity (Lateral Crossover Speed) <--- NEW!
    df['Delta_Ball_X'] = df.groupby('Video_ID')['Rel_Ball_X'].diff()
    df['Delta_Ball_X'] = df['Delta_Ball_X'].fillna(0.0)   

    # 3. Overwrite the CSV with the new columns
    df.to_csv(csv_path, index=False)
    print(f"✅ Successfully injected 4 new features. Total columns: {len(df.columns)}")

if __name__ == "__main__":
    # Update these paths if your CSVs are located somewhere else!
    inject_new_features("../data/bskt/training_dataset_smooth.csv")
    inject_new_features("../data/bskt/validation_dataset_smooth.csv")