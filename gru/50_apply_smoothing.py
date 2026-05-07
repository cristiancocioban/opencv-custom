import pandas as pd
import numpy as np

def convert_and_smooth(df_video, label_col):
    """
    Takes a single video's dataframe, finds blocks of 1s, isolates the floor impact,
    and applies a Gaussian curve.
    """
    raw_labels = df_video[label_col].values
    y_coords = df_video['Rel_Ball_Y'].values
    
    # 1. Create a blank canvas for the new discrete labels
    new_labels = np.zeros_like(raw_labels, dtype=float)
    
    # 2. Find all the continuous blocks of 1s
    in_block = False
    block_start = 0
    blocks = []
    
    for i in range(len(raw_labels)):
        if raw_labels[i] == 1.0 and not in_block:
            in_block = True
            block_start = i
        elif raw_labels[i] == 0.0 and in_block:
            in_block = False
            blocks.append((block_start, i - 1))
            
    # Catch a block if the video ends while dribbling
    if in_block:
        blocks.append((block_start, len(raw_labels) - 1))
        
    # 3. Find the exact floor impact (Max Y) for each block
    for (start, end) in blocks:
        block_y_coords = y_coords[start:end+1]
        impact_idx = start + np.argmax(block_y_coords)
        new_labels[impact_idx] = 1.0

    # 4. Apply the Gaussian Smoothing Curve around our new anchor points
    smoothed = np.zeros_like(new_labels, dtype=float)
    for i in range(len(new_labels)):
        if new_labels[i] == 1.0:
            smoothed[i] = 1.0
            # 1 frame before/after
            if i - 1 >= 0: smoothed[i-1] = max(smoothed[i-1], 0.75)
            if i + 1 < len(smoothed): smoothed[i+1] = max(smoothed[i+1], 0.75)
            # 2 frames before/after
            if i - 2 >= 0: smoothed[i-2] = max(smoothed[i-2], 0.25)
            if i + 2 < len(smoothed): smoothed[i+2] = max(smoothed[i+2], 0.25)
            
    return smoothed

def process_dataset(input_csv, output_csv):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Sort strictly by video and frame
    df = df.sort_values(['Video_ID', 'Frame_ID']).reset_index(drop=True)
    
    print("Converting continuous Dribble blocks to Gaussian curves...")
    dribble_smoothed = []
    for video_id, group in df.groupby('Video_ID', sort=False):
        dribble_smoothed.extend(convert_and_smooth(group, 'Dribble'))
    df['Dribble'] = dribble_smoothed

    print("Converting continuous Crossover blocks to Gaussian curves...")
    crossover_smoothed = []
    for video_id, group in df.groupby('Video_ID', sort=False):
        crossover_smoothed.extend(convert_and_smooth(group, 'Crossover'))
    df['Crossover'] = crossover_smoothed

    mask = (df['Crossover'] > 0) & (df['Dribble'] == 0)
    copied = int(mask.sum())
    df.loc[mask, 'Dribble'] = df.loc[mask, 'Crossover']
    print(f"Copied {copied} Crossover values into Dribble where Dribble was 0.")

    # Save it to the NEW file path
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! Smoothed dataset saved safely to: {output_csv}\n")

if __name__ == "__main__":
    # Input File -> Output File
    #process_dataset("../data/bskt/validation_dataset.csv", "../data/bskt/validation_dataset_smooth.csv")
    
    # You can also do the same for your training dataset later!
    process_dataset("../data/bskt/training_dataset.csv", "../data/bskt/training_dataset_smooth.csv")