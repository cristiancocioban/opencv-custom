import pandas as pd
import xml.etree.ElementTree as ET
import os

# 1. Load your master auto-labeled CSV
master_csv_path = '../data/bskt/training_data.csv'
df = pd.read_csv(master_csv_path)

# 2. Reset the label columns to 0 (we are going to overwrite them with pure ground truth)
df['Dribble'] = 0
df['Crossover'] = 0
df['Hand_Touch'] = 0

# 3. List the corrected XML files you downloaded from CVAT
# Format: {'Video_ID_in_CSV': 'path_to_xml_file'}
corrected_files = {
    'crossover_02_05_1.mp4': '../data/bskt/cvat_exports/crossover_02_05_1_cvat_final.xml',
    # Add your other videos here...
}

# 4. Loop through each XML and update the master DataFrame
for video_id, xml_path in corrected_files.items():
    if not os.path.exists(xml_path):
        print(f"Skipping {xml_path} - File not found.")
        continue
        
    print(f"Applying ground truth for {video_id}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # CVAT XML stores frames in <image> tags
    for image in root.findall('image'):
        frame_id = int(image.get('id'))
        
        # Look for the tags you added/kept in CVAT
        for tag in image.findall('tag'):
            label = tag.get('label')
            
            # Find the exact row in the DataFrame and update it
            row_indexer = (df['Video_ID'] == video_id) & (df['Frame_ID'] == frame_id)
            
            if label == 'Crossover':
                df.loc[row_indexer, 'Crossover'] = 1
            elif label == 'Dribble':
                df.loc[row_indexer, 'Dribble'] = 1
            elif label == 'Hand_Touch':
                df.loc[row_indexer, 'Hand_Touch'] = 1   

# 5. Save the final, perfect dataset!
output_path = '../data/bskt/training_dataset.csv'
df.to_csv(output_path, index=False)
print(f"Success! Master Ground Truth saved to {output_path}")