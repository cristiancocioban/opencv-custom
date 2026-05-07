import pandas as pd
import os

def generate_cvat_xmls():
    csv_file = '../data/bskt/training_data.csv'
    output_folder = '../data/bskt/cvat_exports'
    
    # 1. Check if the CSV exists
    if not os.path.exists(csv_file):
        print(f"Error: Cannot find {csv_file}. Make sure it is in the same folder.")
        return

    # 2. Load the Master CSV
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # 3. Create a folder to hold all the new XML files so your main folder stays clean
    os.makedirs(output_folder, exist_ok=True)

    # 4. Find all unique videos in the dataset
    unique_videos = df['Video_ID'].unique()
    print(f"Found {len(unique_videos)} unique videos. Generating XMLs...")
    print("-" * 30)

    # 5. Loop through each video and create its specific XML
    for video_id in unique_videos:
        # Filter the data so we are only looking at rows for THIS specific video
        video_df = df[df['Video_ID'] == video_id]
        
        # Start building the CVAT XML string
        xml_out = '<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n  <version>1.1</version>\n'
        
        # Loop through every frame for this specific video
        for index, row in video_df.iterrows():
            frame_id = int(row['Frame_ID'])
            
            # Create an image block for the frame
            xml_out += f'  <image id="{frame_id}" name="frame_{frame_id:06d}.png" width="1920" height="1080">\n'
            
            # Add Crossover tag if flagged
            if row['Crossover'] == 1:
                xml_out += f'    <tag label="Crossover" source="auto"></tag>\n'
                
            # Add Dribble tag if flagged
            if row['Dribble'] == 1:
                xml_out += f'    <tag label="Dribble" source="auto"></tag>\n'
                
            # Add Hand_Touch tag if flagged
            if row['Hand_Touch'] == 1:
                xml_out += f'    <tag label="Hand_Touch" source="auto"></tag>\n'
            xml_out += f'  </image>\n'
        
        xml_out += '</annotations>'
        
        # 6. Save the XML file using the video's name (e.g., video1_cvat.xml)
        # We strip the '.mp4' extension to make the filename cleaner
        clean_name = os.path.splitext(video_id)[0]
        xml_filename = f"{clean_name}_cvat.xml"
        xml_filepath = os.path.join(output_folder, xml_filename)
        
        with open(xml_filepath, 'w') as f:
            f.write(xml_out)
            
        print(f"[{clean_name}] -> Saved {xml_filename} ({len(video_df)} frames processed)")

    print("-" * 30)
    print(f"Done! All XML files are saved in the '{output_folder}' folder.")

if __name__ == "__main__":
    generate_cvat_xmls()