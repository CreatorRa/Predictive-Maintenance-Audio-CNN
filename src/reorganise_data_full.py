import shutil
from pathlib import Path

def pool_all_pump_data(source_dir: str, target_dir: str):
    """
    Scans through all pump directories, renames audio files to include the pump ID, 
    and copies them into master 'Normal' and 'Abnormal' folders.
    
    OVERWRITE BEHAVIOR: 
    If a file already exists in the destination folder (e.g., from a previous test run),
    this script will safely overwrite it. It will NOT create duplicates (like file(1).wav).
    """
    # Convert string paths to Path objects for cross-platform compatibility
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 1. Define the master destination folders exactly as required
    master_normal = target_path / "Normal (Master)"
    master_abnormal = target_path / "Abnormal (Master)"
    
    # 2. Create the master folders if they don't exist
    # exist_ok=True ensures the script doesn't crash if the folders are already there
    master_normal.mkdir(parents=True, exist_ok=True)
    master_abnormal.mkdir(parents=True, exist_ok=True)

    print("Starting full data pool extraction. Overwriting existing files if found...\n")

    # 3. Iterate through everything inside the main pump directory
    for pump_dir in source_path.iterdir():
        
        # 4. Filter for valid pump folders: Must be a directory AND start with "id_"
        # This safely ignores files or the 'Master Raw Data' folder itself
        if pump_dir.is_dir() and pump_dir.name.startswith("id_"):
            pump_name = pump_dir.name
            print(f"Processing directory: {pump_name}...")
            
            # 5. Look for 'normal' and 'abnormal' subdirectories within the current pump folder
            for condition in ["normal", "abnormal"]:
                condition_dir = pump_dir / condition
                
                # Check if the normal/abnormal folder actually exists for this pump
                if condition_dir.exists() and condition_dir.is_dir():
                    
                    # 6. Find all .wav files in this specific condition directory
                    audio_files = list(condition_dir.glob("*.wav"))
                    
                    for audio_file in audio_files:
                        
                        # 7. Construct the new unique filename (e.g., "id_00_00000000.wav")
                        # This prevents files from different pumps with the same name from clashing
                        new_filename = f"{pump_name}_{audio_file.name}"
                        
                        # 8. Route the file to the correct custom-named master folder
                        if condition == "normal":
                            dest_path = master_normal / new_filename
                        else:
                            dest_path = master_abnormal / new_filename
                            
                        # 9. Perform the copy and OVERWRITE
                        # shutil.copy2 copies the file and its metadata (timestamps).
                        # If a file with 'new_filename' already exists at 'dest_path' (like from our test),
                        # it is automatically and silently OVERWRITTEN here.
                        shutil.copy2(audio_file, dest_path)
                        
            print(f"  -> Finished copying all normal and abnormal files for {pump_name}")

    print(f"\nSUCCESS: All data pooled safely into: {target_path.absolute()}")


if __name__ == "__main__":
    # Using raw strings (r"") to perfectly handle Windows absolute paths and backslashes
    
    # RAW_DATA_DIR points to the parent folder containing id_00, id_02, id_04, id_06
    RAW_DATA_DIR = r"C:\Users\Carter\OneDrive\Documents\KLU\KLU Studies\Deep learning and Machine Learning\-6_dB_pump\pump"       
    
    # MASTER_DATA_DIR points to where the "Normal (Master)" and "Abnormal (Master)" folders will live
    MASTER_DATA_DIR = r"C:\Users\Carter\OneDrive\Documents\KLU\KLU Studies\Deep learning and Machine Learning\-6_dB_pump\pump\Master Raw Data"  
    
    # Execute the function
    pool_all_pump_data(RAW_DATA_DIR, MASTER_DATA_DIR)