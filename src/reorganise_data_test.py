'''This script is commented out to prevent accidental execution. The script is a test to process files for quick verification before running on the entire dataset.
It is designed to pool audio files from a structured directory of pump data into two master folders: "Normal (Master)" and "Abnormal (Master)".
Each file is renamed to include its original pump ID for traceability. '''

'''import shutil
from pathlib import Path

def pool_pump_data(source_dir: str, target_dir: str, test_mode: bool = False, test_limit: int = 4):
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Define the master destination folders exactly as requested
    master_normal = target_path / "Normal (Master)"
    master_abnormal = target_path / "Abnormal (Master)"
    
    master_normal.mkdir(parents=True, exist_ok=True)
    master_abnormal.mkdir(parents=True, exist_ok=True)

    if test_mode:
        print(f"--- RUNNING IN TEST MODE: Limiting to {test_limit} files per condition ---\n")

    # Iterate through the main pump directory
    for pump_dir in source_path.iterdir():
        # CRITICAL FIX: Only process directories that start with "id_"
        # This prevents the script from accidentally trying to process the 'Master Raw Data' folder
        if pump_dir.is_dir() and pump_dir.name.startswith("id_"):
            pump_name = pump_dir.name
            
            # Look for 'normal' and 'abnormal' subdirectories within the id_xx folder
            for condition in ["normal", "abnormal"]:
                condition_dir = pump_dir / condition
                
                if condition_dir.exists() and condition_dir.is_dir():
                    audio_files = list(condition_dir.glob("*.wav"))
                    
                    if test_mode:
                        audio_files = audio_files[:test_limit]
                    
                    for audio_file in audio_files:
                        # Construct the new unique filename: e.g., "id_00_00000000.wav"
                        new_filename = f"{pump_name}_{audio_file.name}"
                        
                        # Route to the correct custom-named master folder
                        if condition == "normal":
                            dest_path = master_normal / new_filename
                        else:
                            dest_path = master_abnormal / new_filename
                            
                        # Copy the file with metadata preserved
                        shutil.copy2(audio_file, dest_path)
                        
            print(f"Successfully processed files for {pump_name}")

    print(f"\nAll data pooled successfully into: {target_path.absolute()}")

if __name__ == "__main__":
    # Using raw strings (r"") to perfectly handle Windows absolute paths
    RAW_DATA_DIR = r"C:\Users\Carter\OneDrive\Documents\KLU\KLU Studies\Deep learning and Machine Learning\-6_dB_pump\pump"       
    MASTER_DATA_DIR = r"C:\Users\Carter\OneDrive\Documents\KLU\KLU Studies\Deep learning and Machine Learning\-6_dB_pump\pump\Master Raw Data"  
    
    # Keep this True to test 4 files per condition first. 
    # Change to False when you are ready to process everything.
    TEST_MODE = True 
    
    pool_pump_data(RAW_DATA_DIR, MASTER_DATA_DIR, test_mode=TEST_MODE)

'''