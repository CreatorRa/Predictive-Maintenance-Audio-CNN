import shutil
from pathlib import Path

def pool_pump_data(source_dir: str, target_dir: str, test_mode: bool = False, test_limit: int = 4):
    """
    Scans through pump directories, renames audio files to include the pump name, 
    and copies them into master 'normal' and 'abnormal' folders.
    
    Args:
        source_dir: Path to the raw, separated pump folders.
        target_dir: Path to the master destination folder.
        test_mode: If True, only processes `test_limit` files per condition.
        test_limit: The maximum number of files to copy per condition during a test.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Define and create the master destination folders
    master_normal = target_path / "normal"
    master_abnormal = target_path / "abnormal"
    
    master_normal.mkdir(parents=True, exist_ok=True)
    master_abnormal.mkdir(parents=True, exist_ok=True)

    if test_mode:
        print(f"--- RUNNING IN TEST MODE: Limiting to {test_limit} files per condition ---\n")

    # Iterate through each pump directory
    for pump_dir in source_path.iterdir():
        if pump_dir.is_dir() and not pump_dir.name.startswith('.'):
            pump_name = pump_dir.name
            
            # Look for 'normal' and 'abnormal' subdirectories
            for condition in ["normal", "abnormal"]:
                condition_dir = pump_dir / condition
                
                if condition_dir.exists() and condition_dir.is_dir():
                    
                    # Convert generator to list so we can slice it for the test
                    audio_files = list(condition_dir.glob("*.wav"))
                    
                    if test_mode:
                        audio_files = audio_files[:test_limit]
                    
                    for audio_file in audio_files:
                        # Construct the new unique filename: e.g., "Pump1_00000000.wav"
                        new_filename = f"{pump_name}_{audio_file.name}"
                        
                        # Route to the correct master folder
                        if condition == "normal":
                            dest_path = master_normal / new_filename
                        else:
                            dest_path = master_abnormal / new_filename
                            
                        # Copy the file with metadata preserved
                        shutil.copy2(audio_file, dest_path)
                        
            print(f"Successfully processed files for {pump_name}")

    print(f"\nAll data pooled successfully into: {target_path.absolute()}")

if __name__ == "__main__":
    # Path(__file__).resolve() gets the absolute path to this script
    # .parent gets the 'src' folder
    # .parent.parent gets the 'Predictive-Maintenance-Audio-CNN' root folder
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Now build absolute paths from the project root
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_pumps"       
    MASTER_DATA_DIR = PROJECT_ROOT / "data" / "master_pool_TEST"  
    
    # Keep this True for your 8-file test. 
    TEST_MODE = True 
    
    pool_pump_data(RAW_DATA_DIR, MASTER_DATA_DIR, test_mode=TEST_MODE)