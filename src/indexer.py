import os
import pydicom
import pandas as pd
from pathlib import Path
import json

def build_patient_index(data_dir: str = "./data", output_db: str = "patient_index.csv"):
    """
    The 'Overnight Batch Processor'. 
    Scans every DICOM file to extract physical properties for deterministic Agent querying.
    """
    root_path = Path(data_dir)
    if not root_path.exists():
        print(f"❌ Error: Data directory {data_dir} not found.")
        return

    print("🔍 Initializing ETL Pipeline... Scanning for DICOM files...")
    
    # Grab every single dcm file
    dcm_files = list(root_path.rglob("*.dcm"))
    print(f"📂 Found {len(dcm_files)} total slices. Extracting metadata...")
    
    database = []
    
    for idx, filepath in enumerate(dcm_files):
        if idx % 500 == 0 and idx > 0:
            print(f"   ...Processed {idx}/{len(dcm_files)} files")
            
        try:
            # stop_before_pixels=True makes this incredibly fast (milliseconds per file)
            dcm = pydicom.dcmread(str(filepath), stop_before_pixels=True)
            
            # Extract Z-coordinate for spatial alignment
            # ImagePositionPatient is an array: [X, Y, Z] in millimeters
            z_coord = None
            if 'ImagePositionPatient' in dcm:
                z_coord = float(dcm.ImagePositionPatient[2])
                
            # Standardize the Modality tags
            raw_desc = str(getattr(dcm, 'SeriesDescription', 'Unknown'))
            desc_lower = raw_desc.lower()
            if 't1' in desc_lower and 'post' in desc_lower: tag = 'T1-Post Contrast'
            elif 't1' in desc_lower: tag = 'T1-Pre Contrast'
            elif 'flair' in desc_lower: tag = 'FLAIR'
            elif 'perf' in desc_lower: tag = 'Perfusion'
            elif 'dti' in desc_lower: tag = 'Diffusion'
            elif 't2' in desc_lower: tag = 'T2-Weighted'
            else: tag = raw_desc

            # Append to our structured record
            database.append({
                "PatientID": getattr(dcm, 'PatientID', 'Unknown'),
                "StudyDate": getattr(dcm, 'StudyDate', 'Unknown'),
                "Modality": tag,
                "RawDescription": raw_desc,
                "Z_Coordinate": z_coord, # <-- The holy grail for longitudinal tracking
                "SliceThickness": getattr(dcm, 'SliceThickness', None),
                "PixelSpacing": str(getattr(dcm, 'PixelSpacing', None)), 
                "SeriesUID": getattr(dcm, 'SeriesInstanceUID', 'Unknown'),
                "FilePath": str(filepath)
            })
            
        except Exception as e:
            continue

    # Convert to Pandas DataFrame and save
    df = pd.DataFrame(database)
    
    # Drop files that don't have spatial coordinates (e.g., structured reports)
    df = df.dropna(subset=['Z_Coordinate'])
    
    # Sort chronologically, then by Modality, then by Z-Coordinate (Bottom of head to Top)
    df = df.sort_values(by=["PatientID", "StudyDate", "Modality", "Z_Coordinate"])
    
    df.to_csv(output_db, index=False)
    print(f"\n✅ ETL Complete! Database saved to '{output_db}'.")
    print(f"🏥 Indexed {len(df)} clinically viable slices.")

if __name__ == "__main__":
    build_patient_index()