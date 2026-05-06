import os
import pydicom
import numpy as np
import cv2
from PIL import Image
import json
from pathlib import Path
from collections import defaultdict

# ==========================================
# VLM IMAGE OPTIMIZATION TOOLS
# ==========================================

def apply_windowing(img_array, dcm):
    """Applies the radiologist's Window Center and Width to fix contrast."""
    try:
        if 'WindowCenter' in dcm and 'WindowWidth' in dcm:
            center = dcm.WindowCenter
            width = dcm.WindowWidth
            
            if isinstance(center, pydicom.multival.MultiValue): center = center[0]
            if isinstance(width, pydicom.multival.MultiValue): width = width[0]
            
            min_val = center - (width / 2.0)
            max_val = center + (width / 2.0)
            img_array = np.clip(img_array, min_val, max_val)
            
            return ((img_array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    except Exception:
        pass
        
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max > img_min:
        return ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return img_array.astype(np.uint8)

def crop_to_brain(img_array):
    """Removes dead black space so the brain fills the VLM's context window."""
    mask = img_array > 10
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img_array 
        
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    pad = 5
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(img_array.shape[0], y_max + pad)
    x_max = min(img_array.shape[1], x_max + pad)
    
    return img_array[y_min:y_max, x_min:x_max]

def apply_clahe(img_array):
    """Applies adaptive histogram equalization to make textures pop for the AI."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_array)

def extract_image_from_dicom(dcm_path: str, output_filename: str = "temp_scan.png") -> str:
    """Extracts and optimizes a single DICOM slice for the Vision Language Model."""
    try:
        dcm = pydicom.dcmread(dcm_path)
        img_array = dcm.pixel_array
        
        normalized_img = apply_windowing(img_array, dcm)
        cropped_img = crop_to_brain(normalized_img)
        final_img = apply_clahe(cropped_img)
            
        img = Image.fromarray(final_img)
        os.makedirs("data", exist_ok=True)
        output_path = os.path.join("data", output_filename)
        img.save(output_path)
        
        print(f"✅ Successfully created VLM-optimized slice at {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to parse DICOM file: {e}")
        return None

# ==========================================
# AGENT TIMELINE COMPRESSION TOOL
# ==========================================

def extract_patient_timeline(patient_id: str, data_dir: str = "./data") -> str:
    """
    Tool: Timeline_Compressor
    Description: Fetches and compresses the chronological MRI scanning history for a given patient.
    
    Args:
        patient_id (str): The ID of the patient (e.g., 'UPENN-GBM-00045').
        data_dir (str): The root directory containing the dataset manifests. Default is './data'.
        
    Returns:
        str: A JSON-formatted string mapping dates to standardized modalities and folder UUIDs. 
    """
    root_path = Path(data_dir)
    
    if not root_path.exists():
        return json.dumps({"error": f"Root data directory '{data_dir}' not found."})

    # Dynamically locate the patient directory across any manifest folder
    patient_dirs = list(root_path.rglob(patient_id))
    if not patient_dirs or not patient_dirs[0].is_dir():
        return json.dumps({"error": f"Patient ID '{patient_id}' not found in '{data_dir}'."})
    
    target_dir = patient_dirs[0]
    sequence_map = {}
    
    # Extract raw metadata using rglob
    for uuid_dir in target_dir.iterdir():
        if uuid_dir.is_dir():
            dcm_files = list(uuid_dir.rglob('*.dcm'))
            if dcm_files:
                sample_dcm = dcm_files[0]
                try:
                    ds = pydicom.dcmread(sample_dcm, stop_before_pixels=True)
                    desc = getattr(ds, 'SeriesDescription', 'Unknown Sequence')
                    date = getattr(ds, 'StudyDate', 'Unknown Date')
                    sequence_map[uuid_dir.name] = {"Description": desc, "Date": date}
                except Exception:
                    pass

    # Compress into token-efficient timeline
    timeline = defaultdict(list)
    for uuid, data in sequence_map.items():
        date = data.get('Date', 'Unknown')
        seq_desc = data.get('Description', 'Unknown')
        seq_lower = seq_desc.lower()
        
        # Standardize tags for agent comprehension
        if 't1' in seq_lower and 'post' in seq_lower: tag = 'T1-Post Contrast'
        elif 't1' in seq_lower: tag = 'T1-Pre Contrast'
        elif 'flair' in seq_lower: tag = 'FLAIR'
        elif 't2' in seq_lower: tag = 'T2-Weighted'
        # --- NEW SURGICAL MODALITIES ---
        elif 'bold' in seq_lower or 'task' in seq_lower: tag = 'fMRI-Functional'
        elif 'lang' in seq_lower: tag = 'fMRI-Language'  
        elif 'motor' in seq_lower: tag = 'fMRI-Motor'
        elif 'dti' in seq_lower or 'tensor' in seq_lower: tag = 'DTI-Tractography'
        elif 'dwi' in seq_lower or 'diffusion' in seq_lower: tag = 'DWI'
        elif 'perf' in seq_lower or 'asl' in seq_lower: tag = 'Perfusion'
        else: tag = seq_desc
            
        timeline[date].append({'modality': tag, 'uuid_reference': uuid})
        
    return json.dumps(dict(sorted(timeline.items())))

if __name__ == "__main__":
    # Test 1: Image Extraction
    print("Testing Image Extraction...")
    test_dcm_file = "/Users/nithingodi/Desktop/gemma4bt/data/manifest-1776580364344/upenn_gbm/UPENN-GBM-00001/1.3.6.1.4.1.14519.5.2.1.303922662365557240161187739636975756435/1.3.6.1.4.1.14519.5.2.1.337491152254065288399657726162931889194/3ce0549e-870b-49e5-ab08-15ad0e92ff65.dcm" 
    
    if os.path.exists(test_dcm_file):
        extract_image_from_dicom(test_dcm_file)
    else:
        print(f"Could not find the test file at: {test_dcm_file}")
        
    # Test 2: Timeline Extraction Tool
    print("\nTesting Timeline Compressor Tool...")
    test_patient = "UPENN-GBM-00045"
    timeline_result = extract_patient_timeline(test_patient)
    if "error" not in timeline_result:
        print(f"✅ Successfully extracted timeline for {test_patient} (Length: {len(timeline_result)} chars)")
    else:
        print(timeline_result)