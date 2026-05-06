import os
import pydicom
import json
from pathlib import Path

def extract_middle_slice_paths(patient_id: str, data_dir: str = "./data") -> str:
    """
    Tool: Middle_Slice_Locator
    Description: Locates the exact file paths for the middle slices of all available MRI sequences for a patient.
                 Useful for passing a representative image to a Vision Language Model.
    
    Args:
        patient_id (str): The ID of the patient (e.g., 'UPENN-GBM-00001').
        data_dir (str): The root directory containing the dataset manifests. Default is './data'.
        
    Returns:
        str: A JSON-formatted string mapping the sequence UUIDs to their exact .dcm file paths, 
             standardized sequence names, and Z-coordinates.
    """
    root_path = Path(data_dir)
    
    if not root_path.exists():
        return json.dumps({"error": f"Root data directory '{data_dir}' not found."})

    # Dynamically locate the patient directory just like our other tool
    patient_dirs = list(root_path.rglob(patient_id))
    if not patient_dirs or not patient_dirs[0].is_dir():
        return json.dumps({"error": f"Patient ID '{patient_id}' not found in '{data_dir}'."})
    
    target_dir = patient_dirs[0]
    slice_map = {}
    
    for root, dirs, files in os.walk(target_dir):
        # Sort files to ensure we are actually picking from the middle of the spatial sequence
        dcm_files = sorted([f for f in files if f.endswith('.dcm')])
        
        if dcm_files:
            # Dynamically grab the true middle slice of the volume
            middle_index = len(dcm_files) // 2
            target_file_path = os.path.join(root, dcm_files[middle_index])
            
            try:
                dcm = pydicom.dcmread(target_file_path, stop_before_pixels=True)
                desc = dcm.get('SeriesDescription', 'Unknown Sequence')
                seq_lower = desc.lower()
                
                # Standardize tags for agent comprehension (matching timeline_compressor)
                if 't1' in seq_lower and 'post' in seq_lower: tag = 'T1-Post Contrast'
                elif 't1' in seq_lower: tag = 'T1-Pre Contrast'
                elif 'flair' in seq_lower: tag = 'FLAIR'
                elif 't2' in seq_lower: tag = 'T2-Weighted'
                elif 'bold' in seq_lower or 'task' in seq_lower: tag = 'fMRI-Functional'
                elif 'lang' in seq_lower: tag = 'fMRI-Language'  
                elif 'motor' in seq_lower: tag = 'fMRI-Motor'
                elif 'dti' in seq_lower or 'tensor' in seq_lower: tag = 'DTI-Tractography'
                elif 'dwi' in seq_lower or 'diffusion' in seq_lower: tag = 'DWI'
                elif 'perf' in seq_lower or 'asl' in seq_lower: tag = 'Perfusion'
                else: tag = desc

                # Extract Z-coordinate to anchor the AI's spatial reasoning
                z_coord = "Unknown"
                if 'ImagePositionPatient' in dcm:
                    z_coord = round(float(dcm.ImagePositionPatient[2]), 2)
                
                # Use the UUID folder name as the key to align with your timeline tool
                uuid_folder = os.path.basename(root)
                slice_map[uuid_folder] = {
                    "Sequence": tag,
                    "RawDescription": desc,
                    "Z_Coordinate_mm": z_coord,
                    "FilePath": target_file_path
                }
                
            except Exception:
                pass
                
    if not slice_map:
        return json.dumps({"error": "No .dcm files found for this patient."})

    # Return as a structured JSON string for the agent
    return json.dumps(slice_map, indent=4)

if __name__ == "__main__":
    # How the agent will test/use the tool:
    test_patient = "UPENN-GBM-00001"
    print(f"Agent executing tool on: {test_patient}...\n")
    
    tool_result = extract_middle_slice_paths(test_patient)
    print(tool_result)