import numpy as np
import pydicom
import json
from pathlib import Path

def compute_activation_centroid(dcm_series_path: str) -> dict:
    """
    Finds peak intensity in a 3D volume, now with a safety filter
    for inconsistent DICOM slice shapes.
    """
    dcm_files = sorted(Path(dcm_series_path).rglob("*.dcm"))
    if not dcm_files:
        return {"error": "No DICOM files found"}
    
    temp_slices = []
    temp_files = []

    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, 'pixel_array'):
                # We store both the array and the filename to keep them in sync
                temp_slices.append(ds.pixel_array.astype(np.float32))
                temp_files.append(f)
        except:
            continue
    
    if not temp_slices:
        return {"error": "Could not extract pixel data from DICOMs"}

    # --- THE FIX: Shape Consistency Check ---
    # We find the most common shape in the series (usually the actual scans)
    shapes = [s.shape for s in temp_slices]
    primary_shape = max(set(shapes), key=shapes.count)
    
    final_slices = []
    final_files = []
    for s, f in zip(temp_slices, temp_files):
        if s.shape == primary_shape:
            final_slices.append(s)
            final_files.append(f)

    # Now stack safely
    volume = np.stack(final_slices, axis=0) 
    
    peak_idx = np.unravel_index(np.argmax(volume), volume.shape)
    slice_idx, row, col = peak_idx
    
    # Use the file that corresponds to the specific slice in our filtered list
    ds = pydicom.dcmread(str(final_files[slice_idx]), force=True)
    
    try:
        origin = np.array([float(x) for x in ds.ImagePositionPatient])
        orientation = np.array([float(x) for x in ds.ImageOrientationPatient])
        row_dir = orientation[:3]
        col_dir = orientation[3:]
        pixel_spacing = [float(x) for x in ds.PixelSpacing]
        
        world_coord = (origin 
                       + col * pixel_spacing[1] * row_dir 
                       + row * pixel_spacing[0] * col_dir)
                       
        return {
            "peak_voxel": list(int(x) for x in peak_idx),
            "world_coord_mm": world_coord.tolist(),
            "series_path": str(dcm_series_path)
        }
    except Exception as e:
        return {"error": f"Spatial metadata failure: {str(e)}"}

def compute_eloquent_displacement(
    patient_id: str, 
    structure_tag: str, 
    date_baseline: str, 
    date_followup: str, 
    data_dir: str = "./data"
) -> str:
    """
    Computes 3D displacement of functional hotspots between two dates.
    Now includes fuzzy matching for messy clinical DICOM descriptions.
    """
    root = Path(data_dir)
    # Aggressive search for patient folder
    patient_dirs = [p for p in root.glob(f"**/{patient_id}") if p.is_dir()]
    
    if not patient_dirs:
        return json.dumps({"error": f"Patient {patient_id} folder not found."})
    
    target_dir = patient_dirs[0]
    
    def find_series_fuzzy(date, tag):
        matches = []
        found_descriptions = [] # For debugging
        
        # Normalize target tag (e.g., 'DTI-Tractography' -> 'dtitractography')
        t_norm = tag.lower().replace('-', '').replace('_', '').replace(' ', '')
        
        for uuid_dir in target_dir.iterdir():
            if not uuid_dir.is_dir(): continue
            dcm_files = list(uuid_dir.rglob("*.dcm"))
            if not dcm_files: continue
            
            try:
                ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                d_str = str(getattr(ds, 'StudyDate', ''))
                desc = str(getattr(ds, 'SeriesDescription', ''))
                
                # Normalize header description (e.g., 'ep2d_DTI_30dir' -> 'ep2ddti30dir')
                d_norm = desc.lower().replace('-', '').replace('_', '').replace(' ', '')
                
                if d_str == str(date):
                    found_descriptions.append(desc)
                    # Check if tag is in description OR description is in tag
                    if t_norm in d_norm or d_norm in t_norm:
                        matches.append(uuid_dir)
            except:
                continue
        return matches, found_descriptions

    base_matches, base_all = find_series_fuzzy(date_baseline, structure_tag)
    foll_matches, foll_all = find_series_fuzzy(date_followup, structure_tag)
    
    if not base_matches or not foll_matches:
        return json.dumps({
            "error": f"Sequence mismatch for tag '{structure_tag}'",
            "reason": "Could not find matching series on both dates.",
            "baseline_date": date_baseline,
            "baseline_available": base_all,
            "followup_date": date_followup,
            "followup_available": foll_all,
            "solution": "Try a broader tag like 'DTI', 'perf', or 't1'."
        }, indent=4)

    b_res = compute_activation_centroid(str(base_matches[0]))
    f_res = compute_activation_centroid(str(foll_matches[0]))
    
    if "error" in b_res or "error" in f_res:
        return json.dumps({"error": "Coordinate computation failed", "details": [b_res, f_res]})

    b = np.array(b_res["world_coord_mm"])
    f = np.array(f_res["world_coord_mm"])
    
    dist = float(np.linalg.norm(f - b))
    
    # Clinical logic
    if dist < 4:
        risk = "STABLE — Displacement within expected registration error."
    elif dist < 10:
        risk = "MODERATE SHIFT — Likely due to mass effect or tumor growth."
    else:
        risk = "CRITICAL DISPLACEMENT — Significant reorganization detected. Re-evaluate margins."

    return json.dumps({
        "patient_id": patient_id,
        "modality_match": structure_tag,
        "displacement_mm": round(dist, 2),
        "vector_xyz": [round(x, 2) for x in (f - b).tolist()],
        "clinical_assessment": risk
    }, indent=4)