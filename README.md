# MEDICAL-IMAGE-ANALYSIS

# Advanced Medical Imaging Analysis Project

## Overview
This project implements a comprehensive pipeline for medical CT scan processing and analysis. It leverages deep learning techniques for automatic organ segmentation and provides tools for volumetric analysis.

## Features
- CT scan processing pipeline using PyTorch and MONAI (Medical Open Network for AI)
- Preprocessing workflows for DICOM medical images with multi-orientation visualization
- Automatic organ identification using a pre-trained deep learning segmentation model
- Volumetric analysis for organ quantification from segmentation masks
- Advanced visualization of anatomical cross-sections and segmentation overlays

## Technologies Used
- **PyTorch**: Deep learning framework
- **MONAI**: Medical imaging specific AI toolkit
- **pydicom**: DICOM file handling
- **rt_utils**: Radiotherapy utilities
- **TCIA**: Cancer Imaging Archive data access tools
- **matplotlib**: Advanced visualization methods
- **numpy**: Numerical processing

## Installation
```bash
pip install torch pydicom matplotlib numpy monai rt_utils tcia_utils
```

## Usage
1. Download CT data from TCIA:
```python
from tcia_utils import nbia
cart_name = "nbia-56561691129779503"
cart_data = nbia.getSharedCart(cart_name)
df = nbia.downloadSeries(cart_data, format="df", path=datadir)
```

2. Load and preprocess CT images:
```python
preprocessing_pipeline = Compose([
    LoadImaged(keys='image', image_only=True),
    EnsureChannelFirstd(keys='image'),
    Orientationd(keys='image', axcodes='LPS')
])
data = preprocessing_pipeline({'image': CT_folder})
```

3. Apply organ segmentation model:
```python
with torch.no_grad():
    data['pred'] = inferer(data['image'].unsqueeze(0), network=model)
data = postprocessing(data)
```

4. Calculate organ volumes:
```python
number_bladder_voxels = (segmentation==13).sum().item()
voxel_volume_cm3 = np.prod(CT.meta['spacing']/10)
bladder_volume = number_bladder_voxels * voxel_volume_cm3
```

## Results
The project successfully:
- Processes CT scans with proper Hounsfield unit calibration
- Visualizes data in different anatomical planes
- Identifies organs using deep learning segmentation
- Calculates organ volumes with high precision

## Future Work
- Implement additional segmentation models
- Add support for other imaging modalities (MRI, PET)
- Develop automated reporting features

## Data Source
All medical imaging data was obtained from The Cancer Imaging Archive (TCIA).
