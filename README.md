# 3D Object Detection with MMDetection3D

This project demonstrates 3D object detection inference using MMDetection3D models on KITTI and nuScenes datasets. It provides automated inference, visualization, and evaluation pipelines with support for multiple state-of-the-art models.

## Features

- 🚗 **Multiple Models**: PointPillars, 3DSSD, CenterPoint
- 📊 **Dual Dataset Support**: KITTI and nuScenes
- 🎨 **Rich Visualizations**: 2D projections, 3D point clouds, Open3D rendering
- 💾 **Automated Exports**: JSON predictions, PLY files, comparison metrics
- 🎬 **Demo Video Generation**: Automated multi-model comparison videos
- ⚡ **Flexible Deployment**: CPU and CUDA support

> 📊 **Full evaluation results available in [HW_REPORT.md](HW_REPORT.md)**

## Quick Links

- [Setup & Installation](#setup--installation)
- [Data Preparation](#data-preparation)
- [Running Inference](#running-inference)
- [Model Comparison](#model-comparison)
- [Troubleshooting](#troubleshooting)

---

## Setup & Installation

### System Requirements

| Component | Requirement | Notes |
|-----------|------------|-------|
| **Python** | 3.10 | Install via `winget install Python.Python.3.10` |
| **GPU** | NVIDIA (optional) | GTX 1650+ recommended for CUDA models |
| **CUDA Toolkit** | 11.3+ | Required for 3DSSD and CenterPoint |
| **Storage** | ~5GB | For models and sample data |

### 1. Environment Setup

Create and activate a virtual environment:

```powershell
# Create virtual environment
py -3.10 -m venv .venv

# Activate environment
& .\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

***Choose based on your hardware:***

<details>
<summary><b>🖥️ CPU-Only Installation</b></summary>

```powershell
python -m pip install -U pip
pip install openmim open3d opencv-python-headless==4.8.1.78 opencv-python==4.8.1.78 \
    matplotlib tqdm moviepy pandas seaborn
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4
mim install mmengine
pip install mmcv==2.1.0 mmdet==3.2.0
mim install mmdet3d
```

**Limitations**: Only PointPillars models supported (slower inference).

</details>

<details>
<summary><b>🚀 CUDA Installation (Recommended)</b></summary>

```powershell
python -m pip install -U pip
pip install openmim open3d opencv-python-headless==4.8.1.78 opencv-python==4.8.1.78 \
    matplotlib tqdm moviepy pandas seaborn
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4
mim install mmengine
pip install mmcv==2.1.0 mmdet==3.2.0
mim install mmdet3d
```

**Verify CUDA**:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

</details>

> ⚠️ **Important**: NumPy 1.26.x and OpenCV 4.8.1 are pinned to prevent ABI conflicts with MMDetection3D sparse operations.

### 3. Download Pretrained Models

```powershell
# PointPillars models (CPU/CUDA compatible)
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest checkpoints/kitti_pointpillars
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class --dest checkpoints/kitti_pointpillars_3class
mim download mmdet3d --config pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d --dest checkpoints/nuscenes_pointpillars

# Advanced models (CUDA required)
mim download mmdet3d --config 3dssd_4x4_kitti-3d-car --dest checkpoints/3dssd
mim download mmdet3d --config centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest checkpoints/nuscenes_centerpoint
```

---

## Data Preparation

### Option 1: Quick Start with Demo Data

Use pre-packaged samples from MMDetection3D:

```powershell
# Clone MMDetection3D (if not already done)
git clone https://github.com/open-mmlab/mmdetection3d.git external/mmdetection3d

# Copy KITTI sample
Copy-Item external\mmdetection3d\demo\data\kitti\000010.bin data\kitti\training\velodyne\
Copy-Item external\mmdetection3d\demo\data\kitti\000010.png data\kitti\training\image_2\
Copy-Item external\mmdetection3d\demo\data\kitti\000010.txt data\kitti\training\label_2\
python scripts/export_kitti_calib.py `
  external/mmdetection3d/demo/data/kitti/000010.pkl `
  data/kitti/training/calib/000010.txt

# Copy nuScenes sample
Copy-Item external\mmdetection3d\demo\data\nuscenes\*CAM*jpg data\nuscenes_demo\images\
Copy-Item external\mmdetection3d\demo\data\nuscenes\n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin `
  data\nuscenes_demo\lidar\sample2.pcd.bin
```

### Option 2: Use Full Datasets

<details>
<summary><b>KITTI Dataset</b></summary>

Download from [KITTI 3D Object Detection Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

- Left color images (12 GB)
- Velodyne point clouds (29 GB)
- Camera calibration matrices (16 MB)
- Training labels (5 MB)

Organize into:
```
data/kitti/training/
  ├── velodyne/     # .bin files
  ├── image_2/      # .png files
  ├── calib/        # .txt files
  └── label_2/      # .txt files
```

</details>

<details>
<summary><b>nuScenes Dataset</b></summary>

Download from [nuScenes](https://www.nuscenes.org/nuscenes#download):

- **Mini version** (4GB): 10 scenes - perfect for testing
- **Full version** (350GB): Complete dataset

For this project, only LiDAR `.pcd.bin` files are required:
```
data/nuscenes_demo/lidar/
  └── *.pcd.bin
```

</details>

---

## Running Inference

### Model Capabilities

### Model Capabilities

| Model | Dataset | Classes | CPU | CUDA | Speed (GPU) | Notes |
|-------|---------|---------|-----|------|-------------|-------|
| **PointPillars** | KITTI | Car | ✅ | ✅ | ~0.1s | General purpose |
| **PointPillars 3-class** | KITTI | Car, Pedestrian, Cyclist | ✅ | ✅ | ~0.1s | Multi-class detection |
| **PointPillars** | nuScenes | 10 classes | ✅ | ✅ | ~0.15s | Urban scenes |
| **3DSSD** | KITTI | Car | ❌ | ✅ | ~0.2s | High recall, tune threshold |
| **CenterPoint** | nuScenes | 10 classes | ❌ | ✅ | ~0.15s | State-of-the-art |

### Common Parameters

```powershell
--dataset      # Dataset type: 'kitti', 'any', 'nuscene'
--input-path   # Path to data folder or file
--frame-number # Frame ID (for KITTI) or -1 for all
--out-dir      # Output directory
--device       # 'cpu' or 'cuda:0'
--headless     # Save visualizations instead of displaying
--score-thr    # Detection confidence threshold (0.0-1.0)
```

### Quick Examples

<details>
<summary><b>KITTI PointPillars (Single Class)</b></summary>

```powershell
# CPU version
python mmdet3d_inference2.py `
  --dataset kitti `
  --input-path data\kitti\training `
  --frame-number 000010 `
  --model checkpoints\kitti_pointpillars\pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py `
  --checkpoint checkpoints\kitti_pointpillars\hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth `
  --out-dir outputs\kitti_pointpillars `
  --device cpu `
  --headless `
  --score-thr 0.2

# CUDA version (faster)
python mmdet3d_inference2.py `
  --dataset kitti `
  --input-path data\kitti\training `
  --frame-number 000010 `
  --model checkpoints\kitti_pointpillars\pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py `
  --checkpoint checkpoints\kitti_pointpillars\hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth `
  --out-dir outputs\kitti_pointpillars_gpu `
  --device cuda:0 `
  --headless `
  --score-thr 0.2
```

</details>

<details>
<summary><b>KITTI PointPillars (Multi-Class)</b></summary>

Detects Cars, Pedestrians, and Cyclists:

```powershell
python mmdet3d_inference2.py `
  --dataset kitti `
  --input-path data\kitti\training `
  --frame-number 000010 `
  --model checkpoints\kitti_pointpillars_3class\pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py `
  --checkpoint checkpoints\kitti_pointpillars_3class\hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth `
  --out-dir outputs\kitti_pointpillars_3class `
  --device cuda:0 `
  --headless `
  --score-thr 0.2
```

</details>

<details>
<summary><b>KITTI 3DSSD (CUDA Only)</b></summary>

> ⚠️ Use higher threshold (0.6-0.7) to reduce false positives

```powershell
python mmdet3d_inference2.py `
  --dataset kitti `
  --input-path data\kitti\training `
  --frame-number 000010 `
  --model checkpoints\3dssd\3dssd_4x4_kitti-3d-car.py `
  --checkpoint checkpoints\3dssd\3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth `
  --out-dir outputs\3dssd `
  --device cuda:0 `
  --headless `
  --score-thr 0.6
```

</details>

<details>
<summary><b>nuScenes PointPillars</b></summary>

```powershell
python mmdet3d_inference2.py `
  --dataset any `
  --input-path data\nuscenes_demo\lidar\sample2.pcd.bin `
  --model checkpoints\nuscenes_pointpillars\pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py `
  --checkpoint checkpoints\nuscenes_pointpillars\hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth `
  --out-dir outputs\nuscenes_pointpillars `
  --device cuda:0 `
  --headless `
  --score-thr 0.2
```

</details>

<details>
<summary><b>nuScenes CenterPoint (CUDA Only)</b></summary>

```powershell
python mmdet3d_inference2.py `
  --dataset any `
  --input-path data\nuscenes_demo\lidar\sample2.pcd.bin `
  --model checkpoints\nuscenes_centerpoint\centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py `
  --checkpoint checkpoints\nuscenes_centerpoint\centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth `
  --out-dir outputs\nuscenes_centerpoint `
  --device cuda:0 `
  --headless `
  --score-thr 0.2
```

</details>

### Generated Outputs

Each inference run produces:

| File Type | Description | Format |
|-----------|-------------|--------|
| `*_predictions.json` | Raw detection results with scores, labels, boxes | JSON |
| `*_2d_vis.png` | 2D image with projected bounding boxes | PNG |
| `*_points.ply` | Colored point cloud | PLY |
| `*_pred_bboxes.ply` | Predicted 3D bounding boxes | PLY |
| `*_pred_labels.ply` | Text labels for predictions | PLY |
| `*_gt_bboxes.ply` | Ground truth boxes (if available) | PLY |
| `*_axes.ply` | Coordinate frame reference | PLY |
| `preds/*.json` | Formatted prediction files | JSON |

---

## Visualization & Analysis

### Open3D Interactive Viewer

**Interactive mode** (requires display):
```powershell
python scripts/open3d_view_saved_ply.py `
  --dir outputs\kitti_pointpillars `
  --basename 000010 `
  --width 1600 --height 1200
```

Controls: Mouse rotate, Right-click pan, Scroll zoom, `Q` to quit

**Headless mode** (save screenshot):
```powershell
python scripts/open3d_view_saved_ply.py `
  --dir outputs\kitti_pointpillars `
  --basename 000010 `
  --width 1600 --height 1200 `
  --save-path outputs\kitti_pointpillars\000010_open3d.png `
  --no-show
```

### Create Demo Video

Compile multiple visualizations into a comparison video:

```powershell
python create_demo.py
```

Or manually with MoviePy:
```powershell
python -c "from moviepy import ImageClip, concatenate_videoclips; import os; frames=['outputs/kitti_pointpillars/000010_2d_vis.png','outputs/kitti_pointpillars/000010_open3d.png','outputs/3dssd/000010_2d_vis.png','outputs/3dssd/000010_open3d.png','outputs/kitti_pointpillars_3class/000010_2d_vis.png','outputs/nuscenes_pointpillars/sample_open3d.png']; clips=[ImageClip(f).with_duration(3) for f in frames if os.path.exists(f)]; concatenate_videoclips(clips, method='compose').write_videofile('outputs/detections_demo.mp4', fps=24, codec='libx264', audio=False)"
```

---

## Model Comparison

### Run Comparison Analysis

```powershell
python compare_models_metrics.py
```

**Generates**:
- Detection count statistics
- Score distributions (mean, std, min, max)
- Class-wise breakdowns
- Performance rankings
- Comparison tables

**Output files**:
- `outputs/inference_stats.json` - Per-model statistics
- `outputs/combined_stats.json` - Aggregated metrics
- `metrics_output.txt` - Human-readable report

### Manual Statistics Generation

```powershell
python -c "import json, numpy as np; mappings={'kitti':{0:'Car'},'nuscenes':{0:'car',1:'truck',2:'construction_vehicle',3:'bus',4:'trailer',5:'barrier',6:'motorcycle',7:'bicycle',8:'pedestrian',9:'traffic_cone'}}; files={'kitti':'outputs/kitti_pointpillars/000010_predictions.json','nuscenes':'outputs/nuscenes_pointpillars/sample.pcd_predictions.json'}; aggregated={};
for name,path in files.items():
    data=json.load(open(path))
    scores=np.array(data.get('scores_3d', []), dtype=float)
    labels=data.get('labels_3d', [])
    class_map=mappings[name]
    counts={}
    for lab in labels:
        cls=class_map.get(lab, str(lab))
        counts[cls]=counts.get(cls,0)+1
    aggregated[name]={
        'detections': len(labels),
        'mean_score': float(scores.mean()) if scores.size else None,
        'score_std': float(scores.std()) if scores.size else None,
        'max_score': float(scores.max()) if scores.size else None,
        'min_score': float(scores.min()) if scores.size else None,
        'class_counts': counts
    }
json.dump(aggregated, open('outputs/inference_stats.json','w'), indent=2)"
```

---

## Troubleshooting

### Installation & Dependencies

**NumPy ABI errors**
- Ensure NumPy 1.26.x installed: `pip show numpy`
- NumPy 2.x breaks mmcv sparse ops
- Reinstall: `pip install numpy==1.26.4 --force-reinstall`

**Open3D import failures**
- Verify inside venv: `pip show open3d`
- Reinstall: `pip install open3d --force-reinstall`

**Missing checkpoints**
- Run `mim download` commands from Setup section
- Check `checkpoints/` folder exists

### CUDA Issues

**CUDA not available**
```powershell
# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

**CUDA out of memory**
- Use CPU for PointPillars: `--device cpu`
- Close other GPU applications
- Reduce point cloud density (if using custom data)

**Sparse convolution errors**
- CenterPoint and 3DSSD require CUDA
- Use PointPillars on CPU as alternative

### Model-Specific Issues

| Problem | Solution |
|---------|----------|
| **3DSSD false positives** | Increase `--score-thr 0.6` or `0.7` |
| **PointPillars low nuScenes scores** | Expected behavior, use `--score-thr 0.15` |
| **CenterPoint/3DSSD on CPU** | Not supported, use `--device cuda:0` |
| **Long CPU inference** | Normal (~10-12s/frame), use CUDA for speed |

### Data Issues

**KITTI label format mismatch**
- Label files should contain object annotations, not calibration data
- Format: `Car 0.88 3 -0.69 0.00 192.37 402.31 374.00 1.60 1.57 3.23 -2.70 1.74 3.68 -1.29`
- Download from [KITTI website](http://www.cvlibs.net/datasets/kitti/)

**nuScenes data requirements**
- Only LiDAR `.pcd.bin` files required
- Camera images optional (not used in inference)
- No calibration files needed

---

## Project Structure

```
3D-object-detection/
├── mmdet3d_inference2.py          # Main inference script
├── compare_models_metrics.py       # Model comparison tool
├── organize_results.py             # Results organization
├── README.md                       # This file
├── REPORT.md                       # Detailed evaluation
├── scripts/
│   ├── export_kitti_calib.py      # KITTI calibration converter
│   └── open3d_view_saved_ply.py   # 3D visualization tool
├── checkpoints/                    # Model weights
│   ├── kitti_pointpillars/
│   ├── kitti_pointpillars_3class/
│   ├── 3dssd/
│   ├── nuscenes_pointpillars/
│   └── nuscenes_centerpoint/
├── data/
│   ├── kitti/training/            # KITTI dataset
│   │   ├── velodyne/              # .bin point clouds
│   │   ├── image_2/               # .png images
│   │   ├── calib/                 # .txt calibration
│   │   └── label_2/               # .txt labels
│   └── nuscenes_demo/             # nuScenes samples
│       ├── images/                # Camera images
│       └── lidar/                 # .pcd.bin files
└── outputs/                        # All inference results
    ├── combined_stats.json
    ├── inference_stats.json
    ├── inference_times.json
    ├── kitti_pointpillars/
    │   ├── 000010_predictions.json
    │   ├── 000010_2d_vis.png
    │   ├── 000010_points.ply
    │   ├── 000010_pred_bboxes.ply
    │   ├── 000010_pred_labels.ply
    │   ├── 000010_gt_bboxes.ply
    │   ├── 000010_axes.ply
    │   └── preds/
    │       └── 000010.json
    ├── kitti_pointpillars_3class/
    ├── 3dssd/
    ├── nuscenes_pointpillars/
    ├── nuscenes_centerpoint/
    └── detections_demo.mp4
```

---

## Additional Resources

### Documentation
- **[REPORT.md](REPORT.md)** - Full evaluation with metrics, visualizations, and analysis
- **[MMDetection3D Docs](https://mmdetection3d.readthedocs.io/)** - Official documentation
- **[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)** - Dataset homepage
- **[nuScenes](https://www.nuscenes.org/)** - Dataset homepage

### Key Scripts
- `mmdet3d_inference2.py` - Enhanced inference with visualization
- `compare_models_metrics.py` - Multi-model performance comparison
- `scripts/open3d_view_saved_ply.py` - Interactive 3D viewer
- `scripts/export_kitti_calib.py` - Calibration file converter



---

## Open3d Visualization

The helper script supports both interactive and headless viewing.

### Capture Screenshot (headless)
```powershell
python scripts/open3d_view_saved_ply.py --dir outputs\kitti_pointpillars --basename 000010 `
  --width 1600 --height 1200 --save-path outputs\kitti_pointpillars\000010_open3d.png --no-show
```

### Interactive Exploration
```powershell
python scripts/open3d_view_saved_ply.py --dir outputs\kitti_pointpillars --basename 000010 --width 1600 --height 1200
```
- Mouse rotate, right-click pan, scroll zoom, `Q` to close.
- Repeat with `--dir outputs\nuscenes_pointpillars --basename sample2.pcd` for nuScenes.

## Demo Video Assembly

A short stitched video (`outputs/detections_demo.mp4`) is produced with MoviePy:

```powershell
python -c "from moviepy import ImageClip, concatenate_videoclips; import os; frames=['outputs/kitti_pointpillars/000010_2d_vis.png','outputs/kitti_pointpillars/000010_open3d.png','outputs/3dssd/000010_2d_vis.png','outputs/3dssd/000010_open3d.png','outputs/kitti_pointpillars_3class/000010_2d_vis.png','outputs/nuscenes_pointpillars/sample_open3d.png']; clips=[ImageClip(f).with_duration(3) for f in frames if os.path.exists(f)]; concatenate_videoclips(clips, method='compose').write_videofile('outputs/detections_demo.mp4', fps=24, codec='libx264', audio=False)"
```

Alternatively, use the helper script:
```powershell
python create_demo.py
```

Inline preview (GIF):

![3D Detection Demo](outputs/detections_demo.gif)

## Runtime & Score Stats

- `outputs/inference_times.json` – measured wall-clock runtime per frame using PowerShell’s `Measure-Command`.
- `outputs/inference_stats.json` – mean/max/min detection scores and raw class counts.
- `outputs/combined_stats.json` – merged view adding runtime and top-three class tallies.

To regenerate stats:

```powershell
python -c "import json, numpy as np; mappings={'kitti':{0:'Car'},'nuscenes':{0:'car',1:'truck',2:'construction_vehicle',3:'bus',4:'trailer',5:'barrier',6:'motorcycle',7:'bicycle',8:'pedestrian',9:'traffic_cone'}}; files={'kitti':'outputs/kitti_pointpillars/000010_predictions.json','nuscenes':'outputs/nuscenes_pointpillars/sample.pcd_predictions.json'}; aggregated={};
for name,path in files.items():
    data=json.load(open(path))
    scores=np.array(data.get('scores_3d', []), dtype=float)
    labels=data.get('labels_3d', [])
    class_map=mappings[name]
    counts={}
    for lab in labels:
        cls=class_map.get(lab, str(lab))
        counts[cls]=counts.get(cls,0)+1
    aggregated[name]={
        'detections': len(labels),
        'mean_score': float(scores.mean()) if scores.size else None,
        'score_std': float(scores.std()) if scores.size else None,
        'max_score': float(scores.max()) if scores.size else None,
        'min_score': float(scores.min()) if scores.size else None,
        'class_counts': counts
    }
json.dump(aggregated, open('outputs/inference_stats.json','w'), indent=2)"
```

## Model Comparison

Compare all models using the comparison script:

```powershell
python compare_models_metrics.py
```

This generates:
- Detailed metrics for each model (detection counts, score statistics)
- Comparison table
- Summary statistics
- Best performer analysis

See `REPORT.md` for comprehensive analysis and results.

## Troubleshooting

### CUDA Issues
- **CUDA not available:** Ensure PyTorch CUDA version matches your CUDA toolkit. Install with `--index-url https://download.pytorch.org/whl/cu118`
- **CUDA out of memory:** Reduce batch size or use CPU for PointPillars models
- **Sparse conv errors:** CenterPoint requires CUDA. Use PointPillars on CPU if GPU unavailable

### Model-Specific Issues
- **3DSSD false positives:** Use higher score threshold (`--score-thr 0.6` or `0.7`)
- **PointPillars low scores on nuScenes:** This is expected; consider filtering with higher threshold
- **CenterPoint/3DSSD CPU errors:** These models require CUDA. Use PointPillars for CPU inference

### General Issues
- **NUMPY ABI errors:** Ensure NumPy 1.26.x remains installed; newer 2.x builds break mmcv's compiled ops
- **Open3D import failures:** Confirm `pip show open3d` inside the active venv
- **Long runtimes:** CPU inference is slow (~10-12s per frame); use CUDA for faster inference
- **Missing checkpoints:** Run `mim download` commands to fetch model weights

## Key Outputs (for reference)

### 2D Visualizations
- `outputs/kitti_pointpillars_gpu/000010_2d_vis.png` - PointPillars (KITTI)
- `outputs/kitti_pointpillars_3class/000010_2d_vis.png` - PointPillars 3-class (KITTI)
- `outputs/3dssd/000010_2d_vis.png` - 3DSSD (KITTI)
- `outputs/nuscenes_centerpoint/` - CenterPoint (nuScenes)

### 3D Visualizations
- `outputs/*/000010_points.ply` - Point clouds
- `outputs/*/000010_pred_bboxes.ply` - 3D bounding boxes
- `outputs/*/000010_pred_labels.ply` - Labels

### Data Files
- `outputs/*/000010_predictions.json` - Raw predictions
- `outputs/detections_demo.mp4` - Demo video (if generated)
- `metrics_output.txt` - Model comparison metrics

## Documentation

- **REPORT.md** - Comprehensive evaluation report with:
  - Setup instructions
  - Model specifications
  - Detailed metrics and results
  - Performance analysis
  - Visualizations and screenshots
  - Conclusions and recommendations

