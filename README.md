# YouCount: A YOLO11-based Framework for Automated Steller Sea Lion Population Count

## Abstract

This repository presents YouCount, a robust automated framework for counting Steller sea lions from aerial imagery, addressing a critical ecological monitoring challenge characterized by dense crowds, occlusions, and varied object scales. Traditional detection and segmentation approaches demonstrate significant limitations in this domain due to instance overlaps and annotation inconsistencies. Our methodology employs the state-of-the-art YOLO11 architecture complemented by a novel Adaptive Bounding Boxes Coordinates Annotation Framework (ABC). The proposed approach adaptively adjusts bounding boxes and implements dual patch generation strategies to effectively capture dense and overlapping instances. Comprehensive experimentation on the NOAA Fisheries Steller Sea Lion Population Count dataset demonstrates the superiority of our approach, achieving an RMSE of 11.65312 and securing 2nd place in the Kaggle competition. Ablation studies further validate the efficacy of our adaptive annotation and training strategies.

## Table of Contents

- [Introduction](#introduction)
- [Technical Innovation](#technical-innovation)
- [Repository Structure](#repository-structure)
- [Environment Configuration](#environment-configuration)
- [Usage](#usage)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
  - [Adaptive Bounding Box Annotation (ABC)](#adaptive-bounding-box-annotation-abc)
  - [Dual Patch Generation Strategy](#dual-patch-generation-strategy)
  - [Training Configuration](#training-configuration)
  - [Inference and Post-Processing](#inference-and-post-processing)
- [Experimental Results](#experimental-results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Introduction

The Steller sea lion population in the western Aleutian Islands has experienced a dramatic 94% decline over the past three decades, designating the western Pacific population as endangered and a critical focus for conservation efforts. The NOAA Fisheries Alaska Fisheries Science Center conducts annual population surveys using aerial imagery collected via aircraft and unoccupied aircraft systems. These surveys are essential for understanding factors contributing to the lack of recovery in specific regions.

Current manual counting methodologies present significant limitations. Biologists require up to four months to count sea lions from thousands of images collected annually by NOAA Fisheries. Following individual counts, tallies must undergo reconciliation to ensure reliability, with results being time-sensitive for conservation purposes. The automation of this process through advanced computer vision techniques would substantially enhance monitoring efficiency, redirecting valuable resources toward conservation initiatives while benefiting endangered species protection efforts.

Aerial sea lion counting presents multiple technical challenges:
- Dense aggregation and overlapping of sea lion individuals leading to frequent missed detections
- Variations in image capture height and position introducing inconsistencies in spatial distribution and size differences
- Differentiation between sea lion categories and difficulty in reliably detecting pups due to their small size

## Technical Innovation

YouCount introduces two primary innovations to address these challenges:

1. **Adaptive Bounding Boxes Coordinates Annotation Framework (ABC)**: Unlike conventional approaches that employ fixed-radius circles or fixed-size squares centered at annotation points as bounding boxes, our ABC framework dynamically adjusts bounding box dimensions based on sea lion density and proximity. This adaptation is achieved through computation of Intersection over Union (IoU) between all bounding box pairs, following standard object detection practices.

2. **Dual Patch Generation Strategy**: Considering the large dimensions of original images, we implement two complementary patching strategies:
   - **Overfitting-based Patching**: Samples patches centered around bounding boxes, with the number of patches set to 30% of the total bounding boxes in the image, constrained between 1-3 patches.
   - **Tiling-based Patching**: Employs fixed patch sizes with a sliding window approach, using a stride of 95% of the patch size to ensure adjacent patch overlap.

## Repository Structure

```
.
├── src/
│   ├── pre-process-detection.py    # ABC framework implementation and patch generation
│   ├── train.py                    # YOLO11x training pipeline
│   ├── test.py                     # Inference and evaluation pipeline
│   ├── post-process.py             # Post-processing for improved recognition
│   └── dataset_detection.py        # Dataset utilities and configuration
├── visualizations/
│   ├── patches/                    # Sample patch visualizations with detections
│   ├── statistics/                 # Statistical analysis plots
│   ├── heatmaps/                   # Detection density heatmaps
│   └── full_images/                # Full image detection visualizations
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Environment Configuration

### Prerequisites
- Python 3.8+
- PyTorch 2.0.1
- CUDA-compatible GPU (8GB+ recommended)

### Installation

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv sealion-counting
   source sealion-counting/bin/activate  # On Windows: sealion-counting\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional packages:
   ```bash
   pip install ultralytics==8.0.196 opencv-python==4.8.0 pandas==2.0.3 numpy==1.24.3 matplotlib==3.7.1
   ```

## Usage

### Preprocessing for Training Data Generation

Use the preprocessing pipeline to generate JSON annotation files from the original point-based annotations:

```bash
python src/pre-process-detection.py
```

This script performs the following operations:
- Implements the Adaptive Bounding Boxes Coordinates Annotation Framework (ABC)
- Converts point annotations to adaptive bounding boxes based on IoU calculations
- Generates dual patch strategy samples (overfitting-based and tiling-based patches)
- Outputs YOLO-format JSON annotation files for training
- Creates training patches with corresponding labels in `data/patches/`

**Key outputs:**
- `data/train/labels/` - YOLO format labels with ABC annotation
- `data/patches/` - Generated training patches (640×640 from 1280×1280 originals)
- Training-ready JSON files with adaptive bounding box coordinates

### Training the Model

```bash
python src/train.py
```

### Inference and Post-Processing

For optimal recognition results, use the inference pipeline with integrated post-processing:

```bash
python src/test.py
```

The post-processing pipeline enhances recognition through:
- **Boundary weighting strategy**: Adjusts confidence scores based on detection proximity to patch edges
- **Category-specific scaling**: Applies empirically-derived scaling factors to correct class-specific biases
- **Non-Maximum Suppression**: Filters overlapping detections with optimized IoU thresholds
- **Multi-patch aggregation**: Combines predictions from overlapping patches for comprehensive coverage

**Post-processing improvements:**
- Reduces RMSE by 0.8 points through bias correction
- Handles edge artifacts from patch-based inference
- Corrects category-specific detection imbalances (e.g., 30% increase for pups, 55% increase for adult males)

### Additional Post-Processing

For enhanced post-processing operations, you can also use:

```bash
python src/post-process.py
```

## Dataset Description

The dataset utilized in this study is from the NOAA Fisheries Steller Sea Lion Population Count competition and comprises:
- 946 training images with corresponding point-based annotation images showing the location of each animal
- Test image collection
- The objective is to count five distinct classes of sea lions: Adult males, Subadult males, Adult females, Juveniles, and Pups

### Sea Lion Categories
| Class ID | Category | Description |
|----------|----------|-------------|
| 0 | Adult males | Large, dominant males |
| 1 | Subadult males | Younger males |
| 2 | Adult females | Mature females |
| 3 | Juveniles | Young sea lions |
| 4 | Pups | Newborn/very young |

## Methodology

### Adaptive Bounding Box Annotation (ABC)

To localize sea lions in high-resolution images, we initially generate bounding boxes by drawing squares centered at each annotated point with a fixed side length. Subsequently, we compute the Intersection over Union (IoU) between all bounding box pairs to determine appropriate scaling factors following standard object detection practices. A minimum allowable dimension is established to prevent bounding boxes from becoming excessively small. Based on horizontal and vertical distances between point pairs, we determine the optimal direction for resizing each bounding box.

The ABC framework dynamically adjusts bounding box sizes based on IoU calculations:

| IoU Threshold | Primary Scale Factor | Secondary Scale Factor |
|---------------|---------------------|----------------------|
| ≥ 0.50 | 0.70 | 0.85 |
| ≥ 0.45 | 0.72 | 0.86 |
| ≥ 0.40 | 0.74 | 0.87 |
| ≥ 0.35 | 0.76 | 0.88 |
| ≥ 0.30 | 0.78 | 0.89 |

The adjustment strategy:
- Calculate IoU between all bounding box pairs
- Apply scaling factors based on IoU thresholds
- Prioritize dimension with larger displacement between centers
- Maintain minimum box size constraints

### Dual Patch Generation Strategy

#### Strategy 1: Overfitting-based Patching
- **Sampling rate**: 30% of total bounding boxes
- **Patch constraints**: 1-3 patches per image
- **Center selection**: Random subset of bounding boxes
- **Purpose**: High-quality positive samples with guaranteed sea lion presence

#### Strategy 2: Tiling-based Patching
- **Method**: Systematic grid-based extraction
- **Stride**: 95% of patch size (1216 pixels overlap)
- **Coverage**: Comprehensive spatial coverage
- **Purpose**: Capture edge cases and diverse spatial contexts

For both methods, patches are extracted from the image, and corresponding bounding boxes are adjusted to remain within patch boundaries. If a bounding box is clipped such that its area becomes less than 20% of its original area, it is discarded. Additionally, patches with a black pixel ratio exceeding 0.5 are discarded to ensure visual usability.

### Training Configuration

#### Model Architecture
- **Base Model**: YOLO11x (enhanced from YOLOv8x)
- **Improvements**: 22% parameter reduction with higher mAP
- **Input Resolution**: 640×640 pixels
- **Classes**: 5 sea lion categories

#### Hyperparameters
```python
# Core training parameters
TOTAL_EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 2
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.7
```

#### Data Augmentation Strategy

Our comprehensive augmentation pipeline enhances model robustness:

##### Color Space Augmentation
- Hue shift: ±1.5%
- Saturation adjustment: ±70%
- Brightness adjustment: ±40%

##### Geometric Transformations
- Random rotation: ±3°
- Translation: ±2%
- Scaling range: ±30%
- Copy-Paste: 8% probability
- Random Erasing: 8% probability
- MixUp: 4% probability

### Inference and Post-Processing

During inference, each image patch (originally sized 1440×1440) is resized to 640×640 to match the input requirements of the YOLO model. The model extracts features and predicts bounding boxes with associated confidence scores and class probabilities. Non-Maximum Suppression (NMS) is applied to filter highly overlapping predictions, retaining only the most confident ones.

#### Boundary Weighting Strategy
```python
# Weight detections based on proximity to patch edges
if near_one_edge:
    weight = 0.7
elif near_multiple_edges:
    weight = 0.5
else:
    weight = 1.0
```

#### Category-wise Post-Processing
Empirically-derived scaling factors to correct category-specific biases:

```python
CATEGORY_SCALING = {
    "pups": 1.3,           # 30% increase
    "juveniles": 0.85,     # 15% decrease  
    "adult_females": 0.96,  # 4% decrease
    "adult_males": 1.55,   # 55% increase
    "subadult_males": 1.2  # 20% increase
}
```

## Experimental Results

### Competition Performance
| Metric | Score |
|--------|-------|
| **Final Private Score** | **11.65312 RMSE** |
| Public Leaderboard | 11.68292 RMSE |
| **Final Ranking** | **2nd Place** |

### Leaderboard Comparison
| Rank | Team Score | 
|------|------------|
| 1st | 10.85644 |
| **2nd (Ours)** | **11.65312** |
| 3rd | 12.50888 |
| 4th | 13.03257 |
| 5th | 13.18968 |
| 6th | 13.92760 |

### Ablation Study Results

#### Impact of Resolution and Post-Processing
| Test Patch Size | Post-Processing | RMSE | Improvement |
|-----------------|----------------|------|-------------|
| 640/1280 | No | 16.59 | Baseline |
| 640/1440 | No | 12.45 | -4.14 |
| 640/1440 | Yes | **11.65** | **-0.80** |

Key insights:
- **Resolution increase**: 4.1 RMSE improvement through better scale matching
- **Post-processing**: Additional 0.8 RMSE improvement via bias correction
- **Scale consistency**: Larger patches better match test data characteristics

The visualization of predicted bounding boxes across entire images demonstrates that the model accurately detects individual sea lions, even in crowded regions with partial overlaps. Notably, the model successfully distinguishes between adjacent individuals without merging them into single bounding boxes, exhibiting robustness in handling occlusion. Detection density heatmaps align with actual spatial distributions of sea lions in original images, with no significant false positive hotspots observed in background regions such as ocean areas.

## Conclusion

The YouCount system represents a significant advancement in automated aerial sea lion counting. Through our innovative Adaptive Bounding Boxes Coordinates Annotation Framework (ABC) and dual patch generation strategy, we effectively address the core challenges of processing dense, overlapping instances. Our design choices based on the YOLO11x architecture, combined with meticulous data augmentation and rigorous quality control, enable our method to achieve an RMSE of 11.65312, securing 2nd place in the Kaggle competition.

Extensive experimentation and ablation studies validate the effectiveness of our adaptive annotation and training strategies. The approach provides a solid foundation for wildlife monitoring and can be readily generalized to similar ecological counting tasks. Future research directions include native support for multi-scale training, integration of attention mechanisms, and exploration of semi-supervised learning methods, all of which promise to further enhance system performance under complex environmental conditions.

## Acknowledgments

- NOAA Fisheries Alaska Fisheries Science Center for providing the dataset
- Kaggle for hosting the competition platform
- Ultralytics for the YOLO11 framework
- Conservation community working to protect Steller sea lions

## Competition Link

[NOAA Fisheries Steller Sea Lion Population Count Kaggle Competition](https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count)
