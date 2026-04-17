# Dataset Management

This guide covers the dataset management features in X-AnyLabeling: importing and exporting full datasets, managing annotation versions, partitioning images into train/test/val splits, and using Smart Select mode for faster segmentation.

## Table of Contents

- [1. Dataset Import](#1-dataset-import)
- [2. Dataset Export](#2-dataset-export)
- [3. Version Control](#3-version-control)
- [4. Split Management](#4-split-management)
- [5. Smart Select Mode](#5-smart-select-mode)

---

## 1. Dataset Import

**Location:** `File` > `Import Dataset`

Import an entire dataset folder or ZIP archive. X-AnyLabeling auto-detects the annotation format and directory structure, then converts everything to the native XLABEL format so you can begin editing immediately.

### Supported Formats

| Format | Tasks |
|--------|-------|
| YOLO | HBB detection, OBB detection, Segmentation, Pose estimation |
| COCO | Detection, Segmentation, Pose estimation |
| VOC | Detection, Segmentation |
| DOTA | Oriented object detection |
| XLABEL | Native X-AnyLabeling format |

### How to Import

1. Open `File` > `Import Dataset`.
2. Select a dataset folder or a `.zip` archive.
3. X-AnyLabeling auto-detects the format and split structure. Confirm or adjust the detected settings in the dialog.
4. Click **OK**. Annotations are converted to XLABEL format and placed alongside the images.

### Auto-Detection of Split Structure

If your dataset contains `train/`, `test/`, or `val/` subdirectories, they are detected automatically and mapped to the internal split assignments (see [Split Management](#4-split-management)).

### Expected Input Directory Structures

**YOLO dataset:**

```
my_dataset/
  train/
    images/
      img_001.jpg
      img_002.jpg
    labels/
      img_001.txt
      img_002.txt
  val/
    images/
      img_003.jpg
    labels/
      img_003.txt
  data.yaml
```

**COCO dataset:**

```
my_dataset/
  train/
    images/
      img_001.jpg
      img_002.jpg
    annotations/
      instances_train.json
  val/
    images/
      img_003.jpg
    annotations/
      instances_val.json
```

**VOC dataset:**

```
my_dataset/
  JPEGImages/
    img_001.jpg
    img_002.jpg
  Annotations/
    img_001.xml
    img_002.xml
  ImageSets/
    Main/
      train.txt
      val.txt
```

> **Tip:** You can also import a `.zip` file containing any of the above structures. X-AnyLabeling extracts it to a temporary directory, detects the format, and imports the annotations.

---

## 2. Dataset Export

**Location:** `File` > `Export Dataset`

Export your annotated images and labels into a structured dataset ready for training. Unlike the per-format annotation export under `File` > `Export Annotations` (which exports labels only for the current working directory), **Export Dataset** produces a complete, organized output with proper directory structure and split folders.

### Supported Output Formats

All major formats are supported: YOLO (HBB/OBB/Seg/Pose), COCO (Det/Seg/Pose), VOC (Det/Seg), DOTA, and XLABEL.

### Export Options

| Option | Description |
|--------|-------------|
| **Format** | Target annotation format (YOLO, COCO, VOC, DOTA, XLABEL) |
| **Include images** | Copy source images into the output directory |
| **Skip empty labels** | Exclude images that have no annotations |
| **Create ZIP** | Package the output as a `.zip` archive |
| **Split structure** | Organize output into `train/`, `val/`, `test/` folders based on split assignments |

### How to Export

1. Open `File` > `Export Dataset`.
2. Choose the target format and configure options.
3. Select the output directory.
4. Click **OK**.

### Output Directory Structures

**YOLO export:**

```
output/
  train/
    images/
      img_001.jpg
    labels/
      img_001.txt
  val/
    images/
      img_003.jpg
    labels/
      img_003.txt
  data.yaml
```

The generated `data.yaml` file contains the class names and paths, ready for use with Ultralytics or other YOLO training frameworks.

**VOC export:**

```
output/
  JPEGImages/
    img_001.jpg
    img_002.jpg
  Annotations/
    img_001.xml
    img_002.xml
  ImageSets/
    Main/
      train.txt
      val.txt
```

The generated `ImageSets/Main/` files list the image basenames belonging to each split.

**COCO export:**

```
output/
  train/
    images/
      img_001.jpg
    annotations/
      instances_train.json
  val/
    images/
      img_003.jpg
    annotations/
      instances_val.json
```

> **Note:** Split assignments are pulled from the Split Management tool. If no splits have been assigned, all images are exported as a single set.

---

## 3. Version Control

**Location:** `Tools` > `Version Control`

Create snapshots of your annotation state so you can review history, compare changes, and restore previous versions. This is useful when experimenting with different labeling strategies or when multiple annotators work on the same dataset.

### Storage

Version snapshots are stored in a `.xanylabeling_versions/` directory within your image folder. Each version contains a copy of all annotation files at the time of the snapshot.

### Creating a Version

1. Open `Tools` > `Version Control`.
2. Click **Create Version**.
3. Enter a name (e.g., `v1-initial-labels`) and an optional description.
4. Click **OK**. The current state of all annotations is saved.

### Browsing Version History

The Version Control dialog displays a table of all saved versions with the following details:

- **Name** -- the version label you assigned
- **Timestamp** -- when the version was created
- **Stats** -- number of images and shapes in the snapshot
- **Size** -- disk space used by the snapshot

### Comparing Two Versions

1. Select two versions in the history table.
2. Click **Compare**.
3. The comparison view shows per-image differences:
   - **Added shapes** -- shapes present in the newer version but not the older one
   - **Removed shapes** -- shapes present in the older version but not the newer one
   - **Modified shapes** -- shapes that changed between versions

### Restoring a Version

1. Select a version in the history table.
2. Click **Restore**.
3. Confirm the action. Your current annotations are replaced with the selected snapshot.

> **Warning:** Restoring a version overwrites your current annotations. Create a new version before restoring if you want to preserve your current state.

### Deleting a Version

Select a version and click **Delete** to remove it and free disk space.

---

## 4. Split Management

**Location:** `Tools` > `Split Management`

Partition your images into **train**, **test**, and **val** sets. Split assignments are saved in a `.xanylabeling_splits.json` file in your image directory and are used by the Dataset Export feature to organize output folders.

### Auto-Split

1. Open `Tools` > `Split Management`.
2. Set the desired ratios (e.g., `70 / 20 / 10` for train/val/test).
3. Choose a splitting strategy:
   - **Random** -- shuffles images and assigns splits by ratio
   - **Stratified** -- distributes images so that each split has a similar class distribution
4. Click **Apply**. Images are assigned to splits automatically.

### Manual Assignment

Right-click any image in the file list and select **Assign Split** to manually assign it to `train`, `val`, `test`, or remove its split assignment.

### Visual Indicators

After splits are assigned, colored circles appear next to each filename in the file list:

| Color | Split |
|-------|-------|
| Blue | Train |
| Orange | Val |
| Green | Test |

### Statistics

The Split Management dialog includes a statistics table showing the number of images and per-class annotation counts for each split, so you can verify that your data is distributed as intended.

### Integration with Export

When you use `File` > `Export Dataset`, the export respects split assignments and creates the corresponding `train/`, `val/`, and `test/` subdirectories in the output.

---

## 5. Smart Select Mode

**Location:** Auto Labeling panel > **Smart Select** button

Smart Select is a streamlined interactive segmentation mode for SAM and SAM2 models. It combines positive and negative point placement into a single workflow, eliminating the need to switch between separate +Point and -Point buttons.

### Prerequisites

Load a SAM or SAM2 model from the Auto Labeling panel before using Smart Select.

### How to Use

1. Click the **Smart Select** button in the Auto Labeling panel.
2. **Left-click** on the image to place a **positive point** (mark an area to segment).
3. **Right-click** on the image to place a **negative point** (mark an area to exclude).
4. The segmentation mask updates in real time as you add points.
5. Click **Undo Mark** to remove the last prompt point if needed.
6. Press `f` to finalize and accept the generated shape.

### Comparison with Standard SAM Workflow

| Action | Standard workflow | Smart Select |
|--------|-------------------|--------------|
| Add positive point | Click +Point button, then click image | Left-click image |
| Add negative point | Click -Point button, then click image | Right-click image |
| Remove last point | Not available | Click **Undo Mark** |
| Finalize | Press `f` | Press `f` |

Smart Select works with all SAM and SAM2 model variants available in X-AnyLabeling, including SAM-HQ, MobileSAM, EfficientViT-SAM, and EdgeSAM.
