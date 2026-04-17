# Quick Start Guide

## Quick Install (Fork Edition)

> **Placeholder notice:** This guide uses `https://github.com/Kababey/X-AnyLabeling` as a placeholder URL for the fork. **Replace it with your actual GitHub repository URL** before publishing or sharing this documentation.

This fork adds features on top of upstream X-AnyLabeling: multi-project management, dataset import/export (YOLO, COCO, VOC, DOTA) with auto-detection, annotation version control, train/test/val split management, image resolution consistency, a class manager, a dataset health dashboard, and SAM2 Smart Select.

You can install and run this fork **without cloning the source code**. Choose the tier that matches your environment:

### Tier 1 — pip install from GitHub (for Python developers)

Requires Python **3.11 or newer** and `pip`. Install directly from the fork's Git branch:

```bash
pip install "x-anylabeling-cvhub[cpu] @ git+https://github.com/Kababey/X-AnyLabeling.git@feature/dataset-management-tools"
xanylabeling
```

Swap `[cpu]` for `[gpu]` (CUDA 12.x) or `[gpu-cu11]` (CUDA 11.x) if you have an NVIDIA GPU:

```bash
# CUDA 12.x
pip install "x-anylabeling-cvhub[gpu] @ git+https://github.com/Kababey/X-AnyLabeling.git@feature/dataset-management-tools"

# CUDA 11.x
pip install "x-anylabeling-cvhub[gpu-cu11] @ git+https://github.com/Kababey/X-AnyLabeling.git@feature/dataset-management-tools"
```

> **Tip:** Use a fresh virtual environment (`python -m venv .venv` then activate it) to avoid dependency conflicts with other projects.

### Tier 2 — Pre-built wheel from GitHub Releases (recommended for most users)

Once the maintainer tags a release, a pre-built wheel is attached to the [Releases page](https://github.com/Kababey/X-AnyLabeling/releases). Download the `.whl` and install it with pip — no compiler, no Git, no source checkout required.

```bash
# After downloading x_anylabeling_cvhub-<version>-py3-none-any.whl from the Releases page:
pip install "x_anylabeling_cvhub-4.0.0-py3-none-any.whl[cpu]"
xanylabeling
```

Replace `[cpu]` with `[gpu]` or `[gpu-cu11]` as needed. This is the fastest path for users who already have Python installed but do not want to build from source.

### Tier 3 — Standalone executable (for non-developers)

No Python installation needed. Download the pre-packaged binary for your OS from the [Releases page](https://github.com/Kababey/X-AnyLabeling/releases) and double-click to run:

| Platform | File |
|----------|------|
| Windows | `X-AnyLabeling-<version>-win.exe` |
| macOS | `X-AnyLabeling-<version>.dmg` |
| Linux | `X-AnyLabeling-<version>.AppImage` |

> **Note:** Standalone builds may lag slightly behind the latest source. For the newest features, use Tier 1 or Tier 2.

---

## Running the App

Once installed via Tier 1 or Tier 2, the following commands are available on your `PATH`.

### Launch the GUI

```bash
xanylabeling
```

### CLI conversion between annotation formats

```bash
xanylabeling convert                # list all supported conversion tasks
xanylabeling convert <task>         # show help and examples for a specific task
xanylabeling convert xlabel2yolo    # example: convert XLABEL annotations to YOLO format
```

### Other useful commands

```bash
xanylabeling checks                 # show system and version information
xanylabeling version                # print the installed version
xanylabeling config                 # print the config file path
xanylabeling --help                 # list all CLI options
```

### Config and data locations

X-AnyLabeling stores settings and cached data in your home directory:

| Path | Purpose |
|------|---------|
| `~/.xanylabelingrc` | User configuration file (UI preferences, default paths, etc.) |
| `~/.xanylabeling/` | Projects registry — metadata for the multi-project home screen |
| `~/.xanylabeling_data/` | Downloaded model weights and inference caches |

On Windows, `~` resolves to `%USERPROFILE%` (typically `C:\Users\<you>`).

To reset your UI configuration, delete `~/.xanylabelingrc` or run:

```bash
xanylabeling --reset-config
```

---

## 1. Installation and Deployment (Upstream / Alternative Methods)

> The sections below are the **original upstream installation instructions** from the parent X-AnyLabeling project. They still work, but most fork users should prefer the Quick Install tiers above.

X-AnyLabeling provides multiple installation methods. You can install the official package directly via `pip` to get the latest stable version, install from source by cloning the official GitHub repository, or use the convenient GUI installer package.

> [!NOTE]
> **Advanced Features**: The following advanced features are only available through Git clone installation. Please refer to the corresponding documentation for configuration instructions.
>
> 0. **Remote Inference Service**: X-AnyLabeling-Server based remote inference service - [Installation Guide](https://github.com/CVHub520/X-AnyLabeling-Server)
> 1. **Video Object Tracking**: Segment-Anything based video object tracking - [Installation Guide](../../examples/interactive_video_object_segmentation)
> 2. **Bounding Box Generation**: UPN-based bounding box generation - [Installation Guide](../../examples/detection/hbb/README.md)
> 3. **Interactive Detection & Segmentation**: Interactive object detection and segmentation with visual and text prompts - [Installation Guide](../../examples/detection/hbb/README.md)
> 4. **Smart Detection & Segmentation**: Object detection and segmentation with visual prompts, text prompts, and prompt-free modes - [Installation Guide](../../examples/grounding/)
> 5. **One-Click Training Platform**: Ultralytics framework-based training platform - [Installation Guide](../../examples/training/ultralytics/README.md)
> 6. **Multimodal Vision Model**: Rex-Omni based object detection, keypoint detection (person/hand/animal), OCR, Pointing, visual prompt localization - [Installation Guide](../../examples/vision_language/rexomni/README.md)

### 1.1 Prerequisites

#### 1.1.1 Miniconda

If you are already using Miniconda, follow these steps:

**Step 0.** Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 1.** Create a conda environment with Python 3.11 ~ 3.13 (Python 3.12 is recommended) and activate it.

> [!NOTE]
> Other Python versions require compatibility verification on your own.

```bash
# CPU Environment [Windows/Linux/macOS]
conda create --name x-anylabeling-cpu python=3.12 -y
conda activate x-anylabeling-cpu

# CUDA 11.x Environment [Windows/Linux]
conda create --name x-anylabeling-cu11 python=3.12 -y
conda activate x-anylabeling-cu11

# CUDA 12.x Environment [Windows/Linux]
conda create --name x-anylabeling-cu12 python=3.12 -y
conda activate x-anylabeling-cu12
```

#### 1.1.2 Venv

You can also use Python's built-in `venv` module to create virtual environments:

```bash
# CPU [Windows/Linux/macOS]
python3.12 -m venv venv-cpu
source venv-cpu/bin/activate  # Linux/macOS
# venv-cpu\Scripts\activate    # Windows

# CUDA 12.x [Windows/Linux]
python3.12 -m venv venv-cu12
source venv-cu12/bin/activate  # Linux
# venv-cu12\Scripts\activate    # Windows

# CUDA 11.x [Windows/Linux]
python3.12 -m venv venv-cu11
source venv-cu11/bin/activate  # Linux
# venv-cu11\Scripts\activate    # Windows
```

#### 1.1.3 uv

**Step 0.** Install uv:

```bash
# Linux / macOS / WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Step 1.** Create a virtual environment (uv downloads the required Python automatically) and activate it:

```bash
# CPU Environment [Windows/Linux/macOS]
uv venv --python 3.12 .venv-cpu
source .venv-cpu/bin/activate      # Linux/macOS/WSL2
# .venv-cpu\Scripts\activate       # Windows

# CUDA 12.x Environment [Windows/Linux]
uv venv --python 3.12 .venv-cu12
source .venv-cu12/bin/activate     # Linux
# .venv-cu12\Scripts\activate      # Windows

# CUDA 11.x Environment [Windows/Linux]
uv venv --python 3.12 .venv-cu11
source .venv-cu11/bin/activate     # Linux
# .venv-cu11\Scripts\activate      # Windows
```

### 1.2 Installation

#### 1.2.1 Pip Installation (Stable Version)

You can easily install the latest stable version of X-AnyLabeling with the following commands (using `uv pip` is recommended):

```bash
pip install -U uv

# CPU [Windows/Linux/macOS]
uv pip install x-anylabeling-cvhub[cpu]

# CUDA 12.x is the default GPU option [Windows/Linux]
uv pip install x-anylabeling-cvhub[gpu]

# CUDA 11.x [Windows/Linux]
uv pip install x-anylabeling-cvhub[gpu-cu11]
```

If you want to try the latest beta pre-release, add the `--pre` flag to the install command, for example:

```bash
# CPU [Windows/Linux/macOS]
uv pip install --pre x-anylabeling-cvhub[cpu]

# CUDA 12.x [Windows/Linux]
uv pip install --pre x-anylabeling-cvhub[gpu]

# CUDA 11.x [Windows/Linux]
uv pip install --pre x-anylabeling-cvhub[gpu-cu11]
```

#### 1.2.2 Git Clone (Recommended)

**Step a.** Clone the repository.

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
cd X-AnyLabeling
```

After cloning the repository, you can choose to install the dependencies in either developer mode or regular mode according to your needs.

**Step b.** Install the dependencies.

```bash
pip install -U uv

# CPU [Windows/Linux/macOS]
uv pip install -e .[cpu]

# CUDA 12.x is the default GPU option [Windows/Linux]
uv pip install -e .[gpu]

# CUDA 11.x [Windows/Linux]
uv pip install -e .[gpu-cu11]
```

If you need to perform secondary development or package compilation, you can install the `dev` dependencies simultaneously, for example:

```bash
uv pip install -e .[cpu,dev]
```

> [!NOTE]
> If you switch to a new project directory, rerun the install command there, otherwise the environment will reference the source code from the previous directory.

After installation, you can verify it by running the following command:

```bash
xanylabeling checks   # Display system and version information
```

You can also run the following commands to get other information:

```bash
xanylabeling help     # Display help information
xanylabeling version  # Display version number
xanylabeling config   # Display configuration file path
```

After verification, you can run the application directly:

```bash
xanylabeling
```

> [!TIP]
> You can use `xanylabeling --help` to view all available command line options. Please refer to the **Command Line Parameters** table below for complete parameter descriptions.

| Option                     | Description                                                                                                   |
|----------------------------|---------------------------------------------------------------------------------------------------------------|
| `filename`                 | Specify the image or label filename. If a directory path is provided, all files in the folder will be loaded. |
| `--help`, `-h`             | Display help information and exit.                                                                            |
| `--reset-config`           | Reset Qt configuration, clearing all settings.                                                                |
| `--logger-level`           | Set the logging level: "debug", "info", "warning", "fatal", "error".                                          |
| `--output`, `-O`, `-o`     | Specify the output file or directory. Paths ending with `.json` are treated as files.                         |
| `--config`                 | Specify a configuration file or YAML-formatted configuration string. Defaults to a user-specific path.        |
| `--work-dir`               | Specify the working directory for configuration and data files. Defaults to home directory.                   |
| `--nodata`                 | Prevent storing image data in JSON files.                                                                     |
| `--autosave`               | Enable automatic saving of annotation data.                                                                   |
| `--nosortlabels`           | Disable label sorting.                                                                                        |
| `--flags`                  | Comma-separated list of flags or path to a file containing flags.                                             |
| `--labelflags`             | YAML-formatted string for label-specific flags or a file containing a JSON-formatted string.                  |
| `--labels`                 | Comma-separated list of labels or path to a file containing labels.                                           |
| `--validatelabel`          | Specify the type of label validation.                                                                         |
| `--keep-prev`              | Keep annotations from the previous frame.                                                                     |
| `--no-auto-update-check`   | Disable automatic update checks on startup.                                                                   |

> [!NOTE]
> Please refer to the X-AnyLabeling [pyproject.toml](../../pyproject.toml) file for a list of dependencies. Note that all the examples above install all required dependencies.

We also supports batch conversion between multiple annotation formats:

```bash
xanylabeling convert         # List all supported conversion tasks
xanylabeling convert <task>  # Show detailed help and examples for a specific task, i.e., xlabel2yolo
```

> [!IMPORTANT]
> For GPU acceleration, please follow the instructions below to ensure that your local CUDA and cuDNN versions are compatible with the ONNX Runtime version, and install the required dependencies to ensure GPU-accelerated inference works properly:
>
> - Ⅰ. [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
> - Ⅱ. [Get started with ONNX Runtime in Python](https://onnxruntime.ai/docs/get-started/with-python.html)
> - Ⅲ. [ONNX Runtime Compatibility](https://onnxruntime.ai/docs/reference/compatibility.html)

> [!WARNING]
> For `CUDA 11.x` environments, please ensure that the versions meet the following requirements:
> - `onnx >= 1.15.0, < 1.16.1`
> - `onnxruntime-gpu >= 1.15.0, < 1.19.0`

**Optional Step**: Refresh Translation and Resource Files

If you update UI text or maintain localization files, refresh the translation artifacts for all supported interface languages (`en_US`, `zh_CN`, `ja_JP`, `ko_KR`) with:

```bash
# Regenerate .ts catalogs from source strings
python scripts/generate_languages.py

# Compile .qm files and rebuild Qt resources
python scripts/compile_languages.py
```

**Optional Step**: Set Environment Variables

```bash
# Linux or macOS
export PYTHONPATH=/path/to/X-AnyLabeling

# Windows
set PYTHONPATH=C:\path\to\X-AnyLabeling
```

> [!CAUTION]
> **Avoid Dependency Conflicts**: To avoid conflicts with third-party packages, please uninstall the old version first:
> ```bash
> pip uninstall anylabeling -y
> ```

> [!NOTE]
> **Special Note for Fedora KDE Users**: If you encounter slow mouse movement or response lag, try using the `--qt-platform xcb` parameter to improve performance:
> ```bash
> xanylabeling --qt-platform xcb
> ```

#### 1.2.3 GUI Installer Package

> **Download Link**: [GitHub Releases](https://github.com/CVHub520/X-AnyLabeling/releases)

Compared to running from source code, the GUI installer package provides a more convenient user experience. Users don't need to understand the underlying implementation and can use it directly after extraction. However, the GUI installer package also has some limitations:

- **Difficult Troubleshooting**: If crashes or errors occur, it may be difficult to quickly identify the specific cause, increasing the difficulty of troubleshooting.
- **Feature Lag**: The GUI version may lag behind the source code version in functionality, potentially leading to missing features and compatibility issues.
- **GPU Acceleration Limitations**: Given the diversity of hardware and operating system environments, current GPU inference acceleration services require users to compile from source code as needed.

Therefore, it is recommended to choose between running from source code and using the GUI installer package based on your specific needs and usage scenarios to optimize the user experience.

## 2. Usage

For detailed instructions on how to use X-AnyLabeling, please refer to the corresponding [User Guide](./user_guide.md).

## 3. Packaging and Compilation

> [!NOTE]
> Please note that the following steps are optional. This section is intended for users who may need to customize and compile the software to adapt to specific deployment scenarios. If you use the software without such requirements, you can skip this section.

<details>
<summary>Expand/Collapse</summary>

To facilitate users running `X-AnyLabeling` on different platforms, this tool provides packaging and compilation instructions along with relevant notes. Before executing the following packaging commands, please modify the `__preferred_device__` parameter in the [app_info.py](../../anylabeling/app_info.py) file according to your environment and requirements to select the appropriate GPU or CPU version for building.

### 3.1 Notes

- **Modify Device Configuration**: Before compiling, ensure that the `__preferred_device__` parameter in the `anylabeling/app_info.py` file has been modified according to the required GPU/CPU version.

- **Verify GPU Environment**: If compiling the GPU version, please activate the corresponding GPU runtime environment first and execute `pip list | grep onnxruntime-gpu` to ensure it is properly installed.

- **Windows-GPU Compilation**: Manually modify the `datas` list parameter in the `packaging/pyinstaller/specs/x-anylabeling-win-gpu.spec` file to add the relevant `*.dll` files of the local `onnxruntime-gpu` dynamic library to the list.

- **Linux-GPU Compilation**: Manually modify the `datas` list parameter in the `packaging/pyinstaller/specs/x-anylabeling-linux-gpu.spec` file to add the relevant `*.so` files of the local `onnxruntime-gpu` dynamic library to the list. Additionally, ensure that you download a matching `onnxruntime-gpu` package according to your CUDA version. For detailed compatibility information, please refer to the [official documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

### 3.2 Build Commands

```bash
# Windows-CPU
bash scripts/build_executable.sh win-cpu

# Windows-GPU
bash scripts/build_executable.sh win-gpu

# Linux-CPU
bash scripts/build_executable.sh linux-cpu

# Linux-GPU
bash scripts/build_executable.sh linux-gpu

# macOS
bash scripts/build_executable.sh macos
```

> [!TIP]
> If you encounter permission issues when executing the above commands on Windows, after ensuring the above preparation steps are completed, you can directly execute the following commands:
> ```bash
> pyinstaller --noconfirm packaging/pyinstaller/specs/x-anylabeling-win-cpu.spec
> pyinstaller --noconfirm packaging/pyinstaller/specs/x-anylabeling-win-gpu.spec
> ```

</details>
