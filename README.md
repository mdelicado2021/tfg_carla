# Autonomous Lane Perception and Mapping in CARLA

Benchmarking framework focused on autonomous lane perception and mapping in the CARLA Simulator. The project integrates various lane detection algorithms, performs 2D-to-3D spatial projection using depth sensors, and implements mathematical lane modeling for map generation.

## Features

* **Multi-Algorithm Pipeline:** Implementation and comparison of classical computer vision filters (Thresholding, Canny, Hough, Sliding Windows) against Deep Learning architectures (U-Net, UFLD).
* **3D Spatial Projection:** Conversion of 2D pixel detections into real-world 3D coordinates using CARLA's depth sensors and camera intrinsic matrices.
* **ROS 2 Integration:** Real-time extraction and publishing of lane data as `PointCloud2` messages for robotic mapping and visualization.
* **Latency Benchmarking:** Strict, isolated measurement of pure algorithmic computational times (CPU-bound) for fair hardware-agnostic evaluation.

## Prerequisites

* [CARLA Simulator](https://carla.org/)
* [ROS 2](https://docs.ros.org/en/humble/index.html) (Humble recommended)
* Python 3.12+
* PyTorch, OpenCV, Pygame, NumPy

## Installation

Clone the main repository and set up the virtual environment:
```bash
git clone https://github.com/your-username/tfg_carla.git
cd tfg_carla
python3 -m venv carla_env
source carla_env/bin/activate
pip install -r requirements.txt
```

## Deploying third-party models
To run the neural network comparison using UFLD, you must clone the original repository into the workspace before running the inference script:

```bash
mkdir -p src/external
cd src/external
git clone https://github.com/jkdxbns/autonomous-driving-carla.git
```

## Project Structure

```Plaintext
├── plots/                  # Generated latency comparison graphs (.pdf)
├── saved_models/           # Neural network weights (U-Net, UFLD)
├── src/
│   ├── deepL/              # Deep Learning inference scripts (inference.py, inference_ufld.py)
│   ├── external/           # Third-party repositories (ignored in version control)
│   └── plot_times.py       # Matplotlib script for benchmark visualization
├── times/                  # Raw execution time logs (.txt)
├── .gitignore
└── README.md
```

## Usage
1. Start the CARLA server in the background:
```bash
./init_carla.sh
```

2. (Optional) Source the ROS 2 environment if 3D point cloud publishing is required:
```bash
source /opt/ros/humble/setup.bash
```

3. Run the desired perception pipeline to collect 100 frames of latency data:
```bash
python3 src/deepL/inference_ufld.py
```

4. Generate the comparative latency plots:
```bash
python3 src/plot_times.py
```

## Academic Context
This repository contains the source code for a Bachelor's Thesis in Robotics Software Engineering, developed at Universidad Rey Juan Carlos.
