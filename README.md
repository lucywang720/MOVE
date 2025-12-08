# MOVE: Motion-Based Variability Enhancement for Spatial Generalization in Robotic Manipulation

> 🚀 **联合发布：清华大学 (Tsinghua University) & 智源人工智能研究院 (Beijing Academy of Artificial Intelligence - BAAI)**

This repository contains the official implementation of **MOVE (MOtion-Based Variability Enhancement)**, a novel and simple data collection paradigm that significantly enhances spatial generalization and data efficiency for robotic manipulation tasks through dynamic demonstrations.

## 🌟 Overview (项目概览)

Imitation learning methods in robotics are often constrained by **data scarcity** and a lack of **spatial generalization**. Standard data collection relies on *static* demonstrations, where the environment's spatial configuration (object, target, camera pose) remains fixed throughout a trajectory. This leads to **spatial sparsity**—the policy only succeeds around training points.

**MOVE** addresses this by injecting **motion** into movable objects (pickup object, target object) and the camera during the expert demonstration process. This augmentation implicitly generates a dense and diverse set of spatial configurations within a single trajectory, leading to:
* **Richer Spatial Information:** One trajectory encodes information from multiple spatial locations.
* **Superior Spatial Generalization:** Policies trained with MOVE data generalize much better to unseen spatial configurations.
* **Enhanced Data Efficiency:** MOVE achieves comparable performance with significantly less data than static methods.

## 🔥 Key Results (核心成果)

### Simulation (Meta-World)
MOVE consistently and significantly outperforms the static data collection paradigm.

| Metric | Static Baseline | MOVE | Improvement |
| :--- | :--- | :--- | :--- |
| **Average Success Rate** | 22.2% | **39.1%** | **76.1% Relative Gain** |
| **Data Efficiency** | 100k timesteps | **20k-50k timesteps** | Up to **2-5x** gain on certain tasks (e.g., Pick-Place-Wall) |

### Real-World (Pick-and-Place)
In a real-world pick-and-place task with highly randomized configurations, MOVE dramatically reduces the required data scale.

| Dataset Size | Static Method (Success Rate) | MOVE (Success Rate) |
| :--- | :--- | :--- |
| **35k Timesteps** | 3.3% | **23.3%** |
| **75k Timesteps** | 23.3% | **36.7%** |

> MOVE with 35k timesteps matches the performance of the static method trained with over twice the data (75k).

## ⚙️ Implementation Details (实现细节)

### Dynamic Feature Augmentation
MOVE implements the following controlled kinematic motions to enrich the training data:

1.  **Object Translation ($V_{o}$):** Linear motion with boundary bounce for the pickup/target object.
2.  **Object Rotation ($\omega_{i}$):** 1-D rotation around the vertical z-axis.
3.  **Camera Movement ($V_{c}$):** Camera moves along a constrained cylindrical path relative to the scene.

### Staged Augmentation
In multi-phase tasks like Pick-Place, the motion augmentation is applied strategically:
* **Pick Phase:** Motion is applied only to the **pickup object**.
* **Placement Phase:** Motion is applied only to the **target object**.

## 🚀 Getting Started (快速开始)

### Installation

```bash
# 克隆仓库
git clone [https://github.com/lucywang720/MOVE.git](https://github.com/lucywang720/MOVE.git)
cd MOVE
```

### data generation

```
bash generate/scripts/gen_demonstration_metaworld.sh 
```

### diffusion training

```
bash MOVE/diffusion_policy/train.sh
```

## Citation

if you find this work helpful, please consider citing our paper:

```
@misc{wang2025movesimplemotionbaseddata,
      title={MOVE: A Simple Motion-Based Data Collection Paradigm for Spatial Generalization in Robotic Manipulation}, 
      author={Huanqian Wang and Chi Bene Chen and Yang Yue and Danhua Tao and Tong Guo and Shaoxuan Xie and Denghang Huang and Shiji Song and Guocai Yao and Gao Huang},
      year={2025},
      eprint={2512.04813},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.04813}, 
}
```
