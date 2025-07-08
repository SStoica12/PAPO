<div align="center">

# **PAPO: Perception-Aware Policy Optimization for Multimodal Reasoning**

</div>

<div align="center">

[![Project Page](https://img.shields.io/badge/🌐%20Project%20Page-Visit-blue)](https://mikewangwzhl.github.io/PAPO)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![GitHub](https://img.shields.io/badge/💻%20GitHub-Code-green)](https://github.com/mikewangwzhl/PAPO)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/PAPO)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Data-yellow)](https://huggingface.co/collections/PAPO)

</div>

**PAPO** is a simple yet effective extension of GRPO that encourages multimodal models to learn to perceive while learning to reason. By introducing an Implicit Perception Loss, PAPO significantly improves multimodal reasoning capabilities without requiring additional data curation, external reward models, or proprietary models.

## 🌟 **Key Highlights**

- **4.4% overall improvement** on diverse multimodal benchmarks
- **8.0% improvement** on vision-dependent tasks  
- **30.5% reduction** in perception errors
- **No additional data or external models** required
- Compatible with existing RLVR frameworks

## 📖 **Methodology**

### **Perception Bottleneck**

We identified that **67% of errors** in current multimodal reasoning models stem from poor perception rather than logical reasoning failures.

<div align="center">
<img src="./static/images/teaser.png" alt="PAPO Overview" width="800"/>
</div>

### **PAPO Algorithm**

**PAPO** extends GRPO by adding an **Implicit Perception Loss** that maximizes the KL divergence between model outputs on original vs. corrupted (masked) images:

<div align="center">
<img src="./static/images/method.png" alt="PAPO Method" width="940"/>
</div>

The core intuition is that a well-behaved multimodal model should produce significantly different outputs when visual information is corrupted, indicating reliance on meaningful visual content.

<div align="center">
<img src="./static/images/method_objective.png" alt="PAPO Objective" width="940"/>
</div>

### **Main Results**

PAPO consistently outperforms GRPO across all benchmarks, with particularly pronounced improvements on vision-dependent tasks:

<div align="center">
<img src="./static/images/main_results.png" alt="Main Results" width="1200"/>
</div>

#### **PAPO + Remove Reference KL**

PAPO is highly compatible with removing the reference KL penalty, achieving further improvements:

<div align="center">
<img src="./static/images/remove_kl_table.png" alt="PAPO Remove KL Results" width="450"/>
</div>

## 📊 **Data**

### **Training Data**
We adapt [TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) and [FanqingM/MMK12](https://huggingface.co/datasets/FanqingM/MMK12) to train **PAPO**:
- `PAPO/papo_virl39k_train`: [Hugging Face Dataset](https://huggingface.co/datasets/PAPO/papo_virl39k_train)
- `PAPO/papo_mm_eureka_test`: [Hugging Face Dataset](https://huggingface.co/datasets/PAPO/papo_mm_eureka_test)

### **Evaluation Data**
We adapted 8 different multimodal reasoning datasets to evaluate **PAPO**, which are further splitted into `General Reasoning` and `Vision-Dependent Reasoning` evaluation datasets:
- **General Reasoning**
    - `hiyouga/geometry3k`: [Hugging Face Dataset](https://huggingface.co/datasets/hiyouga/geometry3k), [Data Source](https://github.com/lupantech/InterGPS)
    - `AI4Math/MathVista`: [Hugging Face Dataset](https://huggingface.co/datasets/AI4Math/MathVista)
    - `We-Math/We-Math`: [Hugging Face Dataset](https://huggingface.co/datasets/We-Math/We-Math)
    - `FanqingM/MMK12`: [Hugging Face Dataset](https://huggingface.co/datasets/FanqingM/MMK12)
    - `AI4Math/MathVerse`: [Hugging Face Dataset](https://huggingface.co/datasets/AI4Math/MathVerse)
- **Vision-Dependent Reasoning**
    - `lscpku/LogicVista`: [Hugging Face Dataset](https://huggingface.co/datasets/lscpku/LogicVista)
    - `BUAADreamer/clevr_count_70k`: [Hugging Face Dataset](https://huggingface.co/datasets/BUAADreamer/clevr_count_70k)
    - `MMMU/MMMU_Pro`: [Hugging Face Dataset](https://huggingface.co/datasets/MMMU/MMMU_Pro)
    - `MathVerse_V` (vision-dependent subset): Adapted from [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse)

## 🚀 **Quick Start**

### **Environment Setup**

We provide multiple ways to set up the environment:

#### **Option 1: Using conda with environment.yaml**
```bash
conda env create -f environment.yaml
conda activate papo
```

#### **Option 2: Using pip**
```bash
pip install -e .
```

### **Training**

We support training with different configurations for both `Qwen2.5-VL 3B` and `7B` models:
- **Qwen2.5-VL 3B:** We typically use 2 `80G H100` GPUs
- **Qwen2.5-VL 7B:** We typically use 4 `80G H100` GPUs

#### **GRPO Baseline**
```bash
# 3B model
cd PAPO
bash examples/qwen2_5_vl_3b_grpo.sh

# 7B model  
cd PAPO
bash examples/qwen2_5_vl_7b_grpo.sh
```

#### PAPO (γ = 0.01)
```bash
# 3B model
cd PAPO
bash examples/qwen2_5_vl_3b_papo.sh

# 7B model  
cd PAPO
bash examples/qwen2_5_vl_7b_papo.sh
```

#### PAPO_H (γ = 0.02)
```bash
# 3B model
cd PAPO
bash examples/qwen2_5_vl_3b_papo_high.sh

# 7B model  
cd PAPO
bash examples/qwen2_5_vl_7b_papo_high.sh
```

#### PAPO + No Reference KL
```bash
# 3B model
cd PAPO
bash examples/qwen2_5_vl_3b_papo_no_kl_ref.sh

# 7B model  
cd PAPO
bash examples/qwen2_5_vl_7b_papo_no_kl_ref.sh
```


## 🥰 Acknowledgements

We thank the [EasyR1](https://github.com/deepseek-ai/DeepSeek-R1) team for providing the foundational codebase that we adapted to implement PAPO. Our implementation builds upon their efficient RLVR framework and extends it with perception-aware optimization methodologies. We also acknowledge the open-source community for providing the datasets and evaluation benchmarks that made this research possible.

## 📝 Citation

If you find PAPO useful in your research, please kindly cite our paper:

```bibtex
@article{wang2025papo,
  title={PAPO: Perception-Aware Policy Optimization for Multimodal Reasoning},
  author={Wang, Zhenhailong and Guo, Xuehang and Stoica, Sofia and Xu, Haiyang and Wang, Hongru and Ha, Hyeonjeong and Chen, Xiusi and Chen, Yangyi and Yan, Ming and Huang, Fei and Ji, Heng},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Learning to perceive while learning to reason!**

[🌐 Project Page](https://mikewangwzhl.github.io/PAPO)  |  [📄 Paper](https://arxiv.org/abs/XXXX.XXXXX)  |  [💻 GitHub](https://github.com/mikewangwzhl/PAPO)  |  [🤗 Models](https://huggingface.co/collections/PAPO)  |  [🤗 Data](https://huggingface.co/collections/PAPO)

</div>