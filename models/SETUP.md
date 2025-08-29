# 🛠️ Setup Guide - Deep Learning Projects Collection

This guide will help you set up your environment to run all the deep learning projects in this collection.

## 📋 Prerequisites

- **Python 3.10+** (recommended: Python 3.11)
- **Git** for version control
- **8GB+ RAM** (16GB recommended for larger models)
- **GPU** (optional but recommended for faster training)

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Beginners)

1. Open any project notebook in Google Colab
2. All dependencies are pre-installed
3. Free GPU access available
4. No local setup required!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Option 2: Local Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/deep-learning-projects-collection.git
cd deep-learning-projects-collection
```

#### Step 2: Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv dl_projects_env

# Activate environment
# On Windows:
dl_projects_env\Scripts\activate
# On macOS/Linux:
source dl_projects_env/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Or install minimal requirements for specific projects
pip install tensorflow numpy matplotlib jupyter scikit-learn
```

#### Step 4: Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

### Option 3: Conda Environment

```bash
# Create conda environment
conda create -n dl_projects python=3.11
conda activate dl_projects

# Install packages
conda install tensorflow pytorch scikit-learn matplotlib jupyter -c conda-forge
pip install -r requirements.txt
```

## 🖥️ GPU Setup (Optional)

### NVIDIA GPU (CUDA)

```bash
# Check if CUDA is available
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# For PyTorch
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### Apple Silicon (M1/M2)

```bash
# TensorFlow Metal support
pip install tensorflow-metal

# PyTorch MPS support is included by default
```

## 📁 Project Structure

```
deep-learning-projects-collection/
├── README.md                     # Main documentation
├── SETUP.md                      # This setup guide
├── requirements.txt              # Python dependencies
├── 01_Deep_Learning_Basics/      # Beginner project
│   ├── README.md
│   └── deep_learning_basics.ipynb
├── 02_CIFAR10_CNN_Classifier/    # CNN project
├── 03_Fashion_MNIST_ResNet/      # Advanced CNN
├── ...                           # More projects
└── 12_Diffusion_Models/          # Advanced project
```

## 🧪 Testing Your Setup

Run this quick test to verify everything is working:

```python
# Test script - save as test_setup.py
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn

print("✅ TensorFlow version:", tf.__version__)
print("✅ PyTorch version:", torch.__version__)
print("✅ NumPy version:", np.__version__)
print("✅ Scikit-learn version:", sklearn.__version__)

# Test GPU availability
print("🖥️ TensorFlow GPU:", len(tf.config.list_physical_devices('GPU')) > 0)
print("🖥️ PyTorch GPU:", torch.cuda.is_available())

print("\n🎉 Setup complete! Ready to start learning!")
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall packages
pip install --upgrade tensorflow torch matplotlib
```

**2. Jupyter Kernel Issues**
```bash
# Install kernel
python -m ipykernel install --user --name=dl_projects_env
```

**3. Memory Issues**
- Reduce batch sizes in notebooks
- Close other applications
- Use Google Colab for resource-intensive projects

**4. CUDA Issues**
```bash
# Check CUDA version
nvidia-smi

# Reinstall TensorFlow with specific CUDA version
pip install tensorflow[and-cuda]
```

## 📚 Learning Path

**Beginner (Projects 1-4):**
- Start with Project 01: Deep Learning Basics
- Learn fundamental concepts
- No prior ML experience needed

**Intermediate (Projects 5-8):**
- Requires basic understanding of neural networks
- Introduces specialized architectures
- Real-world applications

**Advanced (Projects 9-12):**
- Modern architectures (Transformers, GANs, Diffusion)
- State-of-the-art techniques
- Research-level implementations

## 🆘 Getting Help

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check individual project READMEs
- **Community**: Join our Discord/Slack (links in main README)

## 🔄 Updates

Keep your environment updated:

```bash
# Update packages
pip install --upgrade -r requirements.txt

# Pull latest changes
git pull origin main
```

---

**Happy Learning! 🚀**

Ready to dive into deep learning? Start with [Project 01: Deep Learning Basics](01_Deep_Learning_Basics/README.md)!
