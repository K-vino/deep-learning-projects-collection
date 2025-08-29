# Project 02: CIFAR-10 CNN Classifier

## 🎯 Project Overview

This project introduces **Convolutional Neural Networks (CNNs)** through the CIFAR-10 dataset. You'll learn how to build, train, and optimize CNNs for image classification, including advanced techniques like data augmentation and transfer learning.

## 📚 Learning Objectives

By completing this project, you will:
- Understand CNN architecture and components (Conv2D, MaxPooling, etc.)
- Learn about data augmentation techniques
- Implement dropout and batch normalization
- Compare different CNN architectures
- Visualize feature maps and filters
- Achieve >85% accuracy on CIFAR-10

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **OpenCV** - Image processing (optional)

## 📊 Dataset: CIFAR-10

- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 60,000 (50,000 training + 10,000 test)
- **Size**: 32x32 pixels, RGB
- **Challenge**: Small image size, similar classes

## 📁 Project Structure

```
02_CIFAR10_CNN_Classifier/
├── README.md                    # This file
├── cifar10_cnn.ipynb           # Main notebook
├── models/                     # Saved models
│   ├── basic_cnn.h5
│   ├── improved_cnn.h5
│   └── best_model.h5
└── assets/                     # Generated plots
    ├── training_history.png
    ├── confusion_matrix.png
    └── feature_maps.png
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow matplotlib numpy seaborn
```

### Running the Project

1. Open `cifar10_cnn.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. Experiment with different architectures
4. Try various data augmentation techniques

## 🏗️ CNN Architectures Implemented

### 1. Basic CNN
- 2 Convolutional layers
- MaxPooling layers
- Dense layers for classification
- **Expected Accuracy**: ~70%

### 2. Improved CNN
- More convolutional layers
- Batch normalization
- Dropout for regularization
- **Expected Accuracy**: ~80%

### 3. Advanced CNN
- Deeper architecture
- Data augmentation
- Learning rate scheduling
- **Expected Accuracy**: >85%

## 🎯 Key Concepts Covered

### CNN Components
- **Conv2D**: Feature extraction with filters
- **MaxPooling2D**: Spatial dimension reduction
- **BatchNormalization**: Training stabilization
- **Dropout**: Overfitting prevention

### Data Augmentation
- Random rotations and flips
- Brightness and contrast adjustments
- Zoom and shift transformations
- Cutout/Random erasing

### Optimization Techniques
- Adam optimizer with learning rate decay
- Early stopping
- Model checkpointing
- Cross-validation

## 📈 Expected Results

| Model | Accuracy | Training Time | Parameters |
|-------|----------|---------------|------------|
| Basic CNN | ~70% | 10 min | ~100K |
| Improved CNN | ~80% | 20 min | ~500K |
| Advanced CNN | >85% | 30 min | ~1M |

## 🔍 Visualizations

The notebook includes:
- CIFAR-10 dataset samples
- Training/validation curves
- Confusion matrix
- Feature map visualizations
- Filter visualizations
- Misclassified examples analysis

## 🎓 Advanced Experiments

Try these extensions:
1. **Transfer Learning**: Use pre-trained models (ResNet, VGG)
2. **Ensemble Methods**: Combine multiple models
3. **AutoAugment**: Automated data augmentation
4. **Mixed Precision**: Faster training with FP16

## 🔧 Hyperparameter Tuning

Key parameters to experiment with:
- Learning rate: [0.001, 0.01, 0.1]
- Batch size: [32, 64, 128]
- Number of filters: [32, 64, 128, 256]
- Dropout rate: [0.2, 0.3, 0.5]

## 🚀 Next Steps

After completing this project:
1. Experiment with different architectures
2. Try other datasets (CIFAR-100, ImageNet)
3. Implement custom data augmentation
4. Move on to **Project 03: Fashion-MNIST with ResNet**

## 📚 Additional Resources

- [CS231n: Convolutional Neural Networks](http://cs231n.github.io/)
- [Deep Learning for Computer Vision](https://www.pyimagesearch.com/)
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Data Augmentation Techniques](https://github.com/aleju/imgaug)

## 🤝 Contributing

Found ways to improve the model? Submit a pull request with your enhancements!

---

**Ready to dive into CNNs? Let's classify some images! 🖼️**
