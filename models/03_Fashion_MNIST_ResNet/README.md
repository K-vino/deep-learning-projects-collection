# Project 03: Fashion-MNIST with ResNet

## 🎯 Project Overview

This project introduces **Residual Networks (ResNet)** through the Fashion-MNIST dataset. You'll learn about deep CNN architectures, residual connections, and advanced training techniques to achieve state-of-the-art performance on fashion item classification.

## 📚 Learning Objectives

By completing this project, you will:
- Understand ResNet architecture and residual connections
- Implement skip connections and identity mappings
- Learn advanced batch normalization techniques
- Master transfer learning with pre-trained models
- Achieve >95% accuracy on Fashion-MNIST
- Visualize feature maps and learned representations

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical plotting

## 📊 Dataset: Fashion-MNIST

- **Classes**: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Images**: 70,000 (60,000 training + 10,000 test)
- **Size**: 28x28 pixels, grayscale
- **Challenge**: Similar clothing items, fine-grained classification

## 🏗️ ResNet Architectures Implemented

### 1. Mini ResNet-18
- 18 layers with residual connections
- Basic residual blocks
- **Expected Accuracy**: ~92%

### 2. Custom ResNet
- Optimized for Fashion-MNIST
- Advanced residual blocks
- **Expected Accuracy**: ~95%

### 3. Transfer Learning
- Pre-trained ResNet50 adapted
- Fine-tuning techniques
- **Expected Accuracy**: >95%

## 🎯 Key Concepts Covered

### ResNet Components
- **Residual Blocks**: Skip connections for gradient flow
- **Identity Mapping**: Preserving information flow
- **Bottleneck Design**: Efficient parameter usage
- **Global Average Pooling**: Reducing overfitting

### Advanced Techniques
- **Learning Rate Scheduling**: Cosine annealing
- **Data Augmentation**: Fashion-specific transforms
- **Model Ensembling**: Combining multiple models
- **Gradient Clipping**: Training stability

## 📈 Expected Results

| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| Mini ResNet-18 | ~92% | ~500K | 15 min |
| Custom ResNet | ~95% | ~1M | 25 min |
| Transfer Learning | >95% | ~25M | 10 min |

## 🔍 Visualizations

The notebook includes:
- Fashion-MNIST dataset exploration
- ResNet architecture diagrams
- Training curves comparison
- Feature map visualizations
- Confusion matrix analysis
- t-SNE embeddings of learned features

## 🚀 Advanced Experiments

Try these extensions:
1. **ResNet Variants**: ResNeXt, Wide ResNet, DenseNet
2. **Attention Mechanisms**: SE-Net, CBAM
3. **Knowledge Distillation**: Teacher-student training
4. **Neural Architecture Search**: AutoML approaches

## 📚 Additional Resources

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- [ResNet Implementation Guide](https://keras.io/examples/vision/resnet/)

---

**Ready to build deep networks that actually train well? Let's implement ResNet! 🏗️**
