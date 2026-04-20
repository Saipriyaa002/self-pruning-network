# Self-Pruning Neural Network (CNN-Based)

## 1.Overview
This project implements a self-pruning neural network that dynamically learns which connections to remove during training. Unlike traditional pruning methods that operate after training, this model integrates pruning directly into the learning process using learnable gate parameters and L1 regularization.

The goal is to achieve an optimal balance between model accuracy and sparsity, reducing unnecessary parameters while maintaining performance.


## 2.Key Idea

Each weight in the network is associated with a learnable gate:

- Gate value ≈ 1 → connection is active  
- Gate value ≈ 0 → connection is pruned  

These gates are learned during training and controlled using a sparsity-inducing loss.


## 3.Architecture

### Feature Extractor
- Convolutional Neural Network (CNN)
- Conv → BatchNorm → ReLU → MaxPool

### Classifier
- Custom `PrunableLinear` layers
- Learnable gate parameters per weight
- Dropout for regularization


## 4.Loss Function

Total Loss: CrossEntropyLoss + λ × SparsityLoss

- **CrossEntropyLoss** → classification objective  
- **SparsityLoss (L1 on gates)** → encourages pruning  
- **λ (lambda)** → controls sparsity vs accuracy trade-off  


## 5.Dataset

- CIFAR-10
- 60,000 images (32×32 RGB)
- 10 classes


## 6.Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|------------------|-------------|
| 1e-5   | 74.51%           | 0.62%       |
| 1e-4   | 73.61%           | 1.12%       |
| 5e-4   | 73.01%           | 1.35%       |


## 7.Observations

- Increasing λ increases sparsity significantly  
- High sparsity leads to slight reduction in accuracy  
- CNN backbone improves baseline performance compared to fully connected models  
- The model successfully learns to prune itself during training  


## 8.Gate Distribution

The histogram of gate values shows:

- A strong spike near 0 → pruned weights  
- Remaining values → important connections  


## 9.Output file:

gate_distribution.png


## 10.Project Structure

self-pruning-network/

├── model.py # Model + prunable layers

├── utils.py # Sparsity loss and metrics

├── train.py # Training + evaluation

├── report.md # Analysis and results

├── README.md # Documentation

├── requirements.txt # Dependencies

└── gate_distribution.png # Output plot


## 11.Installation

pip install -r requirements.txt

## 12.How to Run

python train.py


## 13.Output

## Console:

Training loss
Accuracy
Sparsity %

## Files generated:

gate_distribution.png


## 14.Key Concepts Demonstrated

Neural network pruning
L1 regularization for sparsity
Custom PyTorch layer implementation
CNN-based feature extraction
Trade-off between model size and accuracy


## 15.Future Improvements

Structured pruning (neuron/channel pruning)
Fine-tuning after pruning
Comparison with baseline (non-pruned model)
Deployment optimization (ONNX / TorchScript)


## Author
Sai Priyaa M
