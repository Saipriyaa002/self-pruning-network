# Self-Pruning Neural Network (CNN Version)

## Why L1 Encourages Sparsity
The L1 penalty adds a cost proportional to gate values. This encourages many gates to shrink toward zero, effectively removing weak connections and creating a sparse network.

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 1e-5   | 74.51%      | 0.62%       |
| 1e-4   | 73.61%      | 1.12%       |
| 5e-4   | 73.01%      | 1.35%       |

## Observations
- Increasing lambda increases sparsity
- High sparsity leads to reduced accuracy
- CNN backbone improves baseline accuracy significantly

## Conclusion
The model successfully learns to prune itself while maintaining reasonable accuracy, demonstrating an effective sparsity–performance trade-off.

## Plot
See gate_distribution.png