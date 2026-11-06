# Unified Autoencoder and Time-Lagged Autoencoder

A reusable PyTorch implementation for molecular feature dimensionality reduction. Set `lag=0` for ordinary AE reconstruction or a positive lag for future-feature prediction.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from autoencoder import AutoEncoder

# X_train and X_val: separate continuous trajectories, shape (frames, features).
model = AutoEncoder(
    lag=1,  # use 0 for ordinary AE
    bottleneck_size=2,
    hidden_dims=(128, 64, 32),
    batch_size=512,
    learning_rate=0.001,
    training_iterations=1000,
    evaluation_iterations=250,
)
model.fit(X_train, X_val)
Z = model.transform(X_val)
model.plot_loss()
model.plot_latent()
```

Prepare numerical features before calling the wrapper. Keep temporal order within each input array; do not concatenate independent trajectories when using a positive lag. Lag is measured in stored frames. Each split must contain more frames than the lag.

With `fit(X)`, the first 80% of a continuous trajectory is used for training and the final 20% for validation by default. With `fit(X_train, X_val)`, the supplied split is used. Standardization is fitted on training frames only and reused for validation and inference. Each `fit()` call creates a new network.

`inverse_transform(Z)` decodes latent representations and restores the original feature scale and shape. With positive lag, it predicts future features rather than reconstructing current inputs.

## Files

- `autoencoder.py`: code exported from the supplied notebook without changing the model logic.
- `Time-lagged autoencoders.ipynb`: original implementation notebook.
- `requirements.txt`: wrapper dependencies; versions are not pinned.

The example demonstrates the interface; it does not reproduce every historical experiment. The earlier alanine dipeptide AE experiment used a separate older implementation without standardization. No trajectory data or trained model weights are included.
