"""Unified AE/TAE implementation exported from the original notebook."""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

#数据输入接口
import torch
from torch.utils.data import TensorDataset


def prepare_dataset(X):
    X = torch.as_tensor(X, dtype=torch.float32)

    if X.ndim < 2:#？？？？？
        raise ValueError(
            "X must have shape (n_samples, ...). "
            "For one feature, use shape (n_samples, 1)."
        )

    if X.numel() == 0:
        raise ValueError

    if not torch.isfinite(X).all():
        raise ValueError
    input_shape = tuple(X.shape[1:])

    X = X.flatten(start_dim=1)

    input_dim = X.shape[1]

    dataset = TensorDataset(X)

    return dataset, input_dim, input_shape

#划分数据集和验证集
from torch.utils.data import random_split
def split_dataset(dataset, val_ratio=0.2, seed=42):##默认值，可以自己改
    # 检查验证集比例
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    # 计算两个数据集的样本数
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    # 保证训练集和验证集都有数据
    if train_size == 0 or val_size == 0:
        raise ValueError(
            "Too few samples for this split. "
            "Adjust val_ratio or provide more data."
        )

    # 随机划分，固定seed方便复现
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_set, val_set

##normalize
def fit_standardizer(X_train):
    X_train = torch.as_tensor(X_train, dtype=torch.float32)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True, unbiased=False)

    # 如果某个特征始终不变，避免除以 0
    std = torch.where(std == 0, torch.ones_like(std), std)

    return mean, std


def standardize(X, mean, std):
    X = torch.as_tensor(X, dtype=torch.float32)

    return (X - mean) / std

# 搭 structure
class VanillaAutoEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        bottleneck_size=2,
        hidden_dims=(128, 64, 32)
    ):
        super().__init__()

        #可以自己设置latent 层数结构eg:(256, 128, 64, 32)
        dims = [input_dim, *hidden_dims, bottleneck_size]

        encoder_layers = []

        for i in range(len(dims) - 1):
            encoder_layers.append(
                nn.Linear(dims[i], dims[i + 1])
            )

            # 只在隐藏层后添加 ReLU
            if i < len(dims) - 2:
                encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder：按相反顺序重构输入
        decoder_dims = list(reversed(dims))

        decoder_layers = []

        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(
                nn.Linear(decoder_dims[i], decoder_dims[i + 1])
            )

            # 最终输出层不添加激活函数
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward_enc(self, x):
        return self.encoder(x)

    def forward_dec(self, x):
        return self.decoder(x)

    def forward(self, x):
        enc = self.forward_enc(x)
        dec = self.forward_dec(enc)

        return enc, dec

def prepare_lagged_dataset(X, lag=1):
    # 检查 lag 是否为非负整数
    if (
        isinstance(lag, (bool, np.bool_))
        or not isinstance(lag, (int, np.integer))
        or lag < 0
    ):
        raise ValueError("lag must be a nonnegative integer.")

    # 整理数据，保持帧的时间顺序
    dataset, input_dim, input_shape = prepare_dataset(X)
    X = dataset.tensors[0]

    # 轨迹必须足够长，才能找到对应的未来帧
    if len(X) <= lag:
        raise ValueError("The number of frames must be greater than lag.")

    # 构造当前帧和目标帧
    if lag == 0:
        X_now = X
        X_future = X
    else:
        X_now = X[:-lag]
        X_future = X[lag:]

    # 把每一对输入和目标绑定在一起
    paired_dataset = TensorDataset(X_now, X_future)

    return paired_dataset, input_dim, input_shape

def train(
    model,
    train_set,
    val_set,
    batch_size=64,
    training_iterations=25000,
    evaluation_iterations=250,
    learning_rate=0.0005,
    verbose=False
):
    """
    Input data should already be appropriately preprocessed.
    If standardization is needed, compute the mean and standard
    deviation using only the training set, and apply the same
    transformation to the validation set.

    """

    # 1. 检查参数和数据
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if training_iterations < 1 or evaluation_iterations < 1:
        raise ValueError("Iteration counts must be positive.")

    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError(
            "Training and validation sets must not be empty."
        )

    # 2. 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 3. 按 batch 读取数据
    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    valloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    # 4. 设置优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # 5. 保存训练记录
    train_losses = []
    validation_losses = []
    encoded_data_per_eval = []

    train_loss_sum = 0.0
    train_sample_count = 0
    step_counter = 0

    # 6. 开始训练
    model.train()

    with tqdm(total=training_iterations) as pbar:

        while step_counter < training_iterations:
            for data, target in trainloader:
                data = data.to(device)
                target = target.to(device)  # 新增 2

                optimizer.zero_grad()

                encoded, reconstruction = model(data)
                loss = torch.mean((target - reconstruction) ** 2)

                loss.backward()
                optimizer.step()

                # 按样本数量累计 loss
                train_loss_sum += loss.item() * data.shape[0]
                train_sample_count += data.shape[0]

                step_counter += 1
                pbar.update(1)

                # 定期验证；最后一步也验证
                if (
                    step_counter % evaluation_iterations == 0
                    or step_counter == training_iterations
                ):
                    model.eval()

                    val_loss_sum = 0.0
                    val_sample_count = 0
                    encoded_evaluations = []

                    with torch.no_grad():

                        for val_data, val_target in valloader:
                            val_data = val_data.to(device)
                            val_target = val_target.to(device)

                            val_encoded, val_reconstruction = model(val_data)

                            val_loss = torch.mean(
                                (val_target - val_reconstruction) ** 2
                            )

                            val_loss_sum += (
                                val_loss.item() * val_data.shape[0]
                            )
                            val_sample_count += val_data.shape[0]

                            encoded_evaluations.append(
                                val_encoded.cpu()
                            )

                    mean_train_loss = train_loss_sum / train_sample_count
                    mean_val_loss = val_loss_sum / val_sample_count

                    train_losses.append(mean_train_loss)
                    validation_losses.append(mean_val_loss)

                    encoded_data_per_eval.append(
                        torch.cat(encoded_evaluations, dim=0).numpy()
                    )

                    if verbose:
                        print(
                            f"Step {step_counter}: "
                            f"train loss = {mean_train_loss:.6f}, "
                            f"validation loss = {mean_val_loss:.6f}"
                        )

                    # 开始下一段训练记录
                    train_loss_sum = 0.0
                    train_sample_count = 0

                    model.train()

                if step_counter >= training_iterations:
                    break

    return (
        model,
        train_losses,
        validation_losses,
        encoded_data_per_eval
    )

# Loss curve
def plot_loss(train_losses, validation_losses):
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")

    plt.xlabel("Evaluation Index")
    plt.ylabel("MSE Loss")
    plt.title("AutoEncoder Training")

    plt.legend()
    plt.show()

def plot_latent(encoded, dims=(0, 1)):
    encoded = np.asarray(encoded)

    if encoded.ndim != 2:
        raise ValueError(
            "encoded must have shape (n_samples, latent_dim)."
        )

    if len(dims) != 2 or dims[0] == dims[1]:
        raise ValueError("Please select two different latent dimensions.")

    for dim in dims:
        if not isinstance(dim, (int, np.integer)):
            raise ValueError("Dimension indices must be integers.")
        if dim < 0 or dim >= encoded.shape[1]:
            raise ValueError("Selected dimension is out of range.")

    plt.figure(figsize=(8, 6))

    plt.scatter(
        encoded[:, dims[0]],
        encoded[:, dims[1]],
        color="steelblue",
        alpha=0.5,
        s=10
    )

    plt.xlabel(f"Latent Dimension {dims[0] + 1}")
    plt.ylabel(f"Latent Dimension {dims[1] + 1}")
    plt.title("Latent Space")

    plt.show()

class AutoEncoder:

    def __init__(
        self,
        bottleneck_size=2,
        hidden_dims=(128, 64, 32),
        batch_size=64,
        training_iterations=25000,
        evaluation_iterations=250,
        learning_rate=0.0005,
        verbose=False,
        lag=1                       # 新增：默认预测下一帧
    ):
        # 保存用户的设置，调用 fit 时才创建和训练模型
        self.bottleneck_size = bottleneck_size
        self.hidden_dims = tuple(hidden_dims)
        self.batch_size = batch_size
        self.training_iterations = training_iterations
        self.evaluation_iterations = evaluation_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose

        # 新增：检查并保存时间间隔
        if (
            isinstance(lag, (bool, np.bool_))
            or not isinstance(lag, (int, np.integer))
            or lag < 0
        ):
            raise ValueError("lag must be a nonnegative integer.")

        self.lag = int(lag)

    def fit(self, X_train, X_val=None, val_ratio=0.2):
        """
        每次调用都会创建并训练一个新模型。

        X_train、X_val 各是一条按时间排列的连续轨迹。
        输入原始特征，标准化由本方法完成。
        """

        # 1. 整理训练数据，记录每一帧原来的形状
        dataset, input_dim, input_shape = prepare_dataset(X_train)
        train_data = dataset.tensors[0]

        # 2. 划分训练集和验证集
        if X_val is None:
            if not 0 < val_ratio < 1:
                raise ValueError(
                    "val_ratio must be between 0 and 1."
                )

            val_size = int(len(train_data) * val_ratio)
            train_size = len(train_data) - val_size

            # 修改：按时间顺序划分，不随机打乱原始帧
            val_data = train_data[train_size:]
            train_data = train_data[:train_size]

        else:
            val_dataset, _, val_input_shape = prepare_dataset(X_val)
            val_data = val_dataset.tensors[0]

            if input_shape != val_input_shape:
                raise ValueError(
                    "Training and validation samples "
                    "must have the same shape."
                )

        # 每一部分都必须有足够的帧进行时间配对
        if (
            len(train_data) <= self.lag
            or len(val_data) <= self.lag
        ):
            raise ValueError(
                "Training and validation data must each contain "
                "more frames than lag."
            )

        # 3. 新增：只用训练数据计算标准化参数
        mean, std = fit_standardizer(train_data)

        train_data = standardize(train_data, mean, std)
        val_data = standardize(val_data, mean, std)

        # 4. 新增：分别在训练集和验证集内部配对
        train_set, _, _ = prepare_lagged_dataset(
            train_data,
            lag=self.lag
        )

        val_set, _, _ = prepare_lagged_dataset(
            val_data,
            lag=self.lag
        )

        # 5. 创建底层网络：保持原样
        model = VanillaAutoEncoder(
            input_dim=input_dim,
            bottleneck_size=self.bottleneck_size,
            hidden_dims=self.hidden_dims
        )

        # 6. 调用训练函数：保持原来的参数
        (
            model,
            train_losses,
            validation_losses,
            encoded_history
        ) = train(
            model,
            train_set,
            val_set,
            batch_size=self.batch_size,
            training_iterations=self.training_iterations,
            evaluation_iterations=self.evaluation_iterations,
            learning_rate=self.learning_rate,
            verbose=self.verbose
        )

        # 保存训练结果
        self.model_ = model
        self.input_dim_ = input_dim
        self.input_shape_ = input_shape
        self.train_losses_ = train_losses
        self.validation_losses_ = validation_losses
        self.encoded_history_ = encoded_history

        # 新增：保存本次训练使用的标准化参数和 lag
        self.mean_ = mean
        self.std_ = std
        self.lag_ = self.lag

        self.model_.eval()

        return self

    def _check_fitted(self):
        """Check that training has completed."""
        if not hasattr(self, "model_"):
            raise RuntimeError(
                "Call fit() before using this method."
            )

    def transform(self, X):
        """把输入构象编码成 latent 表示。"""
        self._check_fitted()

        dataset, _, input_shape = prepare_dataset(X)

        if input_shape != self.input_shape_:
            raise ValueError(
                "Input samples must have the same shape "
                "as the training samples."
            )

        # 新增：使用训练时的参数标准化，不重新计算
        data = standardize(
            dataset.tensors[0],
            self.mean_,
            self.std_
        )
        dataset = TensorDataset(data)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        device = next(self.model_.parameters()).device
        encoded_batches = []

        self.model_.eval()

        with torch.no_grad():
            for (data,) in loader:
                encoded = self.model_.forward_enc(
                    data.to(device)
                )
                encoded_batches.append(encoded.cpu())

        return torch.cat(
            encoded_batches, dim=0
        ).numpy()

    def inverse_transform(self, Z):
        """
        解码 latent，并恢复原始特征的数值尺度和形状。

        lag > 0：输出是未来特征的预测，不是重建当前帧。
        lag = 0：输出是当前帧的重建。
        """
        self._check_fitted()

        Z = torch.as_tensor(Z, dtype=torch.float32)

        if (
            Z.ndim != 2
            or Z.shape[1] != self.bottleneck_size
        ):
            raise ValueError(
                "Z must have shape "
                "(n_samples, bottleneck_size)."
            )

        if (
            Z.shape[0] == 0
            or not torch.isfinite(Z).all()
        ):
            raise ValueError(
                "Z must be nonempty and contain finite values."
            )

        loader = DataLoader(
            TensorDataset(Z),
            batch_size=self.batch_size,
            shuffle=False
        )

        device = next(self.model_.parameters()).device
        reconstructed_batches = []

        self.model_.eval()

        with torch.no_grad():
            for (latent,) in loader:
                reconstruction = self.model_.forward_dec(
                    latent.to(device)
                )
                reconstructed_batches.append(
                    reconstruction.cpu()
                )

        reconstruction = torch.cat(
            reconstructed_batches, dim=0
        )

        # 新增：撤销标准化，恢复原来的数值尺度
        reconstruction = (
            reconstruction * self.std_.cpu()
            + self.mean_.cpu()
        )

        return reconstruction.reshape(
            -1, *self.input_shape_
        ).numpy()

    def plot_loss(self):
        """Plot the recorded training and validation losses."""
        self._check_fitted()

        plot_loss(
            self.train_losses_,
            self.validation_losses_
        )

    def plot_latent(self, dims=(0, 1)):
        """Plot validation latent representations from the final evaluation."""
        self._check_fitted()

        plot_latent(
            self.encoded_history_[-1],
            dims=dims
        )

# 用法示例：

# 1. 只提供一条连续轨迹 X
# 默认 lag=1：用当前帧预测下一帧
# 默认 val_ratio=0.2：前 80% 用于训练，后 20% 用于验证
# ae = AutoEncoder(
#     bottleneck_size=2,
#     hidden_dims=(128, 64, 32),
#     training_iterations=1000
# ).fit(X)

# 2. 自己提供训练轨迹和验证轨迹，并设置 lag
# lag=10：用当前帧预测相隔 10 帧的未来帧
# X_train、X_val 各自必须保持时间顺序
# 不要把多条独立轨迹直接首尾拼接
# ae = AutoEncoder(
#     bottleneck_size=2,
#     hidden_dims=(128, 64, 32),
#     training_iterations=1000,
#     lag=10
# ).fit(X_train, X_val)

# 输入特征的标准化由包内完成
# 验证数据和新数据使用训练时保存的均值、标准差

# 绘制 loss 和最后一次验证的 latent space
# ae.plot_loss()
# ae.plot_latent(dims=(0, 1))
# dims 中的索引不能超过 bottleneck_size - 1

# 编码新数据
# Z = ae.transform(X_new)

# 解码并恢复原始数值尺度
# lag > 0：预测未来特征；lag = 0：重建当前特征
# X_predicted = ae.inverse_transform(Z)

# 使用普通 AE 时，设置 lag=0