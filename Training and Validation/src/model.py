import numpy as np


def _he_init(shape):
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    return (np.random.randn(*shape) * np.sqrt(2.0 / max(fan_in, 1))).astype(np.float32)


def _im2col(x: np.ndarray, kernel_size: int, stride: int, padding: int):
    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    x_pad = (
        np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        if padding > 0 else x
    )
    s0, s1, s2, s3 = x_pad.strides
    col = np.lib.stride_tricks.as_strided(
        x_pad,
        shape=(N, C, kernel_size, kernel_size, H_out, W_out),
        strides=(s0, s1, s2, s3, s2 * stride, s3 * stride),
    )
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1).copy(), H_out, W_out


def _col2im(dcol: np.ndarray, x_shape, kernel_size: int, stride: int, padding: int):
    N, C, H, W = x_shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    dx_pad = np.zeros((N, C, H_padded, W_padded), dtype=dcol.dtype)
    d = dcol.reshape(N, H_out, W_out, C, kernel_size, kernel_size).transpose(0, 3, 4, 5, 1, 2)

    for kh in range(kernel_size):
        for kw in range(kernel_size):
            dx_pad[:, :, kh: kh + stride * H_out: stride,
                         kw: kw + stride * W_out: stride] += d[:, :, kh, kw, :, :]

    return dx_pad[:, :, padding: H_padded - padding, padding: W_padded - padding] if padding > 0 else dx_pad


class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self):
        return []

    def gradients(self):
        return []

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())


class Conv2D(Layer):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = _he_init((out_ch, in_ch, kernel_size, kernel_size))
        self.b = np.zeros(out_ch, dtype=np.float32) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

        self.frozen = False
        self._col_cache = None
        self._x_shape_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        col, H_out, W_out = _im2col(x, self.kernel_size, self.stride, self.padding)
        W_col = self.W.reshape(self.out_ch, -1)
        out = (W_col @ col.T).T.reshape(N, H_out, W_out, self.out_ch).transpose(0, 3, 1, 2)
        if self.b is not None:
            out = out + self.b.reshape(1, -1, 1, 1)
        self._col_cache = col
        self._x_shape_cache = x.shape
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N = dout.shape[0]
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_ch)
        W_col = self.W.reshape(self.out_ch, -1)

        if not self.frozen:
            self.dW = (dout_r.T @ self._col_cache / N).reshape(self.W.shape)
            if self.b is not None:
                self.db = dout_r.sum(axis=0) / N

        dcol = dout_r @ W_col
        return _col2im(dcol, self._x_shape_cache, self.kernel_size, self.stride, self.padding)

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])

    def gradients(self):
        if self.frozen:
            return [np.zeros_like(self.W)] + ([np.zeros_like(self.b)] if self.b is not None else [])
        return [self.dW] + ([self.db] if self.b is not None else [])


class BatchNorm2D(Layer):

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta  = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var  = np.ones(num_features, dtype=np.float32)

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)
        self.training = True
        self.frozen = False

        self._x_norm = None
        self._std = None
        self._x_minus_mu = None
        self._m = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mu  = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mu  = self.running_mean
            var = self.running_var

        std = np.sqrt(var + self.eps).reshape(1, -1, 1, 1)
        x_minus_mu = x - mu.reshape(1, -1, 1, 1)
        x_norm = x_minus_mu / std

        self._x_norm     = x_norm
        self._std        = std
        self._x_minus_mu = x_minus_mu
        self._m          = x.shape[0] * x.shape[2] * x.shape[3]

        return self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if not self.frozen:
            self.dgamma = (dout * self._x_norm).sum(axis=(0, 2, 3))
            self.dbeta  = dout.sum(axis=(0, 2, 3))

        dx_norm = dout * self.gamma.reshape(1, -1, 1, 1)

        dvar  = (-0.5 * dx_norm * self._x_minus_mu / self._std ** 3).sum(axis=(0, 2, 3))
        dmean = ((-dx_norm / self._std).sum(axis=(0, 2, 3))
                 + dvar * (-2.0 / self._m) * self._x_minus_mu.sum(axis=(0, 2, 3)))

        dx = (dx_norm / self._std
              + (2.0 / self._m) * dvar.reshape(1, -1, 1, 1) * self._x_minus_mu
              + dmean.reshape(1, -1, 1, 1) / self._m)
        return dx

    def parameters(self):
        return [self.gamma, self.beta]

    def gradients(self):
        if self.frozen:
            return [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        return [self.dgamma, self.dbeta]


class ReLU(Layer):
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


class Dropout(Layer):

    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training and self.p > 0.0:
            self._mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
            return x * self._mask
        self._mask = None
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask if self._mask is not None else dout


class MaxPool2D(Layer):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self._cache = None
        self._x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        if self.padding > 0:
            x = np.pad(x, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),
                       constant_values=-np.inf)
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        col, _, _ = _im2col(x, self.kernel_size, self.stride, 0)
        k2 = self.kernel_size ** 2
        col_r = col.reshape(N * H_out * W_out * C, k2)
        max_idx = col_r.argmax(axis=1)
        out = col_r[np.arange(len(col_r)), max_idx].reshape(N, H_out, W_out, C).transpose(0, 3, 1, 2)

        self._cache  = (col_r, max_idx, N, C, H_out, W_out)
        self._x_shape = (N, C, H, W)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        col_r, max_idx, N, C, H_out, W_out = self._cache
        k2 = self.kernel_size ** 2
        dcol = np.zeros_like(col_r)
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1)
        dcol[np.arange(len(dcol)), max_idx] = dout_flat
        dcol_full = dcol.reshape(N * H_out * W_out, C * k2)
        return _col2im(dcol_full, self._x_shape, self.kernel_size, self.stride, self.padding)


class AvgPool2D(Layer):
    def __init__(self, kernel_size: int, stride: int = None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self._x_shape = None
        self._H_out = None
        self._W_out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        col, H_out, W_out = _im2col(x, self.kernel_size, self.stride, 0)
        k2 = self.kernel_size ** 2
        out = col.reshape(N * H_out * W_out, C, k2).mean(axis=2)
        self._x_shape = x.shape
        self._H_out, self._W_out = H_out, W_out
        return out.reshape(N, H_out, W_out, C).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, H_out, W_out = dout.shape
        k2 = self.kernel_size ** 2
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        dcol = np.repeat(dout_r[:, :, np.newaxis], k2, axis=2) / k2
        dcol = dcol.reshape(N * H_out * W_out, -1)
        return _col2im(dcol, self._x_shape, self.kernel_size, self.stride, 0)


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.W = _he_init((out_features, in_features))
        self.b = np.zeros(out_features, dtype=np.float32) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self.frozen = False
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        out = x @ self.W.T
        return out + self.b if self.b is not None else out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if not self.frozen:
            N = dout.shape[0]
            self.dW = (dout.T @ self._x) / N
            if self.b is not None:
                self.db = dout.mean(axis=0)
        return dout @ self.W

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])

    def gradients(self):
        if self.frozen:
            return [np.zeros_like(self.W)] + ([np.zeros_like(self.b)] if self.b is not None else [])
        return [self.dW] + ([self.db] if self.b is not None else [])


class DenseLayer:

    def __init__(self, in_channels: int, growth_rate: int):
        bottleneck = 4 * growth_rate
        self.bn1   = BatchNorm2D(in_channels)
        self.relu1 = ReLU()
        self.conv1 = Conv2D(in_channels, bottleneck, kernel_size=1, bias=False)
        self.bn2   = BatchNorm2D(bottleneck)
        self.relu2 = ReLU()
        self.conv2 = Conv2D(bottleneck, growth_rate, kernel_size=3, padding=1, bias=False)

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self._sublayers = [self.bn1, self.relu1, self.conv1, self.bn2, self.relu2, self.conv2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = self.conv1.forward(self.relu1.forward(self.bn1.forward(x)))
        out = self.conv2.forward(self.relu2.forward(self.bn2.forward(out)))
        return np.concatenate([x, out], axis=1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d_identity = dout[:, :self.in_channels]
        d_new      = dout[:, self.in_channels:]

        d = self.conv2.backward(d_new)
        d = self.relu2.backward(d)
        d = self.bn2.backward(d)
        d = self.conv1.backward(d)
        d = self.relu1.backward(d)
        d = self.bn1.backward(d)

        return d_identity + d

    def set_training(self, mode: bool):
        self.bn1.training = mode
        self.bn2.training = mode

    def set_frozen(self, frozen: bool):
        for layer in self._sublayers:
            if hasattr(layer, "frozen"):
                layer.frozen = frozen

    def parameters(self):
        params = []
        for l in self._sublayers:
            params.extend(l.parameters())
        return params

    def gradients(self):
        grads = []
        for l in self._sublayers:
            grads.extend(l.gradients())
        return grads


class DenseBlock:

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        self.layers = []
        ch = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.out_channels = ch

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def set_training(self, mode: bool):
        for layer in self.layers:
            layer.set_training(mode)

    def set_frozen(self, frozen: bool):
        for layer in self.layers:
            layer.set_frozen(frozen)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def gradients(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.gradients())
        return grads


class TransitionLayer:

    def __init__(self, in_channels: int, compression: float = 0.5):
        self.out_channels = int(in_channels * compression)
        self.bn   = BatchNorm2D(in_channels)
        self.relu = ReLU()
        self.conv = Conv2D(in_channels, self.out_channels, kernel_size=1, bias=False)
        self.pool = AvgPool2D(kernel_size=2, stride=2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.relu.forward(self.bn.forward(x))
        return self.pool.forward(self.conv.forward(x))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d = self.pool.backward(dout)
        d = self.conv.backward(d)
        d = self.relu.backward(d)
        return self.bn.backward(d)

    def set_training(self, mode: bool):
        self.bn.training = mode

    def set_frozen(self, frozen: bool):
        self.bn.frozen   = frozen
        self.conv.frozen = frozen

    def parameters(self):
        return self.bn.parameters() + self.conv.parameters()

    def gradients(self):
        return self.bn.gradients() + self.conv.gradients()


class DenseNet121:

    GROWTH_RATE  = 32
    BLOCK_CONFIG = (6, 12, 24, 16)
    INIT_FEATURES = 64

    def __init__(self, num_classes: int = 7, dropout: float = 0.6):
        self.num_classes = num_classes
        self._backbone_frozen = False
        self._training = True

        self.conv0 = Conv2D(3, self.INIT_FEATURES, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = BatchNorm2D(self.INIT_FEATURES)
        self.relu0 = ReLU()
        self.pool0 = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.dense_blocks = []
        self.transitions  = []
        num_features = self.INIT_FEATURES

        for i, num_layers in enumerate(self.BLOCK_CONFIG):
            block = DenseBlock(num_layers, num_features, self.GROWTH_RATE)
            self.dense_blocks.append(block)
            num_features = block.out_channels

            if i < len(self.BLOCK_CONFIG) - 1:
                trans = TransitionLayer(num_features, compression=0.5)
                self.transitions.append(trans)
                num_features = trans.out_channels

        self.norm_final = BatchNorm2D(num_features)
        self.relu_final = ReLU()

        self.dropout = Dropout(p=dropout)
        self.fc = Linear(num_features, num_classes)
        self._gap_shape = None

    def train(self):
        self._training = True
        self.norm0.training = True
        self.dropout.training = True
        for block in self.dense_blocks:
            block.set_training(True)
        for trans in self.transitions:
            trans.set_training(True)
        self.norm_final.training = True

    def eval(self):
        self._training = False
        self.norm0.training = False
        self.dropout.training = False
        for block in self.dense_blocks:
            block.set_training(False)
        for trans in self.transitions:
            trans.set_training(False)
        self.norm_final.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.pool0.forward(self.relu0.forward(self.norm0.forward(self.conv0.forward(x))))

        for i, block in enumerate(self.dense_blocks):
            x = block.forward(x)
            if i < len(self.transitions):
                x = self.transitions[i].forward(x)

        x = self.relu_final.forward(self.norm_final.forward(x))

        self._gap_shape = x.shape
        x = x.mean(axis=(2, 3))

        x = self.dropout.forward(x)
        return self.fc.forward(x)

    def backward(self, dlogits: np.ndarray) -> None:
        d = self.fc.backward(dlogits)
        d = self.dropout.backward(d)

        N, C, H, W = self._gap_shape
        d = np.broadcast_to(
            d[:, :, np.newaxis, np.newaxis] / (H * W), (N, C, H, W)
        ).copy()

        d = self.relu_final.backward(d)
        d = self.norm_final.backward(d)

        for i in range(len(self.dense_blocks) - 1, -1, -1):
            if i < len(self.transitions):
                d = self.transitions[i].backward(d)
            d = self.dense_blocks[i].backward(d)

        d = self.pool0.backward(d)
        d = self.relu0.backward(d)
        d = self.norm0.backward(d)
        self.conv0.backward(d)

    def parameters(self, classifier_only: bool = False):
        params = []
        if not classifier_only:
            params.extend(self.conv0.parameters())
            params.extend(self.norm0.parameters())
            for block in self.dense_blocks:
                params.extend(block.parameters())
            for trans in self.transitions:
                params.extend(trans.parameters())
            params.extend(self.norm_final.parameters())
        params.extend(self.fc.parameters())
        return params

    def gradients(self, classifier_only: bool = False):
        grads = []
        if not classifier_only:
            grads.extend(self.conv0.gradients())
            grads.extend(self.norm0.gradients())
            for block in self.dense_blocks:
                grads.extend(block.gradients())
            for trans in self.transitions:
                grads.extend(trans.gradients())
            grads.extend(self.norm_final.gradients())
        grads.extend(self.fc.gradients())
        return grads

    def named_parameters(self):
        named = [
            ("conv0.weight",  self.conv0.W),
            ("norm0.weight",  self.norm0.gamma),
            ("norm0.bias",    self.norm0.beta),
        ]
        for i, block in enumerate(self.dense_blocks):
            for j, dl in enumerate(block.layers):
                named += [
                    (f"denseblock{i+1}.denselayer{j+1}.norm1.weight", dl.bn1.gamma),
                    (f"denseblock{i+1}.denselayer{j+1}.norm1.bias",   dl.bn1.beta),
                    (f"denseblock{i+1}.denselayer{j+1}.conv1.weight", dl.conv1.W),
                    (f"denseblock{i+1}.denselayer{j+1}.norm2.weight", dl.bn2.gamma),
                    (f"denseblock{i+1}.denselayer{j+1}.norm2.bias",   dl.bn2.beta),
                    (f"denseblock{i+1}.denselayer{j+1}.conv2.weight", dl.conv2.W),
                ]
        for i, trans in enumerate(self.transitions):
            named += [
                (f"transition{i+1}.norm.weight", trans.bn.gamma),
                (f"transition{i+1}.norm.bias",   trans.bn.beta),
                (f"transition{i+1}.conv.weight",  trans.conv.W),
            ]
        named += [
            ("norm5.weight",          self.norm_final.gamma),
            ("norm5.bias",            self.norm_final.beta),
            ("classifier.0.weight",   self.fc.W),
            ("classifier.0.bias",     self.fc.b),
        ]
        return named

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only and self._backbone_frozen:
            return self.fc.num_parameters()
        return sum(p.size for p in self.parameters())

    def freeze_backbone(self):
        self._backbone_frozen = True
        self.conv0.frozen = True
        self.norm0.frozen = True
        for block in self.dense_blocks:
            block.set_frozen(True)
        for trans in self.transitions:
            trans.set_frozen(True)
        self.norm_final.frozen = True
        self.fc.frozen = False

    def unfreeze_backbone(self):
        self._backbone_frozen = False
        self.conv0.frozen = False
        self.norm0.frozen = False
        for block in self.dense_blocks:
            block.set_frozen(False)
        for trans in self.transitions:
            trans.set_frozen(False)
        self.norm_final.frozen = False

    def save(self, path: str):
        weights = {name: param.copy() for name, param in self.named_parameters()}
        np.savez(path, **weights)

    def load(self, path: str):
        data = np.load(path)
        param_dict = dict(self.named_parameters())
        for name in data.files:
            if name in param_dict:
                np.copyto(param_dict[name], data[name])


def build_model(cfg, num_classes: int) -> DenseNet121:
    return DenseNet121(num_classes=num_classes, dropout=cfg.model.dropout)


def freeze_backbone(model: DenseNet121):
    model.freeze_backbone()
    total     = model.num_parameters(trainable_only=False)
    trainable = model.num_parameters(trainable_only=True)
    return trainable, total


def unfreeze_backbone(model: DenseNet121):
    model.unfreeze_backbone()
    return model.num_parameters(trainable_only=False)
