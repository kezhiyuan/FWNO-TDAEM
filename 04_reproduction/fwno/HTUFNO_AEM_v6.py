import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kan import KAN
except Exception:
    KAN = None


torch.manual_seed(0)
np.random.seed(0)

act_dict = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "idt": nn.Identity(),
    "gelu": nn.GELU(),
}

init_dict = {
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "uniform": nn.init.uniform_,
    "norm": nn.init.normal_,
}


def get_activation(name: str) -> nn.Module:
    if name not in act_dict:
        raise KeyError(f"act name '{name}' is not in act_dict")
    return act_dict[name]


def _num_groups(channels: int, max_groups: int = 8) -> int:

    channels = int(channels)
    for g in range(min(int(max_groups), channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _normalized_coordinate(values, transform='linear'):

    if values is None:
        return None
    v = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    mode = str(transform).lower()
    if mode in ['log10', 'log']:
        if torch.any(v <= 0):
            raise ValueError('Positive coordinate values are required for log10 transform.')
        v = torch.log10(v)
    elif mode in ['ln', 'loge']:
        if torch.any(v <= 0):
            raise ValueError('Positive coordinate values are required for ln transform.')
        v = torch.log(v)
    elif mode not in ['linear', 'none', 'identity']:
        raise ValueError(f'Unsupported coordinate transform: {transform!r}')
    span = torch.clamp(v.max() - v.min(), min=1.0e-12)
    return (v - v.min()) / span


def _resolve_modes_xyz(modes_xyz, modes1=None, modes2=None, modes3=None):

    if modes_xyz is None:
        values = [modes1, modes2, modes3]
    elif isinstance(modes_xyz, str):
        text = modes_xyz.replace('[', '').replace(']', '')
        values = [int(v.strip()) for v in text.split(',') if v.strip()]
    elif isinstance(modes_xyz, (list, tuple)):
        values = [int(v) for v in modes_xyz]
    else:
        values = [int(modes_xyz)]
    if len(values) == 1:
        values = values * 3
    if len(values) != 3 or any(v is None or int(v) < 1 for v in values):
        raise ValueError(
            'modes_xyz must contain three positive integers, or modes1/2/3 '
            'must all be provided.'
        )
    return tuple(int(v) for v in values)


def _resolve_spatial_padding(spatial_padding=None, padding=0):

    value = padding if spatial_padding is None else spatial_padding
    if isinstance(value, str):
        text = value.replace('[', '').replace(']', '')
        values = [int(v.strip()) for v in text.split(',') if v.strip()]
    elif isinstance(value, (list, tuple)):
        values = [int(v) for v in value]
    else:
        values = [int(value)]
    if len(values) == 1:
        values = values * 3
    if len(values) != 3 or any(v < 0 for v in values):
        raise ValueError(
            'spatial_padding must be a non-negative integer or an '
            '(x, y, z) sequence of three non-negative integers.'
        )
    return tuple(values)


def _centered_pad_pairs(total_padding):

    return tuple((p // 2, p - p // 2) for p in total_padding)


class PointwiseKAN(nn.Module):

    def __init__(self, layers_hidden, act_func="gelu"):
        super().__init__()
        if KAN is not None:
            self.net = KAN(layers_hidden=layers_hidden)
        else:
            layers = []
            act = get_activation(act_func)
            for i in range(len(layers_hidden) - 2):
                layers += [nn.Linear(layers_hidden[i], layers_hidden[i + 1]), act]
            layers.append(nn.Linear(layers_hidden[-2], layers_hidden[-1]))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        in_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.net(x)
        x = x.reshape(*in_shape, -1)
        return x


class BranchLift(nn.Module):

    def __init__(
        self,
        width: int,
        padding: int = 8,
        use_kan_lift=True,
        act_func="gelu",
        spatial_padding=None,
    ):
        super().__init__()
        self.width = int(width)
        self.padding = padding
        self.spatial_padding = _resolve_spatial_padding(
            spatial_padding=spatial_padding, padding=padding
        )
        self.pad_pairs = _centered_pad_pairs(self.spatial_padding)
        if bool(use_kan_lift):
            self.lift = PointwiseKAN([1, 64, 128, width], act_func=act_func)
        else:
            act = get_activation(act_func)
            self.lift = nn.Sequential(
                nn.Linear(1, 64), act,
                nn.Linear(64, 128), act,
                nn.Linear(128, width),
            )

    def forward(self, x):

        x = self.lift(x)
        x = x.permute(0, 4, 1, 2, 3)
        if any(self.spatial_padding):
            (px0, px1), (py0, py1), (pz0, pz1) = self.pad_pairs
            x = F.pad(x, [pz0, pz1, py0, py1, px0, px1])
        x = x.permute(0, 2, 3, 4, 1)
        return x



class Unet(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super().__init__()
        self.conv1 = self.conv(input_channels, output_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size, stride=1, dropout_rate=dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size, stride=1, dropout_rate=dropout_rate)
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels * 2, output_channels)
        self.deconv0 = self.deconv(input_channels * 2, output_channels)
        self.output_layer = self.output(input_channels * 2, output_channels, kernel_size, stride=1)

    @staticmethod
    def _pad_to_match(src, ref):

        if src.shape == ref.shape:
            return src

        slices = [slice(None)] * src.ndim
        for axis in range(src.ndim - 3, src.ndim):
            excess = max(0, src.shape[axis] - ref.shape[axis])
            low_crop = excess // 2

            slices[axis] = slice(low_crop, low_crop + min(
                src.shape[axis], ref.shape[axis]
            ))
        src = src[tuple(slices)]

        deficits = [
            max(0, ref.shape[axis] - src.shape[axis])
            for axis in range(src.ndim - 3, src.ndim)
        ]
        pad_pairs = [(d // 2, d - d // 2) for d in deficits]
        (pd0, pd1), (ph0, ph1), (pw0, pw1) = pad_pairs
        return F.pad(src, [pw0, pw1, ph0, ph1, pd0, pd1])

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        out_deconv2 = self._pad_to_match(self.deconv2(out_conv3), out_conv2)
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)

        out_deconv1 = self._pad_to_match(self.deconv1(concat2), out_conv1)
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)

        out_deconv0 = self._pad_to_match(self.deconv0(concat1), x)
        concat0 = torch.cat((x, out_deconv0), dim=1)
        return self.output_layer(concat0)

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(
                in_planes, output_channels, kernel_size=kernel_size,
                stride=stride, padding=(kernel_size - 1) // 2, bias=False
            ),
            nn.GroupNorm(_num_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Dropout3d(float(dropout_rate)),
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(output_channels), output_channels),
            nn.GELU(),
        )

    def output(self, input_channels, output_channels, kernel_size, stride):
        return nn.Conv3d(
            input_channels, output_channels, kernel_size=kernel_size,
            stride=stride, padding=(kernel_size - 1) // 2
        )


class SpectralConv3d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes1=None,
        modes2=None,
        modes3=None,
        modes_xyz=None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1, self.modes2, self.modes3 = _resolve_modes_xyz(
            modes_xyz, modes1, modes2, modes3
        )


        self.scale = 1.0 / math.sqrt(max(1, self.in_channels))
        real_imag_scale = self.scale / math.sqrt(2.0)
        shape = (
            in_channels, out_channels,
            self.modes1, self.modes2, self.modes3, 2,
        )
        self.weights1 = nn.Parameter(
            real_imag_scale * torch.randn(*shape, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            real_imag_scale * torch.randn(*shape, dtype=torch.float32)
        )
        self.weights3 = nn.Parameter(
            real_imag_scale * torch.randn(*shape, dtype=torch.float32)
        )
        self.weights4 = nn.Parameter(
            real_imag_scale * torch.randn(*shape, dtype=torch.float32)
        )

    @staticmethod
    def compl_mul3d(input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        input_dtype = x.dtype
        batchsize = x.shape[0]
        sx, sy, sz = x.size(-3), x.size(-2), x.size(-1)
        m1 = min(self.modes1, sx)
        m2 = min(self.modes2, sy)
        m3 = min(self.modes3, sz // 2 + 1)
        device_type = x.device.type if x.device.type in ["cuda", "cpu"] else "cuda"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_float = x.float()
            x_ft = torch.fft.rfftn(x_float, dim=[-3, -2, -1])
            weight1 = torch.view_as_complex(
                self.weights1[:, :, :m1, :m2, :m3, :].contiguous()
            )
            weight2 = torch.view_as_complex(
                self.weights2[:, :, :m1, :m2, :m3, :].contiguous()
            )
            weight3 = torch.view_as_complex(
                self.weights3[:, :, :m1, :m2, :m3, :].contiguous()
            )
            weight4 = torch.view_as_complex(
                self.weights4[:, :, :m1, :m2, :m3, :].contiguous()
            )
            out_ft = torch.zeros(
                batchsize, self.out_channels, sx, sy, sz // 2 + 1,
                dtype=torch.cfloat, device=x.device,
            )
            out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
                x_ft[:, :, :m1, :m2, :m3], weight1
            )
            out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
                x_ft[:, :, -m1:, :m2, :m3], weight2
            )
            out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
                x_ft[:, :, :m1, -m2:, :m3], weight3
            )
            out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
                x_ft[:, :, -m1:, -m2:, :m3], weight4
            )
            out = torch.fft.irfftn(out_ft, s=(sx, sy, sz))
        return out.to(dtype=input_dtype)


class HaarWaveletResidual3d(nn.Module):

    def __init__(
        self,
        channels: int,
        hidden_multiplier: float = 1.0,
        dropout: float = 0.0,
        act_func: str = 'gelu',
    ):
        super().__init__()
        self.channels = int(channels)
        hidden = max(self.channels, int(round(self.channels * float(hidden_multiplier))))
        act = get_activation(act_func)


        self.low_mixer = nn.Sequential(
            nn.Conv3d(self.channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(hidden), hidden),
            act,
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(hidden, self.channels, kernel_size=1, bias=True),
        )



        high_channels = 7 * self.channels
        self.high_mixer = nn.Sequential(
            nn.Conv3d(high_channels, high_channels, kernel_size=1, groups=7, bias=False),
            nn.GroupNorm(7, high_channels),
            act,
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(high_channels, high_channels, kernel_size=1, groups=7, bias=True),
        )


        self.band_mixing = nn.Parameter(torch.zeros(7, 7, dtype=torch.float32))


        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        low = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float32)
        high = torch.tensor([inv_sqrt2, -inv_sqrt2], dtype=torch.float32)
        kernels = []
        for fx in (low, high):
            for fy in (low, high):
                for fz in (low, high):
                    kernels.append(torch.einsum('i,j,k->ijk', fx, fy, fz))
        kernel = torch.stack(kernels, dim=0).unsqueeze(1)
        self.register_buffer('haar_kernel', kernel, persistent=True)
        self.reset_zero_init()

    def reset_zero_init(self):

        nn.init.zeros_(self.low_mixer[-1].weight)
        nn.init.zeros_(self.low_mixer[-1].bias)
        nn.init.zeros_(self.high_mixer[-1].weight)
        nn.init.zeros_(self.high_mixer[-1].bias)
        with torch.no_grad():
            self.band_mixing.zero_()

    def _analysis_weight(self, device, dtype):

        return self.haar_kernel.to(device=device, dtype=dtype).repeat(
            self.channels, 1, 1, 1, 1
        )

    def _dwt(self, x):
        d, h, w = x.shape[-3:]
        pad_d, pad_h, pad_w = d % 2, h % 2, w % 2
        x_pad = F.pad(x, [0, pad_w, 0, pad_h, 0, pad_d])
        weight = self._analysis_weight(x.device, x.dtype)
        coeff = F.conv3d(x_pad, weight, stride=2, groups=self.channels)
        coeff = coeff.reshape(
            x.shape[0], self.channels, 8,
            coeff.shape[-3], coeff.shape[-2], coeff.shape[-1]
        )
        return coeff, (d, h, w)

    def _idwt(self, coeff, output_shape):
        b, c, _, d2, h2, w2 = coeff.shape
        coeff_flat = coeff.reshape(b, c * 8, d2, h2, w2)
        weight = self._analysis_weight(coeff.device, coeff.dtype)
        rec = F.conv_transpose3d(
            coeff_flat, weight, stride=2, groups=self.channels
        )
        d, h, w = output_shape
        return rec[..., :d, :h, :w]

    def forward(self, x):
        input_dtype = x.dtype
        device_type = x.device.type if x.device.type in ['cuda', 'cpu'] else 'cuda'


        with torch.amp.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            coeff, output_shape = self._dwt(x32)
            low = coeff[:, :, 0]
            high = coeff[:, :, 1:]

            low_delta = self.low_mixer(low)


            hb = high.permute(0, 2, 1, 3, 4, 5).contiguous()
            hb_flat = hb.reshape(
                hb.shape[0], 7 * self.channels,
                hb.shape[-3], hb.shape[-2], hb.shape[-1]
            )
            high_delta = self.high_mixer(hb_flat)
            high_delta = high_delta.reshape(
                hb.shape[0], 7, self.channels,
                hb.shape[-3], hb.shape[-2], hb.shape[-1]
            ).permute(0, 2, 1, 3, 4, 5).contiguous()


            band_delta = torch.einsum('ij,bcjdhw->bcidhw', self.band_mixing, high)
            mixed = torch.cat([
                (low + low_delta).unsqueeze(2),
                high + high_delta + band_delta,
            ], dim=2)
            rec = self._idwt(mixed, output_shape)


            correction = rec - x32
        return correction.to(dtype=input_dtype)


class UFNO3d(nn.Module):

    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width,
        layer_fno,
        layer_ufno,
        act_func,
        use_wavelet_branch=True,
        wavelet_layers=3,
        wavelet_hidden_multiplier=1.0,
        wavelet_scale_init=1.0,
        wavelet_dropout=0.0,
        modes_xyz=None,
    ):
        super().__init__()
        self.width = int(width)
        self.layer_fno = int(layer_fno)
        self.layer_ufno = int(layer_ufno)
        self.activation = get_activation(act_func)
        n_layers = self.layer_fno + self.layer_ufno
        self.fno = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.ufno = nn.ModuleList()
        resolved_modes = _resolve_modes_xyz(
            modes_xyz, modes1, modes2, modes3
        )
        self.modes_xyz = resolved_modes
        for _ in range(n_layers):
            self.fno.append(SpectralConv3d(
                width, width, modes_xyz=resolved_modes
            ))
            self.conv.append(nn.Conv3d(width, width, 1))
            self.norm.append(nn.GroupNorm(_num_groups(width), width))
        for _ in range(self.layer_ufno):
            self.ufno.append(Unet(width, width, 3, 0.0))

        self.use_wavelet_branch = bool(use_wavelet_branch)
        self.wavelet_layers = min(max(0, int(wavelet_layers)), n_layers)
        self.wavelet_start = n_layers - self.wavelet_layers
        self.wavelet = nn.ModuleList()
        if self.use_wavelet_branch and self.wavelet_layers > 0:
            for _ in range(self.wavelet_layers):
                self.wavelet.append(HaarWaveletResidual3d(
                    channels=width,
                    hidden_multiplier=wavelet_hidden_multiplier,
                    dropout=wavelet_dropout,
                    act_func=act_func,
                ))
            self.wavelet_scale = nn.Parameter(torch.full(
                (self.wavelet_layers,), float(wavelet_scale_init), dtype=torch.float32
            ))
        else:
            self.register_parameter('wavelet_scale', None)

        self.residual_scale = nn.Parameter(torch.full((n_layers,), 0.5, dtype=torch.float32))

    def reset_wavelet_zero_init(self):
        for module in self.wavelet:
            module.reset_zero_init()

    def forward(self, x):

        x = x.permute(0, 4, 1, 2, 3)
        n_layers = self.layer_fno + self.layer_ufno
        for i in range(n_layers):
            y = self.fno[i](x) + self.conv[i](x)
            if i >= self.layer_fno:
                y = y + self.ufno[i - self.layer_fno](x)
            if self.use_wavelet_branch and i >= self.wavelet_start:
                iw = i - self.wavelet_start
                y = y + torch.tanh(self.wavelet_scale[iw]) * self.wavelet[iw](x)
            y = self.activation(self.norm[i](y))
            x = x + torch.tanh(self.residual_scale[i]) * y
        return x.permute(0, 2, 3, 4, 1)



def _parse_int_list(value, default):
    if value is None:
        value = default
    if isinstance(value, str):
        value = value.replace("[", "").replace("]", "")
        parts = [p.strip() for p in value.split(",") if p.strip()]
        out = [int(p) for p in parts]
    elif isinstance(value, (list, tuple)):
        out = [int(v) for v in value]
    else:
        out = [int(value)]
    return [max(1, int(v)) for v in out]


class MultiScaleFeatureSummary(nn.Module):

    def __init__(
        self,
        width,
        latent_dim=256,
        global_grid=3,
        global_z_bins=8,
        center_windows=(3, 7, 15),
        center_z_bins=8,
        hidden=512,
        dropout=0.0,
        act_func="gelu",
    ):
        super().__init__()
        self.width = int(width)
        self.global_grid = int(global_grid)
        self.global_z_bins = int(global_z_bins)
        self.center_windows = _parse_int_list(center_windows, [3, 7, 15])
        self.center_z_bins = int(center_z_bins)
        act = get_activation(act_func)
        in_dim = self.width * (
            self.global_grid * self.global_grid * self.global_z_bins
            + len(self.center_windows) * self.center_z_bins
        )
        self.global_pool = nn.AdaptiveAvgPool3d((self.global_grid, self.global_grid, self.global_z_bins))
        self.center_z_pool = nn.AdaptiveAvgPool1d(self.center_z_bins)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), act,
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, latent_dim), act,
            nn.LayerNorm(latent_dim),
        )

    @staticmethod
    def _crop_center_window(feat, window):
        _, nx, ny, _, _ = feat.shape
        cx, cy = nx // 2, ny // 2
        half = int(window) // 2
        xs, xe = max(0, cx - half), min(nx, cx + half + 1)
        ys, ye = max(0, cy - half), min(ny, cy + half + 1)
        return feat[:, xs:xe, ys:ye, :, :]

    def forward(self, feat):

        b = feat.shape[0]
        x = feat.permute(0, 4, 1, 2, 3).contiguous()
        global_vec = self.global_pool(x).reshape(b, -1)

        center_vecs = []
        for win in self.center_windows:
            patch = self._crop_center_window(feat, win)
            depth = patch.mean(dim=(1, 2)).permute(0, 2, 1)
            depth = self.center_z_pool(depth).reshape(b, -1)
            center_vecs.append(depth)

        z = torch.cat([global_vec] + center_vecs, dim=-1)
        return self.net(z)


class SpectralConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        self.scale = 1.0 / math.sqrt(max(1, self.in_channels))
        real_imag_scale = self.scale / math.sqrt(2.0)
        self.weights = nn.Parameter(
            real_imag_scale * torch.randn(
                in_channels, out_channels, self.modes, 2,
                dtype=torch.float32,
            )
        )

    @staticmethod
    def compl_mul1d(input, weights):
        return torch.einsum("bim,iom->bom", input, weights)

    def forward(self, x):

        input_dtype = x.dtype
        b, _, t = x.shape
        m = min(self.modes, t // 2 + 1)
        device_type = x.device.type if x.device.type in ["cuda", "cpu"] else "cuda"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_ft = torch.fft.rfft(x.float(), dim=-1)
            weight = torch.view_as_complex(
                self.weights[:, :, :m, :].contiguous()
            )
            out_ft = torch.zeros(b, self.out_channels, t // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :m] = self.compl_mul1d(x_ft[:, :, :m], weight)
            out = torch.fft.irfft(out_ft, n=t, dim=-1)
        return out.to(dtype=input_dtype)


class ChannelLayerNorm1d(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x):
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()


class TemporalFNOBlock(nn.Module):
    def __init__(self, width, modes, act_func="gelu", dropout=0.0):
        super().__init__()
        self.fno = SpectralConv1d(width, width, modes)
        self.pw = nn.Conv1d(width, width, kernel_size=1)
        self.norm = ChannelLayerNorm1d(width)
        self.act = get_activation(act_func)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):

        y = self.fno(x) + self.pw(x)
        y = self.norm(y)
        y = self.dropout(self.act(y))
        return x + y


class TemporalAttentionBlock(nn.Module):

    def __init__(
        self,
        width,
        num_heads=4,
        hidden=None,
        act_func="gelu",
        dropout=0.0,
    ):
        super().__init__()
        width = int(width)
        num_heads = int(num_heads)
        if width % num_heads != 0:
            raise ValueError(
                f'temporal attention width={width} must be divisible by '
                f'num_heads={num_heads}.'
            )
        hidden = int(hidden if hidden is not None else 2 * width)
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, num_heads, dropout=float(dropout), batch_first=True
        )
        self.dropout1 = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            get_activation(act_func),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, width),
        )
        self.dropout2 = nn.Dropout(float(dropout))

    def forward(self, x):

        xt = x.permute(0, 2, 1)
        q = self.norm1(xt)
        attn, _ = self.attn(q, q, q, need_weights=False)
        xt = xt + self.dropout1(attn)
        xt = xt + self.dropout2(self.ffn(self.norm2(xt)))
        return xt.permute(0, 2, 1).contiguous()


class TemporalDownBlock(nn.Module):

    def __init__(self, width, modes, act_func="gelu", dropout=0.0):
        super().__init__()
        self.block = TemporalFNOBlock(width, modes, act_func=act_func, dropout=dropout)
        self.down = nn.Conv1d(width, width, kernel_size=4, stride=2, padding=1)
        self.norm = ChannelLayerNorm1d(width)
        self.act = get_activation(act_func)

    def forward(self, x):
        skip = self.block(x)
        y = self.act(self.norm(self.down(skip)))
        return y, skip


class TemporalUpBlock(nn.Module):

    def __init__(self, width, modes, act_func="gelu", dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose1d(width, width, kernel_size=4, stride=2, padding=1)
        self.merge = nn.Conv1d(width * 2, width, kernel_size=1)
        self.block = TemporalFNOBlock(width, modes, act_func=act_func, dropout=dropout)
        self.norm = ChannelLayerNorm1d(width)
        self.act = get_activation(act_func)

    @staticmethod
    def _match_time(x, ref):

        dt = ref.shape[-1] - x.shape[-1]
        if dt == 0:
            return x
        if dt > 0:
            left = dt // 2
            right = dt - left
            return F.pad(x, [left, right])

        dt = -dt
        left = dt // 2
        right = dt - left
        if right == 0:
            return x[..., left:]
        return x[..., left:-right]

    def forward(self, x, skip):
        x = self.up(x)
        x = self._match_time(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.merge(x)))
        return self.block(x)



class TemporalUFNODecoder(nn.Module):

    def __init__(
        self,
        latent_dim,
        temporal_width=128,
        n_time=34,
        nout=1,
        temporal_modes=12,
        temporal_pad_to=68,
        temporal_depth=2,
        hidden=256,
        dropout=0.0,
        act_func="gelu",
        time_coordinates=None,
        center_padding=True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.temporal_width = int(temporal_width)
        self.n_time = int(n_time)
        self.nout = int(nout)
        self.temporal_pad_to = int(max(temporal_pad_to, n_time))
        self.temporal_depth = int(max(1, temporal_depth))
        self.center_padding = bool(center_padding)
        act = get_activation(act_func)

        if self.center_padding:
            self.pad_left = (self.temporal_pad_to - self.n_time) // 2
        else:
            self.pad_left = 0
        self.pad_right = self.temporal_pad_to - self.n_time - self.pad_left

        if time_coordinates is None:
            output_t = torch.linspace(0.0, 1.0, self.n_time)
        else:
            output_t = torch.as_tensor(
                time_coordinates, dtype=torch.float32
            ).reshape(-1)
            if output_t.numel() != self.n_time:
                raise ValueError(
                    f"time_coordinates has {output_t.numel()} entries but "
                    f"n_time={self.n_time}"
                )
        if self.n_time > 1 and torch.any(output_t[1:] <= output_t[:-1]):
            raise ValueError('time_coordinates must be strictly increasing.')
        output_t = torch.clamp(output_t, 0.0, 1.0)
        uniform_t = torch.linspace(0.0, 1.0, self.n_time)
        self.output_is_uniform = bool(torch.allclose(
            output_t, uniform_t, rtol=1.0e-5, atol=1.0e-6
        ))



        coordinate_features = torch.stack([
            uniform_t, torch.ones_like(uniform_t)
        ], dim=-1).unsqueeze(0)
        self.register_buffer("time_features", coordinate_features)
        self.register_buffer("output_time_coordinates", output_t)


        if self.n_time <= 1:
            reflect_indices = torch.zeros(
                self.temporal_pad_to, dtype=torch.long
            )
        else:
            raw_idx = torch.arange(
                -self.pad_left, self.n_time + self.pad_right,
                dtype=torch.long,
            )
            period = 2 * (self.n_time - 1)
            reflected = torch.remainder(raw_idx, period)
            reflect_indices = torch.where(
                reflected < self.n_time,
                reflected,
                period - reflected,
            )
        self.register_buffer("reflect_indices", reflect_indices)

        self.latent_to_width = nn.Sequential(
            nn.Linear(self.latent_dim, hidden), act,
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.temporal_width),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(2, self.temporal_width), act,
            nn.Linear(self.temporal_width, self.temporal_width),
        )
        self.input_block = TemporalFNOBlock(
            self.temporal_width, temporal_modes, act_func=act_func, dropout=dropout
        )
        self.downs = nn.ModuleList([
            TemporalDownBlock(
                self.temporal_width, temporal_modes, act_func=act_func, dropout=dropout
            ) for _ in range(self.temporal_depth)
        ])
        self.bottleneck = nn.Sequential(*[
            TemporalFNOBlock(
                self.temporal_width, temporal_modes, act_func=act_func, dropout=dropout
            ) for _ in range(2)
        ])
        self.ups = nn.ModuleList([
            TemporalUpBlock(
                self.temporal_width, temporal_modes, act_func=act_func, dropout=dropout
            ) for _ in range(self.temporal_depth)
        ])
        self.out = nn.Sequential(
            nn.Conv1d(self.temporal_width, self.temporal_width, kernel_size=1),
            act,
            nn.Dropout(float(dropout)),
            nn.Conv1d(self.temporal_width, self.nout, kernel_size=1),
        )

    def forward(self, latent):

        b = latent.shape[0]
        base = self.latent_to_width(latent).unsqueeze(-1).expand(
            b, -1, self.n_time
        )
        t_emb = self.time_mlp(
            self.time_features.to(device=latent.device, dtype=latent.dtype)
        )
        t_emb = t_emb.permute(0, 2, 1).expand(b, -1, -1)
        x = base + t_emb
        x = x.index_select(-1, self.reflect_indices)
        x = self.input_block(x)

        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        y = self.out(x)
        y = y[..., self.pad_left:self.pad_left + self.n_time]
        if not self.output_is_uniform:
            sample_x = 2.0 * self.output_time_coordinates.to(
                device=y.device, dtype=torch.float32
            ) - 1.0
            sample_grid = torch.stack([
                sample_x, torch.zeros_like(sample_x)
            ], dim=-1).reshape(1, 1, self.n_time, 2)
            sample_grid = sample_grid.expand(b, -1, -1, -1)


            output_dtype = y.dtype
            device_type = y.device.type if y.device.type in ['cuda', 'cpu'] else 'cuda'
            with torch.amp.autocast(device_type=device_type, enabled=False):
                y = F.grid_sample(
                    y.float().unsqueeze(-2), sample_grid,
                    mode='bilinear', padding_mode='border', align_corners=True,
                ).squeeze(-2)
            y = y.to(dtype=output_dtype)
        return y.permute(0, 2, 1).contiguous()



class HeightInteractionOperator(nn.Module):

    def __init__(
        self,
        latent_dim,
        n_height=7,
        num_heads=4,
        layers=2,
        hidden=512,
        dropout=0.0,
        act_func="gelu",
        height_coordinates=None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_height = int(n_height)
        self.height_embedding = nn.Parameter(
            torch.randn(self.n_height, self.latent_dim) * 0.02
        )
        self.height_coord_mlp = nn.Sequential(
            nn.Linear(1, self.latent_dim), get_activation(act_func),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.blocks = nn.ModuleList()
        for _ in range(int(max(1, layers))):
            self.blocks.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(
                    self.latent_dim, int(num_heads),
                    dropout=float(dropout), batch_first=True
                ),
                "norm1": nn.LayerNorm(self.latent_dim),
                "mlp": nn.Sequential(
                    nn.Linear(self.latent_dim, int(hidden)),
                    get_activation(act_func),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden), self.latent_dim),
                ),
                "norm2": nn.LayerNorm(self.latent_dim),
            }))

        if height_coordinates is None:
            h = torch.linspace(0.0, 1.0, self.n_height)
        else:
            h = torch.as_tensor(height_coordinates, dtype=torch.float32).reshape(-1)
            if h.numel() != self.n_height:
                raise ValueError(
                    f"height_coordinates has {h.numel()} entries but n_height={self.n_height}"
                )
        self.register_buffer("height_grid", h.reshape(1, self.n_height, 1))

    def forward(self, global_latent):
        b = global_latent.shape[0]
        hcoord = self.height_coord_mlp(
            self.height_grid.to(
                device=global_latent.device, dtype=global_latent.dtype
            )
        )
        tokens = global_latent.unsqueeze(1).expand(b, self.n_height, -1)
        tokens = (
            tokens
            + self.height_embedding.unsqueeze(0).to(dtype=global_latent.dtype)
            + hcoord
        )
        for block in self.blocks:
            q = block["norm1"](tokens)
            attn_out, _ = block["attn"](q, q, q, need_weights=False)
            tokens = tokens + attn_out
            tokens = tokens + block["mlp"](block["norm2"](tokens))
        return tokens



class DepthTokenEncoder(nn.Module):

    def __init__(
        self,
        width,
        latent_dim=256,
        depth_bins=16,
        center_windows=(3, 7, 15),
        hidden=384,
        layers=2,
        dropout=0.0,
        act_func='gelu',
        spatial_token_grid=3,
    ):
        super().__init__()
        self.width = int(width)
        self.latent_dim = int(latent_dim)
        self.depth_bins = int(depth_bins)
        self.spatial_token_grid = int(spatial_token_grid)
        if self.spatial_token_grid < 1:
            raise ValueError('spatial_token_grid must be at least 1.')
        self.center_windows = _parse_int_list(center_windows, [3, 7, 15])
        in_dim = self.width * (1 + len(self.center_windows))
        act = get_activation(act_func)

        self.profile_pool = nn.AdaptiveAvgPool1d(self.depth_bins)
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden)), act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, self.latent_dim), act,
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((
            self.spatial_token_grid,
            self.spatial_token_grid,
            self.depth_bins,
        ))
        self.spatial_proj = nn.Sequential(
            nn.LayerNorm(self.width + 3),
            nn.Linear(self.width + 3, int(hidden)), act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.token_type_embedding = nn.Parameter(
            torch.randn(2, self.latent_dim) * 0.02
        )
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'conv': nn.Conv1d(
                    self.latent_dim, self.latent_dim,
                    kernel_size=3, padding=1
                ),
                'norm': nn.GroupNorm(
                    _num_groups(self.latent_dim), self.latent_dim
                ),
            }) for _ in range(int(max(1, layers)))
        ])
        self.act = act
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _adaptive_pool_centers(n_input, n_output, low, high, device, dtype):

        index = torch.arange(n_output, device=device, dtype=torch.float32)
        start = torch.floor(index * float(n_input) / float(n_output))
        end = torch.ceil((index + 1.0) * float(n_input) / float(n_output))
        center = 0.5 * (start + end - 1.0)
        if int(n_input) <= 1:
            coordinate = torch.full_like(center, 0.5 * (float(low) + float(high)))
        else:
            coordinate = float(low) + (
                (float(high) - float(low)) * center / float(n_input - 1)
            )
        return coordinate.to(dtype=dtype)

    @staticmethod
    def _center_profile(feat, window):
        _, nx, ny, _, _ = feat.shape
        cx, cy = nx // 2, ny // 2
        half = int(window) // 2
        xs, xe = max(0, cx - half), min(nx, cx + half + 1)
        ys, ye = max(0, cy - half), min(ny, cy + half + 1)
        return feat[:, xs:xe, ys:ye].mean(dim=(1, 2))

    def forward(self, feat):

        profiles = [feat.mean(dim=(1, 2))]
        for win in self.center_windows:
            profiles.append(self._center_profile(feat, win))
        profile = torch.cat(profiles, dim=-1)
        profile = profile.permute(0, 2, 1)
        profile = self.profile_pool(profile).permute(0, 2, 1)
        profile_tokens = self.proj(profile)
        profile_z = self._adaptive_pool_centers(
            feat.shape[3], self.depth_bins, 0.0, 1.0,
            profile_tokens.device, profile_tokens.dtype,
        ).reshape(1, self.depth_bins, 1)
        profile_tokens = profile_tokens + self.depth_mlp(
            profile_z
        )
        x = profile_tokens.permute(0, 2, 1)
        for block in self.blocks:
            y = self.act(block['norm'](block['conv'](x)))
            x = x + self.dropout(y)
        profile_tokens = x.permute(0, 2, 1).contiguous()

        spatial = self.spatial_pool(
            feat.permute(0, 4, 1, 2, 3).contiguous()
        )
        spatial = spatial.permute(0, 2, 3, 4, 1).reshape(
            feat.shape[0], -1, self.width
        )
        gx = self._adaptive_pool_centers(
            feat.shape[1], self.spatial_token_grid, -1.0, 1.0,
            spatial.device, spatial.dtype,
        )
        gy = self._adaptive_pool_centers(
            feat.shape[2], self.spatial_token_grid, -1.0, 1.0,
            spatial.device, spatial.dtype,
        )
        gz = self._adaptive_pool_centers(
            feat.shape[3], self.depth_bins, 0.0, 1.0,
            spatial.device, spatial.dtype,
        )
        cx, cy, cz = torch.meshgrid(gx, gy, gz, indexing='ij')
        coords = torch.stack([cx, cy, cz], dim=-1).reshape(
            1, -1, 3
        ).expand(feat.shape[0], -1, -1)
        spatial_tokens = self.spatial_proj(torch.cat([spatial, coords], dim=-1))

        type_embedding = self.token_type_embedding.to(dtype=profile_tokens.dtype)
        profile_tokens = profile_tokens + type_embedding[0].reshape(1, 1, -1)
        spatial_tokens = spatial_tokens + type_embedding[1].reshape(1, 1, -1)
        return torch.cat([
            profile_tokens, spatial_tokens
        ], dim=1).contiguous()


class HeightTimeResidualCorrector(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        n_height=7,
        n_time=34,
        nout=1,
        num_heads=4,
        hidden=384,
        temporal_modes=10,
        temporal_layers=2,
        height_layers=1,
        dropout=0.0,
        act_func='gelu',
        time_coordinates=None,
        height_coordinates=None,
        anchor_gates=3,
        init_scale=0.05,
        temporal_operator='attention',
        temporal_attention_heads=None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_height = int(n_height)
        self.n_time = int(n_time)
        self.nout = int(nout)
        self.anchor_gates = max(1, min(int(anchor_gates), self.n_time))
        self.temporal_operator = str(temporal_operator).lower()
        act = get_activation(act_func)

        if time_coordinates is None:
            t = torch.linspace(0.0, 1.0, self.n_time)
        else:
            t = torch.as_tensor(time_coordinates, dtype=torch.float32).reshape(-1)
        if height_coordinates is None:
            h = torch.linspace(0.0, 1.0, self.n_height)
        else:
            h = torch.as_tensor(height_coordinates, dtype=torch.float32).reshape(-1)
        if t.numel() != self.n_time or h.numel() != self.n_height:
            raise ValueError('Residual corrector coordinate length mismatch.')
        self.register_buffer('time_grid', t.reshape(1, 1, self.n_time, 1))
        self.register_buffer('height_grid', h.reshape(1, self.n_height, 1, 1))

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.latent_dim), act,
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.height_mlp = nn.Sequential(
            nn.Linear(1, self.latent_dim), act,
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.base_mlp = nn.Sequential(
            nn.Linear(2 * self.nout, self.latent_dim), act,
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.query_norm = nn.LayerNorm(self.latent_dim)
        self.depth_norm = nn.LayerNorm(self.latent_dim)
        self.cross_attn = nn.MultiheadAttention(
            self.latent_dim, int(num_heads),
            dropout=float(dropout), batch_first=True
        )
        self.cross_ffn_norm = nn.LayerNorm(self.latent_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden)), act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.latent_dim),
        )

        temporal_layers = int(max(1, temporal_layers))
        if self.temporal_operator in ['attention', 'attn', 'transformer']:
            temporal_heads = int(
                num_heads if temporal_attention_heads is None
                else temporal_attention_heads
            )
            self.temporal_blocks = nn.ModuleList([
                TemporalAttentionBlock(
                    self.latent_dim,
                    num_heads=temporal_heads,
                    hidden=int(hidden),
                    act_func=act_func,
                    dropout=dropout,
                ) for _ in range(temporal_layers)
            ])
        elif self.temporal_operator in ['spectral', 'fno', 'fourier']:
            self.temporal_blocks = nn.ModuleList([
                TemporalFNOBlock(
                    self.latent_dim, int(temporal_modes),
                    act_func=act_func, dropout=dropout
                ) for _ in range(temporal_layers)
            ])
        else:
            raise ValueError(
                "temporal_operator must be 'attention' or 'spectral'."
            )
        self.height_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(self.latent_dim),
                'attn': nn.MultiheadAttention(
                    self.latent_dim, int(num_heads),
                    dropout=float(dropout), batch_first=True
                ),
                'ffn_norm': nn.LayerNorm(self.latent_dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.latent_dim, int(hidden)), act,
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden), self.latent_dim),
                ),
            }) for _ in range(int(max(0, height_layers)))
        ])

        self.shape_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(hidden)), act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.nout),
        )
        self.amplitude_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(hidden)), act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.nout),
        )
        self.shape_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.amplitude_scale = nn.Parameter(
            torch.tensor(float(init_scale) * 0.5)
        )

    def forward(self, base_y, height_tokens, depth_tokens, return_aux=False):

        b, h, t, _ = base_y.shape
        slope = torch.zeros_like(base_y)
        slope[:, :, 1:] = base_y[:, :, 1:] - base_y[:, :, :-1]
        base_features = torch.cat([base_y, slope], dim=-1)

        t_emb = self.time_mlp(
            self.time_grid.to(device=base_y.device, dtype=base_y.dtype)
        )
        h_emb = self.height_mlp(
            self.height_grid.to(device=base_y.device, dtype=base_y.dtype)
        )
        q = (
            height_tokens.unsqueeze(2)
            + t_emb
            + h_emb
            + self.base_mlp(base_features)
        )
        q_flat = q.reshape(b, h * t, self.latent_dim)
        qn = self.query_norm(q_flat)
        dn = self.depth_norm(depth_tokens)
        cross, _ = self.cross_attn(qn, dn, dn, need_weights=False)
        x = q_flat + cross
        x = x + self.cross_ffn(self.cross_ffn_norm(x))
        x = x.reshape(b, h, t, self.latent_dim)


        xt = x.reshape(b * h, t, self.latent_dim).permute(0, 2, 1)
        for block in self.temporal_blocks:
            xt = block(xt)
        x = xt.permute(0, 2, 1).reshape(b, h, t, self.latent_dim)


        for block in self.height_blocks:
            xh = x.permute(0, 2, 1, 3).reshape(b * t, h, self.latent_dim)
            xn = block['norm'](xh)
            attn, _ = block['attn'](xn, xn, xn, need_weights=False)
            xh = xh + attn
            xh = xh + block['ffn'](block['ffn_norm'](xh))
            x = xh.reshape(b, t, h, self.latent_dim).permute(0, 2, 1, 3)

        raw_shape = self.shape_head(x)
        anchor = raw_shape[:, :, :self.anchor_gates].mean(
            dim=2, keepdim=True
        )
        shape_residual = raw_shape - anchor
        amplitude_residual = self.amplitude_head(x.mean(dim=2)).unsqueeze(2)

        corrected = (
            base_y
            + self.shape_scale * shape_residual
            + self.amplitude_scale * amplitude_residual
        )
        if return_aux:
            return corrected, {
                'base_prediction': base_y,
                'shape_residual': shape_residual,
                'amplitude_residual': amplitude_residual,
                'shape_scale': self.shape_scale,
                'amplitude_scale': self.amplitude_scale,
            }
        return corrected


class HeightCurveNormalizer:

    def __init__(self, x, n_height, n_time, eps=1.0e-6):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        self.n_height = int(n_height)
        self.n_time = int(n_time)
        self.eps = float(eps)
        if x.ndim != 4 or x.shape[1] != self.n_height * self.n_time:
            raise ValueError(
                f"HeightCurveNormalizer expects [N,H*T,1,nout], got {tuple(x.shape)} "
                f"for H={self.n_height}, T={self.n_time}."
            )
        nout = int(x.shape[-1])
        xht = x.reshape(x.shape[0], self.n_height, self.n_time, 1, nout)
        self.mean_curve = xht.mean(dim=0, keepdim=True)
        residual = xht - self.mean_curve

        var = torch.mean(residual * residual, dim=(0, 2), keepdim=True)
        self.std_height = torch.sqrt(var + self.eps)

    def encode(self, x):
        shape = x.shape
        xht = x.reshape(shape[0], self.n_height, self.n_time, 1, shape[-1])
        z = (xht - self.mean_curve.to(device=x.device, dtype=x.dtype)) / \
            self.std_height.to(device=x.device, dtype=x.dtype)
        return z.reshape(shape)

    def decode(self, z):
        shape = z.shape
        zht = z.reshape(shape[0], self.n_height, self.n_time, 1, shape[-1])
        x = zht * self.std_height.to(device=z.device, dtype=z.dtype) + \
            self.mean_curve.to(device=z.device, dtype=z.dtype)
        return x.reshape(shape)

    def to(self, device):
        self.mean_curve = self.mean_curve.to(device)
        self.std_height = self.std_height.to(device)
        return self

    def cuda(self, device=None):
        return self.to(torch.device('cuda' if device is None else device))

    def cpu(self):
        return self.to(torch.device('cpu'))



class AmplitudeShapeTemporalDecoder(nn.Module):

    def __init__(
        self,
        latent_dim,
        temporal_width=128,
        n_time=34,
        nout=1,
        temporal_modes=12,
        temporal_pad_to=68,
        temporal_depth=2,
        hidden=256,
        dropout=0.0,
        act_func='gelu',
        parameterization='amplitude_shape',
        anchor_gates=3,
        time_coordinates=None,
        center_padding=True,
    ):
        super().__init__()
        self.n_time = int(n_time)
        self.nout = int(nout)
        self.parameterization = str(parameterization).lower()
        self.anchor_gates = max(1, min(int(anchor_gates), self.n_time))
        act = get_activation(act_func)
        self.amplitude_head = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden)), act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.nout),
        )
        self.temporal_head = TemporalUFNODecoder(
            latent_dim=latent_dim,
            temporal_width=temporal_width,
            n_time=n_time,
            nout=nout,
            temporal_modes=temporal_modes,
            temporal_pad_to=temporal_pad_to,
            temporal_depth=temporal_depth,
            hidden=hidden,
            dropout=dropout,
            act_func=act_func,
            time_coordinates=time_coordinates,
            center_padding=center_padding,
        )

    def forward(self, latent, return_aux=False):
        amp = self.amplitude_head(latent).unsqueeze(1)
        raw = self.temporal_head(latent)

        if self.parameterization in ['initial_increment', 'increment', 'increments']:
            delta = raw[:, 1:, :]
            zero = torch.zeros_like(amp)
            shape = torch.cumsum(torch.cat([zero, delta], dim=1), dim=1)
            y = amp + shape
        elif self.parameterization in ['amplitude_shape', 'shape', 'residual']:
            anchor = raw[:, :self.anchor_gates, :].mean(dim=1, keepdim=True)
            shape = raw - anchor
            y = amp + shape
        else:
            raise ValueError(
                f"Unsupported output_parameterization={self.parameterization!r}. "
                "Use 'amplitude_shape' or 'initial_increment'."
            )

        if return_aux:
            return y, {
                'amplitude': amp,
                'shape': shape,
                'raw_temporal': raw,
            }
        return y



class HTUFNO_AEM_v4_Hybrid(nn.Module):

    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width,
        nout,
        n_height=7,
        n_time=34,
        init_func='xavier_normal',
        layer_fno=3,
        layer_ufno=3,
        act_func='gelu',
        padding=8,
        use_kan_lift=True,
        use_coord_channels=True,
        use_wavelet_branch=True,
        wavelet_layers=3,
        wavelet_hidden_multiplier=1.0,
        wavelet_scale_init=1.0,
        wavelet_dropout=0.0,
        latent_dim=256,
        summary_hidden=512,
        summary_dropout=0.0,
        summary_global_grid=3,
        summary_global_z_bins=8,
        summary_center_windows=(3, 7, 15),
        summary_center_z_bins=8,
        use_height_interaction=True,
        height_interaction_layers=2,
        height_interaction_heads=4,
        height_interaction_hidden=512,
        temporal_width=128,
        temporal_modes=12,
        temporal_pad_to=68,
        temporal_depth=2,
        temporal_hidden=256,
        temporal_dropout=0.0,
        height_specific=False,
        output_parameterization='amplitude_shape',
        amplitude_anchor_gates=3,
        time_values=None,
        height_values=None,
        temporal_time_transform='log10',
        temporal_center_padding=True,
        use_residual_corrector=True,
        depth_token_bins=16,
        depth_token_windows=(3, 7, 15),
        depth_token_hidden=384,
        depth_token_layers=2,
        residual_heads=4,
        residual_hidden=384,
        residual_temporal_modes=10,
        residual_temporal_layers=2,
        residual_height_layers=1,
        residual_dropout=0.0,
        residual_init_scale=0.05,
        modes_xyz=None,
        spatial_padding=None,
        spatial_token_grid=3,
        residual_temporal_operator='attention',
        residual_temporal_attention_heads=None,
    ):
        super().__init__()
        self.width = int(width)
        self.padding = padding
        self.spatial_padding = _resolve_spatial_padding(
            spatial_padding=spatial_padding, padding=padding
        )
        self.spatial_pad_pairs = _centered_pad_pairs(self.spatial_padding)
        self.modes_xyz = _resolve_modes_xyz(
            modes_xyz, modes1, modes2, modes3
        )
        self.nout = int(nout)
        self.n_height = int(n_height)
        self.n_time = int(n_time)
        self.height_specific = bool(height_specific)
        self.use_coord_channels = bool(use_coord_channels)
        self.use_height_interaction = bool(use_height_interaction)
        self.use_residual_corrector = bool(use_residual_corrector)
        self.use_wavelet_branch = bool(use_wavelet_branch)
        self.requested_kan_lift = bool(use_kan_lift)
        self.using_kan_lift = self.requested_kan_lift and KAN is not None
        if self.requested_kan_lift and not self.using_kan_lift:
            warnings.warn(
                "use_kan_lift=True but `from kan import KAN` failed; the model "
                "is using the MLP fallback. Install the same KAN implementation "
                "as the original experiment or set use_kan_lift=false so the "
                "experiment is labeled correctly.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.spatial_token_grid = int(spatial_token_grid)
        self.residual_temporal_operator = str(
            residual_temporal_operator
        ).lower()
        if init_func not in init_dict:
            raise KeyError(f"init name {init_func!r} is not in init_dict")
        self.initializer = init_dict[init_func]

        time_coordinates = _normalized_coordinate(
            time_values, transform=temporal_time_transform
        )
        if time_coordinates is None:
            time_coordinates = torch.linspace(0.0, 1.0, self.n_time)
        height_coordinates = _normalized_coordinate(
            height_values, transform='linear'
        )
        if height_coordinates is None:
            height_coordinates = torch.linspace(0.0, 1.0, self.n_height)

        in_channels = 1 + (3 if self.use_coord_channels else 0)
        self.branch = BranchLift(
            width=width, padding=padding,
            use_kan_lift=use_kan_lift, act_func=act_func,
            spatial_padding=self.spatial_padding,
        )
        if in_channels != 1:
            act = get_activation(act_func)
            if bool(use_kan_lift):
                self.branch.lift = PointwiseKAN(
                    [in_channels, 64, 128, width], act_func=act_func
                )
            else:
                self.branch.lift = nn.Sequential(
                    nn.Linear(in_channels, 64), act,
                    nn.Linear(64, 128), act,
                    nn.Linear(128, width),
                )

        self.ufno = UFNO3d(
            modes1, modes2, modes3, width,
            layer_fno, layer_ufno, act_func,
            use_wavelet_branch=use_wavelet_branch,
            wavelet_layers=wavelet_layers,
            wavelet_hidden_multiplier=wavelet_hidden_multiplier,
            wavelet_scale_init=wavelet_scale_init,
            wavelet_dropout=wavelet_dropout,
            modes_xyz=self.modes_xyz,
        )
        self.summary = MultiScaleFeatureSummary(
            width=width,
            latent_dim=latent_dim,
            global_grid=summary_global_grid,
            global_z_bins=summary_global_z_bins,
            center_windows=summary_center_windows,
            center_z_bins=summary_center_z_bins,
            hidden=summary_hidden,
            dropout=summary_dropout,
            act_func=act_func,
        )

        if self.use_height_interaction:
            self.height_operator = HeightInteractionOperator(
                latent_dim=latent_dim,
                n_height=n_height,
                num_heads=height_interaction_heads,
                layers=height_interaction_layers,
                hidden=height_interaction_hidden,
                dropout=summary_dropout,
                act_func=act_func,
                height_coordinates=height_coordinates,
            )
        else:
            self.height_embeddings = nn.Parameter(
                torch.randn(self.n_height, latent_dim) * 0.02
            )

        n_decoders = self.n_height if self.height_specific else 1
        self.decoders = nn.ModuleList([
            AmplitudeShapeTemporalDecoder(
                latent_dim=latent_dim,
                temporal_width=temporal_width,
                n_time=n_time,
                nout=nout,
                temporal_modes=temporal_modes,
                temporal_pad_to=temporal_pad_to,
                temporal_depth=temporal_depth,
                hidden=temporal_hidden,
                dropout=temporal_dropout,
                act_func=act_func,
                parameterization=output_parameterization,
                anchor_gates=amplitude_anchor_gates,
                time_coordinates=time_coordinates,
                center_padding=temporal_center_padding,
            ) for _ in range(n_decoders)
        ])

        if self.use_residual_corrector:
            self.depth_encoder = DepthTokenEncoder(
                width=width,
                latent_dim=latent_dim,
                depth_bins=depth_token_bins,
                center_windows=depth_token_windows,
                hidden=depth_token_hidden,
                layers=depth_token_layers,
                dropout=residual_dropout,
                act_func=act_func,
                spatial_token_grid=spatial_token_grid,
            )
            self.residual_corrector = HeightTimeResidualCorrector(
                latent_dim=latent_dim,
                n_height=n_height,
                n_time=n_time,
                nout=nout,
                num_heads=residual_heads,
                hidden=residual_hidden,
                temporal_modes=residual_temporal_modes,
                temporal_layers=residual_temporal_layers,
                height_layers=residual_height_layers,
                dropout=residual_dropout,
                act_func=act_func,
                time_coordinates=time_coordinates,
                height_coordinates=height_coordinates,
                anchor_gates=amplitude_anchor_gates,
                init_scale=residual_init_scale,
                temporal_operator=residual_temporal_operator,
                temporal_attention_heads=residual_temporal_attention_heads,
            )

        self._init_layers()


        self.ufno.reset_wavelet_zero_init()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.initializer(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(
                m,
                (nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose3d)
            ):
                if m.weight is not None:
                    nn.init.kaiming_normal_(
                        m.weight, nonlinearity='leaky_relu'
                    )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _coordinate_channels(x):
        b, nx, ny, nz, _ = x.shape
        device, dtype = x.device, x.dtype
        gx = torch.linspace(
            -1.0, 1.0, nx, device=device, dtype=dtype
        ).view(1, nx, 1, 1, 1).expand(b, nx, ny, nz, 1)
        gy = torch.linspace(
            -1.0, 1.0, ny, device=device, dtype=dtype
        ).view(1, 1, ny, 1, 1).expand(b, nx, ny, nz, 1)
        gz = torch.linspace(
            0.0, 1.0, nz, device=device, dtype=dtype
        ).view(1, 1, 1, nz, 1).expand(b, nx, ny, nz, 1)
        return torch.cat([gx, gy, gz], dim=-1)

    def encode_model(self, x):
        nx, ny, nz = x.shape[1:4]
        if self.use_coord_channels:
            x = torch.cat([x, self._coordinate_channels(x)], dim=-1)
        feat = self.branch(x)
        feat = self.ufno(feat)
        (px0, _), (py0, _), (pz0, _) = self.spatial_pad_pairs
        feat = feat[
            :, px0:px0 + nx, py0:py0 + ny, pz0:pz0 + nz, :
        ]
        return feat

    def _height_tokens(self, feat):
        global_latent = self.summary(feat)
        if self.use_height_interaction:
            return self.height_operator(global_latent)
        return (
            global_latent.unsqueeze(1).expand(-1, self.n_height, -1)
            + self.height_embeddings.unsqueeze(0)
        )

    def _base_decode(self, h_tokens, return_aux=False):
        aux_all = []
        if self.height_specific:
            ys = []
            for ih in range(self.n_height):
                if return_aux:
                    y_h, aux_h = self.decoders[ih](
                        h_tokens[:, ih, :], return_aux=True
                    )
                    aux_all.append(aux_h)
                else:
                    y_h = self.decoders[ih](
                        h_tokens[:, ih, :], return_aux=False
                    )
                ys.append(y_h)
            y = torch.stack(ys, dim=1)
            if return_aux:
                aux_all = {
                    key: torch.stack(
                        [height_aux[key] for height_aux in aux_all], dim=1
                    )
                    for key in aux_all[0]
                }
        else:
            b, h, d = h_tokens.shape
            if return_aux:
                y_flat, aux_flat = self.decoders[0](
                    h_tokens.reshape(b * h, d), return_aux=True
                )
                y = y_flat.reshape(b, h, self.n_time, self.nout)
                aux_all = {
                    k: v.reshape(b, h, *v.shape[1:])
                    for k, v in aux_flat.items()
                }
            else:
                y = self.decoders[0](
                    h_tokens.reshape(b * h, d), return_aux=False
                )
                y = y.reshape(b, h, self.n_time, self.nout)
        return y, aux_all

    def forward_from_features(self, feat, return_aux=False):
        h_tokens = self._height_tokens(feat)
        base_y, base_aux = self._base_decode(
            h_tokens, return_aux=return_aux
        )

        residual_aux = None
        if self.use_residual_corrector:
            depth_tokens = self.depth_encoder(feat)
            if return_aux:
                y, residual_aux = self.residual_corrector(
                    base_y, h_tokens, depth_tokens, return_aux=True
                )
            else:
                y = self.residual_corrector(
                    base_y, h_tokens, depth_tokens, return_aux=False
                )
        else:
            y = base_y

        out = y.reshape(
            y.shape[0], self.n_height * self.n_time, 1, self.nout
        )
        if return_aux:
            return out, {
                'base': base_aux,
                'residual': residual_aux,
            }
        return out

    def forward(self, x, query=None, return_aux=False, *args, **kwargs):
        feat = self.encode_model(x)
        return self.forward_from_features(feat, return_aux=return_aux)

    def set_spatial_encoder_trainable(self, trainable=True):
        for module in [self.branch, self.ufno]:
            for p in module.parameters():
                p.requires_grad = bool(trainable)

    def encoder_parameters(self):
        for module in [self.branch, self.ufno]:
            yield from module.parameters()

    def residual_parameters(self):
        if self.use_residual_corrector:
            yield from self.depth_encoder.parameters()
            yield from self.residual_corrector.parameters()

    def base_decoder_parameters(self):
        excluded = {id(p) for p in self.encoder_parameters()}
        excluded.update(id(p) for p in self.residual_parameters())
        for p in self.parameters():
            if id(p) not in excluded:
                yield p


HTUFNO_AEM_v4 = HTUFNO_AEM_v4_Hybrid
HTUFNO_AEM_v4_Fixed = HTUFNO_AEM_v4_Hybrid
HTUFNO_AEM_v3_Fixed = HTUFNO_AEM_v4_Hybrid
HTUFNO_AEM_Fixed = HTUFNO_AEM_v4_Hybrid
HDF_MFHT_Attn = HTUFNO_AEM_v4_Hybrid
HDF = HTUFNO_AEM_v4_Hybrid


class HTUFNO_AEM_v5_Hybrid(HTUFNO_AEM_v4_Hybrid):
    pass


HTUFNO_AEM_v5_Corrected = HTUFNO_AEM_v5_Hybrid
HTUFNO_AEM_v5_corrected = HTUFNO_AEM_v5_Hybrid
HTUFNO_AEM_v5 = HTUFNO_AEM_v5_Hybrid
