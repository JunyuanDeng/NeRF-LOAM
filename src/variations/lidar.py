import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.embedding_size = mapping_size

    def forward(self, x):
        # x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, in_dim, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs
        self.embedding_size = multires*in_dim*2 + in_dim

    def forward(self, x):
        # x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class Same(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.embedding_size = in_dim

    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self,
                 depth=8,
                 width=258,
                 in_dim=3,
                 sdf_dim=128,
                 skips=[4],
                 multires=6,
                 embedder='none',
                 point_dim=3,
                 local_coord=False,
                 **kwargs) -> None:
        super().__init__()
        self.D = depth
        self.W = width
        self.skips = skips
        self.point_dim = point_dim
        if embedder == 'nerf':
            self.pe = Nerf_positional_embedding(in_dim, multires)
        elif embedder == 'none':
            self.pe = Same(in_dim)
        elif embedder == 'gaussian':
            self.pe = GaussianFourierFeatureTransform(in_dim)
        else:
            raise NotImplementedError("unknown positional encoder")
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.pe.embedding_size, width)] + [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + self.pe.embedding_size, width) for i in range(depth-1)])
        self.sdf_out = nn.Linear(width, 1)

    def get_values(self, input):
        x = self.pe(input)
        # point = input[:, -3:]
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # outputs = self.output_linear(h)
        # outputs[:, :3] = torch.sigmoid(outputs[:, :3])
        sdf_out = self.sdf_out(h)

        return sdf_out

    def forward(self, inputs):
        outputs = self.get_values(inputs)

        return {
            'sdf': outputs,
            # 'depth': outputs[:, 1]
        }
