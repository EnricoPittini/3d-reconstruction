import math
from inspect import isfunction
from functools import partial
from einops import rearrange

import torch
from torch import nn, einsum


##### AUXILIARY FUNCTIONS

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



##### UTILITY FUNCTIONS/CLASSES FOR BUILDING THE MODEL

class Residual(nn.Module):
    """Module for adding a skip connection to the arbitrary given function

    Parameters
    ----------
    fn : function
        The function to wrap with a skip connection
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Upsample(nn.Module):
    """Module for upsampling the image tensor

    Parameters
    ----------
    dim : int
        Number of channels (input/output)
    device : str
    """
    def __init__(self, dim, device='cpu'):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1, device=device)

    def forward(self, x):
        return self.convt(x)


class Downsample(nn.Module):
    """Module for downsampling the image tensor

    Parameters
    ----------
    dim : int
        Number of channels (input/output)
    device : str
    """
    def __init__(self, dim, device='cpu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1, device=device)

    def forward(self, x):
        return self.conv(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """Module for creating the embedding of the timestep.

    The embedding has dimensions (b, c), where 'b' is the batch size and 'c' is the number of channels (i.e. `dim`).

    Parameters
    ----------
    dim : int
        Number of dimensions/channels of the embedding
    device : str
    """
    def __init__(self, dim, device='cpu'):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Block which consists in a convolution followed by Group normalization and SiLU.

    The number of channels changes, while the spatial dimensions stay unchanced.

    This block is used as component for other bigger blocks: `ResnetBlock` and `ConvNextBlock`.

    Parameters
    ----------
    dim_in : int
        Number of input channels
    dim_out : int
        Number of output channels
    groups : int
        Number of groups for the Group normalization
    device : str
    """
    def __init__(self, dim_in, dim_out, groups=8, device='cpu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, padding=1, device=device)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim_out, device=device)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Resnet block. It consists in applying two times the block `Block` and then applying a final convolution, with a skip 
    connection. https://arxiv.org/abs/1512.03385

    The number of channels changes, while the spatial dimensions stay unchanced.

    This block takes in input both the image tensor `x` and the emnbedded timestep `t`, appropriately combining them (sum).

    The blocks `ResnetBlock` and `ConvNextBlock` are used in alternative.

    Parameters
    ----------
    dim_in : int
        Number of input channels
    dim_out : int
        Number of output channels
    time_emb_dim : int
        Number of channels/dimensions of the timestep embedding
    groups : int
        Number of groups for the Group normalization
    device : str
    """    
    def __init__(self, dim_in, dim_out, *, groups=8, device='cpu'):
        super().__init__()

        self.block1 = Block(dim_in, dim_out, groups=groups, device=device)
        self.block2 = Block(dim_out, dim_out, groups=groups, device=device)
        self.res_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1, device=device) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)  # First block

        h = self.block2(h)  # Second block
        return h + self.res_conv(x)  # Skip connection


class ConvNextBlock(nn.Module):
    """ConvNextBlock block. https://arxiv.org/abs/2201.03545

        The number of channels changes, while the spatial dimensions stay unchanced.

        This block takes in input both the image tensor `x` and the emnbedded timestep `t`, appropriately combining them (sum).

        The blocks `ResnetBlock` and `ConvNextBlock` are used in alternative.

        Parameters
        ----------
        dim_in : int
            Number of input channels
        dim_out : int
            Number of output channels
        time_emb_dim : int
            Number of channels/dimensions of the timestep embedding
        mult : int
            Number of groups for the Group normalization
        norm : bool
        device : str
    """  

    def __init__(self, dim, dim_out, *, mult=2, norm=True, device='cpu'):
        super().__init__()

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, device='cpu')

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim, device=device) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1, device=device),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult, device=device),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1, device=device),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1, device=device) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.ds_conv(x)

        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """Block implementing the regular multi-head self-attention (as used in the Transformer).

    The blocks `Attention` and `LinearAttention` are used in alternative.

    Parameters
    ----------
    dim : int
        Number of channels
    heads : int
        Number of attention heads
    dim_heads : int
        Number of dimensions of each query-value-key
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """Block implementing the linear attention variant, whose time and memory requirements scale linear in the sequence 
    length, as opposed to quadratic for regular attention.

    The blocks `Attention` and `LinearAttention` are used in alternative.

    Parameters
    ----------
    dim : int
        Number of channels
    heads : int
        Number of attention heads
    dim_heads : int
        Number of dimensions of each query-value-key
    device : str
    """
    def __init__(self, dim, heads=4, dim_head=32, device='cpu'):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False, device=device)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1, device=device), 
                                    nn.GroupNorm(1, dim, device=device))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    """Block for applying Group normalization before the attention

    Parameters
    ----------
    dim : int
        Number of channels
    fn : function
        Function to apply after the normalization
    """
    def __init__(self, dim, fn, device='cpu'):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim, device=device)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)



##### MODEL

class HuggingFaceModel(nn.Module):
    """UNet diffusion model with ResNet blocks (or ConvNext blocks), positional encoding for the timestep and linear attention.
    The downsampling is done with strided convolution and the upsampling is done by Transpose Convolution.
    Group Normalization and SiLU activation function are used.

    This model has been taken from https://huggingface.co/blog/annotated-diffusion.
    The only difference is that we 

    Similar to the UNet Bottleneck residual, but a different implementation with linear attention is used is used (inspired 
    from Huggingface).

    We always use LinearAttention instead of the quadratic Attention. The original implementation uses quadratic attention in
    the intermediate ResNet part.

    Parameters
    ----------
    dim : int
        Input spatial dimension. It is assumed that height and width are equal.
        This information is used for determining several dimensions, like the number of channels of the first UNet stage.
    device : str
    init_dim : int, optional
        Number of channels of the first UNet stage, by default None.
        If not given, it is computed as $dim // 3 * 2$
    out_dim : int, optional
        Number of output channels.
    dim_mults : list of int, optional
        Multiplicative factors used for determining the output number of channels after each UNet stage.
        By default [1, 2, 4, 8].
    in_channels : int, optional
        Number of input channels, by default 3
    with_time_emb : bool, optional
        Whether to use the timestep information or not, by default True
    resnet_block_groups : int, optional
        Number of groups for the GroupNormalization applied in the Resnet blocks, by default 8
    use_convnext : bool optional
        Whether to use ConvNext blocks instead of ResNet blocks, by default True
    convnext_mult : int, optional
        Number of mult for the ConvNext blocks, by default 2
    """
    def __init__(
        self,
        img_size=(480,640),
        device='cpu',
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        in_channels=3,
        n_stem_layers=3,
        resnet_block_groups=6, # 8
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        self.device = device

        self.in_channels = in_channels

        dim = min(img_size)

        # Determine the number of channels for the first UNet stage
        # init_dim = default(init_dim, dim // 3 * 2)
        init_dim = default(init_dim, dim // 40)  # 8, 16, 32, 40
        self.init_conv_list = nn.ModuleList([])
        for i in range(n_stem_layers):
            if i==0:
                self.init_conv_list.append(nn.Conv2d(in_channels, init_dim, 7, padding=3, stride=2, device=device))
            else:
                self.init_conv_list.append(nn.Conv2d(i*init_dim, (i+1)*init_dim, 7, padding=3, stride=2, device=device))
        init_dim = n_stem_layers*init_dim
        """self.init_conv = nn.Conv2d(in_channels, init_dim, 7, padding=3, stride=2, device=device)  # Added stride 2
        self.init_conv1 = nn.Conv2d(init_dim, 2*init_dim, 7, padding=3, stride=2, device=device)  # Added another stride 2
        self.init_conv2 = nn.Conv2d(2*init_dim, 3*init_dim, 7, padding=3, stride=2, device=device)  # Added another stride 2
        init_dim = 3*init_dim"""
        """init_dim = default(init_dim, dim // 8)
        self.init_conv1 = nn.Conv2d(in_channels, init_dim, 3, padding=1, stride=2, device=device)
        self.init_conv2 = nn.Conv2d(init_dim, 2*init_dim, 3, padding=1, stride=2, device=device)
        #self.init_conv3 = nn.Conv2d(2*in_channels, 3*init_dim, 3, padding=1, stride=2, device=device)
        init_dim = 2*init_dim"""

        # List with the number of channels for each UNet stage
        # dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        # List of tuples containing the input number of channels and the output number of channels for each UNet stage
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Basic building block used in each UNet stage. Either Resnet block or Convnext block.
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult, device=device)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups, device=device)

        """if with_time_emb:  
            # The timestep information is used
            time_dim = dim*4  # Dimension of the timestep embedding
            # Multi-layer perceptron 
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim, device=device),
                nn.Linear(dim, time_dim, device=device),
                nn.GELU(),
                nn.Linear(time_dim, time_dim, device=device),
            )
        else:
            time_dim = None
            self.time_mlp = None"""

        # List of down UNet stages
        # List of up UNet stages
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Down UNet stages
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # A down stage consists of:
            #       - two blocks (either ResNet or ConvNext)
            #       - LinearAttention with Groupnormalization (before the attention) and skip connection
            #       - final downsampling

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out),
                        block_klass(dim_out, dim_out),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out, device=device), device=device)),
                        Downsample(dim_out, device=device) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Central UNet stage.
        # It consists of:
        #       - block (either ResNet or ConvNext)
        #       - LinearAttention with Groupnormalization (before the attention) and skip connection
        #       - another block 
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        # TODO : remember this change
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, device=device), device=device)) #Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        # Up UNet stages
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            # An up stage consists of:
            #       - two blocks (either ResNet or ConvNext)
            #       - LinearAttention with Groupnormalization (before the attention) and skip connection
            #       - final upsampling

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in),
                        block_klass(dim_in, dim_in),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in, device=device), device=device)),
                        Upsample(dim_in, device=device) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Number of output channels
        out_dim = default(out_dim, 1)
        # Final convolution
        self.final_conv_list = nn.ModuleList([])
        self.final_conv_list.append(block_klass(init_dim, init_dim))
        for i in range(n_stem_layers): 
            if i==n_stem_layers-1:
                self.final_conv_list.append(nn.ConvTranspose2d(in_channels=init_dim//n_stem_layers, out_channels=out_dim, kernel_size=2, padding=0, stride=2, device=device))
            else:
                self.final_conv_list.append(nn.ConvTranspose2d(in_channels=(n_stem_layers-i)*init_dim//n_stem_layers, out_channels=(n_stem_layers-i-1)*init_dim//n_stem_layers, kernel_size=2, padding=0, stride=2, device=device))

        """self.final_conv = nn.Sequential(
            # block_klass(dim, dim), nn.Conv2d(init_dim, out_dim, 1, device=device)                              
            block_klass(init_dim, init_dim),
            nn.ConvTranspose2d(in_channels=init_dim, out_channels=2*init_dim//3, kernel_size=2, padding=0, stride=2, device=device),  # Add another transposed
            nn.ConvTranspose2d(in_channels=2*init_dim//3, out_channels=init_dim//3, kernel_size=2, padding=0, stride=2, device=device),  # Add another transposed
            nn.ConvTranspose2d(in_channels=init_dim//3, out_channels=out_dim, kernel_size=2, padding=0, stride=2, device=device)  # Add transposed
        )"""

    def forward(self, x):
        # Initial convolution on the image tensor
        for init_conv in self.init_conv_list:
            x = init_conv(x)
        """x = self.init_conv(x)
        x = self.init_conv1(x)
        x = self.init_conv2(x)"""
        """x = self.init_conv1(x)
        x = self.init_conv2(x)
        print(x.shape)"""
        #print(x.shape)

        # List storing the output tensors after each down UNet stage
        h = []

        # Down UNet stages
        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            #print(x.shape)

        # Bottleneck
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        #print(x.shape)

        # Up UNet stages
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)  # Skip connection
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = upsample(x)
            #print(x.shape)

        # Final convolution
        #return self.final_conv(x)
        for final_conv in self.final_conv_list:
            x = final_conv(x)
        return x
