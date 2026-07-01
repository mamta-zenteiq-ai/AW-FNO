from typing import List, Optional, Tuple, Union
# typing is built-in python standard library, no need to install. For Python 3.9+,
# don't need to import List, Tuple, etc. from typing, can just use list, tuple, etc. directly.
# List: list of items
# Optional: a type that can be either the specified type or None
# Tuple: a fixed-length tuple (e.g., Tuple[int, int] is a tuple of two integers). "tuple" is immutable.
# Union: a type that can be one of several specified types (e.g., Union[int, float] can be either an int or a float)

from ..utils.scaling import validate_scaling_factor
# . means from the current package, .. means from the parent package( go up one level in the package hierarchy).
# Here, scaling.py is in the "utils" package, which is a sibling of the "layers" package where this file is located. 
# We import the validate_scaling_factor function from scaling.py to use it in this file.
# validate_scaling_factor is a function that checks if the provided resolution scaling factor is valid and 
# converts it to a standardized format (e.g., list of lists) for use in the SpectralConv layer.
# simple e.g.: # User passes a single number
               # validate_scaling_factor(0.5, n_dim=2)
               # returns [0.5, 0.5]  ← one value per dimension

import torch # Pytorch is a popular open-source deep learning framework.
from torch import nn # "nn" is a submodule in PyTorch that provides classes and functions for building neural networks.

import tensorly as tl # "tensorly" is a Python library for (math of tensors) tensor learning and decomposition. 
# It provides tools for working with multi-dimensional arrays (tensors) and 
# supports various tensor decompositions (e.g., CP, Tucker, Tensor Train).
# CP decomposition factorizes a tensor into a sum of rank-1 tensors.
# Tucker decomposition factorizes a tensor into a core tensor multiplied by factor matrices along each dimension.
# Tensor Train (TT) decomposition factorizes a tensor into a sequence of 3D tensors (cores) connected in a train-like structure.

from tensorly.plugins import use_opt_einsum
# tensorly is module, which we installed and imported as tl. plugins is a submodule of tensorly that provides 
# a way to use optimized einsum implementations. einsum is a powerful function for performing tensor multiplications
#  and contractions using Einstein summation notation. 
# use_opt_einsum is a function that allows us to specify which einsum implementation to use.

from tltorch.factorized_tensors.core import FactorizedTensor
# tltorch .i.e. tensorly-torch is an extension of tensorly that provides support for PyTorch tensors. It lets us 
# to store and manipulate factorized(compressed) tensors (e.g., CP, Tucker, TT) in PyTorch instead of full dense tensors.
# These factorized tensors can be weights in neural networks.
# facorized_tensors is a submodule of tltorch that provides classes and functions for working with factorized tensors.
# core is a submodule (python file: core.py) of tltorch.factorized_tensors that provides the base class FactorizedTensor.
# FactorizedTensor is a base class for all factorized tensor types (e.g., CP (cp.py), Tucker (tucker.py), TT (tt.py)). 
# It provides methods for creating, initializing, and manipulating factorized tensors. 
# All factorizations inherit from this base class "FactorizedTensor".
# core -> toolbox. FactorizedTensor -> main tool inside the toolbox. 
# cp.py, tucker.py, tt.py -> specific tools for specific tensor factorizations.

from .base_spectral_convolution import BaseSpectralConv
# . means from the current package, i.e. awfno/layers. base_spectral_convolution.py is a sibling file of this file.
# BaseSpectralConv is a base class for spectral convolution layers. It provides common functionality and 
# interface for spectral convolution layers, such as handling device placement and common methods for forward passes.
 
tl.set_backend("pytorch")
# tl.set_backend("pytorch") sets the backend for tensorly to use PyTorch.
'''
tl.set_backend("pytorch") means: "Whenever tensorly does tensor operations, use PyTorch under the hood."

1. With the above line:
import tensorly as tl
tl.set_backend("pytorch")   # from now on, use PyTorch
x = tl.tensor([1.0, 2.0, 3.0])
type(x)   # → torch.Tensor   (not numpy array)

2. Without the above line:
# default backend is numpy
x = tl.tensor([1.0, 2.0, 3.0])
type(x)   # → numpy.ndarray

'''
use_opt_einsum("optimal")
# use_opt_einsum("optimal") tells tensorly to use the optimized einsum implementation for tensor multiplications and contractions.

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
#  einsum_symbols is a sequence of characters that can be used as symbols in Einstein summation notation for tensor operations.
# For example, in an einsum operation like tl.einsum('abc,cd->abd', A, B), the characters 'a', 'b', 'c', 'd' 
# represent the dimensions of the tensors A and B, and how they are contracted together.


def _contract_dense(x, weight, separable=False):
    """
    Computes the contraction of dense weights with the input tensor x in Fourier space.
    
    If separable=True, performs element-wise multiplication.
    If separable=False, performs a matrix multiplication along the channel dimension.
    """
    order = tl.ndim(x)
    # tl.ndim(x) returns the number of dimensions (order) of the input tensor x.
    # 2-D input x will have 4 dimensions (order  = 4) as : batch-size, in_channels, x, y
    
    x_syms = list(einsum_symbols[:order])
    # einsum_symbols is a string of characters and [:order] = [:4] = "abcd" .i.e. first 4 characters of einsum_symbols.
    # list(einsum_symbols[:order]) = list("abcd") = ['a', 'b', 'c', 'd'] = x_syms 
    # x_syms is a list of characters representing the dimensions of x.

   
    weight_syms = list(x_syms[1:])  # no batch-size
    # weight_syms = list(x_syms[1:]) = list(['b', 'c', 'd']) = ['b', 'c', 'd'] 
    # x_syms[1:] means including second element of x_syms to the end. i.e. ['b', 'c', 'd'] = in_channels, x, y.

   
    if separable: # separable=True then following line is executed. 
        out_syms = [x_syms[0]] + list(weight_syms)
        # out_syms = [x_syms[0]] + list(weight_syms) = ['a'] + ['b', 'c', 'd'] = ['a', 'b', 'c', 'd']
        # When separable = True, then weight_syms represents "in_channels, x, y" and it is ['b', 'c', 'd'].
        # When separable = True, then out_syms represents "batch_size, in_channels, x, y.
        # Here, (in_channels = out_channels). Hence, out_syms is also called as batch_size, out_channels, x, y.
        # and it(out_syms) is ['a', 'b', 'c', 'd'].
        
    else: # separable=False then following line is executed.
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        # einsum_symbols[order] = einsum_symbols[4] = 'e' (the 5th character, since indexing starts at 0)
        # weight_syms.insert(1, 'e') means insert 'e' at index 1 of weight_syms.
        # weight_syms was ['b', 'c', 'd'] and after insertion becomes ['b', 'e', 'c', 'd'].
        # When separable = False, then weight_syms represents "in_channels, out_channels, x, y".

        out_syms = list(weight_syms) # out_syms = list(weight_syms) = ['b', 'e', 'c', 'd']
        out_syms[0] = x_syms[0] # out_syms = ['a', 'e', 'c', 'd'] 
        # When separable = False, then out_syms represents "batch_size, out_channels, x, y".

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'
    # eq is the Einstein summation equation that defines how the input tensor x and the weight tensor are contracted together.
    # Case i) separable=True:
    # x_syms = ['a', 'b', 'c', 'd']  (batch_size, in_channels, x, y)
    # weight_syms = ['b', 'c', 'd']  (in_channels, x, y)
    # out_syms = ['a', 'b', 'c', 'd'] (batch_size, out_channels, x, y)
    # eq = 'abcd,bcd->abcd'  (element-wise multiplication in Fourier space).
    # For this einsum, No summation happens along any dimension, just element-wise multiplication.
    # As, weight_syms does not have out_channels dimension, so output channels equal input channels automatically.
    # Case ii) separable=False:
    # x_syms = ['a', 'b', 'c', 'd']  (batch_size, in_channels, x, y)
    # weight_syms = ['b', 'e', 'c', 'd']  (in_channels, out_channels, x, y)
    # out_syms = ['a', 'e', 'c', 'd'] (batch_size, out_channels, x, y)
    # eq = 'abcd,becd->aecd'  (matrix multiplication along in_channels dimension in Fourier space).
    # For this einsum, summation happens along the in_channels dimension (symbol 'b'), which is the second dimension 
    # of x and the first dimension of weight. This corresponds to a matrix multiplication along the channel dimension 
    # in Fourier space, while keeping the spatial dimensions (x, y) intact and it will be repeated for each spatial
    # point ('c', 'd') and for each output channel ('e') and for each batch ('a').

    if not torch.is_tensor(weight): # factorized tensors are not torch tensors, they are instances of FactorizedTensor class.
    # Hence, if weight = factorized tensor then, this "if not torch.is_tensor(weight)" is True.

        # if not : this phrase means that given condition is False, then execute the following line. 
        # "if not" is opposite of "if". 
        # torch.is_tensor() method in torch module checks if the given input is a PyTorch tensor.
        #  It returns True if the input is a tensor and False otherwise.

        weight = weight.to_tensor() 
        # to_tensor() is a method of the FactorizedTensor class that reconstructs the full dense tensor from its
        # factorized representation. If weight is a factorized tensor (e.g., CP, Tucker, TT), 
        # this line will convert it to a full dense tensor that can be used for the contraction in Fourier space.

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return tl.einsum(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight) # tl.einsum() is tensorly's implementation of Einstein summation. 
    # for separable=True, einsum('abcd,bcd->abcd', x, weight) will perform element-wise multiplication in Fourier space.
    # for separable=False, einsum('abcd,becd->aecd', x, weight) will perform matrix multiplication along 
    # the in_channels dimension in Fourier space.

'''
Note: "tl.einsum(eq, x, weight)" is equivalent to "torch.einsum(eq, x, weight)" when the backend is set to "pytorch".
.i.e. pytorch's einsum is used under the hood for the actual computation, 
but we call it through tensorly's interface to maintain compatibility with factorized tensors 
and to allow for potential future backends.
'''

def _contract_dense_separable(x, weight, separable):
    """
    Computes the contraction of dense separable weights with the input tensor x.
    This corresponds to element-wise multiplication in Fourier space.
    This is equivalent to _contract_dense with separable=True, but it is provided as a separate function 
    for "performance optimization", as it avoids the overhead of einsum string construction and parsing when we just 
    want element-wise multiplication, this is directly given by x * weight.
    """
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight


def _contract_cp(x, cp_weight, separable=False):
    """
    Computes the contraction of CP-decomposed weights with the input tensor x.
    Uses efficient Einstein summation (einsum) to contract factors directly.
    """
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    """
    Computes the contraction of Tucker-decomposed weights with the input tensor x.
    Uses efficient Einstein summation (einsum) to contract the core and factors directly.
    """
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    """
    Computes the contraction of Tensor Train (TT) decomposed weights with the input tensor x.
    Constructs an Einstein summation equation to contract the TT cores directly.
    """
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return tl.einsum(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors.
        if False,
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


Number = Union[int, float]


class SpectralConv(BaseSpectralConv):
    """SpectralConv implements the Spectral Convolution component of a Fourier layer
    described. 
    
    It is implemented as described in [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : int or int tuple
        Number of modes to use for contraction in Fourier domain during training.

        .. warning::

            We take care of the redundancy in the Fourier modes, therefore, for an input
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes
            as the real FFT is redundant along that last dimension. For more information on
            mode truncation, refer to :ref:`fourier_layer_impl`


        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.

        This can be updated dynamically during training.

    complex_data : bool, optional
        Whether data takes on complex values in the spatial domain, by default False.
        If True, uses different logic for FFT contraction and uses full FFT instead of real-valued.
    max_n_modes : int tuple or None, optional
        * If not None, **maximum** number of modes to keep in Fourier Layer, along each dim
            The number of modes (`n_modes`) cannot be increased beyond that.
        * If None, all the n_modes are used.
        By default None.
    bias : bool, optional
        Whether to add a learnable bias to the output, by default True.
    separable : bool, optional
        Whether to use separable implementation of contraction.
        If True, contracts factors of factorized tensor weight individually.
        By default False.
    resolution_scaling_factor : float, list of float, or None, optional
        Scaling factor(s) for resolution scaling. If provided, the output resolution
        will be scaled by this factor along each spatial dimension.
        By default None.
    fno_block_precision : str, optional
        Precision mode for FNO block operations. Options: 'full', 'half', 'mixed'.
        By default 'full'.
    rank : float, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0.
        Ignored if ``factorization is None``.
    factorization : str or None, optional
        Tensor factorization type. Options: {'tucker', 'cp', 'tt'}.
        If None, a single dense weight is learned for the FNO.
        Otherwise, that weight, used for the contraction in the Fourier domain
        is learned in factorized form. In that case, `factorization` is the
        tensor factorization of the parameters weight used.
        By default None.
    implementation : {'factorized', 'reconstructed'}, optional
        If factorization is not None, forward mode to use:
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
        Ignored if ``factorization is None``.
        By default 'reconstructed'.
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False.
        Ignored if ``factorization is None``.
    decomposition_kwargs : dict or None, optional
        Optional additional parameters to pass to the tensor decomposition.
        Ignored if ``factorization is None``.
        By default None.
    init_std : float or 'auto', optional
        Standard deviation to use for weight initialization, by default 'auto'.
        If 'auto', uses (2 / (in_channels + out_channels)) ** 0.5.
    fft_norm : str, optional
        FFT normalization parameter, by default 'forward'.
    device : torch.device or None, optional
        Device to place the layer on, by default None.

    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    .. [2] :

    Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
        Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
        TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=1.0,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.complex_data = complex_data

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        if factorization is None:
            factorization = "Dense"  # No factorization

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
        self.separable = separable

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

        # Create/init spectral weight tensor
        self.weight = FactorizedTensor.new(
            weight_shape,
            rank=self.rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **tensor_kwargs,
            dtype=torch.cfloat,
        )
        self.weight.normal_(0, init_std)

        self._contract = get_contract_fun(
            self.weight, implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def transform(self, x, output_shape=None):
        """
        Transforms the input x, optionally rescaling the spatial dimensions.
        This method is typically used for resolution scaling or resampling.
        """
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return torch.nn.functional.interpolate(x, size=out_shape, mode='bilinear' if len(out_shape)==2 else 'trilinear' if len(out_shape)==3 else 'linear', align_corners=False)

    @property
    def n_modes(self):
        """
        Property to get or set the number of modes used in the spectral convolution.
        When setting, handles redundancy for real-valued FFTs.
        """
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off informtion from the last mode
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def _get_weight_indices(self, fft_size):
        """Computes indices for slicing the weight tensor."""
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(
                fft_size, self.n_modes, self.max_n_modes
            )
        ]

        if self.separable:
            slices_w = [slice(None)]  # channels
        else:
            slices_w = [slice(None), slice(None)]  # in_channels, out_channels

        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            # The last mode already has redundant half removed in real FFT
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        return tuple(slices_w)

    def _get_input_indices(self, fft_size, weight_shape):
        """Computes indices for slicing the input tensor."""
        # if separable conv, weight tensor only has one channel dim
        if self.separable:
            weight_start_idx = 1
        # otherwise drop first two dims (in_channels, out_channels)
        else:
            weight_start_idx = 2

        slices_x = [slice(None), slice(None)]  # Batch_size, channels
        kept_modes_list = weight_shape[weight_start_idx:]

        for all_modes, kept_modes in zip(fft_size, kept_modes_list):
            # After fft-shift, the 0th frequency is located at n // 2 in each direction
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2

            # this slice represents the desired indices along each dim
            slices_x.append(slice(center - negative_freqs, center + positive_freqs))

        # Special handling for the last dimension
        if kept_modes_list[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, kept_modes_list[-1])
        else:
            slices_x[-1] = slice(None)

        return tuple(slices_x)

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        # x: (batch, in_channels, d1, ..., dN)
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient

        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        # 1. FFT
        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]

        # x (complex): (batch, in_channels, d1, ..., dN)
        # x (real):    (batch, in_channels, d1, ..., dN // 2 + 1)

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        if self.fno_block_precision == "mixed":
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat

        # out_fft: (batch, out_channels, d1, ..., dN) (or dN // 2 + 1 for real)
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype
        )

        # 2. Contraction
        slices_w = self._get_weight_indices(fft_size)
        weight = self.weight[slices_w]

        slices_x = self._get_input_indices(fft_size, weight.shape)
        
        # Contraction in Fourier domain
        # x[slices_x]: (batch, in_channels, modes_1, ..., modes_N)
        # weight:      (in_channels, out_channels, modes_1, ..., modes_N) (or similar depending on factorization)
        # Result:      (batch, out_channels, modes_1, ..., modes_N)
        out_fft[slices_x] = self._contract(
            x[slices_x], weight, separable=self.separable
        )

        # 3. Resample (Inverse)
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)
                ]
            )

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])

        # Inverse FFT
        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(
                out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
            )
        
        # x: (batch, out_channels, d1', ..., dN')

        if self.bias is not None:
            x = x + self.bias

        return x