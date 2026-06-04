from .fno import FNO, FNOBlock
from .wno import WNO1d, WNO2d
from .awfno import AWFNO1d, AWFNO2d, AWFNOBlock1d, AWFNOBlock2d, AdaptiveGatedFusion1d, AdaptiveGatedFusion2d
from .awfno_v2 import AWFNOv2_1d, AWFNOv2_2d
from .base_model import BaseModel, get_model, available_models


def build_model(name: str, **kwargs):
    """
    Instantiate a model by name string.

    Supported: ``"awfno"``, ``"awfno_v2"``, ``"fno"``, ``"wno"``.
    Dimensionality is inferred from ``n_modes`` or ``size`` kwargs.
    """
    name = name.lower().strip()

    def _ndim() -> int:
        ref = kwargs.get("n_modes") or kwargs.get("size")
        if ref is None:
            raise ValueError("Pass n_modes or size to infer spatial dimensionality.")
        return len(ref) if hasattr(ref, "__len__") else 1

    if name == "awfno":
        return AWFNO2d(**kwargs) if _ndim() == 2 else AWFNO1d(**kwargs)

    if name == "awfno_v2":
        return AWFNOv2_2d(**kwargs) if _ndim() == 2 else AWFNOv2_1d(**kwargs)

    if name == "fno":
        return FNO(**kwargs)

    if name == "wno":
        _kw = dict(
            in_channels=kwargs.get("in_channels", 1),
            out_channels=kwargs.get("out_channels", 1),
            width=kwargs.get("width", kwargs.get("hidden_channels", 32)),
            size=kwargs["size"],
            level=kwargs.get("level", kwargs.get("wno_level", 2)),
            n_layers=kwargs.get("n_layers", 4),
            padding=kwargs.get("padding", 2 if _ndim() == 2 else 0),
            wavelet=kwargs.get("wavelet", kwargs.get("wno_wavelet", "db4")),
        )
        return WNO2d(**_kw) if _ndim() == 2 else WNO1d(**_kw)

    raise ValueError(f"Unknown model '{name}'. Choices: awfno, awfno_v2, fno, wno")


def count_parameters(model) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "AWFNO1d", "AWFNO2d",
    "AWFNOv2_1d", "AWFNOv2_2d",
    "FNO", "FNOBlock",
    "WNO1d", "WNO2d",
    "BaseModel",
    "build_model",
    "count_parameters",
]
