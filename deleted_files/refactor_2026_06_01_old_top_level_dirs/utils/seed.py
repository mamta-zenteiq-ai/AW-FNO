"""Thin re-export so experiments can import from utils.seed uniformly."""

from awfno.utils.seed import set_seed

__all__ = ["set_seed"]
