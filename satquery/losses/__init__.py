"""satquery/losses/__init__.py"""
from satquery.losses.compound_loss import FocalLoss, DiceLoss, CompoundChangeLoss

__all__ = ["FocalLoss", "DiceLoss", "CompoundChangeLoss"]
