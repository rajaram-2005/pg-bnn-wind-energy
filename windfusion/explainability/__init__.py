"""Explainability: routing, physics residuals, saliency and advisory reports."""

from .report import DISCLAIMER, explain, gradient_saliency, render_report

__all__ = ["DISCLAIMER", "explain", "gradient_saliency", "render_report"]
