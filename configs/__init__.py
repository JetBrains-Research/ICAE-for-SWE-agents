"""Config package initialization."""

from .templates import TemplateManager
from .config import get_config


__all__ = [
    "TemplateManager", 
    "get_config"
] 