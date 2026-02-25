"""
Configuration loader for extraction configurations.
Loads YAML configuration files and provides access to extraction configurations.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from .models.extraction_config import ExtractionConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manager for extraction configurations."""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, ExtractionConfig] = {}
    
    def load_all(self) -> Dict[str, ExtractionConfig]:
        """Load all configurations from YAML files in config_dir."""
        configs = {}
        
        if not self.config_dir.exists():
            logger.warning(f"Config directory does not exist: {self.config_dir}")
            return configs
        
        for file in self.config_dir.glob("*.yaml"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # Load from file
                config = ExtractionConfig(**data)
                
                # Redis override support can be added here if needed
                # For now, configuration is loaded solely from YAML files
                
                configs[config.id] = config
                logger.debug(f"Loaded config: {config.id} (version {config.version})")
                
            except Exception as e:
                logger.error(f"Failed to load config file {file}: {e}")
        
        self._cache = configs
        logger.info(f"Loaded {len(configs)} extraction configurations")
        return configs
    
    def get_config(self, config_id: str) -> Optional[ExtractionConfig]:
        """Get configuration by ID. Reloads cache if not found."""
        if config_id not in self._cache:
            self.load_all()  # Reload if not found
        
        return self._cache.get(config_id)
    
    def list_configs(self) -> Dict[str, ExtractionConfig]:
        """List all available configurations."""
        if not self._cache:
            self.load_all()
        
        return self._cache.copy()