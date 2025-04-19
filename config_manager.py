# config_manager.py
import json
import logging
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import constants from main file
from youtube_downloader import (
    APP_ROOT,
    CONFIG_PATH,
    DOWNLOADS_ROOT,
    AppConfig
)

class ConfigManager:
    """Manage application configuration with validation"""

    def __init__(self):
        self.config = AppConfig()
        self._load_config()

    def _load_config(self):
        """Load configuration from file with fallback to defaults"""
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    # Update configuration while preserving defaults for missing values
                    self.config = AppConfig(**{
                        **asdict(self.config),
                        **config_data
                    })
                    
                # Validate loaded configuration
                if not self.config.validate():
                    logger.warning("Invalid configuration loaded, using defaults")
                    self.config = AppConfig()
            else:
                logger.info("No configuration file found, using defaults")
                self.save_config()  # Create default config file
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = AppConfig()  # Use defaults on error

    def save_config(self):
        """Save current configuration to file"""
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=4)
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values
        
        Args:
            updates (Dict[str, Any]): Dictionary of configuration updates
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Create new config with updates
            new_config = AppConfig(**{
                **asdict(self.config),
                **updates
            })
            
            # Validate new configuration
            if not new_config.validate():
                logger.error("Invalid configuration update")
                return False
                
            self.config = new_config
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False

    def get_download_dir(self) -> Path:
        """Get validated download directory path"""
        download_dir = Path(self.config.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        return download_dir

    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            self.config = AppConfig()
            return self.save_config()
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False
