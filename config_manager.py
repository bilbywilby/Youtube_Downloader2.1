import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict

CONFIG_FILE = Path.home() / '.youtube_downloader' / 'config.json'

@dataclass
class AppConfig:
    """Application configuration settings."""
    download_dir: str = str(Path.home() / '.youtube_downloader' / 'downloads')
    max_workers: int = 4
    default_format: str = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'
    default_resolution: str = 'best'
    default_audio_format: str = 'mp3'
    include_metadata: bool = True
    extract_thumbnail: bool = True
    theme: str = 'Reddit'
    use_ffmpeg: bool = False
    auto_queue: bool = False
    show_advanced: bool = False
    max_download_size: int = 2048  # MB
    rate_limit: int = 0  # KB/s, 0 for no limit
    history_enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    min_disk_space: int = 1024  # MB
    preferred_protocol: str = 'https'
    resume_downloads: bool = True
    verify_ssl: bool = True
    proxy: str = ''
    download_archive: bool = True
    archive_path: str = str(Path.home() / '.youtube_downloader' / 'downloaded.txt')
    notify_on_complete: bool = True

    def validate(self) -> bool:
        """Validates the configuration values."""
        try:
            self.max_workers = max(1, min(int(self.max_workers), 10))
            self.rate_limit = max(0, int(self.rate_limit))
            self.max_download_size = max(0, int(self.max_download_size))
            self.max_retries = max(1, min(int(self.max_retries), 10))
            self.retry_delay = max(1, min(int(self.retry_delay), 30))
            self.min_disk_space = max(100, int(self.min_disk_space))

            download_dir = Path(self.download_dir)
            if not download_dir.exists():
                download_dir.mkdir(parents=True, exist_ok=True)
            if not download_dir.is_dir():
                raise ValueError(f"Download directory '{self.download_dir}' is not a valid directory.")

            archive_path = Path(self.archive_path)
            if archive_path.parent.exists() and not archive_path.parent.is_dir():
                raise ValueError(f"Archive directory '{archive_path.parent}' is not a valid directory.")

            return True
        except ValueError as ve:
            logging.error(f"Config validation error: {ve}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during config validation: {e}")
            return False

class ConfigManager:
    """Manages application configuration, including loading, saving, updating, and reloading."""
    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        """Loads the configuration from the JSON file or creates a default one if it doesn't exist or is invalid."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                config = AppConfig(**data)
                if not config.validate():
                    raise ValueError("Invalid config values found in file.")
                logging.info(f"Configuration loaded from {self.config_path}")
                return config
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logging.warning(f"Failed to load config from {self.config_path}, using defaults: {e}")
        # If loading fails, use defaults and save
        default_config = AppConfig()
        default_config.validate()
        self._save_config(default_config)
        return default_config

    def _save_config(self, config: AppConfig) -> None:
        """Saves the configuration to the JSON file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(asdict(config), f, indent=4)
            logging.info(f"Configuration saved to {self.config_path}")
        except IOError as e:
            logging.error(f"Failed to save config to {self.config_path}: {e}")

    def get_config(self) -> AppConfig:
        """Returns the current application configuration."""
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Updates the configuration with the provided dictionary and saves it."""
        updated = False
        for key, value in updates.items():
            if hasattr(self.config, key):
                attr_type = type(getattr(self.config, key))
                try:
                    # Attempt to cast value to correct type
                    setattr(self.config, key, attr_type(value))
                    updated = True
                except Exception:
                    logging.warning(f"Type mismatch for '{key}': expected {attr_type.__name__}, got {type(value).__name__}")
            else:
                logging.warning(f"Ignoring unknown configuration key: {key}")

        if updated and self.config.validate():
            self._save_config(self.config)
        elif updated and not self.config.validate():
            logging.error("Config update failed validation. Previous config is still active.")

    def reload(self) -> None:
        """Reloads the configuration from disk."""
        self.config = self._load_config()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_manager = ConfigManager()
    config = config_manager.get_config()
    logging.info(f"Initial config: {asdict(config)}")

    # Example update
    config_manager.update_config({'max_workers': 6, 'default_resolution': '1080p'})
    updated_config = config_manager.get_config()
    logging.info(f"Updated config: {asdict(updated_config)}")

    # Example of invalid update
    config_manager.update_config({'max_workers': 'invalid'})
    invalid_config = config_manager.get_config()
    logging.info(f"Config after invalid update: {asdict(invalid_config)}")

    # Example of creating config directory and file
    temp_config_path = Path.home() / '.temp_downloader' / 'test_config.json'
    temp_config_manager = ConfigManager(config_path=temp_config_path)
    temp_config = temp_config_manager.get_config()
    logging.info(f"Temp config: {asdict(temp_config)}")
    # Clean up the temporary directory if empty
    try:
        temp_config_path.unlink()
        temp_config_path.parent.rmdir()
    except Exception:
        pass
    # Clean up the original config file if empty
    try:
        CONFIG_FILE.unlink()
        CONFIG_FILE.parent.rmdir()
    except Exception:
        pass
    # Clean up the original config directory if empty
    try:
        CONFIG_FILE.parent.rmdir()
    except Exception:
        pass
    # Clean up the original config file if empty
    try:
        CONFIG_FILE.unlink()
    except Exception:
        pass