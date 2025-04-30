# -*- coding: utf-8 -*-
"""
YouTube Downloader GUI Application
Version: 2.1.0
Author: Programming Assistant (Refactored from bilbywilby's original)
Updated: 2025-04-19
"""

# ----------- Standard Library Imports -----------
import json
import logging
import logging.handlers
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Union

# ----------- Third-Party Library Imports -----------
try:
    import psutil
    import PySimpleGUI as sg
    import yt_dlp
except ImportError as e:
    print(f"Error: Missing dependency - {e.name}.")
    print("Please install the required libraries:")
    print("pip install psutil PySimpleGUI yt-dlp")
    sys.exit(1)


# ----------- Application Metadata -----------
APP_NAME = "YouTube Downloader"
VERSION = "2.1.0"
AUTHOR = "Programming Assistant (Refactored from bilbywilby)"
UPDATED = datetime.now().strftime("%Y-%m-%d")


# ----------- App Directories & Constants -----------
APP_ROOT = Path.home() / ".youtube_downloader_pro"
CONFIG_PATH = APP_ROOT / "config.json"
HISTORY_PATH = APP_ROOT / "history.db"  # Changed to SQLite
QUEUE_PATH = APP_ROOT / "queue.json"
LOG_DIR = APP_ROOT / "logs"
DOWNLOADS_ROOT = Path.home() / "Downloads" / "YouTubeDownloader"
TEMP_DIR = APP_ROOT / "temp"
CACHE_DIR = APP_ROOT / "cache"
LOG_FILE = LOG_DIR / "downloader.log"

# Default values (can be overridden by config)
DEFAULT_THEME = "DarkBlue3"
DEFAULT_MAX_WORKERS = os.cpu_count() or 4
DEFAULT_MIN_DISK_SPACE_MB = 1024  # 1GB

# File formats and limits
ALLOWED_VIDEO_FORMATS = {"mp4", "mkv", "webm"}
ALLOWED_AUDIO_FORMATS = {"mp3", "wav", "m4a", "aac", "flac"}
MAX_FILENAME_LENGTH = 150  # Increased flexibility
DEFAULT_VIDEO_FORMAT = "mp4"
DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_VIDEO_QUALITY = "1080p" # Default to a common quality

# Network & Download Constants
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for downloads (used by yt-dlp internally)
USER_AGENT = f"{APP_NAME}/{VERSION} (Python/{platform.python_version()}; {platform.system()})"


# ----------- Logging Setup -----------
def setup_logging() -> logging.Logger:
    """Configures logging for the application."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        log_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(threadName)s:%(filename)s:%(lineno)d] - %(message)s"
        )
        log_level = logging.INFO # Default level

        # File Handler (Rotating)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(log_level)

        # Console Handler (Optional - good for debugging)
        # console_handler = logging.StreamHandler(sys.stdout)
        # console_handler.setFormatter(log_formatter)
        # console_handler.setLevel(log_level)

        # Configure Root Logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear() # Remove default handlers if any
        root_logger.addHandler(file_handler)
        # root_logger.addHandler(console_handler) # Uncomment for console output

        # Suppress verbose logs from libraries if needed
        logging.getLogger("yt_dlp").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        app_logger = logging.getLogger(APP_NAME)
        app_logger.info("=" * 50)
        app_logger.info(f"Logging initialized for {APP_NAME} v{VERSION}")
        app_logger.info(f"Log file: {LOG_FILE}")
        return app_logger

    except Exception as e:
        # Fallback basic logging if setup fails
        logging.basicConfig(level=logging.WARNING)
        logging.error(f"Failed to configure logging: {e}", exc_info=True)
        return logging.getLogger(APP_NAME)

logger = setup_logging()

# ----------- Directory Creation -----------
def create_initial_directories():
    """Creates necessary application directories."""
    logger.info("Ensuring application directories exist...")
    for dir_path in [APP_ROOT, LOG_DIR, DOWNLOADS_ROOT, TEMP_DIR, CACHE_DIR]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {dir_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)
            # Handle critical directory creation failure (e.g., exit or notify user)
            if dir_path in [APP_ROOT, LOG_DIR]:
                sg.popup_error(f"Critical Error: Cannot create directory {dir_path}.\nPlease check permissions.\n{e}", title="Initialization Error")
                sys.exit(1)
            else:
                 sg.popup_warning(f"Warning: Could not create directory {dir_path}.\nDownloads may fail.\n{e}", title="Initialization Warning")

create_initial_directories()


# ----------- Error Classes -----------
class DownloaderError(Exception):
    """Base exception for downloader-specific errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)
        logger.error(f"{self.__class__.__name__}: {message}", exc_info=original_exception if original_exception else False)

class ValidationError(DownloaderError):
    """Errors related to input validation (URLs, settings)."""
    pass

class DiskSpaceError(DownloaderError):
    """Errors related to insufficient disk space."""
    pass

class NetworkError(DownloaderError):
    """Errors related to network connectivity or timeouts."""
    pass

class ConfigError(DownloaderError):
    """Errors related to loading or saving configuration."""
    pass

class DownloadProcessError(DownloaderError):
    """Errors occurring during the actual download process (yt-dlp)."""
    pass

class FileSystemError(DownloaderError):
    """Errors related to file system operations (permissions, etc.)."""
    pass


# ----------- Data Classes -----------
@dataclass
class AppConfig:
    """Stores application configuration settings."""
    download_dir: str = str(DOWNLOADS_ROOT)
    max_workers: int = DEFAULT_MAX_WORKERS
    default_video_format: str = DEFAULT_VIDEO_FORMAT
    default_audio_format: str = DEFAULT_AUDIO_FORMAT
    default_quality: str = DEFAULT_VIDEO_QUALITY
    include_metadata: bool = True
    extract_thumbnail: bool = True
    theme: str = DEFAULT_THEME
    use_ffmpeg_if_available: bool = True # Renamed for clarity
    auto_start_queue: bool = False
    show_advanced_options: bool = False # Placeholder for future use
    max_log_size_mb: int = 10
    history_limit: int = 100 # Max entries in history display
    max_retries: int = MAX_RETRIES
    retry_delay: int = RETRY_DELAY
    min_disk_space_mb: int = DEFAULT_MIN_DISK_SPACE_MB
    preferred_protocol: str = "https"
    resume_downloads: bool = True
    verify_ssl: bool = True
    proxy: str = ""
    download_archive_enabled: bool = False # Optional feature
    archive_path: str = str(APP_ROOT / "downloaded_archive.txt")
    notify_on_complete: bool = True
    rate_limit_kbs: int = 0 # 0 means no limit

    def __post_init__(self):
        """Validate and sanitize configuration after initialization."""
        self.validate()

    def validate(self):
        """Validates configuration values and applies constraints."""
        logger.debug("Validating AppConfig...")
        try:
            self.max_workers = max(1, min(int(self.max_workers), os.cpu_count() * 2 if os.cpu_count() else 8))
            self.rate_limit_kbs = max(0, int(self.rate_limit_kbs))
            self.max_retries = max(0, min(int(self.max_retries), 10))
            self.retry_delay = max(1, min(int(self.retry_delay), 60))
            self.min_disk_space_mb = max(50, int(self.min_disk_space_mb))
            self.max_log_size_mb = max(1, int(self.max_log_size_mb))
            self.history_limit = max(10, int(self.history_limit))

            # Ensure download directory exists
            download_path = Path(self.download_dir)
            if not download_path.exists():
                logger.warning(f"Download directory '{self.download_dir}' not found. Attempting to create.")
                try:
                    download_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created download directory: {download_path}")
                except OSError as e:
                    logger.error(f"Failed to create download directory '{download_path}': {e}. Resetting to default.", exc_info=True)
                    self.download_dir = str(DOWNLOADS_ROOT)
                    DOWNLOADS_ROOT.mkdir(parents=True, exist_ok=True) # Ensure default exists

            # Validate theme
            if self.theme not in sg.theme_list():
                logger.warning(f"Invalid theme '{self.theme}' found in config. Resetting to default '{DEFAULT_THEME}'.")
                self.theme = DEFAULT_THEME

            # Validate formats
            if self.default_video_format not in ALLOWED_VIDEO_FORMATS:
                logger.warning(f"Invalid default video format '{self.default_video_format}'. Resetting to '{DEFAULT_VIDEO_FORMAT}'.")
                self.default_video_format = DEFAULT_VIDEO_FORMAT
            if self.default_audio_format not in ALLOWED_AUDIO_FORMATS:
                 logger.warning(f"Invalid default audio format '{self.default_audio_format}'. Resetting to '{DEFAULT_AUDIO_FORMAT}'.")
                 self.default_audio_format = DEFAULT_AUDIO_FORMAT

            logger.debug("AppConfig validation successful.")

        except (ValueError, TypeError) as e:
            logger.error(f"Configuration validation failed: {e}. Resetting to defaults might be needed.", exc_info=True)
            # Consider resetting specific fields or the whole config on severe errors
            raise ConfigError(f"Invalid configuration value encountered: {e}", original_exception=e)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        # Filter data to only include keys defined in the dataclass
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class DownloadItem:
    """Represents a single item in the download queue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())) # Unique ID for tracking
    url: str
    title: Optional[str] = None
    video_format: str = DEFAULT_VIDEO_FORMAT
    audio_format: str = DEFAULT_AUDIO_FORMAT
    quality: str = DEFAULT_VIDEO_QUALITY
    is_audio_only: bool = False
    is_playlist: bool = False
    download_subtitles: bool = False
    target_directory: Optional[str] = None # Overrides global config if set
    status: str = "Pending"  # Pending, Fetching, Downloading, Converting, Completed, Failed, Canceled
    progress: float = 0.0  # Percentage (0.0 to 100.0)
    speed_mbps: float = 0.0
    eta_seconds: int = 0
    error_message: Optional[str] = None
    video_id: Optional[str] = None
    final_filepath: Optional[str] = None
    filesize_bytes: Optional[int] = None
    added_time: str = field(default_factory=lambda: datetime.now().isoformat())
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadItem":
        # Filter data to only include keys defined in the dataclass
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        # Handle potential missing keys if loading older queue/history data
        for key in valid_keys:
            if key not in filtered_data:
                # Set a default value based on the type hint or field default
                 default_val = cls.__dataclass_fields__[key].default_factory() if callable(cls.__dataclass_fields__[key].default_factory) else cls.__dataclass_fields__[key].default
                 filtered_data[key] = default_val
                 logger.warning(f"Missing key '{key}' in loaded DownloadItem data, using default: {default_val}")

        return cls(**filtered_data)

    def get_display_name(self) -> str:
        """Returns a user-friendly name for display."""
        if self.title:
            return self.title
        elif self.video_id:
            return f"ID: {self.video_id}"
        else:
            # Truncate long URLs
            max_url_len = 40
            display_url = self.url if len(self.url) <= max_url_len else self.url[:max_url_len-3] + "..."
            return display_url

    def get_status_summary(self) -> str:
        """Provides a concise status summary for the UI."""
        if self.status == "Downloading":
            eta_str = f"{self.eta_seconds}s" if self.eta_seconds else "N/A"
            return f"Downloading ({self.progress:.1f}%) at {self.speed_mbps:.2f} MB/s, ETA: {eta_str}"
        elif self.status == "Completed":
            return f"Completed ({datetime.fromisoformat(self.end_time).strftime('%Y-%m-%d %H:%M') if self.end_time else ''})"
        elif self.status == "Failed":
            return f"Failed: {self.error_message or 'Unknown error'}"
        return self.status


# ----------- Utility Classes -----------

class URLValidator:
    """Validates and processes URLs, specifically for YouTube."""

    # More comprehensive patterns including playlists, shorts, live, music
    YOUTUBE_PATTERNS = [
        re.compile(r"^https?://(?:www\.)?youtube\.com/watch\?v=([\w-]+)", re.IGNORECASE),
        re.compile(r"^https?://(?:www\.)?youtube\.com/shorts/([\w-]+)", re.IGNORECASE),
        re.compile(r"^https?://(?:www\.)?youtube\.com/live/([\w-]+)", re.IGNORECASE),
        re.compile(r"^https?://youtu\.be/([\w-]+)", re.IGNORECASE),
        re.compile(r"^https?://(?:www\.)?youtube\.com/playlist\?list=([\w\d_-]+)", re.IGNORECASE),
        re.compile(r"^https?://music\.youtube\.com/watch\?v=([\w-]+)", re.IGNORECASE),
        re.compile(r"^https?://music\.youtube\.com/playlist\?list=([\w\d_-]+)", re.IGNORECASE),
        # Handle googleusercontent proxy links (example structure)
        re.compile(r"^https?://[\w-]+\.googleusercontent\.com/youtube\.com/\d+/watch\?v=([\w-]+)", re.IGNORECASE),
        re.compile(r"^https?://[\w-]+\.googleusercontent\.com/youtube\.com/\d+/shorts/([\w-]+)", re.IGNORECASE),
    ]
    PLAYLIST_PATTERN = re.compile(r"list=([\w\d_-]+)", re.IGNORECASE)

    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        """Checks if the URL matches known YouTube patterns."""
        if not url or not isinstance(url, str):
            return False
        try:
            return any(pattern.search(url) for pattern in URLValidator.YOUTUBE_PATTERNS)
        except Exception as e:
            logger.error(f"URL validation regex error for '{url}': {e}", exc_info=True)
            return False # Treat regex errors as invalid

    @staticmethod
    @lru_cache(maxsize=256) # Cache results for frequently checked URLs
    def extract_id(url: str) -> Optional[Dict[str, str]]:
        """Extracts Video ID or Playlist ID from a YouTube URL."""
        if not url or not isinstance(url, str):
            return None

        try:
            # Prioritize playlist ID if present
            playlist_match = URLValidator.PLAYLIST_PATTERN.search(url)
            if playlist_match:
                playlist_id = playlist_match.group(1)
                # Sanity check playlist ID format (basic)
                if playlist_id and len(playlist_id) > 10: # Playlist IDs are usually longer
                     logger.debug(f"Extracted Playlist ID: {playlist_id} from {url}")
                     return {"type": "playlist", "id": playlist_id}

            # Check standard video patterns
            for pattern in URLValidator.YOUTUBE_PATTERNS:
                match = pattern.search(url)
                if match:
                    video_id = match.group(1) # The capturing group should contain the ID
                    # Validate video ID format (11 chars, base64-like)
                    if video_id and re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
                        logger.debug(f"Extracted Video ID: {video_id} from {url}")
                        return {"type": "video", "id": video_id}

            logger.warning(f"Could not extract a valid ID from URL: {url}")
            return None
        except Exception as e:
            logger.error(f"Error extracting ID from URL '{url}': {e}", exc_info=True)
            return None

    @staticmethod
    def sanitize_url(url: str) -> str:
        """Removes potentially harmful characters and excessive whitespace."""
        if not url or not isinstance(url, str):
            return ""
        try:
            # Basic sanitization: strip whitespace, remove common injection chars
            sanitized = url.strip()
            sanitized = re.sub(r'[<>"\'`;(){}]', '', sanitized)
            # Further validation could be added here if needed
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing URL '{url}': {e}", exc_info=True)
            return "" # Return empty string on error

class FileManager:
    """Handles file and directory operations safely."""

    # Stricter pattern for safe filenames, allows underscores, hyphens, spaces
    INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]' # Control chars included
    MAX_PATH_LEN = 255 # Common limit on many OSes

    @staticmethod
    def sanitize_filename(filename: str, replacement: str = '_') -> str:
        """Creates a safe filename by removing/replacing invalid characters."""
        if not filename:
            return f"download_{int(time.time())}" # Default name if empty

        # Remove invalid characters
        sanitized = re.sub(FileManager.INVALID_FILENAME_CHARS, replacement, filename)

        # Replace multiple spaces/underscores with a single one
        sanitized = re.sub(r'[\s_]+', replacement, sanitized).strip(replacement)

        # Truncate if too long (considering extension length)
        base, ext = os.path.splitext(sanitized)
        if len(sanitized) > MAX_FILENAME_LENGTH:
            available_len = MAX_FILENAME_LENGTH - len(ext) - 1 # Account for dot
            base = base[:available_len]
            sanitized = f"{base}{ext}"
            logger.warning(f"Original filename truncated: {filename} -> {sanitized}")

        # Handle reserved names (Windows mainly) - basic check
        reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
        name_part, _ = os.path.splitext(sanitized)
        if name_part.upper() in reserved_names:
            sanitized = f"_{sanitized}"
            logger.warning(f"Filename conflicts with reserved name. Prepended underscore: {sanitized}")

        if not sanitized: # If sanitization resulted in empty string
            return f"download_{int(time.time())}{ext}"

        return sanitized

    @staticmethod
    def check_disk_space(dir_path: Union[str, Path], min_mb_required: int) -> bool:
        """Checks if sufficient disk space is available in the given directory."""
        try:
            path = Path(dir_path)
            usage = psutil.disk_usage(str(path.anchor)) # Check space on the drive root
            available_mb = usage.free / (1024 * 1024)
            logger.debug(f"Disk space check for '{path.anchor}': Available={available_mb:.2f} MB, Required={min_mb_required} MB")
            return available_mb >= min_mb_required
        except FileNotFoundError:
            logger.error(f"Disk space check failed: Directory '{dir_path}' not found.")
            return False
        except Exception as e:
            logger.error(f"Error checking disk space for '{dir_path}': {e}", exc_info=True)
            return False # Assume insufficient space on error

    @staticmethod
    def get_available_disk_space_mb(dir_path: Union[str, Path]) -> Optional[float]:
        """Returns available disk space in MB for the given path's drive."""
        try:
            path = Path(dir_path)
            usage = psutil.disk_usage(str(path.anchor))
            return usage.free / (1024 * 1024)
        except Exception as e:
            logger.error(f"Could not get available disk space for '{dir_path}': {e}", exc_info=True)
            return None

    @staticmethod
    def safe_delete(filepath: Union[str, Path]):
        """Safely deletes a file, logging errors."""
        try:
            p = Path(filepath)
            if p.exists():
                p.unlink()
                logger.info(f"Deleted file: {filepath}")
        except OSError as e:
            logger.error(f"Error deleting file '{filepath}': {e}", exc_info=True)
        except Exception as e:
             logger.error(f"Unexpected error deleting file '{filepath}': {e}", exc_info=True)


class SystemManager:
    """Handles system-related operations like checking dependencies and opening folders."""

    _ffmpeg_path: Optional[str] = None
    _ffmpeg_checked: bool = False

    @classmethod
    def find_ffmpeg(cls) -> Optional[str]:
        """Checks if FFmpeg is available in PATH or common locations."""
        if cls._ffmpeg_checked:
            return cls._ffmpeg_path

        cls._ffmpeg_checked = True
        cls._ffmpeg_path = shutil.which("ffmpeg")
        if cls._ffmpeg_path:
            logger.info(f"FFmpeg found via PATH: {cls._ffmpeg_path}")
            return cls._ffmpeg_path

        # Add checks for common install locations if needed (e.g., within app dir)
        logger.warning("FFmpeg not found in system PATH. Some features (e.g., format conversion, thumbnail embedding) might be unavailable.")
        return None

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Gathers basic system information."""
        info = {
            "app_version": VERSION,
            "python_version": platform.python_version(),
            "yt_dlp_version": getattr(yt_dlp, '__version__', 'Unknown'),
            "os": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory_total_gb": f"{psutil.virtual_memory().total / (1024**3):.2f}",
            "ffmpeg_path": SystemManager.find_ffmpeg() or "Not Found",
        }
        logger.debug(f"Gathered system info: {info}")
        return info

    @staticmethod
    def open_folder(path: Union[str, Path]):
        """Opens the specified folder in the system's file explorer."""
        try:
            folder_path = str(Path(path).resolve())
            if not os.path.isdir(folder_path):
                logger.error(f"Cannot open folder: '{folder_path}' is not a valid directory.")
                sg.popup_error(f"Cannot open folder:\n'{folder_path}'\nIt is not a valid directory.", title="Error")
                return

            logger.info(f"Attempting to open folder: {folder_path}")
            system = platform.system()

            if system == "Windows":
                # Use os.startfile for broader compatibility on Windows
                os.startfile(folder_path)
            elif system == "Darwin": # macOS
                subprocess.run(["open", folder_path], check=True)
            else: # Linux and other Unix-like
                # Try xdg-open first, fallback to alternatives if needed
                try:
                    subprocess.run(["xdg-open", folder_path], check=True)
                except FileNotFoundError:
                    logger.warning("xdg-open not found. Trying 'gnome-open' or 'kde-open'.")
                    # Add fallbacks if necessary, e.g., gnome-open, kde-open
                    # For simplicity, we'll just report the error if xdg-open fails
                    raise OSError("Could not find a suitable command to open the folder (xdg-open missing).")
                except subprocess.CalledProcessError as e:
                     raise OSError(f"Command 'xdg-open {folder_path}' failed: {e}") from e

            logger.info(f"Successfully requested to open folder: {folder_path}")

        except FileNotFoundError:
             logger.error(f"Cannot open folder: The path '{path}' does not exist.")
             sg.popup_error(f"Cannot open folder:\nThe path '{path}' does not exist.", title="Error")
        except PermissionError as e:
            logger.error(f"Permission denied trying to open folder '{path}': {e}", exc_info=True)
            sg.popup_error(f"Permission denied:\nCould not open folder '{path}'.\nPlease check permissions.", title="Error")
        except Exception as e:
            logger.error(f"Failed to open folder '{path}': {e}", exc_info=True)
            sg.popup_error(f"An unexpected error occurred while trying to open the folder:\n{e}", title="Error")


class FormatDetector:
    """Detects available video and audio formats using yt-dlp."""

    # Cache format detection results to avoid repeated network requests for the same URL
    _format_cache = {} # Simple dict cache for this session
    _cache_lock = threading.Lock()

    @staticmethod
    def detect_formats(url: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Detects available formats for a given URL, using a cache."""
        sanitized_url = URLValidator.sanitize_url(url)
        if not URLValidator.is_valid_youtube_url(sanitized_url):
            logger.warning(f"Format detection skipped for invalid URL: {url}")
            return None

        # Check cache first
        with FormatDetector._cache_lock:
            if sanitized_url in FormatDetector._format_cache:
                logger.debug(f"Using cached formats for: {sanitized_url}")
                return FormatDetector._format_cache[sanitized_url]

        logger.info(f"Detecting formats for: {sanitized_url}")
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False, # Get detailed format info
            'forcejson': True, # Get JSON output directly
            'skip_download': True,
            'simulate': True, # Ensure no download happens
            'ignoreerrors': True, # Try to get info even if some parts fail
            'user_agent': USER_AGENT,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Use extract_info which is designed for getting metadata
                info_dict = ydl.extract_info(sanitized_url, download=False)

            if not info_dict or 'formats' not in info_dict:
                logger.warning(f"No format information found for URL: {sanitized_url}")
                return None # Indicate failure clearly

            detected_formats = {"video": [], "audio": [], "video+audio": [], "metadata": {}}

            # Extract basic metadata
            detected_formats["metadata"] = {
                "title": info_dict.get("title", "N/A"),
                "uploader": info_dict.get("uploader", "N/A"),
                "duration": info_dict.get("duration"),
                "thumbnail": info_dict.get("thumbnail"),
                "is_live": info_dict.get("is_live", False),
            }

            # Process formats
            for fmt in info_dict.get("formats", []):
                # Basic info common to all formats
                format_info = {
                    "id": fmt.get("format_id"),
                    "ext": fmt.get("ext"),
                    "filesize": fmt.get("filesize") or fmt.get("filesize_approx"), # Approx is better than nothing
                    "tbr": fmt.get("tbr"), # Total bitrate
                    "format_note": fmt.get("format_note"),
                    "protocol": fmt.get("protocol"),
                    "url": fmt.get("url"), # Direct URL (might expire)
                    "is_hdr": "hdr" in (fmt.get("dynamic_range") or "").lower(),
                    "vcodec": fmt.get("vcodec", "none").lower(),
                    "acodec": fmt.get("acodec", "none").lower(),
                }

                is_video = format_info["vcodec"] != "none"
                is_audio = format_info["acodec"] != "none"

                # Video-specific info
                if is_video:
                    format_info.update({
                        "height": fmt.get("height"),
                        "width": fmt.get("width"),
                        "fps": fmt.get("fps"),
                        "dynamic_range": fmt.get("dynamic_range"),
                    })
                    # Categorize: video-only or combined
                    if not is_audio:
                        detected_formats["video"].append(format_info)
                    else:
                        detected_formats["video+audio"].append(format_info)

                # Audio-specific info
                elif is_audio: # It's audio-only if not video
                    format_info.update({
                        "abr": fmt.get("abr"), # Audio bitrate
                        "asr": fmt.get("asr"), # Audio sample rate
                    })
                    detected_formats["audio"].append(format_info)

            # Sort formats by quality (heuristic)
            # Video: Height > FPS > Bitrate
            # Audio: Bitrate > Sample Rate
            # Combined: Height > Video Bitrate > Audio Bitrate
            sort_key_video = lambda x: (x.get("height") or 0, x.get("fps") or 0, x.get("tbr") or 0)
            sort_key_audio = lambda x: (x.get("abr") or 0, x.get("asr") or 0)
            sort_key_combined = lambda x: (x.get("height") or 0, x.get("tbr") or 0, x.get("abr") or 0)

            detected_formats["video"].sort(key=sort_key_video, reverse=True)
            detected_formats["audio"].sort(key=sort_key_audio, reverse=True)
            detected_formats["video+audio"].sort(key=sort_key_combined, reverse=True)

            logger.info(f"Successfully detected formats for {sanitized_url}. "
                        f"Video-only: {len(detected_formats['video'])}, "
                        f"Audio-only: {len(detected_formats['audio'])}, "
                        f"Combined: {len(detected_formats['video+audio'])}")

            # Store in cache
            with FormatDetector._cache_lock:
                 FormatDetector._format_cache[sanitized_url] = detected_formats

            return detected_formats

        except yt_dlp.utils.DownloadError as e:
             # Handle specific yt-dlp errors gracefully
             if "Unsupported URL" in str(e):
                 logger.error(f"Format detection failed: Unsupported URL - {sanitized_url}")
             elif "urlopen error [Errno 11001] getaddrinfo failed" in str(e) or "nodename nor servname provided, or not known" in str(e):
                 logger.error(f"Format detection failed: Network/DNS error for {sanitized_url}", exc_info=True)
             else:
                 logger.error(f"yt-dlp format detection error for {sanitized_url}: {e}", exc_info=True)
             return None # Indicate failure
        except Exception as e:
            logger.error(f"Unexpected error during format detection for {sanitized_url}: {e}", exc_info=True)
            return None # Indicate failure

    @staticmethod
    def clear_cache():
        """Clears the format detection cache."""
        with FormatDetector._cache_lock:
            FormatDetector._format_cache.clear()
        logger.info("Format detection cache cleared.")


# ----------- Persistence Managers -----------

class ConfigManager:
    """Manages loading and saving of the AppConfig."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> AppConfig:
        """Loads configuration from JSON file or returns defaults."""
        logger.info(f"Loading configuration from: {self.config_path}")
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    logger.debug(f"Loaded raw config data: {config_data}")
                    # Validate and create AppConfig object
                    return AppConfig.from_dict(config_data)
            else:
                logger.warning(f"Config file not found at {self.config_path}. Using default settings.")
                return AppConfig() # Return default config
        except (json.JSONDecodeError, TypeError, KeyError, ConfigError) as e:
            logger.error(f"Failed to load or validate config file '{self.config_path}': {e}. Using default settings.", exc_info=True)
            # Optionally backup corrupted config here
            return AppConfig() # Return default config on error
        except Exception as e:
             logger.error(f"Unexpected error loading config '{self.config_path}': {e}. Using defaults.", exc_info=True)
             return AppConfig()

    def save_config(self):
        """Saves the current configuration to the JSON file."""
        logger.info(f"Saving configuration to: {self.config_path}")
        try:
            # Ensure the directory exists before saving
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # Validate before saving
            self.config.validate()
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=4, ensure_ascii=False)
            logger.debug("Configuration saved successfully.")
        except (TypeError, OSError, ConfigError) as e:
            logger.error(f"Failed to save config file '{self.config_path}': {e}", exc_info=True)
            raise ConfigError(f"Failed to save configuration: {e}", original_exception=e)
        except Exception as e:
            logger.error(f"Unexpected error saving config '{self.config_path}': {e}", exc_info=True)
            raise ConfigError(f"Unexpected error saving configuration: {e}", original_exception=e)

# --- History Manager (Using SQLite for better scalability) ---
class HistoryManager:
    """Manages download history using an SQLite database."""
    DB_VERSION = 1 # For potential future schema migrations

    def __init__(self, history_path: Path = HISTORY_PATH):
        self.db_path = history_path
        self._lock = threading.Lock() # Ensure thread safety for DB operations
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite database."""
        try:
            # `check_same_thread=False` is needed because we might access
            # the DB from different threads (main GUI, download threads).
            # Access is protected by self._lock.
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10) # Increased timeout
            conn.row_factory = sqlite3.Row # Access columns by name
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to history database '{self.db_path}': {e}", exc_info=True)
            raise DataAccessError(f"Could not connect to history DB: {e}", original_exception=e)

    def _init_db(self):
        """Initializes the database table if it doesn't exist."""
        logger.info(f"Initializing history database: {self.db_path}")
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            video_format TEXT,
            audio_format TEXT,
            quality TEXT,
            is_audio_only BOOLEAN,
            is_playlist BOOLEAN,
            status TEXT NOT NULL,
            error_message TEXT,
            video_id TEXT UNIQUE,
            final_filepath TEXT,
            filesize_bytes INTEGER,
            added_time TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT NOT NULL -- Use end_time for ordering completed downloads
        );
        """
        # Add index for faster lookups
        create_index_sql = "CREATE INDEX IF NOT EXISTS idx_history_endtime ON history (end_time);"
        create_vid_index_sql = "CREATE INDEX IF NOT EXISTS idx_history_videoid ON history (video_id);"


        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;") # Improve concurrency
                cursor.execute(create_table_sql)
                cursor.execute(create_index_sql)
                cursor.execute(create_vid_index_sql)
                conn.commit()
                logger.debug("History database schema ensured.")
                self._check_schema_version(cursor) # Basic schema migration placeholder
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize history database table: {e}", exc_info=True)
            # Consider more drastic action if DB init fails (e.g., delete and retry?)
            raise DataAccessError(f"Failed to initialize history DB: {e}", original_exception=e)

    def _check_schema_version(self, cursor: sqlite3.Cursor):
         """ Placeholder for future schema migration logic. """
         # Example: Check a version table or pragma user_version
         pass


    def add_or_update_entry(self, item: DownloadItem):
        """Adds a new entry or updates an existing one based on video_id."""
        if not item.end_time: # Only log completed or failed items permanently
            logger.debug(f"Skipping history entry for non-finalized item: {item.id} ({item.status})")
            return

        # Check if video ID already exists (for updates on retry/overwrite)
        existing_entry = self.get_entry_by_video_id(item.video_id) if item.video_id else None

        sql = ""
        if existing_entry:
             logger.info(f"Updating history entry for Video ID: {item.video_id}")
             sql = """
             UPDATE history SET
                 url=?, title=?, video_format=?, audio_format=?, quality=?, is_audio_only=?,
                 is_playlist=?, status=?, error_message=?, final_filepath=?, filesize_bytes=?,
                 added_time=?, start_time=?, end_time=?
             WHERE video_id = ?;
             """
             params = (
                 item.url, item.title, item.video_format, item.audio_format, item.quality,
                 item.is_audio_only, item.is_playlist, item.status, item.error_message,
                 item.final_filepath, item.filesize_bytes, item.added_time, item.start_time,
                 item.end_time, item.video_id
             )
        else:
            logger.info(f"Adding new history entry: {item.id} ({item.title or item.url})")
            sql = """
            INSERT INTO history (
                id, url, title, video_format, audio_format, quality, is_audio_only,
                is_playlist, status, error_message, video_id, final_filepath, filesize_bytes,
                added_time, start_time, end_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            params = (
                item.id, item.url, item.title, item.video_format, item.audio_format, item.quality,
                item.is_audio_only, item.is_playlist, item.status, item.error_message,
                item.video_id, item.final_filepath, item.filesize_bytes,
                item.added_time, item.start_time, item.end_time
            )

        try:
            with self._lock, self._get_connection() as conn:
                conn.execute(sql, params)
                conn.commit()
            logger.debug(f"History entry {'updated' if existing_entry else 'added'}: {item.id}")
        except sqlite3.IntegrityError as e:
             # This might happen if two threads try to insert the same ID simultaneously
             logger.error(f"History DB integrity error for ID {item.id} or Video ID {item.video_id}: {e}", exc_info=True)
             # Could retry or handle differently
        except sqlite3.Error as e:
            logger.error(f"Failed to add/update history entry for {item.id}: {e}", exc_info=True)

    def get_entry_by_video_id(self, video_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Retrieves a history entry by its video ID."""
        if not video_id:
            return None
        sql = "SELECT * FROM history WHERE video_id = ?;"
        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (video_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
             logger.error(f"Failed to get history entry by video ID {video_id}: {e}", exc_info=True)
             return None

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves the most recent history entries."""
        sql = "SELECT * FROM history ORDER BY end_time DESC LIMIT ?;"
        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (limit,))
                # Convert rows to dictionaries for easier use
                history_list = [dict(row) for row in cursor.fetchall()]
                logger.debug(f"Retrieved {len(history_list)} history entries (limit {limit}).")
                return history_list
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve download history: {e}", exc_info=True)
            return []

    def clear_history(self):
        """Clears all entries from the history table."""
        logger.warning("Clearing entire download history.")
        sql_delete = "DELETE FROM history;"
        sql_vacuum = "VACUUM;" # Reclaims disk space
        try:
            with self._lock, self._get_connection() as conn:
                conn.execute(sql_delete)
                conn.execute(sql_vacuum) # Optional: run vacuum after delete
                conn.commit()
            logger.info("Download history cleared successfully.")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear download history: {e}", exc_info=True)
            raise DataAccessError(f"Failed to clear history: {e}", original_exception=e)

    def prune_history(self, keep_limit: int):
        """Keeps only the most recent 'keep_limit' entries."""
        if keep_limit <= 0:
            logger.warning("Prune history called with non-positive limit. Clearing history instead.")
            self.clear_history()
            return

        logger.info(f"Pruning history, keeping the latest {keep_limit} entries.")
        # Find the end_time of the oldest entry to keep
        sql_find_cutoff = "SELECT end_time FROM history ORDER BY end_time DESC LIMIT 1 OFFSET ?;"
        # Delete entries older than or equal to the cutoff time
        sql_delete_old = "DELETE FROM history WHERE end_time <= ?;"
        sql_vacuum = "VACUUM;"

        try:
             with self._lock, self._get_connection() as conn:
                 cursor = conn.cursor()
                 cursor.execute(sql_find_cutoff, (keep_limit -1,)) # Offset is 0-based
                 cutoff_row = cursor.fetchone()

                 if cutoff_row:
                     cutoff_time = cutoff_row['end_time']
                     logger.debug(f"History pruning cutoff time: {cutoff_time}")
                     result = conn.execute(sql_delete_old, (cutoff_time,))
                     deleted_count = result.rowcount
                     conn.execute(sql_vacuum)
                     conn.commit()
                     logger.info(f"Pruned {deleted_count} old history entries.")
                 else:
                     logger.info("History pruning not needed (fewer than limit entries).")

        except sqlite3.Error as e:
            logger.error(f"Failed to prune download history: {e}", exc_info=True)

# --- Queue Manager ---
class QueueManager:
    """Manages loading and saving the download queue state."""

    def __init__(self, queue_path: Path = QUEUE_PATH):
        self.queue_path = queue_path

    def load_queue(self) -> List[DownloadItem]:
        """Loads the download queue from a JSON file."""
        logger.info(f"Loading download queue from: {self.queue_path}")
        if not self.queue_path.exists():
            logger.info("Queue file not found. Starting with an empty queue.")
            return []

        try:
            with open(self.queue_path, 'r', encoding='utf-8') as f:
                queue_data = json.load(f)
                # Convert dicts back to DownloadItem objects
                loaded_items = [DownloadItem.from_dict(item_data) for item_data in queue_data]
                logger.info(f"Loaded {len(loaded_items)} items from queue file.")
                # Filter out already completed/failed items if necessary (or handle them)
                active_items = [item for item in loaded_items if item.status not in ("Completed", "Failed", "Canceled")]
                if len(active_items) != len(loaded_items):
                    logger.info(f"Filtered out {len(loaded_items) - len(active_items)} completed/failed items from loaded queue.")
                return active_items
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"Failed to load or parse queue file '{self.queue_path}': {e}. Starting fresh.", exc_info=True)
            # Optionally backup corrupted queue file
            self.safe_delete_queue_file() # Avoid loading corrupted data again
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading queue '{self.queue_path}': {e}. Starting fresh.", exc_info=True)
            self.safe_delete_queue_file()
            return []

    def save_queue(self, current_queue: List[DownloadItem]):
        """Saves the current download queue state to a JSON file."""
        logger.info(f"Saving {len(current_queue)} items to queue file: {self.queue_path}")
        try:
            # Ensure directory exists
            self.queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_data = [item.to_dict() for item in current_queue]
            with open(self.queue_path, 'w', encoding='utf-8') as f:
                json.dump(queue_data, f, indent=4, ensure_ascii=False)
            logger.debug("Queue state saved successfully.")
        except (TypeError, OSError) as e:
            logger.error(f"Failed to save queue file '{self.queue_path}': {e}", exc_info=True)
            # Avoid raising error here, as saving queue isn't as critical as config
            sg.popup_warning(f"Warning: Could not save the download queue state.\n{e}", title="Queue Save Error")
        except Exception as e:
             logger.error(f"Unexpected error saving queue '{self.queue_path}': {e}", exc_info=True)
             sg.popup_warning(f"Warning: An unexpected error occurred while saving the queue state.\n{e}", title="Queue Save Error")

    def safe_delete_queue_file(self):
         """ Deletes the queue file safely. """
         logger.warning(f"Deleting potentially corrupted queue file: {self.queue_path}")
         try:
            if self.queue_path.exists():
                self.queue_path.unlink()
         except OSError as e:
              logger.error(f"Failed to delete queue file '{self.queue_path}': {e}", exc_info=True)


# ----------- Download Management -----------
class DownloadManager:
    """Manages the download process, queue, threading, and yt-dlp interaction."""

    def __init__(self, config_manager: ConfigManager, history_manager: HistoryManager, queue_manager: QueueManager, window: sg.Window):
        self.config_manager = config_manager
        self.history_manager = history_manager
        self.queue_manager = queue_manager
        self.window = window # Reference to the GUI window for updates
        self.format_detector = FormatDetector()

        self._download_queue: List[DownloadItem] = [] # Internal list representing the queue
        self._active_downloads: Dict[str, threading.Future] = {} # item.id -> Future
        self._queue_lock = threading.Lock() # Protects access to _download_queue
        self._executor: Optional[ThreadPoolExecutor] = None
        self._cancel_event = threading.Event() # Signals cancellation to worker threads
        self._is_processing = False
        self._processing_thread: Optional[threading.Thread] = None

        # Load persisted queue on startup
        self._load_initial_queue()


    def _load_initial_queue(self):
         """Loads the queue from the QueueManager."""
         with self._queue_lock:
             self._download_queue = self.queue_manager.load_queue()
             logger.info(f"Initialized DownloadManager with {len(self._download_queue)} items from saved queue.")

    def get_queue_snapshot(self) -> List[DownloadItem]:
        """Returns a copy of the current download queue."""
        with self._queue_lock:
            return list(self._download_queue) # Return a copy

    def add_to_queue(self, item: DownloadItem) -> bool:
        """Adds a validated DownloadItem to the queue."""
        logger.info(f"Attempting to add item to queue: {item.url}")

        # 1. Basic Validation
        if not URLValidator.is_valid_youtube_url(item.url):
            logger.error(f"Invalid URL provided: {item.url}")
            raise ValidationError(f"Invalid or unsupported YouTube URL:\n{item.url}")

        # 2. Check Disk Space (Estimate - Actual check happens before download)
        if not FileManager.check_disk_space(
            item.target_directory or self.config_manager.config.download_dir,
            self.config_manager.config.min_disk_space_mb
        ):
             msg = (f"Potentially insufficient disk space in "
                    f"'{item.target_directory or self.config_manager.config.download_dir}'. "
                    f"Need at least {self.config_manager.config.min_disk_space_mb}MB free. "
                    f"Add anyway?")
             logger.warning(msg)
             # Ask user if they want to proceed despite potential space issue
             if sg.popup_yes_no(msg, title="Disk Space Warning", keep_on_top=True) == 'No':
                 logger.info("User chose not to add item due to disk space warning.")
                 return False # User cancelled

        # 3. Add to internal queue (thread-safe)
        with self._queue_lock:
            # Avoid adding exact duplicates (simple URL check)
            if any(existing_item.url == item.url and existing_item.status in ("Pending", "Downloading", "Fetching") for existing_item in self._download_queue):
                 logger.warning(f"Item with URL {item.url} already in queue or downloading. Skipping.")
                 sg.popup_notify(f"URL already in queue:\n{item.url}", title="Duplicate Skipped", keep_on_top=True)
                 return False

            item.status = "Pending"
            self._download_queue.append(item)
            logger.info(f"Item added to queue: {item.id} ({item.get_display_name()})")

        # 4. Update GUI and Persist Queue
        self._update_gui_queue()
        self._persist_queue_state() # Save after adding

        # 5. Auto-start if enabled and not already processing
        if self.config_manager.config.auto_start_queue and not self.is_processing():
             self.start_processing()

        return True

    def remove_from_queue(self, item_id: str) -> bool:
        """Removes an item from the queue by its ID."""
        item_to_remove = None
        with self._queue_lock:
            for i, item in enumerate(self._download_queue):
                if item.id == item_id:
                    if item.status in ("Downloading", "Fetching", "Converting"):
                         logger.warning(f"Cannot remove item {item_id} while it's actively processing.")
                         sg.popup_error("Cannot remove an item while it is actively downloading or processing.", title="Remove Error")
                         return False
                    item_to_remove = self._download_queue.pop(i)
                    break

        if item_to_remove:
            logger.info(f"Removed item from queue: {item_id} ({item_to_remove.get_display_name()})")
            self._update_gui_queue()
            self._persist_queue_state() # Save after removal
            return True
        else:
            logger.warning(f"Could not find item with ID {item_id} to remove from queue.")
            return False

    def clear_queue(self):
        """Clears all pending items from the queue."""
        cleared_items = []
        with self._queue_lock:
            active_items = [item for item in self._download_queue if item.status in ("Downloading", "Fetching", "Converting")]
            if active_items:
                logger.warning("Cannot clear queue while items are actively processing.")
                sg.popup_error("Cannot clear the queue while downloads are in progress.\nPlease cancel active downloads first.", title="Clear Queue Error")
                return

            cleared_items = self._download_queue[:] # Copy before clearing
            self._download_queue.clear()

        if cleared_items:
            logger.info(f"Cleared {len(cleared_items)} items from the queue.")
            self._update_gui_queue()
            self._persist_queue_state() # Save after clearing
        else:
            logger.info("Queue was already empty or only contained active items.")


    def start_processing(self):
        """Starts processing the download queue in a separate thread."""
        if self._is_processing:
            logger.warning("Download processing is already active.")
            return

        if not self.get_queue_snapshot(): # Check if queue is empty using snapshot
             logger.info("Queue is empty. Nothing to process.")
             sg.popup_notify("Download queue is empty.", title="Queue Empty")
             return

        logger.info("Starting download queue processing...")
        self._is_processing = True
        self._cancel_event.clear() # Ensure cancel flag is reset

        # Create executor only when needed
        max_workers = self.config_manager.config.max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Downloader")
        logger.info(f"ThreadPoolExecutor created with max_workers={max_workers}")

        # Start the main processing loop in its own thread
        self._processing_thread = threading.Thread(target=self._queue_processor_loop, name="QueueProcessor", daemon=True)
        self._processing_thread.start()

        # Update GUI state (e.g., disable start button, enable cancel button)
        self.window.write_event_value("-DOWNLOAD_STARTED-", None)


    def stop_processing(self, cancel_active: bool = True):
        """Stops processing the queue and optionally cancels active downloads."""
        if not self._is_processing:
            logger.info("Download processing is not active.")
            return

        logger.info(f"Stopping download queue processing... Cancel active: {cancel_active}")
        self._is_processing = False # Signal loop to stop submitting new tasks

        if cancel_active:
            self._cancel_event.set() # Signal active downloads to cancel
            logger.info("Cancel event set for active downloads.")

        # Shutdown the executor gracefully
        if self._executor:
            logger.debug("Shutting down ThreadPoolExecutor...")
            # Wait=False initially allows GUI to remain responsive
            # The _queue_processor_loop should handle waiting for futures if needed
            self._executor.shutdown(wait=False, cancel_futures=cancel_active)
            self._executor = None
            logger.info("ThreadPoolExecutor shutdown initiated.")

        # Wait for the processor thread to finish (optional, with timeout)
        if self._processing_thread and self._processing_thread.is_alive():
             logger.debug("Waiting for QueueProcessor thread to join...")
             self._processing_thread.join(timeout=10) # Wait max 10 seconds
             if self._processing_thread.is_alive():
                  logger.warning("QueueProcessor thread did not join within timeout.")
             else:
                  logger.debug("QueueProcessor thread joined successfully.")
        self._processing_thread = None


        # Update items that were downloading to 'Canceled' status if cancel_active was True
        if cancel_active:
             with self._queue_lock:
                 for item in self._download_queue:
                      if item.status in ("Downloading", "Fetching", "Converting"):
                          item.status = "Canceled"
                          item.error_message = "Download canceled by user."
                          logger.info(f"Marked item {item.id} as Canceled.")

        # Persist final queue state and update GUI
        self._persist_queue_state()
        self._update_gui_queue()
        self.window.write_event_value("-DOWNLOAD_STOPPED-", None)
        logger.info("Download processing stopped.")


    def is_processing(self) -> bool:
        """Returns True if the queue is currently being processed."""
        return self._is_processing

    def _queue_processor_loop(self):
        """The main loop that pulls items from the queue and submits them to the executor."""
        logger.info("QueueProcessor thread started.")
        active_futures: Dict[threading.Future, str] = {} # Future -> item_id

        while self._is_processing:
            item_to_process = None
            with self._queue_lock:
                 # Find the next 'Pending' item
                 for item in self._download_queue:
                      if item.status == "Pending" and item.id not in self._active_downloads:
                          item_to_process = item
                          break

            if item_to_process:
                 if self._executor and len(self._active_downloads) < self.config_manager.config.max_workers:
                     logger.info(f"Submitting item for download: {item_to_process.id} ({item_to_process.get_display_name()})")
                     item_to_process.status = "Fetching" # Update status before submitting
                     self._update_gui_queue() # Show 'Fetching' status immediately

                     future = self._executor.submit(self._download_task, item_to_process)
                     self._active_downloads[item_to_process.id] = future
                     active_futures[future] = item_to_process.id # Map future back to item ID
                     future.add_done_callback(partial(self._download_task_completed, item_id=item_to_process.id))
                 else:
                      # Wait if executor is full or not ready
                      time.sleep(0.5)
            else:
                 # No pending items found, wait or exit if processing stops
                 if not self._active_downloads and not any(i.status == "Pending" for i in self.get_queue_snapshot()):
                     logger.info("No more pending items and no active downloads. QueueProcessor finishing.")
                     break # Exit loop if queue is truly empty and downloads finished
                 time.sleep(1) # Wait before checking queue again

            # Check for completed futures (optional, done_callback is preferred)
            # done, _ = concurrent.futures.wait(active_futures.keys(), timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED)
            # for future in done:
            #     item_id = active_futures.pop(future)
            #     self._download_task_completed(item_id, future)


        # Loop finished (either stopped or queue empty)
        logger.info("QueueProcessor thread finished main loop.")

        # Wait for any remaining downloads if stop wasn't called with cancel=True
        if not self._cancel_event.is_set() and self._active_downloads:
            logger.info(f"Waiting for {len(self._active_downloads)} remaining downloads to complete...")
            #concurrent.futures.wait(self._active_downloads.values()) # Wait for all submitted tasks
            # The done_callbacks should handle the final updates

        self._is_processing = False
        self.window.write_event_value("-DOWNLOAD_STOPPED-", {"queue_finished": not self._cancel_event.is_set()})
        logger.info("QueueProcessor thread exiting.")


    def _download_task_completed(self, future: threading.Future, item_id: str):
        """Callback executed when a download task future completes."""
        logger.debug(f"Download task completed callback for item ID: {item_id}")

        # Remove from active downloads
        if item_id in self._active_downloads:
            del self._active_downloads[item_id]
        # Remove from future map used in the processor loop (if using that method)


        updated_item = None
        try:
            # Get the result (the updated DownloadItem or None)
            result = future.result() # This will re-raise exceptions from the task
            if isinstance(result, DownloadItem):
                updated_item = result
                logger.info(f"Download task for {item_id} finished with status: {updated_item.status}")
                if updated_item.status == "Completed":
                    # Add to history only on successful completion
                    self.history_manager.add_or_update_entry(updated_item)
                    if self.config_manager.config.notify_on_complete:
                         sg.popup_notify(f"Download Complete:\n{updated_item.get_display_name()}", title="Download Complete", keep_on_top=True)

                elif updated_item.status == "Failed":
                     logger.error(f"Download failed for {item_id}: {updated_item.error_message}")
                     # Optionally notify user of failure
                     sg.popup_notify(f"Download Failed:\n{updated_item.get_display_name()}\n{updated_item.error_message}", title="Download Failed", keep_on_top=True)


            else:
                 # This case shouldn't normally happen if _download_task returns properly
                 logger.error(f"Download task for {item_id} returned unexpected result type: {type(result)}")
                 # Find the item and mark as failed
                 with self._queue_lock:
                      for item in self._download_queue:
                           if item.id == item_id:
                               item.status = "Failed"
                               item.error_message = "Internal error: Task returned unexpected data."
                               updated_item = item
                               break


        except Exception as e:
            # Catch exceptions raised within the _download_task
            logger.error(f"Exception occurred in download task for item {item_id}: {e}", exc_info=True)
            # Find the item and mark as failed
            with self._queue_lock:
                 for item in self._download_queue:
                     if item.id == item_id:
                         item.status = "Failed"
                         item.error_message = f"Task Error: {e}"
                         updated_item = item
                         break
            sg.popup_notify(f"Download Error:\n{updated_item.get_display_name() if updated_item else item_id}\n{updated_item.error_message if updated_item else str(e)}", title="Download Error", keep_on_top=True)


        # Update GUI and persist state regardless of outcome
        self._update_gui_queue()
        self._persist_queue_state()

        # Check if queue processing should stop (if this was the last active download and no more pending)
        # This logic might be better placed at the end of the _queue_processor_loop
        # if not self._active_downloads and not any(i.status == "Pending" for i in self.get_queue_snapshot()):
        #     if self._is_processing: # Ensure it hasn't already been stopped
        #         logger.info("All downloads finished and queue empty. Stopping processing.")
        #         self.stop_processing(cancel_active=False) # Stop naturally


    def _download_task(self, item: DownloadItem) -> DownloadItem:
        """The actual download logic executed by the thread pool worker."""
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Starting download for: {item.id} ({item.get_display_name()})")
        item.start_time = datetime.now().isoformat()
        item.status = "Fetching" # Initial status update
        self._update_gui_specific_item(item.id) # Update GUI

        try:
            # 1. Check Cancellation Flag
            if self._cancel_event.is_set():
                raise OperationCanceledError("Download canceled before start.")

            # 2. Prepare Download Options
            ydl_opts = self._prepare_ydl_options(item)
            logger.debug(f"[{thread_name}] yt-dlp options for {item.id}: {ydl_opts}")

            # 3. Check Disk Space (More accurately before download)
            target_dir = Path(ydl_opts['outtmpl']).parent
            target_dir.mkdir(parents=True, exist_ok=True) # Ensure target exists
            if not FileManager.check_disk_space(target_dir, self.config_manager.config.min_disk_space_mb):
                 raise DiskSpaceError(f"Insufficient disk space in '{target_dir}'. Need {self.config_manager.config.min_disk_space_mb}MB.")

            # 4. Execute Download with yt-dlp
            item.status = "Downloading" # Update status
            self._update_gui_specific_item(item.id) # Update GUI

            # Use context manager for yt-dlp instance
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                 logger.debug(f"[{thread_name}] Calling ydl.download() for {item.url}")
                 # Retries are handled internally by yt-dlp with 'retries' option,
                 # but we can add an outer loop for specific errors if needed.
                 ydl.download([item.url])
                 logger.info(f"[{thread_name}] ydl.download() call completed for {item.id}")


            # 5. Final Checks and Status Update
            # Note: The final filename might differ slightly due to sanitization or yt-dlp behavior.
            # We rely on the progress hook's 'filename' or infer it if necessary.
            # If the hook captured the final path, use it.
            if item.final_filepath and Path(item.final_filepath).exists():
                 logger.info(f"[{thread_name}] Download confirmed successful for {item.id}. Final path: {item.final_filepath}")
                 item.status = "Completed"
                 item.progress = 100.0
                 item.eta_seconds = 0
            else:
                # If hook didn't provide path or file doesn't exist, something went wrong post-download
                # (e.g., postprocessor error not caught, file moved/deleted unexpectedly)
                logger.error(f"[{thread_name}] Download seemed to complete, but final file not found or path missing for {item.id}. Expected near: {ydl_opts['outtmpl']}")
                # Try to find the file based on title/id (less reliable)
                possible_path = self._find_downloaded_file(ydl_opts['outtmpl'], item)
                if possible_path and possible_path.exists():
                     logger.warning(f"[{thread_name}] Found likely file: {possible_path}. Marking as complete.")
                     item.final_filepath = str(possible_path)
                     item.status = "Completed"
                     item.progress = 100.0
                     item.eta_seconds = 0
                else:
                     raise DownloadProcessError("Download finished, but the final output file could not be verified.")


        except OperationCanceledError as e:
             logger.info(f"[{thread_name}] Download canceled for item {item.id}: {e}")
             item.status = "Canceled"
             item.error_message = str(e)
        except DiskSpaceError as e:
             logger.error(f"[{thread_name}] Disk space error for item {item.id}: {e}")
             item.status = "Failed"
             item.error_message = str(e)
        except yt_dlp.utils.DownloadError as e:
            # Handle common yt-dlp errors more specifically
            err_str = str(e).lower()
            if "private video" in err_str or "login required" in err_str:
                 msg = "Video is private or requires login."
            elif "video unavailable" in err_str:
                 msg = "Video is unavailable."
            elif "blocked" in err_str:
                 msg = "Access blocked (geo-restriction or other)."
            elif "network error" in err_str or "connection timed out" in err_str:
                  msg = f"Network error: {e}"
            elif "unable to download webpage" in err_str:
                 msg = f"Could not fetch video page: {e}"
            else:
                 msg = f"yt-dlp error: {e}"
            logger.error(f"[{thread_name}] Download error for item {item.id}: {msg}", exc_info=True) # Log full trace
            item.status = "Failed"
            item.error_message = msg # Provide cleaner message to user
        except Exception as e:
            logger.error(f"[{thread_name}] Unexpected error during download task for {item.id}: {e}", exc_info=True)
            item.status = "Failed"
            item.error_message = f"Unexpected error: {e}"

        finally:
            item.end_time = datetime.now().isoformat()
            # Ensure progress is 100% on completion/failure/cancel unless downloading
            if item.status != "Downloading":
                 item.progress = 100.0 if item.status == "Completed" else item.progress
                 item.speed_mbps = 0.0
                 item.eta_seconds = 0

            logger.info(f"[{thread_name}] Finished processing item {item.id} with status: {item.status}")
            self._update_gui_specific_item(item.id) # Final GUI update for this item

        return item # Return the updated item


    def _prepare_ydl_options(self, item: DownloadItem) -> Dict[str, Any]:
        """Prepares the dictionary of options for yt-dlp based on item and config."""
        config = self.config_manager.config
        output_dir = Path(item.target_directory or config.download_dir)
        # Sanitize title for use in filename template (yt-dlp handles further sanitization)
        # Basic sanitization here prevents issues if title has path separators
        sanitized_title_for_path = re.sub(r'[\\/]', '_', item.title or 'youtube_download')

        # Define output template - organize by Title/Playlist Title if applicable
        # yt-dlp placeholders: https://github.com/yt-dlp/yt-dlp#output-template
        # Example: Downloads/YouTubeDownloader/Video Title/Video Title [1080p].mp4
        # Example: Downloads/YouTubeDownloader/Playlist Name/01 - Video Title.mp4
        if item.is_playlist:
             # Playlist structure (assuming info_dict is available later or using defaults)
             # Requires playlist metadata extraction first - placeholder
             # filename_template = output_dir / "%(playlist_title)s" / "%(playlist_index)s - %(title)s [%(height)sp].%(ext)s"
             # Simpler playlist template for now:
             filename_template = output_dir / f"{sanitized_title_for_path}_playlist" / "%(playlist_index)s - %(title)s.%(ext)s"
        else:
             # Single video template
             # filename_template = output_dir / sanitized_title_for_path / f"%(title)s [%(height)sp].%(ext)s"
             # Simpler single video template:
             filename_template = output_dir / "%(title)s [%(id)s].%(ext)s" # Include ID for uniqueness


        # Ensure the parent directory exists
        filename_template.parent.mkdir(parents=True, exist_ok=True)

        opts = {
            'outtmpl': str(filename_template),
            'format': self._get_format_selection(item),
            'quiet': True, # Suppress console output from yt-dlp
            'progress_hooks': [partial(self._progress_hook, item.id)],
            'postprocessor_hooks': [partial(self._postprocessor_hook, item.id)],
            'final_hooks': [partial(self._final_hook, item.id)],
            'noprogress': True, # Disable yt-dlp's default progress bar
            'noplaylist': not item.is_playlist, # Process as playlist only if specified
            'playlistend': 1 if not item.is_playlist else None, # Download only first item if not playlist
            'retries': config.max_retries,
            'fragment_retries': config.max_retries,
            'retry_sleep': {'http': config.retry_delay, 'fragment': config.retry_delay},
            'ignoreerrors': False, # Stop on errors unless overridden
            'no_warnings': False, # Show warnings in logs
            'writethumbnail': config.extract_thumbnail,
            'writeinfojson': config.include_metadata, # Save video metadata to a .info.json file
            'writesubtitles': item.download_subtitles,
            'writeautomaticsub': item.download_subtitles, # Download auto-generated subs if no others
            'subtitleslangs': ['en', 'en-US', 'en-GB'] if item.download_subtitles else None, # Prioritize English subs
            'subtitlesformat': 'srt/vtt' if item.download_subtitles else None,
            'embedsubtitles': False, # Typically better to keep subs separate
            'embedthumbnail': config.extract_thumbnail, # Embed thumbnail if possible
            'addmetadata': config.include_metadata, # Embed basic metadata if possible
            'postprocessors': [], # Will be added below
            'concurrent_fragment_downloads': config.max_workers,
            'http_chunk_size': CHUNK_SIZE,
            'buffer_size': f"{CHUNK_SIZE * 2}", # Set buffer size (e.g., double chunk size)
            'socket_timeout': 30, # Network timeout in seconds
            'cachedir': str(CACHE_DIR), # Use dedicated cache dir
            'continuedl': config.resume_downloads,
            'ratelimit': config.rate_limit_kbs * 1024 if config.rate_limit_kbs > 0 else None,
            'proxy': config.proxy or None,
            'check_formats': False, # Assume format selection is valid; speeds up start
            'forcejson': False,
            'verbose': False, # Keep logs clean unless debugging
            'merge_output_format': item.video_format if not item.is_audio_only else None, # Ensure final container is correct
            'ffmpeg_location': SystemManager.find_ffmpeg() if config.use_ffmpeg_if_available else None,
            'prefer_ffmpeg': True if config.use_ffmpeg_if_available and SystemManager.find_ffmpeg() else False,
            'geo_bypass': True, # Attempt to bypass geo-restrictions (may not always work)
            'verbose': logger.level == logging.DEBUG, # Enable verbose ytdlp logging if app logging is DEBUG
             'user_agent': USER_AGENT,
             # Archive file to prevent re-downloading (optional)
             'download_archive': config.archive_path if config.download_archive_enabled else None,

        }

        # --- Post-processors ---
        if item.is_audio_only:
            # Extract Audio Postprocessor
            opts['postprocessors'].append({
                'key': 'FFmpegExtractAudio',
                'preferredcodec': item.audio_format,
                'preferredquality': '192', # Standard quality, could be configurable
                'nopostoverwrites': False, # Allow overwriting intermediate files
            })
        elif config.include_metadata or config.extract_thumbnail:
            # Metadata/Thumbnail Embedding (only if FFmpeg is available)
             if opts.get('ffmpeg_location'):
                 if config.include_metadata:
                     opts['postprocessors'].append({
                         'key': 'FFmpegMetadata',
                         'add_metadata': True,
                     })
                 if config.extract_thumbnail:
                      # EmbedThumbnail is added automatically by yt-dlp if writethumbnail=True and ffmpeg is present
                      pass
                      # opts['postprocessors'].append({
                      #     'key': 'EmbedThumbnail',
                      #     'already_have_thumbnail': False, # Let yt-dlp handle finding/downloading
                      # })

        # If merging is likely needed (separate video/audio), ensure FFmpeg is preferred
        # Note: yt-dlp often automatically uses FFmpegMerger PP when needed based on format selection
        # if "+" in opts['format'] and not item.is_audio_only and opts.get('ffmpeg_location'):
        #     opts['postprocessors'].append({'key': 'FFmpegMerger'})


        # --- Conditional Options ---
        if "best" in item.quality or item.quality == "highest":
            # No specific height filter needed for 'best'
             pass
        elif item.quality.endswith('p'): # e.g., "1080p"
            try:
                height = int(item.quality[:-1])
                # Add height filter to format selection if not already implied
                # This is complex as format string might already contain height.
                # A safer approach is to let the complex format string handle it.
                # Example: (Modify format string directly - potentially risky)
                # if f'height<={height}' not in opts['format']:
                #    opts['format'] = opts['format'].replace('bestvideo', f'bestvideo[height<={height}]')
                pass # Rely on _get_format_selection to handle quality string
            except ValueError:
                logger.warning(f"Invalid quality format '{item.quality}'. Using default format selection.")


        return opts


    def _get_format_selection(self, item: DownloadItem) -> str:
        """Determines the yt-dlp format selection string based on item properties."""
        video_pref = f"[ext={item.video_format}]" if item.video_format else ""
        audio_pref = f"[ext={item.audio_format}]" if item.audio_format else "[ext=m4a]" # Default audio merge format

        if item.is_audio_only:
            # Select best audio in the desired format, fallback to best audio overall
            # Use 'ba' shortcut for best audio
            return f"ba{audio_pref}/bestaudio/ba"

        # --- Video Download ---
        quality = item.quality.lower()

        # Simple best quality
        if quality == "best" or quality == "highest":
             # Best video in preferred format + best audio (default m4a), merged into preferred format / fallback to best overall
             return f"bv{video_pref}+ba/b{video_pref}/bv+ba/b"


        # Quality selection (e.g., 1080p, 720p)
        height_filter = ""
        if quality.endswith('p'):
             try:
                 height = int(quality[:-1])
                 # Prefer exact height, then <= height, then best overall
                 # Use vcodec!=none to avoid audio-only formats in video selection
                 height_filter = f"[height={height}]/[height<={height}]"
                 # More robust: Include format preference in each step
                 return (f"bv{video_pref}{height_filter}+ba/b{video_pref}{height_filter}/" # Specific quality video + best audio
                         f"bv{video_pref}+ba/b{video_pref}/bv+ba/b") # Fallbacks
             except ValueError:
                  logger.warning(f"Invalid quality '{item.quality}'. Falling back to 'best'.")
                  return f"bv{video_pref}+ba/b{video_pref}/bv+ba/b" # Fallback to best

        # Default fallback if quality string is unrecognized
        logger.warning(f"Unrecognized quality setting '{item.quality}'. Falling back to 'best'.")
        return f"bv{video_pref}+ba/b{video_pref}/bv+ba/b"


    def _progress_hook(self, item_id: str, d: Dict[str, Any]):
        """yt-dlp progress hook to update GUI."""
        # This hook is called frequently, keep it efficient.
        # Check for cancellation signal
        if self._cancel_event.is_set():
            logger.debug(f"Cancel event detected in progress hook for {item_id}. Raising exception.")
            raise OperationCanceledError("Download canceled by user request.") # Signal yt-dlp to stop

        status = d.get('status')
        item = self._find_item_in_queue(item_id)
        if not item:
            logger.warning(f"Progress hook called for unknown item ID: {item_id}")
            return

        # Update item state based on hook data
        if status == 'downloading':
            item.status = "Downloading"
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded_bytes = d.get('downloaded_bytes')

            if total_bytes and downloaded_bytes:
                 item.progress = (downloaded_bytes / total_bytes) * 100.0
                 item.filesize_bytes = total_bytes # Update file size estimate

            speed_bps = d.get('speed') # Bytes per second
            if speed_bps:
                 item.speed_mbps = speed_bps / (1024 * 1024)
            else:
                 item.speed_mbps = 0.0


            eta = d.get('eta') # Seconds
            if eta:
                 item.eta_seconds = int(eta)
            else:
                 item.eta_seconds = 0

            # Store the potential final filename (can change with post-processing)
            # filename = d.get('filename') or d.get('tmpfilename') # 'filename' is usually the final one after merging
            # if filename:
                 # item.final_filepath = filename # Keep track of the latest known path

            # Get title if not already set (might appear during download)
            info_dict = d.get('info_dict')
            if info_dict and not item.title:
                 item.title = info_dict.get('title')
                 logger.debug(f"Updated title from info_dict for {item_id}: {item.title}")


        elif status == 'finished':
            # This means the download part is done, post-processing might start
            item.progress = 100.0
            item.speed_mbps = 0.0
            item.eta_seconds = 0
            item.status = "Converting" # Assume post-processing might happen
            # Store final filename reported by the download phase
            filename = d.get('filename')
            if filename:
                 item.final_filepath = filename
                 logger.info(f"Download phase finished for {item_id}. File: {filename}")
                 # Check if file exists after download phase
                 if not Path(filename).exists():
                      logger.warning(f"File reported as finished '{filename}' does not exist immediately after download hook.")

        elif status == 'error':
            item.status = "Failed"
            item.error_message = "yt-dlp reported an error during download."
            item.progress = 0 # Reset progress on error
            logger.error(f"yt-dlp progress hook reported error for {item_id}. Raw data: {d}")


        # Send update to GUI thread safely
        self._update_gui_specific_item(item_id)

    def _postprocessor_hook(self, item_id: str, d: Dict[str, Any]):
        """yt-dlp hook called during postprocessing."""
        status = d.get('status')
        postprocessor_name = d.get('postprocessor')
        item = self._find_item_in_queue(item_id)
        if not item: return

        logger.debug(f"Postprocessor hook for {item_id}: Status='{status}', PP='{postprocessor_name}'")

        if status == 'started' or status == 'processing':
             item.status = f"Converting ({postprocessor_name})" if postprocessor_name else "Converting"
        elif status == 'finished':
             item.status = "Converting (done)" # Or maybe back to 'Completed' if this is the last one?
             # The 'final_hook' is usually better for final status updates.

        # Update GUI
        self._update_gui_specific_item(item_id)

    def _final_hook(self, item_id: str, d: Dict[str, Any]):
        """yt-dlp hook called after download and all postprocessing."""
        status = d.get('status') # Should be 'finished' or 'error'
        filename = d.get('filename') # The final filename
        info_dict = d.get('info_dict') # Contains final metadata
        item = self._find_item_in_queue(item_id)
        if not item: return

        logger.info(f"Final hook for {item_id}: Status='{status}', Final File='{filename}'")

        if status == 'finished' and filename:
             item.final_filepath = filename
             item.filesize_bytes = info_dict.get('filesize') or item.filesize_bytes # Update with final size if available
             # Title might also be finalized here
             if info_dict and info_dict.get('title'):
                  item.title = info_dict.get('title')

             # Status is typically set to Completed in the main task after this hook.
             # Avoid setting it here as the main task does verification.
             # item.status = "Completed"
             item.progress = 100.0


        elif status == 'error':
             # Error might have occurred during post-processing
             item.status = "Failed"
             item.error_message = item.error_message or "Error during post-processing." # Keep previous error if any
             logger.error(f"Final hook reported error for {item_id}. Filename: {filename}")


        # Update GUI (optional, as main task completion handles final update)
        # self._update_gui_specific_item(item.id)


    def _find_item_in_queue(self, item_id: str) -> Optional[DownloadItem]:
         """Finds an item in the queue by ID (not thread-safe, use within locked sections or carefully)."""
         # Note: Accessing _download_queue directly is okay here IF called from methods
         # already holding the _queue_lock OR if called from the item's own download thread
         # where modification of *that specific item* is expected. Be cautious.
         for item in self._download_queue:
             if item.id == item_id:
                 return item
         return None

    def _update_gui_queue(self):
        """Sends the entire current queue state to the GUI thread."""
        queue_copy = self.get_queue_snapshot() # Get thread-safe copy
        try:
            self.window.write_event_value("-UPDATE_QUEUE_DISPLAY-", {'queue': queue_copy})
        except Exception as e:
            logger.error(f"Error writing queue update event to GUI: {e}", exc_info=True)


    def _update_gui_specific_item(self, item_id: str):
         """Sends an update for a specific item to the GUI thread."""
         with self._queue_lock: # Need lock to safely access item
            item = self._find_item_in_queue(item_id)
            if item:
                 item_copy = DownloadItem.from_dict(item.to_dict()) # Send a copy
            else:
                 return # Item might have been removed

         try:
             self.window.write_event_value("-UPDATE_ITEM_DISPLAY-", {'item': item_copy})
         except Exception as e:
             # Window might be closing
             logger.warning(f"Could not write item update event to GUI (maybe closing?): {e}")


    def _persist_queue_state(self):
         """Saves the current queue state using QueueManager."""
         queue_copy = self.get_queue_snapshot()
         try:
             self.queue_manager.save_queue(queue_copy)
         except Exception as e:
             # Log error but don't crash the download process
             logger.error(f"Failed to persist queue state: {e}", exc_info=True)

    def _find_downloaded_file(self, output_template: str, item: DownloadItem) -> Optional[Path]:
         """ Attempts to find the downloaded file if the exact path is missing. """
         logger.warning(f"Attempting to locate missing file for item {item.id} near template {output_template}")
         base_dir = Path(output_template).parent
         if not base_dir.exists(): return None

         # Try matching based on video ID or sanitized title
         patterns = []
         if item.video_id:
              patterns.append(f"*{item.video_id}*.*") # Match ID anywhere in name
         if item.title:
              sanitized = FileManager.sanitize_filename(item.title)
              patterns.append(f"{sanitized}*.*") # Match sanitized title start

         for pattern in patterns:
              try:
                   for potential_file in base_dir.glob(pattern):
                        if potential_file.is_file():
                             logger.info(f"Found potential match for {item.id}: {potential_file}")
                             return potential_file
              except Exception as e:
                   logger.error(f"Error searching for file with pattern '{pattern}' in '{base_dir}': {e}")

         return None

# ----------- Custom Exceptions for Specific Cases -----------
class OperationCanceledError(Exception):
     """Exception raised when an operation is canceled by the user."""
     pass

class DataAccessError(DownloaderError):
     """ Errors related to accessing persistent data (DB, files). """
     pass


# ----------- Main GUI Application -----------

class YoutubeDownloaderGUI:
    """Main GUI class for the YouTube Downloader."""

    def __init__(self):
        logger.info("Initializing YouTubeDownloaderGUI...")
        self.config_manager = ConfigManager()
        sg.theme(self.config_manager.config.theme) # Set theme early

        self.history_manager = HistoryManager()
        self.queue_manager = QueueManager()
        self.window: sg.Window = self._create_window()

        # Init DownloadManager *after* window exists
        self.download_manager = DownloadManager(
            self.config_manager,
            self.history_manager,
            self.queue_manager,
            self.window
        )

        # Initialize UI elements
        self._update_queue_display()
        self._update_history_display()
        self._update_button_states()
        self._update_status_bar("Ready.")

        # System Tray (optional, can be enabled via config)
        self.tray: Optional[sg.SystemTray] = self._create_tray()

        logger.info("GUI Initialized successfully.")


    def _create_window(self) -> sg.Window:
        """Creates the main application window layout."""
        config = self.config_manager.config
        menu_def = [
            ['&File', ['&Settings', 'Open &Download Folder', 'Open &Logs Folder', '---', 'E&xit']],
            ['&Queue', ['&Start Downloads', '&Cancel Downloads', 'Clear &Queue', '---', 'Save Queue Now']],
            ['&History', ['Refresh &History', 'Clear &History']],
            ['&Help', ['&About', 'Check for &Updates...']]
        ]

        # --- Input Section ---
        input_frame = sg.Frame('Input URL', [
            [sg.Input(key='-URL-', expand_x=True, tooltip="Enter YouTube video or playlist URL"),
             sg.Button('Paste', key='-PASTE-', button_color=('white', sg.theme_button_color()[1]), tooltip="Paste URL from clipboard (Ctrl+V)")],
             #[sg.Button('Detect Info', key='-DETECT-', tooltip="Fetch video info and formats (Ctrl+D)")]
        ], expand_x=True)

        # --- Options Section ---
        options_layout = [
             [sg.Text("Save To:", size=(8,1)), sg.Input(config.download_dir, key='-DIR-', size=(40,1), enable_events=True, tooltip="Directory to save downloads"), sg.FolderBrowse(target='-DIR-')],
             [sg.Frame('Video', [
                 [sg.Text("Format:"), sg.Combo(list(ALLOWED_VIDEO_FORMATS), default_value=config.default_video_format, key='-VIDEO_FORMAT-', size=(6,1), readonly=True),
                  sg.Text("Quality:"), sg.Combo(['best', '1440p', '1080p', '720p', '480p'], default_value=config.default_quality, key='-QUALITY-', size=(8,1), readonly=True)],
                 [sg.Checkbox('Subtitles', key='-SUBS-', default=False, tooltip="Download subtitles (srt/vtt) if available")]
             ]),
             sg.Frame('Audio', [
                  [sg.Checkbox('Audio Only', key='-AUDIO_ONLY-', default=False, enable_events=True, tooltip="Download only the audio track")],
                  [sg.Text("Format:"), sg.Combo(list(ALLOWED_AUDIO_FORMATS), default_value=config.default_audio_format, key='-AUDIO_FORMAT-', size=(6,1), readonly=True, disabled=True)] # Disabled initially
             ])],
             [sg.Checkbox('Process as Playlist', key='-PLAYLIST-', default=False, tooltip="Treat URL as a playlist and download all items")]
        ]
        options_frame = sg.Frame('Download Options', options_layout, expand_x=True)


        # --- Queue Section ---
        queue_headings = [' Status ', ' Title/URL ', ' Quality ', ' Progress ', ' Speed ', ' ETA ']
        queue_col_widths = [10, 40, 8, 18, 10, 8]
        # Use a Table for better queue visualization
        queue_frame = sg.Frame('Download Queue', [
             [sg.Table(values=[], headings=queue_headings, key='-QUEUE_TABLE-',
                       col_widths=queue_col_widths,
                       auto_size_columns=False,
                       justification='left',
                       num_rows=8,
                       display_row_numbers=False,
                       enable_events=True, # Needed for selection
                       select_mode=sg.TABLE_SELECT_MODE_BROWSE, # Select single rows
                       expand_x=True, expand_y=True,
                       tooltip="Current download queue. Click row to select.")],
             [sg.Button('Add to Queue', key='-ADD-', tooltip="Add the URL with selected options to the queue (Ctrl+A)"),
              sg.Button('Remove Selected', key='-REMOVE-', disabled=True, tooltip="Remove the selected item from the queue"),
              sg.Button('Clear Queue', key='-CLEAR_QUEUE-', tooltip="Remove all pending items from the queue")]
        ], expand_x=True, expand_y=True)


        # --- History Section ---
        history_headings = [' Date ', ' Title ', ' Quality ', ' Status ', ' Path ']
        history_col_widths = [15, 40, 8, 8, 30]
        history_frame = sg.Frame('Download History', [
            [sg.Table(values=[], headings=history_headings, key='-HISTORY_TABLE-',
                      col_widths=history_col_widths,
                      auto_size_columns=False,
                      justification='left',
                      num_rows=6,
                      display_row_numbers=False,
                      enable_events=True,
                      select_mode=sg.TABLE_SELECT_MODE_BROWSE,
                      expand_x=True, expand_y=False, # History height is fixed
                      tooltip="Recently completed or failed downloads. Click row to select.")],
            [sg.Button('Open File Location', key='-OPEN_HISTORY_LOC-', disabled=True, tooltip="Open the folder containing the selected history item"),
             sg.Button('Copy URL', key='-COPY_HISTORY_URL-', disabled=True, tooltip="Copy the URL of the selected history item"),
             sg.Button('Retry Download', key='-RETRY_HISTORY-', disabled=True, tooltip="Add the selected failed item back to the queue"),
             sg.Button('Clear History', key='-CLEAR_HISTORY-', tooltip="Clear all history entries")]
        ], expand_x=True)

        # --- Controls Section ---
        controls_frame = sg.Frame('Controls', [
             [sg.Button(' Start Downloads ', key='-START-', size=(15,1), button_color=('white', 'green'), tooltip="Start processing the download queue (Ctrl+S)"),
              sg.Button(' Cancel Downloads ', key='-CANCEL-', size=(15,1), button_color=('white', 'red'), disabled=True, tooltip="Stop processing and cancel active downloads (Ctrl+C)"),
              sg.Button(' Settings ', key='-SETTINGS-', size=(15,1), tooltip="Open application settings"),
              sg.Button(' Open Folder ', key='-OPEN_FOLDER-', size=(15,1), tooltip="Open the main download folder (Ctrl+O)")]
        ], expand_x=True)


        # --- Status Bar ---
        status_bar = sg.StatusBar("Initializing...", key='-STATUS-', size=(80, 1), relief=sg.RELIEF_SUNKEN, expand_x=True)

        # --- Main Layout ---
        layout = [
            [sg.Menu(menu_def)],
            [input_frame],
            [options_frame],
            [queue_frame],
            [history_frame],
            [controls_frame],
            [status_bar]
        ]

        # Create window
        window = sg.Window(f"{APP_NAME} v{VERSION}", layout,
                           resizable=True, finalize=True, # Finalize allows immediate element access/binding
                           enable_close_attempted_event=True, # Catch closing attempt
                           #icon=get_icon_path() # Add application icon
                          )

        # Bind keyboard shortcuts
        window.bind('<Control-q>', '-EXIT-')
        window.bind('<Control-Q>', '-EXIT-')
        window.bind('<Control-a>', '-ADD-KEY-') # Add to queue
        window.bind('<Control-A>', '-ADD-KEY-')
        window.bind('<Control-s>', '-START-KEY-') # Start download
        window.bind('<Control-S>', '-START-KEY-')
        #window.bind('<Control-d>', '-DETECT-KEY-') # Detect format (removed button for now)
        #window.bind('<Control-D>', '-DETECT-KEY-')
        window.bind('<Control-c>', '-CANCEL-KEY-') # Cancel download
        window.bind('<Control-C>', '-CANCEL-KEY-')
        window.bind('<Control-o>', '-OPEN_FOLDER-KEY-') # Open folder
        window.bind('<Control-O>', '-OPEN_FOLDER-KEY-')
        window.bind('<Control-v>', '-PASTE-KEY-') # Paste URL
        window.bind('<Control-V>', '-PASTE-KEY-')
        window.bind('<Delete>', '-REMOVE-KEY-') # Remove selected queue item


        # Make table columns resizable by dragging header (requires underlying Tkinter access)
        # This is advanced and might be fragile. Simple fixed widths are safer.
        # try:
        #    table = window['-QUEUE_TABLE-'].Widget # type: sg.tk.Treeview
        #    # Add bindings if needed
        # except:
        #     logger.warning("Could not access Treeview widget for queue table resizing.")

        return window

    def _create_tray(self) -> Optional[sg.SystemTray]:
         """ Creates the system tray icon and menu. """
         # Todo: Add option in settings to enable/disable tray
         if True: # Or check config self.config_manager.config.enable_tray:
             logger.info("Creating system tray icon.")
             tray_tooltip = f"{APP_NAME} v{VERSION}"
             # Tray menu definition ('!' denotes disabled)
             tray_menu = ['UNUSED', [
                 'Show/Hide',
                 'Start Downloads',
                 'Cancel Downloads',
                 '---',
                 'Settings',
                 'Exit'
             ]]
             # Use base64 encoded icon data or a path to a .ico/.png file
             # icon_path = get_icon_path(base64=True) # Function to get icon data/path
             try:
                 # Placeholder for icon data/path
                 icon_data = sg.DEFAULT_BASE64_ICON # Default PySimpleGUI icon
                 tray = sg.SystemTray(menu=tray_menu, tooltip=tray_tooltip, data=icon_data, key='-TRAY-')
                 tray.show_message("YouTube Downloader", "Application started.") # Initial notification
                 return tray
             except Exception as e:
                  logger.error(f"Failed to create system tray icon: {e}", exc_info=True)
                  sg.popup_warning("Could not create system tray icon.\nTray functionality will be disabled.", title="Tray Error")
                  return None
         return None


    def _update_queue_display(self, queue_data: Optional[List[DownloadItem]] = None):
        """Updates the queue table in the GUI. If queue_data is None, fetches from DownloadManager."""
        if queue_data is None:
            queue_data = self.download_manager.get_queue_snapshot()

        table_data = []
        for item in queue_data:
            progress_str = f"{item.progress:.1f}%" if item.status == "Downloading" else ""
            speed_str = f"{item.speed_mbps:.2f} MB/s" if item.status == "Downloading" and item.speed_mbps > 0 else ""
            eta_str = f"{item.eta_seconds}s" if item.status == "Downloading" and item.eta_seconds > 0 else ""

            table_data.append([
                item.status,
                item.get_display_name(),
                f"{item.quality}{' (Audio)' if item.is_audio_only else ''}",
                progress_str,
                speed_str,
                eta_str,
                item.id # Hidden data - store item ID for lookup on row selection
            ])

        try:
            self.window['-QUEUE_TABLE-'].update(values=table_data)
            # Re-apply selection if necessary (might be complex)
        except Exception as e:
            logger.error(f"Error updating queue table GUI: {e}", exc_info=True)

    def _update_specific_item_display(self, item: DownloadItem):
        """Updates a single row in the queue table."""
        table: sg.Table = self.window['-QUEUE_TABLE-']
        if not table or not hasattr(table, 'Values'): return # Window closed or element missing

        current_values = table.Values # Get current table data [[row1], [row2], ...]
        updated = False
        for i, row in enumerate(current_values):
            if len(row) > 6 and row[6] == item.id: # Check hidden item ID
                progress_str = f"{item.progress:.1f}%" if item.status == "Downloading" else ""
                speed_str = f"{item.speed_mbps:.2f} MB/s" if item.status == "Downloading" and item.speed_mbps > 0 else ""
                eta_str = f"{item.eta_seconds}s" if item.status == "Downloading" and item.eta_seconds > 0 else ""

                updated_row = [
                    item.status,
                    item.get_display_name(),
                    f"{item.quality}{' (Audio)' if item.is_audio_only else ''}",
                    progress_str,
                    speed_str,
                    eta_str,
                    item.id
                ]
                current_values[i] = updated_row # Update the row in the list
                updated = True
                break

        if updated:
             try:
                table.update(values=current_values) # Update the entire table with modified data
             except Exception as e:
                 logger.error(f"Error updating specific item in table GUI: {e}", exc_info=True)
        else:
             logger.warning(f"Tried to update item ID {item.id} in table, but it wasn't found.")


    def _update_history_display(self):
        """Updates the history table in the GUI."""
        try:
            limit = self.config_manager.config.history_limit
            history_data = self.history_manager.get_history(limit=limit)
            table_data = []
            for entry in history_data:
                # Convert stored ISO time string back to datetime for formatting
                try:
                     dt_obj = datetime.fromisoformat(entry.get('end_time', '')) if entry.get('end_time') else None
                     date_str = dt_obj.strftime('%Y-%m-%d %H:%M') if dt_obj else 'N/A'
                except ValueError:
                     date_str = entry.get('end_time', 'N/A') # Fallback if format is wrong

                table_data.append([
                    date_str,
                    entry.get('title', entry.get('url', 'N/A')), # Show title or URL
                    entry.get('quality', 'N/A'),
                    entry.get('status', 'N/A'),
                    entry.get('final_filepath', 'N/A'),
                    entry.get('id'), # Hidden: item ID
                    entry.get('url'), # Hidden: URL for retry/copy
                    entry.get('final_filepath') # Hidden: full path for opening
                ])

            self.window['-HISTORY_TABLE-'].update(values=table_data)
            self.window['-OPEN_HISTORY_LOC-'].update(disabled=True)
            self.window['-COPY_HISTORY_URL-'].update(disabled=True)
            self.window['-RETRY_HISTORY-'].update(disabled=True)

        except Exception as e:
            logger.error(f"Error updating history table GUI: {e}", exc_info=True)


    def _update_button_states(self):
        """Enables/disables buttons based on application state."""
        is_processing = self.download_manager.is_processing()
        queue_empty = not self.download_manager.get_queue_snapshot()
        selected_queue_rows = self.window['-QUEUE_TABLE-'].SelectedRows
        selected_history_rows = self.window['-HISTORY_TABLE-'].SelectedRows

        try:
            self.window['-START-'].update(disabled=is_processing or queue_empty)
            self.window['-CANCEL-'].update(disabled=not is_processing)
            self.window['-ADD-'].update(disabled=is_processing) # Prevent adding while processing? Maybe allow?
            self.window['-REMOVE-'].update(disabled=is_processing or not selected_queue_rows)
            self.window['-CLEAR_QUEUE-'].update(disabled=is_processing or queue_empty)

            self.window['-OPEN_HISTORY_LOC-'].update(disabled=not selected_history_rows)
            self.window['-COPY_HISTORY_URL-'].update(disabled=not selected_history_rows)
            # Enable retry only if a single failed item is selected in history
            can_retry = False
            if len(selected_history_rows) == 1:
                 try:
                     selected_row_data = self.window['-HISTORY_TABLE-'].Values[selected_history_rows[0]]
                     if len(selected_row_data) > 3 and selected_row_data[3] == 'Failed': # Check status column
                          can_retry = True
                 except IndexError:
                      pass # Row index out of bounds
            self.window['-RETRY_HISTORY-'].update(disabled=not can_retry)


            # Update tray menu (if tray exists) - Requires specific key handling or rebuilding menu
            if self.tray:
                # Example: Disable 'Start' if processing or empty, disable 'Cancel' if not processing
                # Tray updates are more complex, might need helper function
                pass # self.update_tray_menu_state(is_processing, queue_empty)

        except Exception as e:
            # Handle cases where window/elements might not exist yet or are closing
            logger.warning(f"Error updating button states (window might be closing): {e}")


    def _update_status_bar(self, text: str):
        """Updates the text in the status bar."""
        try:
            self.window['-STATUS-'].update(value=text)
        except Exception as e:
            logger.warning(f"Error updating status bar (window might be closing): {e}")


    def _handle_add_to_queue(self, values: Dict[str, Any]):
        """Handles the 'Add to Queue' action."""
        url = URLValidator.sanitize_url(values['-URL-'])
        if not url:
            sg.popup_error("Please enter a YouTube URL.", title="Input Error")
            return

        try:
            # Extract ID to check type (video/playlist) and for uniqueness checks
            extracted = URLValidator.extract_id(url)
            video_id = None
            is_playlist_url = False
            if extracted:
                 if extracted['type'] == 'playlist':
                     is_playlist_url = True
                 # Use video ID even if it's part of a playlist URL for history tracking
                 if extracted['type'] == 'video':
                      video_id = extracted['id']

            # Create DownloadItem from GUI values
            item = DownloadItem(
                url=url,
                video_format=values['-VIDEO_FORMAT-'],
                audio_format=values['-AUDIO_FORMAT-'],
                quality=values['-QUALITY-'],
                is_audio_only=values['-AUDIO_ONLY-'],
                is_playlist=values['-PLAYLIST-'] or is_playlist_url, # Explicit checkbox or detected playlist URL
                download_subtitles=values['-SUBS-'],
                target_directory=values['-DIR-'] or None, # Use None if empty to fallback to global
                video_id=video_id # Store extracted video ID if available
            )

            # Let DownloadManager handle validation and adding
            if self.download_manager.add_to_queue(item):
                 self.window['-URL-'].update('') # Clear URL input after successful add
                 self._update_status_bar(f"Added '{item.get_display_name()}' to queue.")
                 # Focus back on URL input?
                 # self.window['-URL-'].set_focus()


        except (ValidationError, DiskSpaceError) as e:
             logger.error(f"Failed to add item to queue: {e}", exc_info=True)
             sg.popup_error(f"Failed to add to queue:\n{e}", title="Add Error")
             self._update_status_bar(f"Error adding URL: {e}")
        except Exception as e:
            logger.error(f"Unexpected error adding item to queue: {e}", exc_info=True)
            sg.popup_error(f"An unexpected error occurred:\n{e}", title="Add Error")
            self._update_status_bar("Unexpected error adding URL.")


    def _handle_remove_selected(self):
        """Handles removing the selected item from the queue table."""
        selected_rows = self.window['-QUEUE_TABLE-'].SelectedRows
        if not selected_rows:
            sg.popup_error("Please select an item from the queue table to remove.", title="Remove Error")
            return

        try:
            row_index = selected_rows[0]
            item_id_to_remove = self.window['-QUEUE_TABLE-'].Values[row_index][6] # Get hidden ID

            if self.download_manager.remove_from_queue(item_id_to_remove):
                self._update_status_bar("Removed item from queue.")
                self.window['-QUEUE_TABLE-'].update(select_rows=[]) # Deselect rows
            else:
                # Error message already shown by DownloadManager if it failed
                 pass

        except IndexError:
             logger.error("Error removing item: Selected row index out of bounds.")
             sg.popup_error("Could not get data for the selected row. Please try again.", title="Remove Error")
        except Exception as e:
            logger.error(f"Unexpected error removing item from queue: {e}", exc_info=True)
            sg.popup_error(f"An unexpected error occurred:\n{e}", title="Remove Error")


    def _handle_retry_selected_history(self):
         """ Adds a failed item from history back to the queue. """
         selected_rows = self.window['-HISTORY_TABLE-'].SelectedRows
         if not selected_rows:
              sg.popup_error("Please select a failed item from the history table to retry.", title="Retry Error")
              return

         try:
              row_index = selected_rows[0]
              history_entry = self.window['-HISTORY_TABLE-'].Values[row_index]
              # Extract necessary data from the hidden columns or query HistoryManager again
              item_id = history_entry[5]
              url = history_entry[6]
              # Get the full original item details (might need to query DB if not all stored in table)
              # For simplicity, let's recreate a basic item from table data
              # A more robust way is fetch the full item details from DB using item_id

              if not url:
                  sg.popup_error("Could not retrieve URL for the selected history item.", title="Retry Error")
                  return

              # Recreate a DownloadItem (potentially missing some details)
              # Need to map table columns back to DownloadItem fields carefully
              quality = history_entry[2]
              is_audio = '(Audio)' in quality
              quality_clean = quality.replace(' (Audio)', '') if is_audio else quality

              item = DownloadItem(
                  url=url,
                  title=history_entry[1], # Title from table
                  quality=quality_clean,
                  is_audio_only=is_audio,
                  # Need to get video_format, audio_format, playlist status etc.
                  # This requires storing more info in the table or fetching from DB
                  # Using defaults for now:
                  video_format=self.config_manager.config.default_video_format,
                  audio_format=self.config_manager.config.default_audio_format,
                  is_playlist=False, # Assume not playlist unless we store/fetch this
                  download_subtitles=False, # Assume false
              )
              logger.info(f"Retrying download for history item: {item.url}")

              # Add the recreated item to the queue
              if self.download_manager.add_to_queue(item):
                  self._update_status_bar(f"Added '{item.get_display_name()}' back to queue for retry.")
              # else: error handled by add_to_queue

         except IndexError:
              logger.error("Error retrying item: Selected row index out of bounds.")
              sg.popup_error("Could not get data for the selected history row. Please try again.", title="Retry Error")
         except Exception as e:
              logger.error(f"Unexpected error retrying history item: {e}", exc_info=True)
              sg.popup_error(f"An unexpected error occurred during retry:\n{e}", title="Retry Error")


    def _handle_open_history_location(self):
         """ Opens the folder containing the selected history item. """
         selected_rows = self.window['-HISTORY_TABLE-'].SelectedRows
         if not selected_rows: return

         try:
              row_index = selected_rows[0]
              filepath = self.window['-HISTORY_TABLE-'].Values[row_index][7] # Hidden full path column

              if filepath and filepath != 'N/A':
                   folder_path = Path(filepath).parent
                   SystemManager.open_folder(folder_path)
              else:
                   sg.popup_error("File path is not available for the selected history item.", title="Open Location Error")
         except IndexError:
              logger.error("Error opening history location: Selected row index out of bounds.")
         except Exception as e:
              logger.error(f"Unexpected error opening history location: {e}", exc_info=True)
              sg.popup_error(f"An unexpected error occurred:\n{e}", title="Open Location Error")


    def _handle_copy_history_url(self):
         """ Copies the URL of the selected history item to the clipboard. """
         selected_rows = self.window['-HISTORY_TABLE-'].SelectedRows
         if not selected_rows: return

         try:
              row_index = selected_rows[0]
              url = self.window['-HISTORY_TABLE-'].Values[row_index][6] # Hidden URL column

              if url and url != 'N/A':
                   sg.clipboard_set(url)
                   self._update_status_bar(f"Copied URL to clipboard: {url}")
                   sg.popup_notify("URL copied to clipboard!", title="Copied")
              else:
                   sg.popup_error("URL is not available for the selected history item.", title="Copy URL Error")
         except IndexError:
              logger.error("Error copying history URL: Selected row index out of bounds.")
         except Exception as e:
              logger.error(f"Unexpected error copying history URL: {e}", exc_info=True)
              sg.popup_error(f"An unexpected error occurred:\n{e}", title="Copy URL Error")


    def _open_settings_window(self):
        """Opens the settings dialog."""
        logger.debug("Opening settings window.")
        # Create layout for settings - This should be its own function/class
        # Example settings layout:
        config = self.config_manager.config
        themes = sg.theme_list()
        try:
             current_theme_index = themes.index(config.theme)
        except ValueError:
             current_theme_index = themes.index(DEFAULT_THEME) # Fallback

        settings_layout = [
            [sg.Text("Settings", font=("Helvetica", 16))],
            [sg.HorizontalSeparator()],
            [sg.Text("Download Directory:"), sg.Input(config.download_dir, key='-SETT_DIR-', size=(40,1)), sg.FolderBrowse()],
            [sg.Text("Theme:"), sg.Combo(themes, default_value=config.theme, key='-SETT_THEME-', size=(20,1), readonly=True, enable_events=True)],
            [sg.Text("Max Concurrent Downloads:"), sg.Spin([i for i in range(1, (os.cpu_count() or 1)*2 + 1)], initial_value=config.max_workers, key='-SETT_WORKERS-', size=(5,1))],
            [sg.Text("Min Free Disk Space (MB):"), sg.Input(config.min_disk_space_mb, key='-SETT_DISK-', size=(8,1), tooltip="Minimum free space required before adding to queue.")],
            [sg.Text("Rate Limit (KB/s, 0=off):"), sg.Input(config.rate_limit_kbs, key='-SETT_RATE-', size=(8,1))],
            [sg.Checkbox("Use FFmpeg if available (for conversions/metadata)", default=config.use_ffmpeg_if_available, key='-SETT_FFMPEG-'), sg.Text(f"(Found: {SystemManager.find_ffmpeg() or 'No'})", text_color='grey')],
            [sg.Checkbox("Include Metadata (.info.json)", default=config.include_metadata, key='-SETT_META-')],
            [sg.Checkbox("Embed Thumbnail", default=config.extract_thumbnail, key='-SETT_THUMB-')],
            [sg.Checkbox("Auto-Start Queue", default=config.auto_start_queue, key='-SETT_AUTOSTART-')],
            [sg.Checkbox("Notify on Completion", default=config.notify_on_complete, key='-SETT_NOTIFY-')],
            [sg.Text("Network & Advanced:")],
            [sg.Text("Max Retries:"), sg.Spin([i for i in range(0, 11)], initial_value=config.max_retries, key='-SETT_RETRIES-', size=(5,1)),
             sg.Text("Retry Delay (s):"), sg.Spin([i for i in range(1, 61)], initial_value=config.retry_delay, key='-SETT_DELAY-', size=(5,1))],
            [sg.Text("Proxy Server (e.g., socks5://host:port):"), sg.Input(config.proxy, key='-SETT_PROXY-', size=(30,1))],
            [sg.HorizontalSeparator()],
            [sg.Button("Save", key='-SAVE_SETTINGS-'), sg.Button("Cancel")]
        ]

        settings_window = sg.Window("Settings", settings_layout, modal=True, finalize=True)

        # Event loop for settings window
        while True:
            event, values = settings_window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                break
            if event == '-SETT_THEME-':
                 # Preview theme change immediately (optional, requires restart message)
                 sg.theme(values['-SETT_THEME-']) # Change theme
                 # Force redraw or notify user restart is needed
                 sg.popup("Theme changed. Restart the application for the change to fully apply.", title="Theme Changed")
                 # Revert theme for now if not restarting
                 # sg.theme(self.config_manager.config.theme) # Revert if not saving yet
            if event == '-SAVE_SETTINGS-':
                logger.info("Saving settings...")
                # Update config object from settings values
                try:
                     config.download_dir = values['-SETT_DIR-']
                     config.theme = values['-SETT_THEME-']
                     config.max_workers = int(values['-SETT_WORKERS-'])
                     config.min_disk_space_mb = int(values['-SETT_DISK-'])
                     config.rate_limit_kbs = int(values['-SETT_RATE-'])
                     config.use_ffmpeg_if_available = values['-SETT_FFMPEG-']
                     config.include_metadata = values['-SETT_META-']
                     config.extract_thumbnail = values['-SETT_THUMB-']
                     config.auto_start_queue = values['-SETT_AUTOSTART-']
                     config.notify_on_complete = values['-SETT_NOTIFY-']
                     config.max_retries = int(values['-SETT_RETRIES-'])
                     config.retry_delay = int(values['-SETT_DELAY-'])
                     config.proxy = values['-SETT_PROXY-'].strip()

                     # Validate the updated config
                     config.validate()

                     # Save the validated config
                     self.config_manager.save_config()
                     sg.popup_notify("Settings saved successfully!", title="Settings Saved")
                     self._update_status_bar("Settings saved.")
                     # Apply immediate changes if possible (e.g., update default dir display)
                     self.window['-DIR-'].update(config.download_dir)
                     # Theme requires restart, handled above or notify again here.
                     break # Close settings window after saving
                except (ValueError, TypeError) as e:
                     logger.error(f"Invalid setting value entered: {e}")
                     sg.popup_error(f"Invalid value entered. Please check numeric fields.\nError: {e}", title="Settings Error")
                except ConfigError as e:
                     logger.error(f"Failed to save settings: {e}", exc_info=True)
                     sg.popup_error(f"Failed to save settings:\n{e}", title="Settings Save Error")
                except Exception as e:
                    logger.error(f"Unexpected error saving settings: {e}", exc_info=True)
                    sg.popup_error(f"An unexpected error occurred while saving settings:\n{e}", title="Error")

        settings_window.close()
        logger.debug("Settings window closed.")


    def _show_about_window(self):
        """Displays the About window."""
        system_info = SystemManager.get_system_info()
        about_text = f"""
        {APP_NAME}
        Version: {VERSION} ({UPDATED})
        Author: {AUTHOR}

        A graphical interface for downloading YouTube videos using yt-dlp.

        --- System Information ---
        OS: {system_info.get('os', 'N/A')}
        Python: {system_info.get('python_version', 'N/A')}
        yt-dlp: {system_info.get('yt_dlp_version', 'N/A')}
        FFmpeg: {system_info.get('ffmpeg_path', 'N/A')}
        PySimpleGUI: {sg.version}

        Config Path: {CONFIG_PATH}
        History DB: {HISTORY_PATH}
        Log Path: {LOG_FILE}
        """
        sg.popup(about_text, title="About YouTube Downloader", keep_on_top=True)

    def _confirm_exit(self) -> bool:
        """Handles application exit confirmation and cleanup."""
        if self.download_manager.is_processing():
             if sg.popup_yes_no("Downloads are in progress. Are you sure you want to exit?\nActive downloads will be canceled.", title="Confirm Exit", keep_on_top=True) == 'No':
                 return False # Don't exit

        logger.info("Exit requested. Cleaning up...")
        self._update_status_bar("Exiting...")

        # Stop download manager gracefully
        self.download_manager.stop_processing(cancel_active=True)

        # Persist queue state one last time
        self.download_manager._persist_queue_state()

        # Save configuration (optional, could be done only when changed)
        # self.config_manager.save_config()

        # Close tray icon
        if self.tray:
             try:
                 self.tray.close()
             except Exception as e:
                  logger.warning(f"Error closing system tray: {e}")

        logger.info(f"{APP_NAME} finished cleanup. Exiting.")
        return True # Allow exit


    # --- Main Event Loop ---
    def run(self):
        """Runs the main event loop of the GUI."""
        logger.info("Starting GUI event loop...")
        self._update_status_bar("Ready.")

        while True:
            # Read events from window and tray
            # Use a timeout to allow periodic updates (e.g., check download status)
            event, values = self.window.read(timeout=500) # Timeout in ms

            # --- Handle Window Closing ---
            if event in (sg.WIN_CLOSED, 'Exit', '-EXIT-', sg.WINDOW_CLOSE_ATTEMPTED_EVENT):
                 if self._confirm_exit():
                     break # Exit the loop

            # --- Handle Tray Events ---
            if event == '-TRAY-':
                 tray_event = values['-TRAY-'] # Get the specific tray menu item clicked
                 logger.debug(f"Tray event received: {tray_event}")
                 if tray_event == 'Show/Hide':
                      self.window.normal() if self.window.TKroot.state() == 'withdrawn' else self.window.hide()
                 elif tray_event == 'Start Downloads':
                      self.download_manager.start_processing()
                 elif tray_event == 'Cancel Downloads':
                      self.download_manager.stop_processing(cancel_active=True)
                 elif tray_event == 'Settings':
                      self._open_settings_window()
                 elif tray_event == 'Exit':
                      if self._confirm_exit():
                          break
                 # Update button states after tray action if needed
                 self._update_button_states()
                 continue # Processed tray event, continue loop

            # --- Handle Download Manager Events ---
            if event == "-UPDATE_QUEUE_DISPLAY-":
                 self._update_queue_display(values[event]['queue'])
                 self._update_button_states() # Queue change might affect buttons
                 continue
            if event == "-UPDATE_ITEM_DISPLAY-":
                 self._update_specific_item_display(values[event]['item'])
                 # No need to update all buttons usually, status bar maybe?
                 item = values[event]['item']
                 self._update_status_bar(f"{item.get_display_name()}: {item.get_status_summary()}")
                 continue
            if event == "-DOWNLOAD_STARTED-":
                 logger.info("Event loop notified: Download started.")
                 self._update_button_states()
                 self._update_status_bar("Download process started.")
                 continue
            if event == "-DOWNLOAD_STOPPED-":
                 logger.info("Event loop notified: Download stopped.")
                 self._update_button_states()
                 queue_finished = values[event].get('queue_finished', False) if values[event] else False
                 status_msg = "Queue processing finished." if queue_finished else "Download process stopped/canceled."
                 self._update_status_bar(status_msg)
                 if queue_finished:
                     sg.popup_notify("All downloads in the queue are complete!", title="Queue Finished")
                 self._update_history_display() # Refresh history after downloads stop
                 continue


            # --- Handle Keyboard Shortcuts ---
            # Map key events to button events if needed
            if event == '-ADD-KEY-': event = '-ADD-'
            if event == '-START-KEY-': event = '-START-'
            if event == '-CANCEL-KEY-': event = '-CANCEL-'
            if event == '-OPEN_FOLDER-KEY-': event = '-OPEN_FOLDER-'
            if event == '-PASTE-KEY-': event = '-PASTE-'
            if event == '-REMOVE-KEY-': event = '-REMOVE-'

            # --- Handle GUI Element Events ---
            if event == '-URL-':
                 # Optional: Add validation or info detection on URL change/enter
                 pass
            elif event == '-PASTE-' or event == sg.EVENT_SYSTEM_TRAY_ICON_DOUBLE_CLICKED: # Also paste on tray double-click?
                 try:
                    clipboard_content = sg.clipboard_get()
                    self.window['-URL-'].update(clipboard_content)
                    self._update_status_bar("Pasted URL from clipboard.")
                 except Exception as e:
                    logger.error(f"Failed to get clipboard content: {e}")
                    self._update_status_bar("Error pasting from clipboard.")
            elif event == '-AUDIO_ONLY-':
                 # Enable/disable audio format based on checkbox
                 self.window['-AUDIO_FORMAT-'].update(disabled=not values['-AUDIO_ONLY-'])
                 # Maybe also disable video options?
                 self.window['-VIDEO_FORMAT-'].update(disabled=values['-AUDIO_ONLY-'])
                 self.window['-QUALITY-'].update(disabled=values['-AUDIO_ONLY-'])
                 self.window['-SUBS-'].update(disabled=values['-AUDIO_ONLY-'])

            elif event == '-ADD-':
                 self._handle_add_to_queue(values)

            elif event == '-REMOVE-':
                 self._handle_remove_selected()

            elif event == '-CLEAR_QUEUE-':
                 if sg.popup_yes_no("Are you sure you want to clear all pending items from the queue?", title="Confirm Clear Queue", keep_on_top=True) == 'Yes':
                      self.download_manager.clear_queue()
                      self._update_status_bar("Queue cleared.")

            elif event == '-START-':
                 self.download_manager.start_processing()

            elif event == '-CANCEL-':
                 self.download_manager.stop_processing(cancel_active=True)

            elif event == '-SETTINGS-' or event == 'Settings': # From menu or button
                 self._open_settings_window()

            elif event == '-OPEN_FOLDER-' or event == 'Open Download Folder':
                 try:
                     SystemManager.open_folder(self.config_manager.config.download_dir)
                 except Exception as e:
                      # Error already handled and logged by open_folder
                      self._update_status_bar("Error opening download folder.")

            elif event == 'Open Logs Folder':
                 try:
                      SystemManager.open_folder(LOG_DIR)
                 except Exception as e:
                      self._update_status_bar("Error opening logs folder.")

            elif event == 'Save Queue Now':
                 self.download_manager._persist_queue_state()
                 sg.popup_notify("Queue state saved.", title="Queue Saved")

            # --- History Table Events ---
            elif event == '-HISTORY_TABLE-':
                 # Update button states based on selection
                 self._update_button_states()

            elif event == '-OPEN_HISTORY_LOC-':
                 self._handle_open_history_location()

            elif event == '-COPY_HISTORY_URL-':
                 self._handle_copy_history_url()

            elif event == '-RETRY_HISTORY-':
                 self._handle_retry_selected_history()

            elif event == '-CLEAR_HISTORY-' or event == 'Clear History':
                if sg.popup_yes_no("Are you sure you want to clear the entire download history?\nThis cannot be undone.", title="Confirm Clear History", keep_on_top=True) == 'Yes':
                    try:
                        self.history_manager.clear_history()
                        self._update_history_display()
                        self._update_status_bar("History cleared.")
                        sg.popup_notify("Download history cleared.", title="History Cleared")
                    except DataAccessError as e:
                         sg.popup_error(f"Failed to clear history:\n{e}", title="History Error")

            elif event == 'Refresh History':
                self._update_history_display()
                self._update_status_bar("History refreshed.")


             # --- Queue Table Events ---
            elif event == '-QUEUE_TABLE-':
                 # Update button states based on selection
                 self._update_button_states()


            # --- Menu Events ---
            elif event == 'About':
                 self._show_about_window()

            elif event == 'Check for Updates...':
                 # Placeholder for update check logic
                 sg.popup("Update checking is not yet implemented.", title="Update Check")


            # --- Periodic updates (if timeout is used) ---
            if event == sg.TIMEOUT_EVENT:
                 # Can perform periodic checks here if needed
                 # E.g., check disk space periodically?
                 # Keep this light to avoid slowing down the UI
                 pass


        # --- End of Event Loop ---
        self.window.close()
        logger.info("GUI Event loop finished. Window closed.")


# ----------- Main Execution -----------
if __name__ == "__main__":
    # Set high DPI awareness for Windows if applicable
    if platform.system() == "Windows":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1) # Try setting DPI awareness
            logger.info("Set high DPI awareness for Windows.")
        except Exception as e:
            logger.warning(f"Could not set DPI awareness: {e}")

    # Ensure yt-dlp is reasonably up-to-date (optional check)
    try:
        yt_dlp_version = getattr(yt_dlp, '__version__', '0.0.0')
        if tuple(map(int, yt_dlp_version.split('.'))) < (2023, 7, 6): # Example version check
             logger.warning(f"yt-dlp version {yt_dlp_version} might be outdated. Consider updating ('pip install -U yt-dlp').")
    except Exception as e:
         logger.warning(f"Could not check yt-dlp version: {e}")


    # Instantiate and run the application
    try:
        app = YoutubeDownloaderGUI()
        app.run()
    except Exception as e:
        # Catch unexpected top-level errors
        logger.critical(f"An unhandled critical error occurred: {e}", exc_info=True)
        # Try showing a final error message to the user
        try:
            sg.popup_error(f"A critical error occurred:\n{e}\n\nPlease check the logs at:\n{LOG_FILE}\n\nThe application will now exit.", title="Critical Error")
        except Exception as popup_err:
             print(f"CRITICAL ERROR: {e}\n(Failed to show popup: {popup_err})", file=sys.stderr)
        sys.exit(1) # Exit with error code

    sys.exit(0) # Normal exit

