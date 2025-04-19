# history_manager.py
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import asdict
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from youtube_downloader import (
    APP_ROOT,
    HISTORY_PATH,
    DownloadItem
)

logger = logging.getLogger(__name__)

class HistoryManager:
    """
    Advanced history manager with SQLite backend and caching
    Features:
    - Thread-safe operations
    - Automatic cleanup of old entries
    - Statistics tracking
    - Export/Import capabilities
    - Search functionality
    """

    def __init__(self, max_entries: int = 1000, cache_size: int = 100):
        self.db_path = APP_ROOT / "download_history.db"
        self.max_entries = max_entries
        self.cache_size = cache_size
        self._lock = threading.Lock()
        self._cache: Dict[str, DownloadItem] = {}
        self._initialize_db()

    def _initialize_db(self):
        """Initialize SQLite database with optimized schema"""
        try:
            with self._get_db() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS downloads (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT UNIQUE NOT NULL,
                        title TEXT,
                        format TEXT,
                        quality TEXT,
                        download_path TEXT,
                        status TEXT,
                        filesize INTEGER,
                        download_time TIMESTAMP,
                        error TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Create indexes for frequent queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON downloads(url)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON downloads(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_download_time ON downloads(download_time)")
        except Exception as e:
            logger.error(f"Failed to initialize history database: {e}")
            raise

    def _get_db(self) -> sqlite3.Connection:
        """Get database connection with optimized settings"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Better performance with acceptable safety
        return conn

    def add_entry(self, download_item: Union[DownloadItem, Dict]) -> bool:
        """
        Add new download entry to history with automatic cleanup
        
        Args:
            download_item: DownloadItem or dict containing download information
            
        Returns:
            bool: Success status
        """
        try:
            if isinstance(download_item, dict):
                download_item = DownloadItem.from_dict(download_item)

            with self._lock:
                with self._get_db() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO downloads 
                        (url, title, format, quality, download_path, status, 
                         filesize, download_time, error, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        download_item.url,
                        download_item.title,
                        download_item.format,
                        download_item.quality,
                        download_item.download_path,
                        download_item.status,
                        download_item.filesize,
                        download_item.download_time,
                        download_item.error,
                        json.dumps(asdict(download_item))
                    ))

                    # Update cache
                    self._cache[download_item.url] = download_item
                    
                    # Cleanup old entries if needed
                    self._cleanup_old_entries()
                    
            return True
        except Exception as e:
            logger.error(f"Failed to add history entry: {e}")
            return False

    def get_entry(self, url: str) -> Optional[DownloadItem]:
        """Get download history entry with caching"""
        try:
            # Check cache first
            if url in self._cache:
                return self._cache[url]

            with self._get_db() as conn:
                result = conn.execute(
                    "SELECT metadata FROM downloads WHERE url = ?", 
                    (url,)
                ).fetchone()

                if result:
                    entry = DownloadItem.from_dict(json.loads(result['metadata']))
                    # Update cache
                    self._cache[url] = entry
                    return entry

            return None
        except Exception as e:
            logger.error(f"Failed to get history entry: {e}")
            return None

    def search_history(self, 
                      query: str = "", 
                      status: Optional[str] = None,
                      date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None,
                      limit: int = 100) -> List[DownloadItem]:
        """
        Advanced search functionality for download history
        
        Args:
            query: Search term for URL or title
            status: Filter by download status
            date_from: Start date for filter
            date_to: End date for filter
            limit: Maximum number of results
            
        Returns:
            List[DownloadItem]: Matching download items
        """
        try:
            conditions = ["1=1"]
            params = []

            if query:
                conditions.append("(url LIKE ? OR title LIKE ?)")
                params.extend([f"%{query}%", f"%{query}%"])

            if status:
                conditions.append("status = ?")
                params.append(status)

            if date_from:
                conditions.append("download_time >= ?")
                params.append(date_from.isoformat())

            if date_to:
                conditions.append("download_time <= ?")
                params.append(date_to.isoformat())

            sql = f"""
                SELECT metadata 
                FROM downloads 
                WHERE {' AND '.join(conditions)}
                ORDER BY download_time DESC
                LIMIT ?
            """
            params.append(limit)

            with self._get_db() as conn:
                results = conn.execute(sql, params).fetchall()
                return [DownloadItem.from_dict(json.loads(row['metadata'])) 
                        for row in results]

        except Exception as e:
            logger.error(f"Failed to search history: {e}")
            return []

    def get_statistics(self) -> Dict:
        """Get download history statistics"""
        try:
            with self._get_db() as conn:
                stats = {
                    'total_downloads': 0,
                    'successful_downloads': 0,
                    'failed_downloads': 0,
                    'total_size': 0,
                    'formats': {},
                    'qualities': {},
                    'last_download': None
                }

                # Get basic stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        SUM(filesize) as total_size,
                        MAX(download_time) as last_download
                    FROM downloads
                """)
                row = cursor.fetchone()
                
                stats.update({
                    'total_downloads': row['total'],
                    'successful_downloads': row['successful'],
                    'failed_downloads': row['failed'],
                    'total_size': row['total_size'] or 0,
                    'last_download': row['last_download']
                })

                # Get format distribution
                for row in conn.execute("SELECT format, COUNT(*) as count FROM downloads GROUP BY format"):
                    stats['formats'][row['format']] = row['count']

                # Get quality distribution
                for row in conn.execute("SELECT quality, COUNT(*) as count FROM downloads GROUP BY quality"):
                    stats['qualities'][row['quality']] = row['count']

                return stats
        except Exception as e:
            logger.error(f"Failed to get history statistics: {e}")
            return {}

    def export_history(self, export_path: Optional[Path] = None) -> bool:
        """Export download history to JSON file"""
        try:
            if export_path is None:
                export_path = HISTORY_PATH

            with self._get_db() as conn:
                results = conn.execute("SELECT metadata FROM downloads").fetchall()
                history_data = [json.loads(row['metadata']) for row in results]

            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=4)

            return True
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False

    def import_history(self, import_path: Path) -> bool:
        """Import download history from JSON file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

            with ThreadPoolExecutor() as executor:
                executor.map(self.add_entry, history_data)

            return True
        except Exception as e:
            logger.error(f"Failed to import history: {e}")
            return False

    def clear_history(self) -> bool:
        """Clear all download history"""
        try:
            with self._lock:
                with self._get_db() as conn:
                    conn.execute("DELETE FROM downloads")
                self._cache.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False

    def _cleanup_old_entries(self):
        """Clean up old entries when max_entries is exceeded"""
        try:
            with self._get_db() as conn:
                count = conn.execute("SELECT COUNT(*) as count FROM downloads").fetchone()['count']
                if count > self.max_entries:
                    # Delete oldest entries exceeding the limit
                    conn.execute(f"""
                        DELETE FROM downloads 
                        WHERE id IN (
                            SELECT id FROM downloads 
                            ORDER BY download_time ASC 
                            LIMIT {count - self.max_entries}
                        )
                    """)
        except Exception as e:
            logger.error(f"Failed to cleanup old entries: {e}")
