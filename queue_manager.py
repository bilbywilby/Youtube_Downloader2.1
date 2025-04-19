# queue_manager.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from queue import Queue, Empty
import threading
from datetime import datetime, timedelta
import time

from youtube_downloader import (
    APP_ROOT,
    QUEUE_PATH,
    DownloadItem
)

logger = logging.getLogger(__name__)

class QueueManager:
    """
    Advanced queue manager with persistent storage and priority handling
    Features:
    - Priority queue management
    - Auto-save and recovery
    - Queue statistics and monitoring
    - Batch operations
    - Progress tracking
    """

    def __init__(self, auto_save: bool = True, save_interval: int = 300):
        self.queue_path = QUEUE_PATH
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._queue: Queue[DownloadItem] = Queue()
        self._priority_queue: Queue[DownloadItem] = Queue()
        self._processing: Dict[str, DownloadItem] = {}
        self._completed: Dict[str, DownloadItem] = {}
        self._failed: Dict[str, DownloadItem] = {}
        self._lock = threading.Lock()
        self._last_save = datetime.now()
        self._save_thread = None
        self._shutdown = threading.Event()

        # Load saved queue on initialization
        self._load_queue()
        
        if auto_save:
            self._start_auto_save()

    def _start_auto_save(self):
        """Start auto-save background thread"""
        def auto_save_worker():
            while not self._shutdown.is_set():
                time.sleep(self.save_interval)
                self.save_queue()

        self._save_thread = threading.Thread(
            target=auto_save_worker, 
            daemon=True,
            name="QueueAutoSave"
        )
        self._save_thread.start()

    def add_item(self, item: Union[DownloadItem, Dict], priority: bool = False) -> bool:
        """
        Add item to download queue
        
        Args:
            item: DownloadItem or dict to add
            priority: If True, add to priority queue
            
        Returns:
            bool: Success status
        """
        try:
            with self._lock:
                if isinstance(item, dict):
                    item = DownloadItem.from_dict(item)

                if priority:
                    self._priority_queue.put(item)
                else:
                    self._queue.put(item)

                if self.auto_save and (datetime.now() - self._last_save).seconds >= self.save_interval:
                    self.save_queue()

                return True
        except Exception as e:
            logger.error(f"Failed to add item to queue: {e}")
            return False

    def get_next_item(self) -> Optional[DownloadItem]:
        """Get next item from queue, prioritizing priority queue"""
        try:
            # Check priority queue first
            try:
                item = self._priority_queue.get_nowait()
                self._processing[item.url] = item
                return item
            except Empty:
                pass

            # Check regular queue
            try:
                item = self._queue.get_nowait()
                self._processing[item.url] = item
                return item
            except Empty:
                pass

            return None
        except Exception as e:
            logger.error(f"Error getting next queue item: {e}")
            return None

    def mark_completed(self, url: str, status: str = "completed", error: Optional[str] = None):
        """Mark download item as completed or failed"""
        with self._lock:
            if url in self._processing:
                item = self._processing.pop(url)
                item.status = status
                item.error = error
                
                if status == "completed":
                    self._completed[url] = item
                else:
                    self._failed[url] = item

                if self.auto_save:
                    self.save_queue()

    def get_queue_status(self) -> Dict:
        """Get comprehensive queue status"""
        return {
            'queued': self._queue.qsize() + self._priority_queue.qsize(),
            'processing': len(self._processing),
            'completed': len(self._completed),
            'failed': len(self._failed),
            'priority_queued': self._priority_queue.qsize()
        }

    def clear_queue(self):
        """Clear all queues"""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break

            while not self._priority_queue.empty():
                try:
                    self._priority_queue.get_nowait()
                except Empty:
                    break

            self._processing.clear()
            if self.auto_save:
                self.save_queue()

    def save_queue(self) -> bool:
        """Save current queue state to file"""
        try:
            with self._lock:
                queue_data = {
                    'regular_queue': [
                        asdict(item) for item in list(self._queue.queue)
                    ],
                    'priority_queue': [
                        asdict(item) for item in list(self._priority_queue.queue)
                    ],
                    'processing': [
                        asdict(item) for item in self._processing.values()
                    ]
                }

                self.queue_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.queue_path, 'w', encoding='utf-8') as f:
                    json.dump(queue_data, f, indent=4)

                self._last_save = datetime.now()
                return True
        except Exception as e:
            logger.error(f"Failed to save queue: {e}")
            return False

    def _load_queue(self):
        """Load queue from saved file"""
        try:
            if self.queue_path.exists():
                with open(self.queue_path, 'r', encoding='utf-8') as f:
                    queue_data = json.load(f)

                # Load regular queue
                for item_data in queue_data.get('regular_queue', []):
                    self._queue.put(DownloadItem.from_dict(item_data))

                # Load priority queue
                for item_data in queue_data.get('priority_queue', []):
                    self._priority_queue.put(DownloadItem.from_dict(item_data))

                # Load processing items
                for item_data in queue_data.get('processing', []):
                    item = DownloadItem.from_dict(item_data)
                    self._processing[item.url] = item

        except Exception as e:
            logger.error(f"Failed to load queue: {e}")

    def get_queue_items(self, include_processing: bool = True) -> List[DownloadItem]:
        """Get all items in queue"""
        items = []
        with self._lock:
            items.extend(list(self._queue.queue))
            items.extend(list(self._priority_queue.queue))
            if include_processing:
                items.extend(self._processing.values())
        return items

    def remove_item(self, url: str) -> bool:
        """Remove specific item from queue"""
        with self._lock:
            # Helper function to remove from queue
            def filter_queue(q: Queue, url: str) -> Queue:
                new_queue = Queue()
                while not q.empty():
                    try:
                        item = q.get_nowait()
                        if item.url != url:
                            new_queue.put(item)
                    except Empty:
                        break
                return new_queue

            # Remove from regular queue
            self._queue = filter_queue(self._queue, url)
            
            # Remove from priority queue
            self._priority_queue = filter_queue(self._priority_queue, url)
            
            # Remove from processing
            self._processing.pop(url, None)
            
            if self.auto_save:
                self.save_queue()
            
            return True

    def shutdown(self):
        """Shutdown queue manager"""
        self._shutdown.set()
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=5)
        self.save_queue()

