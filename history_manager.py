import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SearchType(Enum):
    """Enumerates the types of search operations."""
    EXACT = "exact"
    CONTAINS = "contains"

@dataclass(frozen=True)
class HistoryEntry:
    """Represents a single entry in the download history."""
    title: str
    url: str
    download_path: str
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'HistoryEntry':
        """Creates a HistoryEntry object from a dictionary."""
        return cls(
            title=data['title'],
            url=data['url'],
            download_path=data['download_path'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

    def to_dict(self) -> Dict[str, str]:
        """Converts the HistoryEntry object to a dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "download_path": self.download_path,
            "timestamp": self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """Returns a string representation of the HistoryEntry."""
        return (f"Title: {self.title}\n"
                f"URL: {self.url}\n"
                f"Path: {self.download_path}\n"
                f"Timestamp: {self.timestamp.isoformat()}\n")

class HistoryManager:
    """Manages the download history, including loading, saving, and searching."""
    def __init__(self, history_file: str = "download_history.json"):
        """Initializes the HistoryManager with the path to the history file."""
        self.history_file = history_file
        self._ensure_history_file_exists()

    def _ensure_history_file_exists(self) -> None:
        """Creates the history file if it does not exist."""
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as file:
                json.dump([], file)
            logging.info(f"History file created at: {self.history_file}")

    def _load_history(self) -> List[HistoryEntry]:
        """Loads the download history from the JSON file."""
        try:
            with open(self.history_file, 'r') as file:
                data = json.load(file)
                return [HistoryEntry.from_dict(entry) for entry in data]
        except FileNotFoundError:
            logging.error(f"History file not found at: {self.history_file}. Creating a new one.")
            self._ensure_history_file_exists()
            return []
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from: {self.history_file}. The file might be corrupted. Returning an empty history.")
            return []

    def _save_history(self, entries: List[HistoryEntry]) -> None:
        """Saves the download history to the JSON file."""
        try:
            with open(self.history_file, 'w') as file:
                json.dump([entry.to_dict() for entry in entries], file, indent=4)
            logging.info(f"History saved to: {self.history_file}")
        except IOError as e:
            logging.error(f"Error saving history to {self.history_file}: {e}")

    def add_entry(self, video_title: str, video_url: str, download_path: str) -> None:
        """Adds a new entry to the download history."""
        timestamp = datetime.now()
        entry = HistoryEntry(
            title=video_title,
            url=video_url,
            download_path=download_path,
            timestamp=timestamp
        )
        history = self._load_history()
        history.append(entry)
        self._save_history(history)
        logging.info(f"Added entry: {entry}")

    def clear_history(self) -> None:
        """Clears the entire download history."""
        self._save_history([])
        logging.info("Download history cleared.")

    def delete_entry(self, search_term: str, search_field: str = "title", search_type: SearchType = SearchType.EXACT) -> None:
        """Deletes entries from the history based on a search term and field."""
        history = self._load_history()
        initial_count = len(history)
        updated_history = [
            entry for entry in history
            if not self._check_match(getattr(entry, search_field), search_term, search_type)
        ]
        if len(updated_history) < initial_count:
            self._save_history(updated_history)
            logging.info(f"Deleted {initial_count - len(updated_history)} entries where {search_field} {search_type.value} '{search_term}'.")
        else:
            logging.info(f"No entries found where {search_field} {search_type.value} '{search_term}'.")

    def _check_match(self, entry_value: Any, search_value: Any, search_type: SearchType) -> bool:
        """Checks if an entry value matches the search value based on the search type."""
        if search_type == SearchType.EXACT:
            return str(entry_value).lower() == str(search_value).lower()
        elif search_type == SearchType.CONTAINS:
            return str(search_value).lower() in str(entry_value).lower()
        return False

    def _filter_entries(self, field: str, value: Any, search_type: SearchType = SearchType.CONTAINS, date_range: Optional[tuple[datetime, datetime]] = None) -> List[HistoryEntry]:
        """Filters history entries based on a field, value, search type, and optional date range."""
        history = self._load_history()
        results = []
        for entry in history:
            entry_value = getattr(entry, field)
            if field == "timestamp":
                if date_range:
                    start_date, end_date = date_range
                    if start_date.date() <= entry_value.date() <= end_date.date():
                        results.append(entry)
                elif search_type == SearchType.EXACT:
                    if entry_value == value if isinstance(value, datetime) else entry_value.isoformat() == value:
                        results.append(entry)
            else:
                if self._check_match(entry_value, value, search_type):
                    results.append(entry)
        return results

    def get_entries_by_title(self, title: str, exact: bool = False) -> List[HistoryEntry]:
        """Retrieves history entries by title."""
        search_type = SearchType.EXACT if exact else SearchType.CONTAINS
        return self._filter_entries("title", title, search_type)

    def get_entries_by_url(self, url: str, exact: bool = False) -> List[HistoryEntry]:
        """Retrieves history entries by URL."""
        search_type = SearchType.EXACT if exact else SearchType.CONTAINS
        return self._filter_entries("url", url, search_type)

    def get_entries_by_download_path(self, path: str, exact: bool = False) -> List[HistoryEntry]:
        """Retrieves history entries by download path."""
        search_type = SearchType.EXACT if exact else SearchType.CONTAINS
        return self._filter_entries("download_path", path, search_type)

    def get_entries_by_timestamp(self, timestamp: Union[str, datetime]) -> List[HistoryEntry]:
        """Retrieves history entries by exact timestamp."""
        target_timestamp = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
        return self._filter_entries("timestamp", target_timestamp, SearchType.EXACT)

    def get_entries_by_date_range(self, start_date: datetime, end_date: datetime) -> List[HistoryEntry]:
        """Retrieves history entries within a specified date range."""
        return self._filter_entries("timestamp", None, date_range=(start_date, end_date))

    def get_all_entries(self) -> List[HistoryEntry]:
        """Retrieves all entries from the download history."""
        return self._load_history()

# Example usage
if __name__ == "__main__":
    history_manager = HistoryManager()

    # Add entries using a helper for custom timestamps
    def add_entry_with_timestamp(title: str, url: str, path: str, ts: datetime):
        history = history_manager.get_all_entries()
        history.append(HistoryEntry(title=title, url=url, download_path=path, timestamp=ts))
        history_manager._save_history(history)

    history_manager.add_entry("Video 1", "http://example.com/video1", "/path/to/video1")
    history_manager.add_entry("Video 2 - Awesome", "http://example.com/video2", "/another/path/video2")
    history_manager.add_entry("Another Video 1", "http://test.com/vid1", "/tmp/vid1")
    add_entry_with_timestamp("Old Video", "http://old.com/vid", "/archive/old", datetime(2024, 1, 15))

    def print_section(title: str, entries: list[HistoryEntry]):
        print(f"\n{title}")
        if not entries:
            print("  (No results)")
        for entry in entries:
            print(entry)

    print_section("All entries:", history_manager.get_all_entries())
    print_section("Search by title (contains 'Video 1'):", history_manager.get_entries_by_title("Video 1"))
    print_section("Search by title (exact 'Video 1'):", history_manager.get_entries_by_title("Video 1", exact=True))
    print_section("Search by URL (contains 'example.com'):", history_manager.get_entries_by_url("example.com"))
    print_section("Search by download path (exact '/path/to/video1'):",
                  history_manager.get_entries_by_download_path("/path/to/video1", exact=True))

    now = datetime.now()
    print_section(f"Search by exact timestamp ({now.isoformat()}):", history_manager.get_entries_by_timestamp(now))

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    print_section(f"Search by date range ({start_date.date()} to {end_date.date()}):",
                  history_manager.get_entries_by_date_range(start_date, end_date))

    # Delete an entry
    history_manager.delete_entry("Another Video 1")
    print_section("History after deleting 'Another Video 1':", history_manager.get_all_entries())

    # Uncomment to clear history
    # history_manager.clear_history()
    # print_section("History after clearing:", history_manager.get_all_entries())

