"""
Object Storage Adapter
========================
Implements IStorageAdapter for local filesystem storage.

In production, this could be replaced with an S3-compatible adapter
without modifying the services that use it (DIP).

Handles the physical storage of:
  - Original medical images (preserved unmodified — RD-13)
  - Processed/anonymized copies
  - Generated clinical reports
"""

import shutil
from pathlib import Path
from typing import Optional

from domain.interfaces.contracts import IStorageAdapter
from config.settings import UPLOAD_DIR, PROCESSED_DIR, REPORTS_DIR
from utils.logger import get_logger
from utils.exceptions import RepositoryError

logger = get_logger("ObjectStorage")


class LocalStorageAdapter(IStorageAdapter):
    """
    Local filesystem implementation of IStorageAdapter.
    
    Stores files in organized subdirectories under the data/ folder.
    Can be swapped for a cloud storage implementation (S3, Azure Blob)
    without changing the service layer.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = base_dir or UPLOAD_DIR
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def store(self, file_data: bytes, filename: str, subdirectory: str = "") -> str:
        """Store file data and return the absolute storage path."""
        try:
            target_dir = self._base_dir / subdirectory if subdirectory else self._base_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            file_path = target_dir / filename
            file_path.write_bytes(file_data)

            logger.info(f"File stored: {file_path} ({len(file_data)} bytes)")
            return str(file_path)

        except Exception as e:
            logger.error(f"Storage failed for '{filename}': {e}")
            raise RepositoryError("store_file", str(e))

    def retrieve(self, file_path: str) -> bytes:
        """Retrieve file data from the given path."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return path.read_bytes()

        except FileNotFoundError:
            raise
        except Exception as e:
            raise RepositoryError("retrieve_file", str(e))

    def delete(self, file_path: str) -> bool:
        """Delete a file from storage."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False

        except Exception as e:
            logger.error(f"Delete failed for '{file_path}': {e}")
            raise RepositoryError("delete_file", str(e))
