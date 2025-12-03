"""
Supabase Client
Connection management for Supabase database
"""

from typing import Optional
import os
from functools import lru_cache

try:
    from supabase import create_client, Client
except ImportError:
    # Fallback for when supabase is not installed
    Client = None
    create_client = None


class SupabaseClient:
    """
    Wrapper for Supabase client with connection management.
    """

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize Supabase client.

        Args:
            url: Supabase project URL (or SUPABASE_URL env var)
            key: Supabase API key (or SUPABASE_KEY env var)
        """
        self.url = url or os.getenv("SUPABASE_URL", "")
        self.key = key or os.getenv("SUPABASE_KEY", "")

        if create_client is None:
            raise ImportError(
                "supabase package not installed. "
                "Install with: pip install supabase"
            )

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key are required. "
                "Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )

        self._client: Client = create_client(self.url, self.key)

    @property
    def client(self) -> Client:
        """Get the Supabase client"""
        return self._client

    def table(self, name: str):
        """Get a table reference"""
        return self._client.table(name)

    def rpc(self, func_name: str, params: dict = None):
        """Call a database function"""
        return self._client.rpc(func_name, params or {})

    def from_(self, table_name: str):
        """Alias for table()"""
        return self.table(table_name)


# Global client instance
_supabase_client: Optional[SupabaseClient] = None


def get_supabase_client(
    url: Optional[str] = None,
    key: Optional[str] = None,
    force_new: bool = False
) -> SupabaseClient:
    """
    Get or create Supabase client singleton.

    Args:
        url: Supabase URL (uses env if not provided)
        key: Supabase key (uses env if not provided)
        force_new: Force create new client

    Returns:
        SupabaseClient instance
    """
    global _supabase_client

    if _supabase_client is None or force_new:
        _supabase_client = SupabaseClient(url, key)

    return _supabase_client
