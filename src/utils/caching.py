"""
Simple in-memory caching utilities.
Addresses Gap 5: No Query Caching.
"""

from functools import lru_cache
import hashlib
import json

def cache_key_builder(*args, **kwargs):
    """Build a deterministic hash key from args/kwargs."""
    # Convert args/kwargs to a JSON string for hashing
    # Note: highly simplified, assumes json-serializable
    key_str = json.dumps([args, kwargs], sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()

def lru_cache_result(maxsize=128):
    """
    Decorator for LRU caching of function results.
    Useful for router and retrieval results.
    """
    return lru_cache(maxsize=maxsize)
