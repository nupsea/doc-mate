"""
Database connection pooling and async helpers.
Addresses Gap 1 (No Pooling) and Gap 2 (Sync DB in Async Context).
"""

import os
import asyncio
import logging
import contextlib
from functools import wraps
from typing import Generator
import psycopg2
import psycopg2.pool

logger = logging.getLogger(__name__)

# Config
DB_CONFIG = {
    "dbname": os.getenv("PG_DB", "booksdb"),
    "user": os.getenv("PG_USER", "bookuser"),
    "password": os.getenv("PG_PASS", "bookpass"),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", 5432),
    # TCP keepalives: prevent idle pool connections from being dropped
    # during long LLM calls (30-120s) in parallel ingestion
    "keepalives": 1,
    "keepalives_idle": 30,       # seconds before first probe
    "keepalives_interval": 10,   # seconds between probes
    "keepalives_count": 5,       # probes before declaring dead
}

class DatabaseManager:
    _pool: psycopg2.pool.ThreadedConnectionPool = None
    _pool_lock = asyncio.Lock()

    @classmethod
    def get_pool(cls) -> psycopg2.pool.ThreadedConnectionPool:
        """Get or create the singleton connection pool."""
        if cls._pool is None:
            try:
                logger.info("[DB] Initializing ThreadedConnectionPool (min=1, max=20)")
                cls._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=20,
                    **DB_CONFIG
                )
            except Exception as e:
                logger.error(f"[DB] Failed to create connection pool: {e}")
                raise
        return cls._pool

    @classmethod
    def close_pool(cls):
        """Close all connections in the pool."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            logger.info("[DB] Connection pool closed")

    @classmethod
    def _is_conn_alive(cls, conn) -> bool:
        """Lightweight check: is this connection still usable?"""
        if conn.closed:
            return False
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            # Clear any implicit transaction state from the probe
            conn.rollback()
            return True
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            return False

    @classmethod
    @contextlib.contextmanager
    def get_connection(cls) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Context manager to get a connection from the pool.
        Yields a raw psycopg2 connection.
        Automatically returns it to the pool on exit.
        Replaces stale connections transparently.
        """
        pool = cls.get_pool()
        conn = pool.getconn()

        # Health check: replace stale connections from the pool
        if not cls._is_conn_alive(conn):
            logger.warning("[DB] Stale connection detected, replacing")
            pool.putconn(conn, close=True)
            conn = pool.getconn()

        try:
            yield conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Connection died mid-operation -- close it so the pool
            # doesn't recycle a broken connection
            logger.warning(f"[DB] Connection error during operation: {e}")
            pool.putconn(conn, close=True)
            conn = None  # prevent double-putconn in finally
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            if conn is not None:
                pool.putconn(conn)

def async_db_task(func):
    """
    Decorator to run a synchronous DB function in a separate thread.
    
    Usage:
        @async_db_task
        def get_summary(slug): ...
        
        # In async code:
        summary = await get_summary(slug)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        # Use a partial to pass args/kwargs correctly to run_in_executor
        from functools import partial
        p_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, p_func)
    return wrapper
