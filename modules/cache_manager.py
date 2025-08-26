"""
Advanced caching system with TTL, LRU eviction, and intelligent promotion.
Handles track metadata caching with performance optimizations.
"""

import asyncio
import time
import logging
from collections import OrderedDict
from typing import Dict, Any, Optional

from modules.metrics import metric_inc

logger = logging.getLogger("Monica.CacheManager")


class CacheManager:
    """Advanced cache manager with TTL, LRU, and intelligent eviction."""
    
    def __init__(self, size_limit: int = 200, ttl_seconds: int = 900):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock: Optional[asyncio.Lock] = None
        self.size_limit = size_limit
        self.ttl_seconds = ttl_seconds
    
    def _ensure_lock(self):
        """Ensure lock is initialized."""
        if self._lock is None:
            self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Optimized cache get with TTL, LRU, and early validation."""
        self._ensure_lock()
        
        # Fast path: check existence without lock first
        entry = self._cache.get(key)
        if not entry:
            metric_inc("cache_miss")
            return None
        
        # Quick TTL check without lock for expired entries
        current_time = time.time()
        ttl = entry.get("ttl", self.ttl_seconds)
        if current_time - entry["ts"] > ttl:
            # Remove expired entry without waiting for cleanup
            async with self._lock:
                self._cache.pop(key, None)
            metric_inc("cache_miss")
            return None
        
        # Update metadata with lock
        async with self._lock:
            # Double-check entry still exists after acquiring lock
            entry = self._cache.get(key)
            if not entry:
                metric_inc("cache_miss")
                return None
                
            # Update access metadata
            entry["ts"] = current_time
            entry["hits"] = entry.get("hits", 0) + 1
            
            # Intelligent TTL promotion based on access patterns
            hits = entry["hits"]
            base_ttl = self.ttl_seconds
            if hits == 5 and entry.get("ttl", base_ttl) == base_ttl:
                entry["ttl"] = base_ttl * 2  # First promotion: 2x TTL
            elif hits == 15 and entry.get("ttl", base_ttl) == base_ttl * 2:
                entry["ttl"] = base_ttl * 3  # Second promotion: 3x TTL for very popular tracks
            
            # Efficient LRU repositioning
            try:
                self._cache.move_to_end(key)
            except KeyError:
                pass  # Entry was removed by another operation
        
        metric_inc("cache_hits")
        return entry["data"]
    
    async def put(self, key: str, data: Dict[str, Any]):
        """Optimized cache put with intelligent eviction and data compression."""
        self._ensure_lock()
        
        # Compress data early to reduce memory footprint
        lean = {
            "title": data.get("title", "")[:200],  # Limit title length
            "webpage_url": data.get("webpage_url", ""),
            "url": data.get("url", ""),
            "thumbnail": data.get("thumbnail", ""),
            "duration": data.get("duration"),
            "uploader": data.get("uploader", "")[:100],  # Limit uploader name
            "is_live": bool(data.get("is_live") or data.get("live_status") in ("is_live", "started")),
        }
        
        async with self._lock:
            current_time = time.time()
            self._cache[key] = {
                "data": lean, 
                "ts": current_time, 
                "ttl": self.ttl_seconds, 
                "hits": 0
            }
            
            # Smart eviction: prioritize removing expired and low-hit entries
            if len(self._cache) > self.size_limit:
                eviction_count = 0
                max_evictions = min(50, len(self._cache) // 4)  # Evict up to 25% at once
                
                # First pass: remove expired entries
                expired_keys = []
                for cache_key, entry in self._cache.items():
                    if cache_key == key:  # Skip the entry we just added
                        continue
                    ttl = entry.get("ttl", self.ttl_seconds)
                    if current_time - entry["ts"] > ttl:
                        expired_keys.append(cache_key)
                        eviction_count += 1
                        if eviction_count >= max_evictions:
                            break
                
                for expired_key in expired_keys:
                    self._cache.pop(expired_key, None)
                
                # Second pass: if still over limit, remove least accessed items
                if len(self._cache) > self.size_limit:
                    remaining_evictions = max_evictions - eviction_count
                    # Sort by hits (ascending) and timestamp (ascending) to find least valuable entries
                    sortable_items = [
                        (cache_key, entry.get("hits", 0), entry.get("ts", 0))
                        for cache_key, entry in self._cache.items()
                        if cache_key != key
                    ]
                    sortable_items.sort(key=lambda x: (x[1], x[2]))  # Sort by hits first, then timestamp
                    
                    for i in range(min(remaining_evictions, len(sortable_items))):
                        self._cache.pop(sortable_items[i][0], None)
                        eviction_count += 1
                
                if eviction_count > 0:
                    logger.debug("Cache evicted %d entries (size: %d)", eviction_count, len(self._cache))
    
    async def cleanup_expired(self):
        """Remove expired cache entries - called by background task."""
        self._ensure_lock()
        current_time = time.time()
        async with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                ttl = entry.get("ttl", self.ttl_seconds)
                if current_time - entry["ts"] > ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._cache.pop(key, None)
            
            if expired_keys:
                logger.debug("Cleaned up %d expired cache entries", len(expired_keys))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "size_limit": self.size_limit,
            "ttl_seconds": self.ttl_seconds
        }
    
    async def clear(self) -> int:
        """Clear all cache entries and return count of cleared items."""
        self._ensure_lock()
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count


# Global cache instance
cache_manager: Optional[CacheManager] = None


def get_cache_manager(size_limit: int = 200, ttl_seconds: int = 900) -> CacheManager:
    """Get or create global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager(size_limit, ttl_seconds)
    return cache_manager


async def cleanup_cache_loop(cache_mgr: CacheManager):
    """Background task for periodic cache cleanup."""
    while True:
        try:
            await cache_mgr.cleanup_expired()
        except Exception:
            logger.exception("Cache cleanup error")
        await asyncio.sleep(60 * 5)  # Clean up every 5 minutes
