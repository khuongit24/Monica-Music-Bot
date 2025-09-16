from collections import deque
import asyncio
from typing import Any, Callable, Optional, List
from modules.metrics import gauge_set
import weakref


_queue_gauge_task: Optional[asyncio.Task] = None


async def _queue_gauge_loop(interval: float = 5.0):
    """Background loop that periodically sets the `queue_size` gauge.

    This helps external exporters observe overall queue size even when no
    queue mutations happen for a while.
    """
    global _queue_gauge_task
    while True:
        try:
            gauge_set("queue_size", AsyncDequeQueue._global_probe())
        except Exception:
            # best-effort; don't crash the loop
            pass
        await asyncio.sleep(interval)


def start_queue_gauge_task(loop: Optional[asyncio.AbstractEventLoop] = None):
    """Start background queue gauge task if not already started."""
    global _queue_gauge_task
    if _queue_gauge_task and not _queue_gauge_task.done():
        return
    if loop is None:
        loop = asyncio.get_event_loop()
    _queue_gauge_task = loop.create_task(_queue_gauge_loop())


async def stop_queue_gauge_task():
    global _queue_gauge_task
    if _queue_gauge_task:
        _queue_gauge_task.cancel()
        try:
            await _queue_gauge_task
        except Exception:
            pass
        _queue_gauge_task = None

class AsyncDequeQueue:
    """Thread-safe async queue implementation using deque with condition variables.
    Provides async methods for putting and getting items with proper synchronization.
    Supports timeout operations and various queue manipulation methods.
    """
    def __init__(self) -> None:
        self._dq: deque[Any] = deque()
        self._cond: asyncio.Condition = asyncio.Condition()
        # Register self in module-global probe list for gauge loop
        try:
            self.__class__._register_instance(self)
        except Exception:
            pass

    # Module-global weak registry for gauge probing. Using a WeakSet means
    # queues won't be kept alive solely by the registry (avoids leaks).
    _GLOBAL_INSTANCES = weakref.WeakSet()

    @classmethod
    def _register_instance(cls, inst: 'AsyncDequeQueue') -> None:
        try:
            cls._GLOBAL_INSTANCES.add(inst)
        except Exception:
            # best-effort registration during construction
            pass

    @classmethod
    def _global_probe(cls) -> int:
        """Return total size across registered queues."""
        try:
            return sum(i.qsize() for i in cls._GLOBAL_INSTANCES)
        except Exception:
            return 0

    async def put(self, item: Any) -> None:
        async with self._cond:
            self._dq.append(item)
            self._cond.notify_all()
        gauge_set("queue_size", len(self._dq))

    async def put_front(self, item: Any) -> None:
        async with self._cond:
            self._dq.appendleft(item)
            self._cond.notify_all()
        gauge_set("queue_size", len(self._dq))

    async def get(self, timeout: Optional[float] = None) -> Any:
        async with self._cond:
            if timeout is None:
                while not self._dq:
                    await self._cond.wait()
            else:
                end_at = asyncio.get_running_loop().time() + timeout
                while not self._dq:
                    remaining = end_at - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise
            item = self._dq.popleft()
            gauge_set("queue_size", len(self._dq))
            return item

    async def clear(self) -> int:
        async with self._cond:
            n = len(self._dq)
            self._dq.clear()
            # update gauge after mutation and before returning
            gauge_set("queue_size", len(self._dq))
            return n

    async def remove_by_pred(self, pred: Callable[[Any], bool]) -> int:
        async with self._cond:
            old = list(self._dq)
            new = [x for x in old if not pred(x)]
            removed = len(old) - len(new)
            self._dq = deque(new)
            # update gauge before returning
            gauge_set("queue_size", len(self._dq))
            return removed

    def snapshot(self, limit: Optional[int] = None) -> List[Any]:
        """Return a snapshot of the queue.

        If `limit` is provided, return at most the first `limit` items (cheap).
        """
        if limit is None:
            return list(self._dq)
        if limit <= 0:
            return []
        # Efficiently take up to limit items without converting entire deque
        res = []
        for i, item in enumerate(self._dq):
            if i >= limit:
                break
            res.append(item)
        return res

    def qsize(self) -> int:
        return len(self._dq)

    def empty(self) -> bool:
        return not self._dq
