from collections import deque
import asyncio
from typing import Any, Callable, List, Optional

class AsyncDequeQueue:
    """Thread-safe async queue implementation using deque with condition variables.
    Provides async methods for putting and getting items with proper synchronization.
    Supports timeout operations and various queue manipulation methods.
    """
    def __init__(self) -> None:
        self._dq: deque[Any] = deque()
        self._cond: asyncio.Condition = asyncio.Condition()

    async def put(self, item: Any) -> None:
        async with self._cond:
            self._dq.append(item)
            self._cond.notify_all()

    async def put_front(self, item: Any) -> None:
        async with self._cond:
            self._dq.appendleft(item)
            self._cond.notify_all()

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
            return self._dq.popleft()

    async def clear(self) -> int:
        async with self._cond:
            n = len(self._dq)
            self._dq.clear()
            return n

    async def remove_by_pred(self, pred: Callable[[Any], bool]) -> int:
        async with self._cond:
            old = list(self._dq)
            new = [x for x in old if not pred(x)]
            removed = len(old) - len(new)
            self._dq = deque(new)
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
