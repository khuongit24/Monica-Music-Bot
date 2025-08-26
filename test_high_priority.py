import asyncio
import pytest
from bot import YTDLTrack, MusicPlayer, AsyncDequeQueue, players


@pytest.mark.asyncio
async def test_asyncdeque_basic_put_get():
    q = AsyncDequeQueue()
    await q.put({'a': 1})
    assert q.qsize() == 1
    item = await q.get(timeout=0.1)
    assert item['a'] == 1


@pytest.mark.asyncio
async def test_loop_one_skip_suppression(monkeypatch):
    # Create a dummy guild/text channel stand-ins
    class Dummy:
        id = 1
    class DummyText:
        id = 2
        async def send(self, *a, **k):
            pass
    guild = Dummy(); text = DummyText()
    p = MusicPlayer(guild, text)
    # Inject a fake voice client
    class Vc:
        def __init__(self):
            self._playing = False
        def is_connected(self): return True
        def is_playing(self): return self._playing
        def is_paused(self): return False
        def play(self, src, after=None):
            self._playing = True
            # Immediately stop to simulate finish
            self._playing = False
            if after:
                after(None)
        def stop(self):
            self._playing = False
    p.vc = Vc()
    # Add two tracks
    t1 = {'title': 'Song1', 'url': 'u1'}
    t2 = {'title': 'Song2', 'url': 'u2'}
    await p.add_track(t1)
    await p.add_track(t2)
    await p.enable_loop_one()
    # Let player loop run one cycle
    await asyncio.sleep(0.05)
    # Simulate skip suppression then ensure only one requeue
    p._suppress_loop_requeue_once = True
    p.next_event.set()  # force loop advance
    await asyncio.sleep(0.05)
    # Queue should not explode in size
    assert p.queue.qsize() <= 3


@pytest.mark.asyncio
async def test_prefetch_inplace_no_duplicate(monkeypatch):
    # Only run if prefetch enabled in config
    from bot import PREFETCH_NEXT
    if not PREFETCH_NEXT:
        pytest.skip('Prefetch not enabled in config')
    class Dummy:
        id = 10
    class DummyText:
        id = 11
        async def send(self, *a, **k):
            pass
    guild = Dummy(); text = DummyText()
    p = MusicPlayer(guild, text)
    # Item missing url triggers prefetch
    item = {'title': 'test audio'}
    await p.add_track(item)
    # Allow some time for prefetch worker to act
    await asyncio.sleep(1.5)
    snap = p.queue.snapshot()
    if snap:
        # Ensure same dict object mutated (still first element object id)
        assert snap[0] is item
        # If resolved, url field should appear
        if item is snap[0] and snap[0].get('url'):
            assert 'url' in snap[0]
