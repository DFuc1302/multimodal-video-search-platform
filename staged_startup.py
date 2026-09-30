"""Keep the ASGI event loop responsive while synchronous startup checks run."""
import asyncio
import threading


def launch_startup(lifespan, app, state):
    async def initialize():
        try:
            async with lifespan(app):
                pass
        except Exception as exc:
            state.update(ready=False, status='failed', phase='failed', error=str(exc))
            # A failed visual load must not take validated OCR/ASR offline.
    thread = threading.Thread(target=lambda: asyncio.run(initialize()),
                              name='search-startup', daemon=True)
    thread.start()
    return thread
