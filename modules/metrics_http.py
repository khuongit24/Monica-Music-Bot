"""Optional Prometheus exposition (simple HTTP server)"""
import asyncio
import logging
import json
from modules.metrics import metrics_snapshot

logger = logging.getLogger("Monica.MetricsHTTP")
_server = None
_running = False

async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        data = await reader.readline()
        if not data:
            writer.close(); return
        # minimal parse
        path = b"/" if b" " not in data else data.split(b" ")[1]
        if path.startswith(b"/metrics"):
            ms = metrics_snapshot()
            lines = []
            for k, v in ms.items():
                if isinstance(v, (int, float)):
                    lines.append(f"monica_{k} {v}")
            body = "\n".join(lines) + "\n"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body.encode())
        elif path.startswith(b"/health") or path.startswith(b"/ready"):
            # Minimal JSON health/readiness response using metrics snapshot
            try:
                ms = metrics_snapshot()
                body = json.dumps({"status": "running", "metrics_count": len(ms)}) + "\n"
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body.encode())
            except Exception:
                writer.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length:0\r\n\r\n")
        elif path.startswith(b"/healthz"):
            # Health check endpoint returning loop status and background tasks count
            try:
                loop = asyncio.get_running_loop()
                loop_running = loop.is_running()
                background_tasks = len([t for t in asyncio.all_tasks(loop) if not t.done()])
                body = json.dumps({
                    "status": "healthy" if loop_running else "unhealthy",
                    "loop_running": loop_running,
                    "background_tasks": background_tasks
                }) + "\n"
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body.encode())
            except Exception as e:
                body = json.dumps({"status": "error", "error": str(e)}) + "\n"
                writer.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body.encode())
        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length:0\r\n\r\n")
        await writer.drain()
    except Exception:
        pass
    finally:
        try: writer.close()
        except Exception: pass

async def start(port: int = 9109):
    global _server, _running
    if _running:
        return
    _server = await asyncio.start_server(_handle, host="0.0.0.0", port=port)
    _running = True
    logger.info("Prometheus metrics server on %s", port)

async def stop():
    global _server, _running
    if _server:
        _server.close()
        await _server.wait_closed()
    _running = False
