import os
import logging
from aiohttp import web

logger = logging.getLogger("health_server")

async def health_check(request):
    return web.Response(text="Trading Bot is ALIVE and running optimally.", status=200)

async def start_health_server():
    try:
        port_raw = os.environ.get("PORT", "8080")
        try:
            port = int(port_raw)
        except (ValueError, TypeError):
            port = 8080

        app = web.Application()
        app.add_routes([web.get('/', health_check), web.get('/health', health_check)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        logger.info(f"Health check server started on port {port}")
        return runner
    except Exception as exc:
        logger.warning(f"Health check server could not be started: {exc}")
        return None


async def stop_health_server(runner):
    if runner is not None:
        try:
            await runner.cleanup()
            logger.info("Health check server stopped.")
        except Exception as exc:
            logger.warning(f"Error stopping health check server: {exc}")
