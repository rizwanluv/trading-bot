import os
import logging
from aiohttp import web

logger = logging.getLogger("health_server")

async def health_check(request):
    return web.Response(text="Trading Bot is ALIVE and running optimally.", status=200)

async def start_health_server():
    port = int(os.environ.get("PORT", 8080))
    app = web.Application()
    app.add_routes([web.get('/', health_check), web.get('/health', health_check)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Health check server started on port {port}")
    
    # We return the runner so it can be cleaned up, but typically it runs forever.
    return runner
