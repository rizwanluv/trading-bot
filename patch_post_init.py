import re

with open('main.py', 'r') as f:
    content = f.read()

post_init_func = """
async def post_init(application: Application) -> None:
    \"\"\"Start the background health check server.\"\"\"
    asyncio.create_task(start_health_server())

"""
if "async def post_init" not in content:
    content = content.replace("def main() -> None:", post_init_func + "def main() -> None:")

content = content.replace("ApplicationBuilder()", "ApplicationBuilder().post_init(post_init)")

# I need to make sure `Application` is imported if not already. Let's see if it is. 
# `ApplicationBuilder` is imported from `telegram.ext`. `Application` is too.
if "from telegram.ext import" in content and "Application," not in content:
    content = content.replace("from telegram.ext import (", "from telegram.ext import (\n    Application,")

with open('main.py', 'w') as f:
    f.write(content)
