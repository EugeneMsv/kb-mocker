import logging

import colorlog
import uvicorn
from fastapi import FastAPI

from src.kb_mocker.api.routes import router

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(asctime)s %(log_color)s[%(levelname)s]%(reset)s %(name)s: %(message)s"
))
logging.basicConfig(level=logging.INFO, handlers=[handler])



app = FastAPI(title="kb-mocker")
app.include_router(router=router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
