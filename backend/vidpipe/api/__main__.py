"""API server entry point for python -m vidpipe.api"""
import uvicorn
from vidpipe.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "vidpipe.api.app:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )
