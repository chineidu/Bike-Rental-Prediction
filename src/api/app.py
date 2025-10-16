import sys
import warnings

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.v1 import health, prediction
from src.api.utilities.utilities import lifespan
from src.config import app_config

warnings.filterwarnings("ignore")


def create_application() -> FastAPI:
    """Create and configure a FastAPI application instance.

    This function initializes a FastAPI application with custom configuration settings,
    adds CORS middleware, and includes API route handlers.

    Returns
    -------
    FastAPI
        A configured FastAPI application instance.
    """
    app = FastAPI(
        title=app_config.api_config.title,
        description=app_config.api_config.description,
        version=app_config.api_config.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.api_config.middleware.cors.allow_origins,
        allow_credentials=app_config.api_config.middleware.cors.allow_credentials,
        allow_methods=app_config.api_config.middleware.cors.allow_methods,
        allow_headers=app_config.api_config.middleware.cors.allow_headers,
    )

    # Include routers
    app.include_router(health.router, prefix=app_config.api_config.prefix)
    app.include_router(prediction.router, prefix=app_config.api_config.prefix)
    # app.include_router(task_status.router, prefix=app_config.api_config.prefix)

    return app


app: FastAPI = create_application()

if __name__ == "__main__":
    try:
        uvicorn.run(
            "src.api.app:app",
            host=app_config.api_config.server.host,
            port=app_config.api_config.server.port,
            reload=app_config.api_config.server.reload,
        )
    except (Exception, KeyboardInterrupt) as e:
        print(f"Error creating application: {e}")
        print("Exiting gracefully...")
        sys.exit(1)
