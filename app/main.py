from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routers import auth, chat, api, dashboard, intake
from .database import engine, Base
from .config import settings

# Create database tables (Supabase Postgres)
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(dashboard.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(api.router)
app.include_router(intake.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
