import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Database configuration
def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")

    # If someone accidentally pastes the placeholder, fail loudly
    if "[YOUR-PASSWORD]" in url:
        raise RuntimeError("DATABASE_URL still contains [YOUR-PASSWORD]. Replace it with the real password.")

    # Supabase sometimes provides 'postgresql://...' which is fine.
    return url

DATABASE_URL = get_database_url()

# Supabase requires SSL. (Most Supabase URLs also work without this, but this is the safe default.)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"sslmode": "require"},
)

def db_ping() -> bool:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
