from typing import Optional, Dict, Any
from sqlalchemy import text
from backend.db import engine

def db_get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    sql = text("""
        SELECT id, email, password_hash, role, is_approved, approved_by, approval_date, created_at
        FROM public.users
        WHERE email = :email
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"email": email}).mappings().first()
        return dict(row) if row else None

def db_create_user(email: str, password_hash: str, role: str) -> Dict[str, Any]:
    sql = text("""
        INSERT INTO public.users (email, password_hash, role, is_approved)
        VALUES (:email, :password_hash, :role, false)
        RETURNING id, email, role, is_approved, approved_by, approval_date, created_at
    """)
    # engine.begin() auto-commits
    with engine.begin() as conn:
        row = conn.execute(sql, {
            "email": email,
            "password_hash": password_hash,
            "role": role
        }).mappings().first()
        return dict(row)
