# We start by importing necessary modules and setting up FastAPI

from datetime import datetime
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from secrets import token_hex
import csv
from pathlib import Path

import os
import json
from google import genai
from fastapi import FastAPI, HTTPException, Depends, Response
from collections import Counter

from fastapi.responses import HTMLResponse

from sqlalchemy import text
from sqlalchemy import create_engine
from backend.db import db_ping, engine

from backend.db_users import db_get_user_by_email, db_create_user


BASE_DIR = Path(__file__).resolve().parent.parent


# -------------------------------------------------
# Password hashing helper
# -------------------------------------------------

import hashlib

def hash_password(password: str) -> str:
    """
    Simple password hashing using SHA-256.
    Note: In production, use bcrypt or similar.
    """
    return hashlib.sha256(password.encode()).hexdigest()



app = FastAPI(title="Rest.Inventory API")

# -------------------------------------------------
# Models for Users
# -------------------------------------------------


class UserRole(str):
    ADMIN = "admin"
    OWNER = "owner"
    KITCHEN = "kitchen"
    FLOOR = "floor"


class UserCreate(BaseModel):
    email: str
    role: str = Field(..., description="one of: admin, owner, kitchen, floor")
    password: str


class User(BaseModel):
    id: int
    email: str
    role: str
    is_approved: bool
    approved_by: Optional[int] = None
    approval_date: Optional[datetime] = None
    created_at: datetime


class UserPublic(BaseModel):
    id: int
    email: str
    role: str
    is_approved: bool
    created_at: datetime


class UserApprovalResponse(BaseModel):
    id: int
    email: str
    role: str
    is_approved: bool
    approved_by: Optional[int] = None
    approval_date: Optional[datetime] = None


class PendingUserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_approved: bool
    created_at: datetime


# -------------------------------------------------
# Models for Auth
# -------------------------------------------------


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    role: str


# -------------------------------------------------
# Models for Catalog
# -------------------------------------------------

class CatalogItemCreate(BaseModel):
    name: str
    category: Optional[str] = None
    unit: Optional[str] = None
    price: Optional[float] = None


class CatalogItem(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    unit: Optional[str] = None
    price: Optional[float] = None
    is_active: bool
    created_by: int
    created_at: datetime


class CatalogRequestCreate(BaseModel):
    item_id: int
    reason: Optional[str] = None


class CatalogRequestStatus(str):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CatalogRequest(BaseModel):
    id: int
    item_id: int
    requested_by: int
    status: str
    reason: Optional[str] = None
    approved_by: Optional[int] = None
    created_at: datetime


# -------------------------------------------------
# Models for Submissions
# -------------------------------------------------

class SubmissionItemCreate(BaseModel):
    catalog_item_id: int = Field(..., description="Catalog item ID")
    quantity: float = Field(..., ge=0, description="Quantity requested")
    comment: Optional[str] = Field(None, description="Optional note")


class SubmissionItem(BaseModel):
    id: int
    submission_id: int
    catalog_item_id: int
    quantity: float
    comment: Optional[str] = None
    created_at: datetime


class SubmissionCreate(BaseModel):
    items: List[SubmissionItemCreate]


class SubmissionStatus(str):
    PENDING = "pending"
    APPROVED = "approved"
    ORDERED = "ordered"


class Submission(BaseModel):
    id: int
    submitted_by_user_id: int
    status: str
    items: List[SubmissionItem]
    created_at: datetime
    approved_at: Optional[datetime] = None
    ordered_at: Optional[datetime] = None
    approved_by: Optional[int] = None


class SubmissionDetail(BaseModel):
    """Submission with full details including item names"""
    id: int
    submitted_by_user_id: int
    status: str
    created_at: datetime
    approved_at: Optional[datetime] = None
    ordered_at: Optional[datetime] = None
    approved_by: Optional[int] = None
    items: List[SubmissionItem]

class AIInsight(BaseModel):
    submission_id: int
    generated_at: datetime
    summary: str
    top_categories: List[str]
    alerts: List[str]
    model: str
    confidence_note: str

class WeeklyAIReport(BaseModel):
    generated_at: datetime
    report_text: str


# -------------------------------------------------
# Dashboard models
# -------------------------------------------------

class DashboardTimelineItem(BaseModel):
    date: str
    submission_count: int
    total_quantity: float
    status_breakdown: Dict[str, int]  # {pending: 5, approved: 3, ordered: 2}


class DashboardTimeline(BaseModel):
    period: str  # "daily", "weekly", "monthly"
    data: List[DashboardTimelineItem]


class DashboardTopItem(BaseModel):
    catalog_item_id: int
    item_name: str
    request_count: int
    total_quantity: float
    category: Optional[str] = None


class DashboardInventoryStatus(BaseModel):
    total_pending: int
    total_approved: int
    total_ordered: int
    pending_quantity: float
    approved_quantity: float
    ordered_quantity: float


class DashboardStatistics(BaseModel):
    total_submissions: int
    avg_items_per_submission: float
    approval_rate: float  # percentage
    submissions_this_week: int
    most_active_user_id: int
    most_active_user_submissions: int


class DashboardSummary(BaseModel):
    generated_at: datetime
    inventory_status: DashboardInventoryStatus
    statistics: DashboardStatistics
    top_items: List[DashboardTopItem]
    recent_timeline: List[DashboardTimelineItem]


# -------------------------------------------------
# Product models and loading from CSV
# -------------------------------------------------


class Product(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    unit: Optional[str] = None
    is_active: bool = True


PRODUCTS: List[Product] = []


def load_products_from_csv() -> None:
    """
    Load products from data/products.csv into the in-memory PRODUCTS list.
    Assumes the CSV has columns: item_id, name, category, unit, is_active.
    """
    global PRODUCTS
    PRODUCTS = []  # reset in case of reload

    csv_path = BASE_DIR / "data" / "products.csv"
    print(f"DEBUG: CSV path being used: {csv_path}")

    if not csv_path.exists():
        print("DEBUG: CSV file does NOT exist at that path!")
        return

    with csv_path.open(newline="", encoding="utf-8") as f:
        # our file is a tab-separated file exported from Excel
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                product = Product(
                    id=int(str(row.get("item_id")).replace("ITEM_", "")),  # e.g. ITEM_0001 -> 1
                    name=(row.get("name") or "").strip(),
                    category=(row.get("category") or "").strip() or None,
                    unit=(row.get("unit") or "").strip() or None,
                    is_active=str(row.get("is_active", "True")).strip().lower()
                    in ("1", "true", "yes"),
                )
                PRODUCTS.append(product)
            except Exception as e:
                print(f"DEBUG: Skipping bad row {row} because: {e}")
                continue

    print(f"DEBUG: Loaded {len(PRODUCTS)} products from CSV.")



# Call loader at startup
@app.on_event("startup")
def startup_event():
    load_products_from_csv()



# -------------------------------------------------
# In-memory storage (for tokens and AI insights)
# -------------------------------------------------

# token -> user_id (for session management)
ACTIVE_TOKENS: Dict[str, int] = {}

# AI insights in memory (can be moved to database later)
AI_INSIGHTS: List[AIInsight] = []


# -------------------------------------------------
# Security dependency
# -------------------------------------------------

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Resolve the Bearer token to a User object from database.
    """
    token = credentials.credentials
    user_id = ACTIVE_TOKENS.get(token)

    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Get user from database
    sql = text("SELECT * FROM public.users WHERE id = :user_id")
    with engine.connect() as conn:
        user = conn.execute(sql, {"user_id": user_id}).mappings().first()
    
    if user is None:
        # token points to a user that no longer exists
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return dict(user)


# -------------------------------------------------
# Health endpoint
# -------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "Rest.Inventory"}

def get_db_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not set")

    # Supabase often requires SSL
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"

    return create_engine(db_url, pool_pre_ping=True)


@app.get("/db/health")
def db_health():
    try:
        db_ping()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": "not connected", "detail": str(e)}



# -------------------------------------------------
# User endpoints
# -------------------------------------------------

@app.post("/users", response_model=UserApprovalResponse)
def create_user(payload: UserCreate):
    """
    Create a new user (admin, owner, kitchen, or floor).
    New users start with is_approved=false and need approval from their superior.
    Admin users are auto-approved to allow system bootstrap.
    """
    # Check if email already exists
    existing = db_get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Validate role
    valid_roles = {UserRole.ADMIN, UserRole.OWNER, UserRole.KITCHEN, UserRole.FLOOR}
    if payload.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}")

    # Hash password
    password_hash = hash_password(payload.password)

    # Create user in DB (starts unapproved)
    created = db_create_user(
        email=payload.email,
        password_hash=password_hash,
        role=payload.role,
    )

    # Auto-approve admin users for system bootstrap
    if payload.role == UserRole.ADMIN:
        sql_approve = text(f"""
            UPDATE public.users
            SET is_approved = true, approved_by = {created['id']}, approval_date = NOW()
            WHERE id = {created['id']}
        """)
        with engine.connect() as conn:
            conn.execute(sql_approve)
            conn.commit()
        created["is_approved"] = True
        created["approved_by"] = created["id"]
        created["approval_date"] = datetime.utcnow().isoformat()

    return UserApprovalResponse(
        id=created["id"],
        email=created["email"],
        role=created["role"],
        is_approved=created.get("is_approved", False),
        approved_by=created.get("approved_by"),
        approval_date=created.get("approval_date"),
    )


@app.get("/users/pending-approval", response_model=List[PendingUserResponse])
def list_pending_approvals(current_user: User = Depends(get_current_user)):
    """
    List all users pending approval.
    - Admin can see all pending users
    - Owner can see pending kitchen/floor staff
    """
    if current_user["role"] not in [UserRole.ADMIN, UserRole.OWNER]:
        raise HTTPException(status_code=403, detail="Only admins and owners can view pending approvals")

    sql = text("""
        SELECT id, email, role, is_approved, created_at
        FROM public.users
        WHERE is_approved = false
    """)

    # If owner (not admin), only show kitchen/floor staff
    if current_user["role"] == UserRole.OWNER:
        sql = text("""
            SELECT id, email, role, is_approved, created_at
            FROM public.users
            WHERE is_approved = false AND role IN ('kitchen', 'floor')
        """)

    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
        return [dict(row) for row in rows]


@app.post("/admin/users/approve/{user_id}", response_model=UserApprovalResponse)
def admin_approve_user(user_id: int, current_user: User = Depends(get_current_user)):
    """
    Admin approves a user (typically the owner).
    Only admins can call this.
    """
    if current_user["role"] != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Only admins can approve users")

    sql_get = text("SELECT * FROM public.users WHERE id = :user_id")
    sql_approve = text("""
        UPDATE public.users 
        SET is_approved = true, approved_by = :admin_id, approval_date = CURRENT_TIMESTAMP
        WHERE id = :user_id
        RETURNING id, email, role, is_approved, approved_by, approval_date
    """)

    with engine.begin() as conn:
        # Check user exists
        user = conn.execute(sql_get, {"user_id": user_id}).mappings().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Approve the user
        approved_user = conn.execute(sql_approve, {"user_id": user_id, "admin_id": current_user["id"]}).mappings().first()
        return dict(approved_user)


@app.post("/owner/users/approve/{user_id}", response_model=UserApprovalResponse)
def owner_approve_user(user_id: int, current_user: User = Depends(get_current_user)):
    """
    Owner approves a user (kitchen/floor staff).
    Only owners can call this.
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owners can approve users")

    sql_get = text("SELECT * FROM public.users WHERE id = :user_id")
    sql_approve = text("""
        UPDATE public.users 
        SET is_approved = true, approved_by = :owner_id, approval_date = CURRENT_TIMESTAMP
        WHERE id = :user_id AND role IN ('kitchen', 'floor')
        RETURNING id, email, role, is_approved, approved_by, approval_date
    """)

    with engine.begin() as conn:
        # Check user exists and is kitchen/floor
        user = conn.execute(sql_get, {"user_id": user_id}).mappings().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user["role"] not in ["kitchen", "floor"]:
            raise HTTPException(status_code=400, detail="Owner can only approve kitchen or floor staff")

        # Approve the user
        approved_user = conn.execute(sql_approve, {"user_id": user_id, "owner_id": current_user["id"]}).mappings().first()
        return dict(approved_user)


# -------------------------------------------------
# Auth endpoints
# -------------------------------------------------


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    """
    Verify email + password and return an access token.
    User must be approved to log in.
    """
    # Get user from database
    user = db_get_user_by_email(payload.email)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Verify password
    password_hash = hash_password(payload.password)
    if user.get("password_hash") != password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Check if user is approved
    if not user.get("is_approved", False):
        raise HTTPException(status_code=403, detail="User account is not approved yet. Contact your administrator.")

    # Generate token and store it
    access_token = token_hex(16)
    ACTIVE_TOKENS[access_token] = user["id"]

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user["id"],
        role=user.get("role", "floor"),
    )


# -------------------------------------------------
# Catalog endpoints
# -------------------------------------------------

@app.post("/catalog/items", response_model=CatalogItem)
def create_catalog_item(
    payload: CatalogItemCreate,
    current_user: dict = Depends(get_current_user),
):
    """
    Create a new catalog item.
    Owners and authorized users can add products directly to the catalog.
    """
    sql = text("""
        INSERT INTO public.catalog_items (name, category, unit, price, is_active, created_by)
        VALUES (:name, :category, :unit, :price, true, :created_by)
        RETURNING id, name, category, unit, price, is_active, created_by, created_at
    """)
    
    with engine.begin() as conn:
        row = conn.execute(sql, {
            "name": payload.name,
            "category": payload.category,
            "unit": payload.unit,
            "price": payload.price,
            "created_by": current_user["id"]
        }).mappings().first()
        return dict(row)


@app.get("/catalog", response_model=List[CatalogItem])
def list_catalog(only_active: bool = True):
    """
    List all catalog items.
    If only_active=True (default), return only active items.
    """
    sql_active = text("SELECT * FROM public.catalog_items WHERE is_active = true ORDER BY category, name")
    sql_all = text("SELECT * FROM public.catalog_items ORDER BY category, name")
    
    sql = sql_active if only_active else sql_all
    
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
        return [dict(row) for row in rows]


@app.get("/catalog/{item_id}", response_model=CatalogItem)
def get_catalog_item(item_id: int):
    """
    Get a single catalog item by ID.
    """
    sql = text("SELECT * FROM public.catalog_items WHERE id = :item_id")
    
    with engine.connect() as conn:
        row = conn.execute(sql, {"item_id": item_id}).mappings().first()
    
    if not row:
        raise HTTPException(status_code=404, detail="Catalog item not found")
    
    return dict(row)


@app.patch("/catalog/{item_id}", response_model=CatalogItem)
def update_catalog_item(
    item_id: int,
    payload: CatalogItemCreate,
    current_user: dict = Depends(get_current_user),
):
    """
    Update a catalog item (anyone can request, owner approves).
    Creates a catalog request for owner to review.
    """
    # Check item exists
    sql_check = text("SELECT * FROM public.catalog_items WHERE id = :item_id")
    with engine.connect() as conn:
        item = conn.execute(sql_check, {"item_id": item_id}).mappings().first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Catalog item not found")
    
    # Create a catalog request for the update
    sql_request = text("""
        INSERT INTO public.catalog_requests (item_id, requested_by, status, reason)
        VALUES (:item_id, :requested_by, 'pending', :reason)
        RETURNING id, item_id, requested_by, status, reason, approved_by, created_at
    """)
    
    reason = f"Update: {payload.name} | Category: {payload.category} | Unit: {payload.unit} | Price: {payload.price}"
    
    with engine.begin() as conn:
        request = conn.execute(sql_request, {
            "item_id": item_id,
            "requested_by": current_user["id"],
            "reason": reason
        }).mappings().first()
    
    # Return the current item (not yet updated)
    return dict(item)


@app.post("/catalog/requests/{request_id}/approve", response_model=CatalogItem)
def approve_catalog_request(
    request_id: int,
    current_user: dict = Depends(get_current_user),
):
    """
    Owner approves a catalog item update request.
    Owner only.
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can approve catalog changes")
    
    # Get the request
    sql_get_request = text("""
        SELECT * FROM public.catalog_requests WHERE id = :request_id
    """)
    
    with engine.connect() as conn:
        request = conn.execute(sql_get_request, {"request_id": request_id}).mappings().first()
    
    if not request:
        raise HTTPException(status_code=404, detail="Catalog request not found")
    
    if request["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Request is already {request['status']}")
    
    # Approve the request
    sql_approve = text("""
        UPDATE public.catalog_requests 
        SET status = 'approved', approved_by = :approved_by
        WHERE id = :request_id
    """)
    
    # Activate the item
    sql_activate = text("""
        UPDATE public.catalog_items 
        SET is_active = true
        WHERE id = :item_id
        RETURNING id, name, category, unit, price, is_active, created_by, created_at
    """)
    
    with engine.begin() as conn:
        conn.execute(sql_approve, {"request_id": request_id, "approved_by": current_user["id"]})
        item = conn.execute(sql_activate, {"item_id": request["item_id"]}).mappings().first()
    
    return dict(item)


@app.post("/catalog/requests/{request_id}/reject", response_model=CatalogRequest)
def reject_catalog_request(
    request_id: int,
    current_user: dict = Depends(get_current_user),
):
    """
    Owner rejects a catalog item request.
    Owner only.
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can reject catalog changes")
    
    # Get the request
    sql_get_request = text("""
        SELECT * FROM public.catalog_requests WHERE id = :request_id
    """)
    
    with engine.connect() as conn:
        request = conn.execute(sql_get_request, {"request_id": request_id}).mappings().first()
    
    if not request:
        raise HTTPException(status_code=404, detail="Catalog request not found")
    
    if request["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Request is already {request['status']}")
    
    # Reject the request
    sql_reject = text("""
        UPDATE public.catalog_requests 
        SET status = 'rejected', approved_by = :approved_by
        WHERE id = :request_id
        RETURNING id, item_id, requested_by, status, reason, approved_by, created_at
    """)
    
    with engine.begin() as conn:
        result = conn.execute(sql_reject, {"request_id": request_id, "approved_by": current_user["id"]}).mappings().first()
    
    return dict(result)


@app.get("/catalog/requests/pending", response_model=List[CatalogRequest])
def get_pending_catalog_requests(current_user: dict = Depends(get_current_user)):
    """
    List all pending catalog requests (owner only).
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can view pending requests")
    
    sql = text("""
        SELECT * FROM public.catalog_requests 
        WHERE status = 'pending'
        ORDER BY created_at DESC
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
        return [dict(row) for row in rows]


# -------------------------------------------------
# Submissions endpoints
# -------------------------------------------------

@app.post("/submissions", response_model=SubmissionDetail)
def create_submission(
    payload: SubmissionCreate,
    current_user: dict = Depends(get_current_user),
):
    """
    Create a new inventory submission with items from the catalog.
    """
    # Validate all catalog items exist
    for item in payload.items:
        sql_check = text("SELECT id FROM public.catalog_items WHERE id = :item_id")
        with engine.connect() as conn:
            check = conn.execute(sql_check, {"item_id": item.catalog_item_id}).scalar()
        if not check:
            raise HTTPException(status_code=404, detail=f"Catalog item {item.catalog_item_id} not found")

    # Create submission
    sql_submission = text("""
        INSERT INTO public.submissions (submitted_by_user_id, status)
        VALUES (:user_id, 'pending')
        RETURNING id, submitted_by_user_id, status, created_at, approved_at, ordered_at, approved_by
    """)

    with engine.begin() as conn:
        submission = conn.execute(sql_submission, {"user_id": current_user["id"]}).mappings().first()
        submission_id = submission["id"]

        # Add items to submission
        sql_items = text("""
            INSERT INTO public.submission_items (submission_id, catalog_item_id, quantity, comment)
            VALUES (:submission_id, :catalog_item_id, :quantity, :comment)
            RETURNING id, submission_id, catalog_item_id, quantity, comment, created_at
        """)

        items = []
        for item in payload.items:
            item_row = conn.execute(sql_items, {
                "submission_id": submission_id,
                "catalog_item_id": item.catalog_item_id,
                "quantity": item.quantity,
                "comment": item.comment
            }).mappings().first()
            items.append(dict(item_row))

    return SubmissionDetail(
        id=submission["id"],
        submitted_by_user_id=submission["submitted_by_user_id"],
        status=submission["status"],
        created_at=submission["created_at"],
        approved_at=submission["approved_at"],
        ordered_at=submission["ordered_at"],
        approved_by=submission["approved_by"],
        items=items
    )


@app.get("/submissions", response_model=List[SubmissionDetail])
def list_submissions(current_user: dict = Depends(get_current_user)):
    """
    List all submissions.
    Only the owner/admin can see all submissions.
    """
    if current_user["role"] not in [UserRole.OWNER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Only owner/admin can view all submissions.",
        )

    sql = text("""
        SELECT id, submitted_by_user_id, status, created_at, approved_at, ordered_at, approved_by
        FROM public.submissions
        ORDER BY created_at DESC
    """)

    with engine.connect() as conn:
        submissions = conn.execute(sql).mappings().all()

    result = []
    for sub in submissions:
        # Get items for this submission
        sql_items = text("""
            SELECT id, submission_id, catalog_item_id, quantity, comment, created_at
            FROM public.submission_items
            WHERE submission_id = :submission_id
        """)
        with engine.connect() as conn:
            items = conn.execute(sql_items, {"submission_id": sub["id"]}).mappings().all()

        result.append(SubmissionDetail(
            id=sub["id"],
            submitted_by_user_id=sub["submitted_by_user_id"],
            status=sub["status"],
            created_at=sub["created_at"],
            approved_at=sub["approved_at"],
            ordered_at=sub["ordered_at"],
            approved_by=sub["approved_by"],
            items=[dict(i) for i in items]
        ))

    return result


@app.get("/submissions/me", response_model=List[SubmissionDetail])
def list_my_submissions(current_user: dict = Depends(get_current_user)):
    """
    List submissions created by the currently authenticated user.
    """
    sql = text("""
        SELECT id, submitted_by_user_id, status, created_at, approved_at, ordered_at, approved_by
        FROM public.submissions
        WHERE submitted_by_user_id = :user_id
        ORDER BY created_at DESC
    """)

    with engine.connect() as conn:
        submissions = conn.execute(sql, {"user_id": current_user["id"]}).mappings().all()

    result = []
    for sub in submissions:
        sql_items = text("""
            SELECT id, submission_id, catalog_item_id, quantity, comment, created_at
            FROM public.submission_items
            WHERE submission_id = :submission_id
        """)
        with engine.connect() as conn:
            items = conn.execute(sql_items, {"submission_id": sub["id"]}).mappings().all()

        result.append(SubmissionDetail(
            id=sub["id"],
            submitted_by_user_id=sub["submitted_by_user_id"],
            status=sub["status"],
            created_at=sub["created_at"],
            approved_at=sub["approved_at"],
            ordered_at=sub["ordered_at"],
            approved_by=sub["approved_by"],
            items=[dict(i) for i in items]
        ))

    return result


class SubmissionStatusUpdate(BaseModel):
    status: str = Field(..., description="One of: pending, approved, ordered")


@app.patch("/submissions/{submission_id}/status", response_model=SubmissionDetail)
def update_submission_status(
    submission_id: int,
    payload: SubmissionStatusUpdate,
    current_user: dict = Depends(get_current_user),
):
    """
    Update the status of a submission.
    Only the owner/admin are allowed to change status.
    """
    if current_user["role"] not in [UserRole.OWNER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Only owner/admin can update submission status.",
        )

    # Validate status
    allowed = {SubmissionStatus.PENDING, SubmissionStatus.APPROVED, SubmissionStatus.ORDERED}
    if payload.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{payload.status}'. Allowed: {', '.join(sorted(allowed))}.",
        )

    # Update submission
    sql_update = text("""
        UPDATE public.submissions 
        SET status = :status, approved_by = :approved_by, approved_at = CURRENT_TIMESTAMP
        WHERE id = :submission_id
        RETURNING id, submitted_by_user_id, status, created_at, approved_at, ordered_at, approved_by
    """)

    with engine.begin() as conn:
        submission = conn.execute(sql_update, {
            "submission_id": submission_id,
            "status": payload.status,
            "approved_by": current_user["id"]
        }).mappings().first()

    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Get items
    sql_items = text("""
        SELECT id, submission_id, catalog_item_id, quantity, comment, created_at
        FROM public.submission_items
        WHERE submission_id = :submission_id
    """)
    with engine.connect() as conn:
        items = conn.execute(sql_items, {"submission_id": submission_id}).mappings().all()

    return SubmissionDetail(
        id=submission["id"],
        submitted_by_user_id=submission["submitted_by_user_id"],
        status=submission["status"],
        created_at=submission["created_at"],
        approved_at=submission["approved_at"],
        ordered_at=submission["ordered_at"],
        approved_by=submission["approved_by"],
        items=[dict(i) for i in items]
    )


@app.get("/submissions/approved", response_model=List[SubmissionDetail])
def get_approved_submissions(current_user: dict = Depends(get_current_user)):
    """
    List all approved submissions ready for purchase (owner only).
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can view approved submissions")

    sql = text("""
        SELECT id, submitted_by_user_id, status, created_at, approved_at, ordered_at, approved_by
        FROM public.submissions
        WHERE status = 'approved'
        ORDER BY created_at DESC
    """)

    with engine.connect() as conn:
        submissions = conn.execute(sql).mappings().all()

    result = []
    for sub in submissions:
        sql_items = text("""
            SELECT id, submission_id, catalog_item_id, quantity, comment, created_at
            FROM public.submission_items
            WHERE submission_id = :submission_id
        """)
        with engine.connect() as conn:
            items = conn.execute(sql_items, {"submission_id": sub["id"]}).mappings().all()

        result.append(SubmissionDetail(
            id=sub["id"],
            submitted_by_user_id=sub["submitted_by_user_id"],
            status=sub["status"],
            created_at=sub["created_at"],
            approved_at=sub["approved_at"],
            ordered_at=sub["ordered_at"],
            approved_by=sub["approved_by"],
            items=[dict(i) for i in items]
        ))

    return result

# -------------------------------------------------
# AI Insight endpoints
# -------------------------------------------------


@app.post("/submissions/{submission_id}/ai-insights", response_model=AIInsight)
def create_ai_insight_for_submission(
    submission_id: int,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate an AI summary for a given submission (owner only).
    Calls Gemini, stores the AI insight in memory, and returns it.
    """
    # Only owner can call this
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(
            status_code=403,
            detail="Only the owner can generate AI insights.",
        )

    # Get submission from database
    sql = text("""
        SELECT id, submitted_by_user_id, status, created_at, approved_at, ordered_at, approved_by
        FROM public.submissions
        WHERE id = :submission_id
    """)

    with engine.connect() as conn:
        submission = conn.execute(sql, {"submission_id": submission_id}).mappings().first()

    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Get submission items
    sql_items = text("""
        SELECT id, submission_id, catalog_item_id, quantity, comment, created_at
        FROM public.submission_items
        WHERE submission_id = :submission_id
    """)

    with engine.connect() as conn:
        items = conn.execute(sql_items, {"submission_id": submission_id}).mappings().all()

    submission_obj = Submission(
        id=submission["id"],
        submitted_by_user_id=submission["submitted_by_user_id"],
        status=submission["status"],
        items=[dict(i) for i in items],
        created_at=submission["created_at"],
        approved_at=submission["approved_at"],
        ordered_at=submission["ordered_at"],
        approved_by=submission["approved_by"]
    )

    # Call Gemini + build AIInsight
    try:
        insight = generate_ai_summary_for_submission(submission_obj)
    except RuntimeError as e:
        # e.g. missing GOOGLE_API_KEY
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print("DEBUG unexpected AI error:", repr(e))
        raise HTTPException(status_code=500, detail=f"LLM error: {repr(e)}")

    # IMPORTANT: actually return the AIInsight object
    print("DEBUG endpoint returning insight: submission_id=", insight.submission_id)
    return insight



@app.get("/ai-insights", response_model=List[AIInsight])
def list_ai_insights(current_user: dict = Depends(get_current_user)):
    """
    List all AI insights (owner only).
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(
            status_code=403, detail="Only the owner can view AI insights."
        )
    return AI_INSIGHTS


@app.get("/ai-insights/{submission_id}", response_model=List[AIInsight])
def list_ai_insights_for_submission(
    submission_id: int,
    current_user: dict = Depends(get_current_user),
):
    """
    List AI insights for a specific submission (owner only).
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(
            status_code=403, detail="Only the owner can view AI insights."
        )

    return [
        insight
        for insight in AI_INSIGHTS
        if insight.submission_id == submission_id
    ]


@app.get("/reports/weekly-ai", response_model=WeeklyAIReport)
def get_weekly_ai_report(current_user: dict = Depends(get_current_user)):
    """
    Return the current 'weekly' AI inventory report as JSON.
    Owner only.
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(
            status_code=403,
            detail="Only the owner can view AI reports.",
        )

    report_text = build_weekly_ai_report(AI_INSIGHTS)
    return WeeklyAIReport(
        generated_at=datetime.utcnow(),
        report_text=report_text,
    )

@app.get("/reports/weekly-ai/download")
def download_weekly_ai_report(current_user: dict = Depends(get_current_user)):
    """
    Download the current 'weekly' AI report as a plain text file.
    Owner only.
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(
            status_code=403,
            detail="Only the owner can download AI reports.",
        )

    report_text = build_weekly_ai_report(AI_INSIGHTS)
    filename = f"weekly_ai_report_{datetime.utcnow().date().isoformat()}.txt"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return Response(content=report_text, media_type="text/plain", headers=headers)


# -------------------------------------------------
# Dashboard endpoints (Owner only)
# -------------------------------------------------

@app.get("/dashboards/inventory-status", response_model=DashboardInventoryStatus)
def get_inventory_status(current_user: dict = Depends(get_current_user)):
    """
    Get current inventory status summary (owner only).
    Shows breakdown by status: pending, approved, ordered.
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can view dashboards")

    sql = text("""
        SELECT 
            status,
            COUNT(*) as count,
            COALESCE(SUM(submission_items.quantity), 0) as total_quantity
        FROM public.submissions
        LEFT JOIN public.submission_items ON submissions.id = submission_items.submission_id
        GROUP BY status
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()

    status_data = {row["status"]: row for row in rows}

    return DashboardInventoryStatus(
        total_pending=int(status_data.get("pending", {}).get("count", 0)),
        total_approved=int(status_data.get("approved", {}).get("count", 0)),
        total_ordered=int(status_data.get("ordered", {}).get("count", 0)),
        pending_quantity=float(status_data.get("pending", {}).get("total_quantity", 0)),
        approved_quantity=float(status_data.get("approved", {}).get("total_quantity", 0)),
        ordered_quantity=float(status_data.get("ordered", {}).get("total_quantity", 0)),
    )


@app.get("/dashboards/top-items", response_model=List[DashboardTopItem])
def get_top_items(current_user: dict = Depends(get_current_user), limit: int = 10):
    """
    Get most frequently requested catalog items (owner only).
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can view dashboards")

    sql = text(f"""
        SELECT 
            catalog_items.id as catalog_item_id,
            catalog_items.name as item_name,
            catalog_items.category,
            COUNT(submission_items.id) as request_count,
            COALESCE(SUM(submission_items.quantity), 0) as total_quantity
        FROM public.catalog_items
        LEFT JOIN public.submission_items ON catalog_items.id = submission_items.catalog_item_id
        GROUP BY catalog_items.id, catalog_items.name, catalog_items.category
        ORDER BY request_count DESC
        LIMIT {limit}
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()

    return [
        DashboardTopItem(
            catalog_item_id=row["catalog_item_id"],
            item_name=row["item_name"],
            request_count=int(row["request_count"]),
            total_quantity=float(row["total_quantity"]),
            category=row["category"]
        )
        for row in rows
    ]


@app.get("/dashboards/statistics", response_model=DashboardStatistics)
def get_statistics(current_user: dict = Depends(get_current_user)):
    """
    Get overall inventory statistics (owner only).
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can view dashboards")

    # Total submissions
    sql_total = text("SELECT COUNT(*) as count FROM public.submissions")
    with engine.connect() as conn:
        total_subs = conn.execute(sql_total).scalar()

    # Average items per submission
    sql_avg = text("""
        SELECT COALESCE(AVG(item_count), 0) as avg_items
        FROM (
            SELECT COUNT(id) as item_count
            FROM public.submission_items
            GROUP BY submission_id
        ) counts
    """)
    with engine.connect() as conn:
        avg_items = float(conn.execute(sql_avg).scalar())

    # Approval rate
    sql_approved = text("""
        SELECT COUNT(*) as approved_count
        FROM public.submissions
        WHERE status IN ('approved', 'ordered')
    """)
    with engine.connect() as conn:
        approved_count = conn.execute(sql_approved).scalar()

    approval_rate = (approved_count / total_subs * 100) if total_subs > 0 else 0

    # Submissions this week
    sql_this_week = text("""
        SELECT COUNT(*) as count
        FROM public.submissions
        WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
    """)
    with engine.connect() as conn:
        this_week = conn.execute(sql_this_week).scalar()

    # Most active user
    sql_active = text("""
        SELECT submitted_by_user_id, COUNT(*) as submission_count
        FROM public.submissions
        GROUP BY submitted_by_user_id
        ORDER BY submission_count DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        active_user = conn.execute(sql_active).mappings().first()

    most_active_user_id = active_user["submitted_by_user_id"] if active_user else 0
    most_active_count = active_user["submission_count"] if active_user else 0

    return DashboardStatistics(
        total_submissions=int(total_subs),
        avg_items_per_submission=avg_items,
        approval_rate=approval_rate,
        submissions_this_week=int(this_week),
        most_active_user_id=most_active_user_id,
        most_active_user_submissions=int(most_active_count),
    )


@app.get("/dashboards/sales-timeline", response_model=DashboardTimeline)
def get_sales_timeline(
    current_user: dict = Depends(get_current_user),
    period: str = "daily",
    days: int = 30
):
    """
    Get sales timeline over specified period (owner only).
    Period can be: daily, weekly, or monthly
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can view dashboards")

    # Build SQL based on period
    if period == "daily":
        date_format = "YYYY-MM-DD"
        interval_str = f"{days} days"
    elif period == "weekly":
        date_format = "YYYY-W"
        interval_str = f"{days * 7} days"
    elif period == "monthly":
        date_format = "YYYY-MM"
        interval_str = f"{days} months"
    else:
        raise HTTPException(status_code=400, detail="Period must be daily, weekly, or monthly")

    # Get raw data with status for each submission
    sql = text(f"""
        SELECT 
            TO_CHAR(submissions.created_at, '{date_format}') as date,
            submissions.status,
            COUNT(DISTINCT submissions.id) as submission_count,
            COALESCE(SUM(submission_items.quantity), 0) as total_quantity
        FROM public.submissions
        LEFT JOIN public.submission_items ON submissions.id = submission_items.submission_id
        WHERE submissions.created_at >= CURRENT_DATE - INTERVAL '{interval_str}'
        GROUP BY TO_CHAR(submissions.created_at, '{date_format}'), submissions.status
        ORDER BY date DESC
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()

    # Aggregate results by date
    aggregated = {}
    for row in rows:
        date = row["date"]
        if date not in aggregated:
            aggregated[date] = {
                "date": date,
                "submission_count": 0,
                "total_quantity": 0,
                "status_breakdown": {}
            }
        aggregated[date]["submission_count"] += row["submission_count"]
        aggregated[date]["total_quantity"] += row["total_quantity"]
        aggregated[date]["status_breakdown"][row["status"]] = row["submission_count"]

    timeline_items = []
    for date in sorted(aggregated.keys(), reverse=True):
        data = aggregated[date]
        timeline_items.append(DashboardTimelineItem(
            date=data["date"],
            submission_count=int(data["submission_count"]),
            total_quantity=float(data["total_quantity"]),
            status_breakdown=data["status_breakdown"]
        ))

    return DashboardTimeline(
        period=period,
        data=timeline_items
    )


@app.get("/dashboards/summary", response_model=DashboardSummary)
def get_dashboard_summary(current_user: dict = Depends(get_current_user)):
    """
    Get comprehensive dashboard summary (owner only).
    Combines inventory status, statistics, top items, and recent timeline.
    """
    if current_user["role"] != UserRole.OWNER:
        raise HTTPException(status_code=403, detail="Only owner can view dashboards")

    # Get all dashboard components
    inventory = get_inventory_status(current_user)
    stats = get_statistics(current_user)
    top_items = get_top_items(current_user, limit=5)
    timeline = get_sales_timeline(current_user, period="daily", days=7)

    return DashboardSummary(
        generated_at=datetime.utcnow(),
        inventory_status=inventory,
        statistics=stats,
        top_items=top_items,
        recent_timeline=timeline.data[:7]
    )


from fastapi.responses import Response

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# -------------------------------------------------
# AI / LLM helpers (Gemini)
# -------------------------------------------------

PROMPT_TEMPLATE = """
You are an AI assistant helping a restaurant owner review weekly inventory requests.

Given a single inventory submission in JSON format:
- Produce a short, clear summary (max 3 lines).
- List the top categories involved.
- Flag anything that looks unusually high or important.

Respond strictly in JSON with the following  s:
- summary
- top_categories
- alerts
"""


def _parse_llm_json(raw: str) -> dict:
    """
    Takes Gemini's text output (possibly wrapped in ```json ``` fences)
    and returns a Python dict.
    """
    cleaned = raw.strip()

    # If it starts with ``` remove the code fences
    if cleaned.startswith("```"):
        # remove first line (``` or ```json)
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1 :]

        # if it starts with 'json' on its own line, skip that too
        if cleaned.lstrip().startswith("json"):
            second_newline = cleaned.find("\n")
            cleaned = cleaned[second_newline + 1 :]

        # drop trailing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]

    return json.loads(cleaned)


def _get_gemini_model():
    """
    Configure and return a Gemini model instance.
    Expects GOOGLE_API_KEY to be set in the environment.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    print("DEBUG GOOGLE_API_KEY prefix:", api_key[:5] if api_key else "NONE")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it before using the AI endpoints."
        )

    genai.configure(api_key=api_key)
    # Same ID you used successfully in the notebook
    return genai.GenerativeModel("gemini-2.5-flash")



def generate_ai_summary_for_submission(submission: Submission) -> AIInsight:
    """
    Call Gemini to generate an AI summary for a given submission,
    store it in AI_INSIGHTS, and return the AIInsight object.
    """
    # 1) Get the Gemini model (uses GOOGLE_API_KEY env var)
    model = _get_gemini_model()

    # 2) Convert submission to JSON-friendly dict (datetime -> ISO string)
    submission_data = submission.model_dump()
    if isinstance(submission_data.get("created_at"), datetime):
        submission_data["created_at"] = submission_data["created_at"].isoformat()

    # 3) Build the prompt text
    prompt_text = f"""{PROMPT_TEMPLATE}

Here is the submission JSON:

```json
{json.dumps(submission_data, indent=2)}
```"""

    # 4) Call Gemini
    response = model.generate_content(prompt_text)
    raw_text = response.text or ""

    # 5) Parse the LLM JSON output
    parsed = _parse_llm_json(raw_text)

    # 6) Build an AIInsight object
    insight = AIInsight(
        submission_id=submission.id,
        generated_at=datetime.utcnow(),
        summary=parsed.get("summary", ""),
        top_categories=parsed.get("top_categories") or [],
        alerts=parsed.get("alerts") or [],
        model="gemini-2.5-flash",
        confidence_note="AI-generated summary. Owner review required.",
    )

    # 7) Store and return it
    AI_INSIGHTS.append(insight)
    print("DEBUG AI insight generated: submission_id=", insight.submission_id)
    return insight


def build_weekly_ai_report(insights: List[AIInsight]) -> str:
    """
    Build a simple weekly-style report from all AI insights in memory.
    For now this looks at all AI_INSIGHTS; later this could filter by date.
    """
    if not insights:
        return "No AI insights available yet. Generate insights on submissions first."

    category_counter = Counter()
    alert_counter = Counter()

    for ins in insights:
        category_counter.update(ins.top_categories or [])
        alert_counter.update(ins.alerts or [])

    lines: List[str] = []
    lines.append("WEEKLY AI INVENTORY SUMMARY")
    lines.append(f"Generated at: {datetime.utcnow().isoformat()}")
    lines.append("")
    lines.append("Top categories observed:")
    for cat, count in category_counter.most_common():
        lines.append(f"- {cat}: {count} submission(s)")
    lines.append("")
    lines.append("Alerts mentioned across submissions:")
    for alert, count in alert_counter.most_common():
        lines.append(f"- {alert} (seen {count} time(s))")

    return "\n".join(lines)


# -------------------------------------------------
# Product endpoints
# -------------------------------------------------


@app.get("/products", response_model=List[Product])
def list_products(only_active: bool = True):
    """
    List all products in the catalog.
    If only_active=True (default), return only active products.
    """

    # DEBUG: if PRODUCTS is empty, try to load again and print info
    if not PRODUCTS:
        print("DEBUG: PRODUCTS was empty inside /products, reloading CSV...")
        load_products_from_csv()
        print(f"DEBUG: after reload, PRODUCTS has {len(PRODUCTS)} items")

    if only_active:
        return [p for p in PRODUCTS if p.is_active]
    return PRODUCTS



@app.get("/products/{product_id}", response_model=Product)
def get_product(product_id: int):
    """
    Get a single product by its ID.
    """
    product = next((p for p in PRODUCTS if p.id == product_id), None)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


UI_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Rest.Inventory Demo UI</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 12px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 14px; }
    input, textarea, select { width: 100%; padding: 8px; margin-top: 6px; margin-bottom: 10px; }
    button { padding: 10px 14px; cursor: pointer; }
    code, pre { background:#f6f6f6; padding:10px; display:block; overflow:auto; }
    .row { display:flex; gap:12px; }
    .row > div { flex:1; }
    .ok { color: #0a7; }
    .err { color: #c00; }
  </style>
</head>
<body>
  <h1>Rest.Inventory — Demo UI</h1>
  <p>This page calls your FastAPI endpoints directly (same server). Token is stored in localStorage.</p>

  <div class="card" id="createUserCard" style="display:none;">
    <h2>1) Create User</h2>
    <div class="row">
      <div>
        <label>Name</label>
        <input id="cu_name" placeholder="Marco" />
      </div>
      <div>
        <label>Email</label>
        <input id="cu_email" placeholder="marco@test.com" />
      </div>
    </div>
    <div class="row">
      <div>
        <label>Role</label>
        <select id="cu_role">
          <option value="admin">admin</option>
          <option value="owner">owner</option>
          <option value="kitchen">kitchen</option>
          <option value="floor">floor</option>
        </select>
      </div>
      <div>
        <label>Password</label>
        <input id="cu_password" placeholder="1234" />
      </div>
    </div>
    <button onclick="createUser()">Create user</button>
  </div>

  <div id="loginSection">
    <div class="card">
      <h2>Login</h2>
    <label>Email</label>
    <input id="li_email" placeholder="owner@test.com" />
    <label>Password</label>
    <input id="li_password" placeholder="1234" />
    <button onclick="login()">Login</button>
    <p>Token: <code id="token_box">(none)</code></p>
    <button onclick="logout()">Logout</button>
  </div>
</div>

<div id="appSection" style="display:none;">

  <div class="card">
    <button onclick="logout()">Logout</button>
  </div>

  <div class="card">
    <h2>3) Add Catalog Item</h2>

    <label>Product Name</label>
    <input id="cat_name" placeholder="Tomato Sauce" />

    <label>Season Category</label>
    <select id="cat_category">
      <option value="Spring">Spring</option>
      <option value="Summer">Summer</option>
      <option value="August">August</option>
      <option value="Winter">Winter</option>
    </select>

    <label>Unit</label>
    <input id="cat_unit" placeholder="bottle" />

    <label>Price</label>
    <input id="cat_price" type="number" step="0.01" placeholder="12.50" />

    <button onclick="createCatalogItem()">Add Catalog Item</button>
  </div>

  <div class="card">
    <h2>4) Create Submission (as logged-in user)</h2>
    <p>Paste items JSON (matches your SubmissionCreate schema):</p>
    <textarea id="sub_json" rows="8">{
  "items": [
    { "product_id": 87, "quantity_needed": 12, "comment": "For tomato sauce for the week" },
    { "product_id": 93, "quantity_needed": 3, "comment": "Running low, used a lot this week" }
  ]
}</textarea>
    <button onclick="createSubmission()">Create submission</button>
  </div>

  <div class="card">
    <h2>5) Generate AI Insight (owner only)</h2>
    <label>Submission ID</label>
    <input id="ai_sub_id" value="1" />
    <button onclick="generateAI()">Generate AI insight</button>
  </div>

  <div class="card">
    <h2>6) Weekly Report</h2>
    <button onclick="getWeeklyReport()">Get weekly report (JSON)</button>
    <p><a href="/reports/weekly-ai/download" target="_blank">Download weekly report (.txt)</a></p>
  </div>

  <div class="card">
    <h2>Output</h2>
    <div id="status"></div>
    <pre id="out"></pre>
  </div>

</div>

<script>
  function setStatus(msg, ok=true){
    const el = document.getElementById("status");
    el.className = ok ? "ok" : "err";
    el.textContent = msg;
  }

  function setOut(obj){
    document.getElementById("out").textContent =
      typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  }

  function getToken(){
    return localStorage.getItem("token") || "";
  }

  function refreshTokenBox(){
    const t = getToken();
    document.getElementById("token_box").textContent = t ? t : "(none)";
  }

  async function api(path, method="GET", body=null){
    const headers = { "accept": "application/json" };
    const token = getToken();
    if(token) headers["Authorization"] = "Bearer " + token;
    if(body !== null) headers["Content-Type"] = "application/json";

    const res = await fetch(path, {
      method,
      headers,
      body: body !== null ? JSON.stringify(body) : null
    });

    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }

    if(!res.ok){
      throw new Error((data && data.detail) ? data.detail : ("HTTP " + res.status + ": " + text));
    }
    return data;
  }

  async function createUser(){
    try{
      const payload = {
        name: document.getElementById("cu_name").value,
        email: document.getElementById("cu_email").value,
        role: document.getElementById("cu_role").value,
        password: document.getElementById("cu_password").value
      };
      const data = await api("/users", "POST", payload);
      setStatus("User created ✅");
      setOut(data);
    }catch(e){
      setStatus("Create user failed ❌ " + e.message, false);
      setOut(e.message);
    }
  }

  async function login(){
    try{
      const payload = {
        email: document.getElementById("li_email").value,
        password: document.getElementById("li_password").value
      };
      const data = await api("/auth/login", "POST", payload);
      localStorage.setItem("token", data.access_token);
      refreshTokenBox();
      showApp();
      setStatus("Logged in ✅");
      setOut(data);
    }catch(e){
      setStatus("Login failed ❌ " + e.message, false);
      setOut(e.message);
    }
  }
  
function logout(){
  localStorage.removeItem("token");
  refreshTokenBox();
  setStatus("Logged out ✅");
  setOut("");
  showLogin();
}

function showApp(){
  document.getElementById("loginSection").style.display = "none";
  document.getElementById("appSection").style.display = "block";
}

function showLogin(){
  document.getElementById("loginSection").style.display = "block";
  document.getElementById("appSection").style.display = "none";
}

async function createCatalogItem(){
  try{
    const payload = {
      name: document.getElementById("cat_name").value,
      category: document.getElementById("cat_category").value,
      unit: document.getElementById("cat_unit").value,
      price: Number(document.getElementById("cat_price").value)
    };

    const data = await api("/catalog/items", "POST", payload);

    setStatus("Catalog item added ✅");
    setOut(data);
  }catch(e){
    setStatus("Add catalog item failed ❌ " + e.message, false);
    setOut(e.message);
  }
}

async function createSubmission(){
  try{
    const raw = document.getElementById("sub_json").value;
    const payload = JSON.parse(raw);
    const data = await api("/submissions", "POST", payload);
    setStatus("Submission created ✅");
    setOut(data);
    document.getElementById("ai_sub_id").value = data.id;
  }catch(e){
    setStatus("Create submission failed ❌ " + e.message, false);
    setOut(e.message);
  }
}

async function generateAI(){
  try{
    const id = document.getElementById("ai_sub_id").value;
    const data = await api(`/submissions/${id}/ai-insights`, "POST", {});
    setStatus("AI insight generated ✅");
    setOut(data);
  }catch(e){
    setStatus("AI insight failed ❌ " + e.message, false);
    setOut(e.message);
  }
}

async function getWeeklyReport(){
  try{
    const data = await api("/reports/weekly-ai", "GET");
    setStatus("Weekly report loaded ✅");
    setOut(data);
  }catch(e){
    setStatus("Weekly report failed ❌ " + e.message, false);
    setOut(e.message);
  }
}

  refreshTokenBox();

  if(getToken()){
    showApp();
  }else{
    showLogin();
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def demo_ui():
    return UI_HTML
