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
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Depends, Response
from collections import Counter

from fastapi.responses import HTMLResponse



BASE_DIR = Path(__file__).resolve().parent.parent



app = FastAPI(title="Rest.Inventory API")

# -------------------------------------------------
# Models for Users
# -------------------------------------------------


class UserRole(str):
    OWNER = "owner"
    KITCHEN = "kitchen"
    FLOOR = "floor"


class UserCreate(BaseModel):
    name: str
    email: str
    role: str = Field(..., description="one of: owner, kitchen, floor")
    password: str


class User(BaseModel):
    id: int
    name: str
    email: str
    role: str
    password: str
    created_at: datetime


class UserPublic(BaseModel):
    id: int
    name: str
    email: str
    role: str
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
# Models for Submissions
# -------------------------------------------------


class ItemSelection(BaseModel):
    product_id: int = Field(..., description="Internal numeric product id (placeholder for now)")
    quantity_needed: float = Field(..., ge=0, description="Quantity requested by staff")
    comment: Optional[str] = Field(None, description="Optional note from staff")


class SubmissionCreate(BaseModel):
    items: List[ItemSelection]


# -------- Submission status helpers --------

class SubmissionStatus(str):
    PENDING = "pending"
    APPROVED = "approved"
    ORDERED = "ordered"


class Submission(BaseModel):
    id: int
    submitted_by_user_id: int
    items: List[ItemSelection]
    created_at: datetime
    status: str = Field(
        default=SubmissionStatus.PENDING,
        description="One of: pending, approved, ordered",
    )

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
# In-memory storage
# -------------------------------------------------

USERS: List[User] = []
SUBMISSIONS: List[Submission] = []

# token -> user_id
ACTIVE_TOKENS: Dict[str, int] = {}

# AI insights in memory
AI_INSIGHTS: List[AIInsight] = []


# -------------------------------------------------
# Security dependency
# -------------------------------------------------

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Resolve the Bearer token to a User object.
    """
    token = credentials.credentials
    user_id = ACTIVE_TOKENS.get(token)

    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = next((u for u in USERS if u.id == user_id), None)
    if user is None:
        # token points to a user that no longer exists
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user


# -------------------------------------------------
# Health endpoint
# -------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "Rest.Inventory"}


# -------------------------------------------------
# User endpoints
# -------------------------------------------------


@app.post("/users", response_model=UserPublic)
def create_user(payload: UserCreate):
    """
    Create a new user (owner, kitchen, or floor).
    """
    # basic uniqueness check on email
    if any(u.email == payload.email for u in USERS):
        raise HTTPException(status_code=400, detail="Email already registered")

    new_id = len(USERS) + 1
    user = User(
        id=new_id,
        name=payload.name,
        email=payload.email,
        role=payload.role,
        password=payload.password,
        created_at=datetime.utcnow(),
    )
    USERS.append(user)
    # Do NOT return password to the client
    return UserPublic(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        created_at=user.created_at,
    )


@app.get("/users", response_model=List[UserPublic])
def list_users():
    """
    List all users (for debugging / setup).
    """
    return [
        UserPublic(
            id=u.id,
            name=u.name,
            email=u.email,
            role=u.role,
            created_at=u.created_at,
        )
        for u in USERS
    ]


# -------------------------------------------------
# Auth endpoints
# -------------------------------------------------


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    """
    Verify email + password and return an access token.
    """
    user = next((u for u in USERS if u.email == payload.email), None)
    if user is None or user.password != payload.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # generate a random token and remember which user it belongs to
    access_token = token_hex(16)
    ACTIVE_TOKENS[access_token] = user.id

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        role=user.role,
    )


# -------------------------------------------------
# Submissions endpoints
# -------------------------------------------------


@app.post("/submissions", response_model=Submission)
def create_submission(
    payload: SubmissionCreate,
    current_user: User = Depends(get_current_user),
):
    """
    Create a new inventory submission linked to the authenticated user.
    """
    new_id = len(SUBMISSIONS) + 1
    submission = Submission(
        id=new_id,
        submitted_by_user_id=current_user.id,
        items=payload.items,
        created_at=datetime.utcnow(),
        status=SubmissionStatus.PENDING,  # NEW
    )
    SUBMISSIONS.append(submission)
    return submission



@app.get("/submissions", response_model=List[Submission])
def list_submissions(current_user: User = Depends(get_current_user)):
    """
    List all submissions.
    Only the owner is allowed to see every submission.
    """
    if current_user.role != UserRole.OWNER:
        raise HTTPException(
            status_code=403,
            detail="Only the owner can view all submissions.",
        )
    return SUBMISSIONS


@app.get("/submissions/me", response_model=List[Submission])
def list_my_submissions(current_user: User = Depends(get_current_user)):
    """
    List submissions created by the currently authenticated user.
    Kitchen/floor staff will use this to see their own requests.
    """
    return [
        s for s in SUBMISSIONS
        if s.submitted_by_user_id == current_user.id
    ]


class SubmissionStatusUpdate(BaseModel):
    status: str = Field(..., description="One of: pending, approved, ordered")


@app.patch("/submissions/{submission_id}/status", response_model=Submission)
def update_submission_status(
    submission_id: int,
    payload: SubmissionStatusUpdate,
    current_user: User = Depends(get_current_user),
):
    """
    Update the status of a submission.
    Only the owner is allowed to change status.
    """
    if current_user.role != UserRole.OWNER:
        raise HTTPException(
            status_code=403,
            detail="Only the owner can update submission status.",
        )

    # validate requested status
    allowed = {
        SubmissionStatus.PENDING,
        SubmissionStatus.APPROVED,
        SubmissionStatus.ORDERED,
    }
    if payload.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{payload.status}'. "
                   f"Allowed: {', '.join(sorted(allowed))}.",
        )

    submission = next((s for s in SUBMISSIONS if s.id == submission_id), None)
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    submission.status = payload.status
    return submission

# -------------------------------------------------
# AI Insight endpoints
# -------------------------------------------------


@app.post("/submissions/{submission_id}/ai-insights", response_model=AIInsight)
def create_ai_insight_for_submission(
    submission_id: int,
    current_user: User = Depends(get_current_user),
):
    """
    Generate an AI summary for a given submission (owner only).
    Calls Gemini, stores the AI insight in memory, and returns it.
    """
    # Only owner can call this
    if current_user.role != UserRole.OWNER:
        raise HTTPException(
            status_code=403,
            detail="Only the owner can generate AI insights.",
        )

    # Find the submission
    submission = next((s for s in SUBMISSIONS if s.id == submission_id), None)
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Call Gemini + build AIInsight
    try:
        insight = generate_ai_summary_for_submission(submission)
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
def list_ai_insights(current_user: User = Depends(get_current_user)):
    """
    List all AI insights (owner only).
    """
    if current_user.role != UserRole.OWNER:
        raise HTTPException(
            status_code=403, detail="Only the owner can view AI insights."
        )
    return AI_INSIGHTS


@app.get("/ai-insights/{submission_id}", response_model=List[AIInsight])
def list_ai_insights_for_submission(
    submission_id: int,
    current_user: User = Depends(get_current_user),
):
    """
    List AI insights for a specific submission (owner only).
    """
    if current_user.role != UserRole.OWNER:
        raise HTTPException(
            status_code=403, detail="Only the owner can view AI insights."
        )

    return [
        insight
        for insight in AI_INSIGHTS
        if insight.submission_id == submission_id
    ]


@app.get("/reports/weekly-ai", response_model=WeeklyAIReport)
def get_weekly_ai_report(current_user: User = Depends(get_current_user)):
    """
    Return the current 'weekly' AI inventory report as JSON.
    Owner only.
    """
    if current_user.role != UserRole.OWNER:
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
def download_weekly_ai_report(current_user: User = Depends(get_current_user)):
    """
    Download the current 'weekly' AI report as a plain text file.
    Owner only.
    """
    if current_user.role != UserRole.OWNER:
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

Respond strictly in JSON with the following keys:
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

  <div class="card">
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

  <div class="card">
    <h2>2) Login (get token)</h2>
    <label>Email</label>
    <input id="li_email" placeholder="owner@test.com" />
    <label>Password</label>
    <input id="li_password" placeholder="1234" />
    <button onclick="login()">Login</button>
    <p>Token: <code id="token_box">(none)</code></p>
    <button onclick="logout()">Logout</button>
  </div>

  <div class="card">
    <h2>3) Create Submission (as logged-in user)</h2>
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
    <h2>4) Generate AI Insight (owner only)</h2>
    <label>Submission ID</label>
    <input id="ai_sub_id" value="1" />
    <button onclick="generateAI()">Generate AI insight</button>
  </div>

  <div class="card">
    <h2>5) Weekly Report</h2>
    <button onclick="getWeeklyReport()">Get weekly report (JSON)</button>
    <p><a href="/reports/weekly-ai/download" target="_blank">Download weekly report (.txt)</a></p>
  </div>

  <div class="card">
    <h2>Output</h2>
    <div id="status"></div>
    <pre id="out"></pre>
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
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def demo_ui():
    return UI_HTML
