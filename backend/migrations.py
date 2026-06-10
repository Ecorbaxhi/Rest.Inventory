"""
Database migration script for Rest.Inventory
Handles schema creation and updates for all phases
Run with: python -m backend.migrations
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, text

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Use DIRECT_URL for migrations (session-mode pooler), fallback to DATABASE_URL
DATABASE_URL = os.environ.get("DIRECT_URL") or os.environ.get("DATABASE_URL")


def get_connection():
    """Create database connection"""
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"sslmode": "require"},
    )
    return engine


def check_table_exists(conn, table_name: str) -> bool:
    """Check if a table exists"""
    result = conn.execute(text(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = '{table_name}'
        )
    """)).scalar()
    return result


def check_column_exists(conn, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table"""
    result = conn.execute(text(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = '{table_name}' 
            AND column_name = '{column_name}'
        )
    """)).scalar()
    return result


def phase_1_user_system(engine):
    """
    PHASE 1: User & Role System Enhancement
    - Create users table if not exists
    - Add admin role support
    - Add approval tracking fields
    """
    print("\n🔄 PHASE 1: User & Role System Enhancement")
    
    with engine.begin() as conn:
        # Create users table if not exists
        if not check_table_exists(conn, "users"):
            print("  ➕ Creating users table...")
            conn.execute(text("""
                CREATE TABLE public.users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL DEFAULT 'floor',
                    is_approved BOOLEAN DEFAULT false,
                    approved_by INTEGER REFERENCES public.users(id),
                    approval_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            print("  ✅ users table created")
        else:
            print("  ⏭️  users table already exists")
            
            # Add is_approved column if not exists
            if not check_column_exists(conn, "users", "is_approved"):
                print("  ➕ Adding is_approved column to users table...")
                conn.execute(text("""
                    ALTER TABLE public.users 
                    ADD COLUMN is_approved BOOLEAN DEFAULT false
                """))
                print("  ✅ is_approved column added")
            else:
                print("  ⏭️  is_approved column already exists")
            
            # Add approved_by column if not exists
            if not check_column_exists(conn, "users", "approved_by"):
                print("  ➕ Adding approved_by column to users table...")
                conn.execute(text("""
                    ALTER TABLE public.users 
                    ADD COLUMN approved_by INTEGER REFERENCES public.users(id)
                """))
                print("  ✅ approved_by column added")
            else:
                print("  ⏭️  approved_by column already exists")
            
            # Add approval_date column if not exists
            if not check_column_exists(conn, "users", "approval_date"):
                print("  ➕ Adding approval_date column to users table...")
                conn.execute(text("""
                    ALTER TABLE public.users 
                    ADD COLUMN approval_date TIMESTAMP
                """))
                print("  ✅ approval_date column added")
            else:
                print("  ⏭️  approval_date column already exists")


def phase_2_catalog_system(engine):
    """
    PHASE 2: Catalog System
    - Create catalog_items table
    - Create catalog_requests table
    """
    print("\n🔄 PHASE 2: Catalog System")
    
    with engine.begin() as conn:
        # Create catalog_items table
        if not check_table_exists(conn, "catalog_items"):
            print("  ➕ Creating catalog_items table...")
            conn.execute(text("""
                CREATE TABLE public.catalog_items (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    category VARCHAR(100),
                    unit VARCHAR(50),
                    price DECIMAL(10, 2),
                    is_active BOOLEAN DEFAULT true,
                    created_by INTEGER NOT NULL REFERENCES public.users(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            print("  ✅ catalog_items table created")
        else:
            print("  ⏭️  catalog_items table already exists")
        
        # Create catalog_requests table
        if not check_table_exists(conn, "catalog_requests"):
            print("  ➕ Creating catalog_requests table...")
            conn.execute(text("""
                CREATE TABLE public.catalog_requests (
                    id SERIAL PRIMARY KEY,
                    item_id INTEGER NOT NULL REFERENCES public.catalog_items(id),
                    requested_by INTEGER NOT NULL REFERENCES public.users(id),
                    status VARCHAR(50) DEFAULT 'pending',
                    reason TEXT,
                    approved_by INTEGER REFERENCES public.users(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            print("  ✅ catalog_requests table created")
        else:
            print("  ⏭️  catalog_requests table already exists")


def phase_3_submissions(engine):
    """
    PHASE 3: Submissions & Request Flow
    - Create submissions table if not exists
    - Update submissions table
    - Create submission_items table
    """
    print("\n🔄 PHASE 3: Submissions & Request Flow")
    
    with engine.begin() as conn:
        # Create submissions table if not exists
        if not check_table_exists(conn, "submissions"):
            print("  ➕ Creating submissions table...")
            conn.execute(text("""
                CREATE TABLE public.submissions (
                    id SERIAL PRIMARY KEY,
                    submitted_by_user_id INTEGER NOT NULL REFERENCES public.users(id),
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    approved_at TIMESTAMP,
                    ordered_at TIMESTAMP,
                    approved_by INTEGER REFERENCES public.users(id)
                )
            """))
            print("  ✅ submissions table created")
        else:
            print("  ⏭️  submissions table already exists")
            
            # Add fields to submissions table if not exists
            if not check_column_exists(conn, "submissions", "approved_at"):
                print("  ➕ Adding approved_at column to submissions table...")
                conn.execute(text("""
                    ALTER TABLE public.submissions 
                    ADD COLUMN approved_at TIMESTAMP
                """))
                print("  ✅ approved_at column added")
            else:
                print("  ⏭️  approved_at column already exists")
            
            if not check_column_exists(conn, "submissions", "ordered_at"):
                print("  ➕ Adding ordered_at column to submissions table...")
                conn.execute(text("""
                    ALTER TABLE public.submissions 
                    ADD COLUMN ordered_at TIMESTAMP
                """))
                print("  ✅ ordered_at column added")
            else:
                print("  ⏭️  ordered_at column already exists")
            
            if not check_column_exists(conn, "submissions", "approved_by"):
                print("  ➕ Adding approved_by column to submissions table...")
                conn.execute(text("""
                    ALTER TABLE public.submissions 
                    ADD COLUMN approved_by INTEGER REFERENCES public.users(id)
                """))
                print("  ✅ approved_by column added")
            else:
                print("  ⏭️  approved_by column already exists")
        
        # Create submission_items table
        if not check_table_exists(conn, "submission_items"):
            print("  ➕ Creating submission_items table...")
            conn.execute(text("""
                CREATE TABLE public.submission_items (
                    id SERIAL PRIMARY KEY,
                    submission_id INTEGER NOT NULL REFERENCES public.submissions(id) ON DELETE CASCADE,
                    catalog_item_id INTEGER NOT NULL REFERENCES public.catalog_items(id),
                    quantity DECIMAL(10, 2) NOT NULL,
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            print("  ✅ submission_items table created")
        else:
            print("  ⏭️  submission_items table already exists")


def phase_4_dashboards(engine):
    """
    PHASE 4: Dashboards & Reporting
    - Create dashboard_data or similar tables for tracking
    """
    print("\n🔄 PHASE 4: Dashboards & Reporting")
    
    with engine.begin() as conn:
        # Create submission_stats table for dashboard data
        if not check_table_exists(conn, "submission_stats"):
            print("  ➕ Creating submission_stats table...")
            conn.execute(text("""
                CREATE TABLE public.submission_stats (
                    id SERIAL PRIMARY KEY,
                    submission_id INTEGER NOT NULL REFERENCES public.submissions(id) ON DELETE CASCADE,
                    item_count INTEGER,
                    total_quantity DECIMAL(10, 2),
                    status VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            print("  ✅ submission_stats table created")
        else:
            print("  ⏭️  submission_stats table already exists")


def run_all_migrations():
    """Run all migration phases"""
    print("=" * 60)
    print("🚀 REST.INVENTORY DATABASE MIGRATIONS")
    print("=" * 60)
    
    try:
        engine = get_connection()
        
        # Test connection
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful\n")
        
        # Run all phases
        phase_1_user_system(engine)
        phase_2_catalog_system(engine)
        phase_3_submissions(engine)
        phase_4_dashboards(engine)
        
        print("\n" + "=" * 60)
        print("✅ ALL MIGRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ MIGRATION FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_migrations()
