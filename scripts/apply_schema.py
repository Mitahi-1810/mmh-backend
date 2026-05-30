"""
Apply pending_migration.sql to Supabase via the database REST proxy.
Uses the service role key + httpx to POST SQL statements one at a time.

Usage:
    python scripts/apply_schema.py
"""
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SQL_FILE = ROOT / "pending_migration.sql"

def split_statements(sql: str) -> list[str]:
    """Split SQL file into individual statements, skipping comments."""
    statements = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or stripped == "":
            continue
        current.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
    return statements


def run_via_rpc(url: str, service_key: str, sql: str) -> tuple[bool, str]:
    """
    Execute a single SQL statement via Supabase pg_net or direct REST.
    Uses the Management API endpoint /rest/v1/rpc/exec (requires pg function to exist).
    Falls back to printing a clear error.
    """
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
    }
    # Try calling a custom exec_sql function (may not exist)
    endpoint = f"{url}/rest/v1/rpc/exec_sql"
    try:
        resp = httpx.post(
            endpoint,
            headers=headers,
            json={"query": sql},
            timeout=15,
        )
        if resp.status_code in (200, 201, 204):
            return True, "ok"
        return False, resp.text
    except Exception as exc:
        return False, str(exc)


def main():
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")

    if not url or not key:
        log.error("SUPABASE_URL / SUPABASE_SERVICE_KEY not set")
        sys.exit(1)

    if not SQL_FILE.exists():
        log.error("SQL file not found: %s", SQL_FILE)
        sys.exit(1)

    sql_text = SQL_FILE.read_text()
    statements = split_statements(sql_text)
    log.info("Found %d SQL statements to apply", len(statements))

    all_ok = True
    failed: list[str] = []

    for i, stmt in enumerate(statements, 1):
        first_line = stmt.splitlines()[0][:80]
        ok, msg = run_via_rpc(url, key, stmt)
        if ok:
            log.info("  [%d/%d] OK: %s", i, len(statements), first_line)
        else:
            log.warning("  [%d/%d] Could not auto-apply via RPC: %s", i, len(statements), first_line)
            log.debug("    Error: %s", msg)
            failed.append(stmt)
            all_ok = False
        time.sleep(0.1)

    if not all_ok:
        fallback_file = ROOT / "pending_migration.sql"
        log.warning("")
        log.warning("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.warning("Some statements could not run automatically.")
        log.warning("Do this ONE-TIME manual step (takes ~30 seconds):")
        log.warning("")
        log.warning("1. Open: https://supabase.com/dashboard/project/ifvuqjlmobaavjxrolbi/sql/new")
        log.warning("2. Paste the contents of: %s", fallback_file)
        log.warning("3. Click 'Run'")
        log.warning("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        log.info("All statements applied successfully!")


if __name__ == "__main__":
    main()
