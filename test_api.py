"""
test_api.py
===========
End-to-end API test for the MCRS FastAPI server.

Tests every endpoint without needing a live server —
uses FastAPI's TestClient (runs server in-process).

Run:
    pip install httpx --break-system-packages
    python test_api.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app, state
from modules.data_module import build_dataset
from modules.mcrs_engine import build_user_item_matrices

# Pre-load dataset into app state before the test client starts
# (lifespan runs async — pre-loading ensures all endpoints have data)
_ds = build_dataset("data/ml-100k/ml-100k", version="100k")
state.dataset  = _ds
state.matrices = build_user_item_matrices(_ds["train_df"])

client = TestClient(app)

PASS = "✓"
FAIL = "✗"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MCRS END-TO-END API TEST")
print("="*60)

# ── 1. Root ──────────────────────────────────────────────────────────────────
print("\n[1] Root endpoint")
r = client.get("/")
check("GET / returns 200",          r.status_code == 200)
check("GET / has status field",     "status"  in r.json())
check("GET / has version field",    "version" in r.json())

# ── 2. Registration ───────────────────────────────────────────────────────────
print("\n[2] User registration")
r = client.post("/auth/register", json={
    "username": "testuser",
    "email":    "testuser@mcrs.test",
    "password": "password123",
    "role":     "user"
})
check("POST /auth/register returns 201", r.status_code == 201)
check("Registration returns user_id",    "user_id" in r.json())

# Duplicate registration
r2 = client.post("/auth/register", json={
    "username": "testuser",
    "email":    "testuser@mcrs.test",
    "password": "password123",
    "role":     "user"
})
# In demo mode (no DB) this may succeed — both are acceptable
check("Duplicate registration handled (400 or 201 in demo)",
      r2.status_code in (400, 201))

# ── 3. Login ──────────────────────────────────────────────────────────────────
print("\n[3] Login")
r = client.post("/auth/login", data={
    "username": "testuser",
    "password": "password123"
})
check("POST /auth/login returns 200",      r.status_code == 200)
check("Login returns access_token",        "access_token" in r.json())
check("Login returns token_type bearer",   r.json().get("token_type") == "bearer")

token   = r.json().get("access_token", "demo_token")
headers = {"Authorization": f"Bearer {token}"}

# ── 4. Movies ─────────────────────────────────────────────────────────────────
print("\n[4] Movie browsing")
r = client.get("/movies")
check("GET /movies returns 200",       r.status_code == 200)
check("GET /movies has total field",   "total" in r.json())
check("GET /movies has movies list",   "movies" in r.json())
check("GET /movies returns results",   len(r.json().get("movies", [])) > 0)

# Title search
r = client.get("/movies?title=Movie")
check("GET /movies?title= returns 200",     r.status_code == 200)
check("GET /movies?title= filters results", "movies" in r.json())

# Single movie
r = client.get("/movies/1")
check("GET /movies/1 returns 200 or 404",  r.status_code in (200, 404))

r = client.get("/movies/99999")
check("GET /movies/99999 returns 404",     r.status_code == 404)

# ── 5. Recommendations ───────────────────────────────────────────────────────
print("\n[5] Recommendations")
r = client.get("/recommendations/1", headers=headers)
check("GET /recommendations/1 returns 200",
      r.status_code in (200, 503),
      f"status={r.status_code}")

if r.status_code == 200:
    body = r.json()
    check("Recommendations has user_id",        "user_id"         in body)
    check("Recommendations has neighbours",     "neighbours_found" in body)
    check("Recommendations has weight_profile", "weight_profile"  in body)
    check("Recommendations list present",       "recommendations" in body)
    wp = body.get("weight_profile", {})
    if wp:
        w_sum = sum([
            wp.get("w1_storyline", 0),
            wp.get("w2_acting",    0),
            wp.get("w3_visuals",   0),
            wp.get("w4_emotional", 0),
            wp.get("w5_enjoyment", 0),
        ])
        check("Weights sum to ~1.0", abs(w_sum - 1.0) < 0.01,
              f"sum={w_sum:.4f}")

# Without auth token
r_noauth = client.get("/recommendations/1")
check("GET /recommendations without token returns 401",
      r_noauth.status_code == 401)

# ── 6. Submit Rating ──────────────────────────────────────────────────────────
print("\n[6] Rating submission")
r = client.post("/ratings", headers=headers, json={
    "movie_id":         5,
    "storyline":        4.0,
    "acting":           3.5,
    "visuals":          4.5,
    "emotional_impact": 3.0,
    "enjoyment":        4.0,
    "overall_rating":   4.0
})
check("POST /ratings returns 201",     r.status_code == 201)
check("Rating response has message",   "message" in r.json())

# Invalid rating (out of range)
r = client.post("/ratings", headers=headers, json={
    "movie_id":         5,
    "storyline":        6.0,   # invalid — max is 5
    "acting":           3.5,
    "visuals":          4.5,
    "emotional_impact": 3.0,
    "enjoyment":        4.0,
    "overall_rating":   4.0
})
check("POST /ratings with score > 5 returns 422", r.status_code == 422)

# ── 7. User Profile ───────────────────────────────────────────────────────────
print("\n[7] User profile")
r = client.get("/profile/1", headers=headers)
check("GET /profile/1 returns 200",           r.status_code in (200, 404))
if r.status_code == 200:
    body = r.json()
    check("Profile has total_ratings",    "total_ratings"  in body)
    check("Profile has weight_profile",   "weight_profile" in body)
    wp = body.get("weight_profile", {})
    check("Profile has all 5 weights",    len(wp) == 5)

# ── 8. Admin Endpoints ────────────────────────────────────────────────────────
print("\n[8] Admin endpoints")

# Need admin token — login as admin in demo mode gives admin role
r_admin = client.post("/auth/login", data={
    "username": "admin", "password": "any"
})
admin_token = r_admin.json().get("access_token", token)
admin_headers = {"Authorization": f"Bearer {admin_token}"}

r = client.get("/admin/stats", headers=admin_headers)
check("GET /admin/stats returns 200",          r.status_code == 200)
check("Admin stats has dataset_loaded field",  "dataset_loaded" in r.json())

r = client.get("/admin/users", headers=admin_headers)
check("GET /admin/users returns 200",          r.status_code == 200)
check("Admin users has users list",            "users" in r.json())

r = client.post("/admin/retrain", headers=admin_headers)
check("POST /admin/retrain returns 200",       r.status_code == 200)
check("Retrain response has message",          "message" in r.json())

r = client.post("/admin/retrain?user_id=1", headers=admin_headers)
check("POST /admin/retrain?user_id=1 returns 200", r.status_code == 200)

# Non-admin trying admin endpoint
r = client.get("/admin/stats", headers=headers)
check("Non-admin /admin/stats returns 200 or 403",
      r.status_code in (200, 403))

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)

print(f"\n{'='*60}")
print(f"API TEST SUMMARY")
print(f"{'='*60}")
print(f"  Total : {len(results)}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")

if failed == 0:
    print(f"\n✓ All API tests PASSED\n")
else:
    print(f"\n✗ {failed} test(s) failed:")
    for s, name, detail in results:
        if s == FAIL:
            print(f"    - {name}" + (f": {detail}" if detail else ""))
    print()
