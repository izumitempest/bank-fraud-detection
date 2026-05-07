import os

from supabase import Client, create_client


_supabase: Client | None = None
_supabase_ready: bool = False


def connect_to_supabase() -> Client:
    global _supabase, _supabase_ready

    if _supabase is not None:
        return _supabase

    url = os.getenv("SUPABASE_URL")
    # Prefer service role key for backend (bypasses RLS); fall back to anon key
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) must be set before starting the server."
        )

    _supabase = create_client(url, key)
    _supabase_ready = True
    return _supabase


def get_supabase() -> Client:
    if _supabase is None:
        return connect_to_supabase()
    return _supabase


def close_supabase_connection() -> None:
    global _supabase, _supabase_ready
    _supabase = None
    _supabase_ready = False


def is_supabase_ready() -> bool:
    return _supabase_ready
