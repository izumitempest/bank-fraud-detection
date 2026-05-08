import os
import httpx
import json
from typing import Any


class SimpleSupabaseClient:
    """Minimal Supabase client using httpx directly."""
    
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            }
        )
    
    def from_(self, table: str):
        return TableQuery(self.client, self.url, table)


class TableQuery:
    def __init__(self, client: httpx.Client, base_url: str, table: str):
        self.client = client
        self.base_url = base_url
        self.table = table
        self.url = f"{base_url}/rest/v1/{table}"
        self.query_params = {}
    
    def insert(self, data):
        self.data = data
        return self
    
    def select(self, *args):
        if args:
            self.query_params["select"] = ",".join(args)
        else:
            self.query_params["select"] = "*"
        return self
    
    def order(self, column, desc=False):
        direction = "desc" if desc else "asc"
        self.query_params["order"] = f"{column}.{direction}"
        return self
    
    def limit(self, num):
        self.query_params["limit"] = str(num)
        return self
    
    def execute(self):
        if hasattr(self, 'data'):
            # INSERT operation
            response = self.client.post(self.url, json=self.data)
        else:
            # SELECT operation
            response = self.client.get(self.url, params=self.query_params)
        
        if response.status_code != 200:
            print(f"DEBUG - Status: {response.status_code}")
            print(f"DEBUG - Response: {response.text}")
        
        response.raise_for_status()
        return SupabaseResponse(response.json())


class SupabaseResponse:
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]


_supabase = None
_supabase_ready = False


def connect_to_supabase():
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

    _supabase = SimpleSupabaseClient(url, key)
    _supabase_ready = True
    return _supabase


def get_supabase():
    if _supabase is None:
        return connect_to_supabase()
    return _supabase


def close_supabase_connection() -> None:
    global _supabase, _supabase_ready
    if _supabase is not None:
        _supabase.client.close()
    _supabase = None
    _supabase_ready = False


def is_supabase_ready() -> bool:
    return _supabase_ready
