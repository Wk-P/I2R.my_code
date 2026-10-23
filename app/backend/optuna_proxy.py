"""v6.1.0: optuna-dashboard (port 8082, systemd my-code-optuna) under /optuna/ of this panel, so it is reachable through
the same tunnel (443). The dashboard's frontend hard-codes /static/, /dashboard and an empty API base; those three
strings are rewritten on the fly (each occurs once, checked against optuna-dashboard 0.20)."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

UPSTREAM = "http://127.0.0.1:8082"
PREFIX = "/optuna"
REWRITE = {
    "index": [('src="/static/', f'src="{PREFIX}/static/')],
    "bundle": [('url_prefix:"/dashboard"', f'url_prefix:"{PREFIX}/dashboard"'),
               ('(null!=e?e:"")}handleResponse', f'(null!=e?e:"{PREFIX}")}}handleResponse')],
}
HOP = {"content-length", "content-encoding", "transfer-encoding", "connection"}

router = APIRouter()
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=60)


@router.get(PREFIX)
@router.get(PREFIX + "/")
async def root():
    return RedirectResponse(PREFIX + "/dashboard/")


@router.api_route(PREFIX + "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(path: str, request: Request):
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "accept-encoding", "content-length"}}
    r = await client.request(request.method, "/" + path, params=request.query_params, content=await request.body(),
                             headers=headers)
    body = r.content
    kind = "bundle" if path == "static/bundle.js" else "index" if "text/html" in r.headers.get("content-type", "") else None
    if kind:
        text = body.decode()
        for a, b in REWRITE[kind]:
            text = text.replace(a, b)
        body = text.encode()
    out = {k: v for k, v in r.headers.items() if k.lower() not in HOP}
    if "location" in out and out["location"].startswith("/"):
        out["location"] = PREFIX + out["location"]
    return Response(body, status_code=r.status_code, headers=out)
