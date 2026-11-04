"""Browser liveness for local Mac launcher; does not terminate active tasks."""
from __future__ import annotations
import threading
import time
import uuid
from fastapi import Body,HTTPException

SESSIONS={};LOCK=threading.RLock();LAST_OPEN=time.monotonic();EVER_OPEN=False

def attach(app,auth,active_checker):
    @app.post('/api/session/open',dependencies=[auth])
    def opened(data:dict=Body(default={})):
        global EVER_OPEN,LAST_OPEN
        token=str(data.get('session_id') or '')
        if not token or len(token)>120:raise HTTPException(400,'Invalid browser session ID')
        with LOCK:
            SESSIONS[token]=time.monotonic();LAST_OPEN=time.monotonic();EVER_OPEN=True
        return {'ok':True}
    @app.post('/api/session/heartbeat',dependencies=[auth])
    def heartbeat(data:dict=Body(default={})):
        sid=str(data.get('session_id') or '')
        with LOCK:
            if sid in SESSIONS:SESSIONS[sid]=time.monotonic();return {'ok':True}
        return {'ok':False}
    @app.post('/api/session/close',dependencies=[auth])
    def closed(data:dict=Body(default={})):
        sid=str(data.get('session_id') or '')
        with LOCK:SESSIONS.pop(sid,None)
        return {'ok':True}
    @app.get('/api/session/status',dependencies=[auth])
    def status():
        with LOCK:
            now=time.monotonic()
            stale=[k for k,v in SESSIONS.items() if now-v>18]
            for k in stale:SESSIONS.pop(k,None)
            return {'ever_open':EVER_OPEN,'browser_sessions':len(SESSIONS),
                    'seconds_since_open':round(now-LAST_OPEN,2),'active_jobs':bool(active_checker())}
