"""Lightweight competition entrypoint: DRES stays online while search boots/fails."""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import secrets
import signal
import socket
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from dres_submission import create_router
from submission_maps import SubmissionMaps

ROOT=Path(__file__).resolve().parent
HOP={'host','connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer','transfer-encoding','upgrade'}

def create_app(*, start_search=True, transport=None, data_root=None, dres_service=None):
    port=int(os.getenv('SEARCH_SERVER_PORT','8081'))
    upstream=f'http://127.0.0.1:{port}'
    isolated=os.getenv('ISOLATE_VISUAL_WORKER','0').lower() in {'1','true','yes'}
    visual_port=int(os.getenv('VISUAL_SERVER_PORT','8082'))
    visual_upstream=f'http://127.0.0.1:{visual_port}'
    state=dict(ready=False,text_ready=False,status='starting',phase='loading_search',submission_ready=True,
               frames=0,error=None,instance=secrets.token_hex(16))
    process=None
    visual_process=None
    visual_restarts=0
    text_snapshot={}
    visual_snapshot={}
    started=time.monotonic()

    async def launch_visual_worker():
        nonlocal visual_process, visual_restarts
        if not isolated:
            raise RuntimeError('Visual worker isolation is disabled')
        if visual_process and visual_process.returncode is None:
            return visual_process
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',visual_port))
        visual_env=dict(os.environ,SERVER_HOST='127.0.0.1',SERVER_PORT=str(visual_port),
                        AIC_GATEWAY_CHILD='1',AIC_SEARCH_INSTANCE=state['instance'],
                        AIC_WORKER_ROLE='visual',AIC_SKIP_REDIS_START='1',
                        STOP_REDIS_ON_EXIT='0',OPEN_BROWSER='0')
        visual_process=await asyncio.create_subprocess_exec(
            'bash',str(ROOT/'run_final_hybrid_wsl.sh'),'run',cwd=ROOT,
            env=visual_env,start_new_session=True)
        visual_restarts += 1
        return visual_process

    @asynccontextmanager
    async def lifespan(app):
        nonlocal process, visual_process, text_snapshot, visual_snapshot
        async with httpx.AsyncClient(transport=transport,trust_env=False,
                                    timeout=httpx.Timeout(180,connect=1),follow_redirects=False) as client:
            app.state.client=client
            if start_search:
                try:
                    # Never proxy to an unrelated listener left on the search port.
                    with socket.socket() as probe:
                        probe.bind(('127.0.0.1',port))
                    env=dict(os.environ,SERVER_HOST='127.0.0.1',SERVER_PORT=str(port),
                             AIC_GATEWAY_CHILD='1',AIC_SEARCH_INSTANCE=state['instance'],
                             STOP_REDIS_ON_EXIT=os.getenv('STOP_REDIS_ON_EXIT','0'),OPEN_BROWSER='0')
                    if isolated:
                        env['AIC_WORKER_ROLE']='text'
                    process=await asyncio.create_subprocess_exec('bash',str(ROOT/'run_final_hybrid_wsl.sh'),'run',
                                                                 cwd=ROOT,env=env,start_new_session=True)
                    if isolated:
                        await launch_visual_worker()
                except (OSError,ValueError) as exc:
                    state.update(status='failed',phase='failed',error=f'Search launch failed: {exc}')
            async def monitor():
                nonlocal text_snapshot, visual_snapshot
                while True:
                    if process and process.returncode is not None:
                        state.update(ready=False,text_ready=False,status='failed',phase='failed',error=f'Search process exited ({process.returncode}). See terminal log.')
                    elif state['status']!='failed':
                        try:
                            response=await client.get(upstream+'/ready',timeout=2)
                            payload=response.json()
                            if payload.get('instance') != state['instance']:
                                raise ValueError('Search instance mismatch')
                            text_snapshot={k:v for k,v in payload.items() if k!='instance'}
                            state.update(text_snapshot)
                            state['submission_ready']=True
                            if state['ready'] and 'search_boot_ms' not in state:
                                state['search_boot_ms']=round((time.monotonic()-started)*1000)
                        except (httpx.HTTPError,ValueError):
                            state.update(ready=False,text_ready=False,status='starting',phase='loading_search')
                    if isolated:
                        if visual_process and visual_process.returncode is not None:
                            visual_snapshot={'ready':False,'status':'failed','phase':'visual_failed',
                                             'error':f'Visual worker exited ({visual_process.returncode}). OCR/ASR remain available.'}
                        else:
                            try:
                                response=await client.get(visual_upstream+'/ready',timeout=2)
                                payload=response.json()
                                if payload.get('instance') != state['instance']:
                                    raise ValueError('Visual instance mismatch')
                                visual_snapshot={k:v for k,v in payload.items() if k!='instance'}
                            except (httpx.HTTPError,ValueError):
                                visual_snapshot={'ready':False,'status':'starting','phase':'loading_visual'}
                        text_ok=bool(text_snapshot.get('text_ready'))
                        visual_ok=bool(visual_snapshot.get('ready'))
                        state.update(text_ready=text_ok,ready=visual_ok,
                                     status='ready' if visual_ok else ('loading_visual' if text_ok else 'starting'),
                                     phase='ready' if visual_ok else visual_snapshot.get('phase','loading_visual'),
                                     frames=visual_snapshot.get('frames',text_snapshot.get('frames',0)),
                                     error=visual_snapshot.get('error'))
                        if visual_ok and 'search_boot_ms' not in state:
                            state['search_boot_ms']=round((time.monotonic()-started)*1000)
                    await asyncio.sleep(1)
            task=asyncio.create_task(monitor())
            print(f'DRES/UI available immediately at http://localhost:{os.getenv("SERVER_PORT","8080")}/#dres',flush=True)
            try:
                yield
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                for child in (process, visual_process):
                    if not child or child.returncode is not None:
                        continue
                    try:
                        os.killpg(child.pid,signal.SIGTERM)
                        await asyncio.wait_for(child.wait(),timeout=10)
                    except asyncio.TimeoutError:
                        os.killpg(child.pid,signal.SIGKILL)
                        await child.wait()
                    except ProcessLookupError:
                        pass

    app=FastAPI(title='Competition UI and DRES',lifespan=lifespan)
    app.state.readiness=state
    maps=SubmissionMaps(data_root or os.getenv('DATA_ROOT',str(ROOT.parent/'final_search_index')),
                        os.getenv('MAPFILES_DIR'))
    app.include_router(create_router(maps.find,service=dres_service))
    app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

    @app.get('/')
    def index():
        return FileResponse(ROOT/'index.html',headers={'Cache-Control':'no-cache'})

    @app.get('/ready')
    def ready():
        return JSONResponse({k:v for k,v in state.items() if k!='instance'},status_code=200 if state['ready'] else 503)

    @app.get('/workers')
    def workers():
        return {'isolated':isolated,
                'text':{**text_snapshot,'pid':process.pid if process and process.returncode is None else None},
                'visual':{**visual_snapshot,'pid':visual_process.pid if visual_process and visual_process.returncode is None else None,
                          'starts':visual_restarts}}

    @app.post('/workers/visual/restart')
    async def restart_visual():
        if not isolated:
            return JSONResponse({'error':'Visual worker isolation is disabled'},status_code=409)
        if visual_process and visual_process.returncode is None:
            return {'status':'already_running','pid':visual_process.pid}
        try:
            child=await launch_visual_worker()
            visual_snapshot.clear()
            visual_snapshot.update(ready=False,status='starting',phase='loading_visual')
            return {'status':'started','pid':child.pid,'starts':visual_restarts}
        except OSError as exc:
            return JSONResponse({'error':f'Cannot restart visual worker: {exc}'},status_code=503)

    @app.api_route('/{path:path}',methods=['GET','POST','PUT','PATCH','DELETE','HEAD','OPTIONS'])
    async def proxy(path: str, request: Request):
        if path=='dres' or path.startswith(('dres/','workers')):
            return JSONResponse({'detail':'Unknown submission endpoint'},status_code=404)
        from startup_access import request_available
        text_capable=request_available({'ready':False,'text_ready':True}, '/'+path,
                                       request.query_params, request.method)
        available=(bool(text_snapshot.get('text_ready')) if isolated and text_capable
                   else bool(visual_snapshot.get('ready')) if isolated
                   else request_available(state, '/'+path, request.query_params, request.method))
        if not available:
            return JSONResponse({'error':'Visual/CLIP đang tải hoặc backend chưa sẵn sàng. Có thể dùng ô OCR và tab Audio khi OCR/ASR sẵn sàng.',
                                 'submission_ready':True,'text_ready':state.get('text_ready',False)},status_code=503,headers={'Retry-After':'2'})
        query=request.url.query
        target=upstream
        if isolated:
            target=upstream if text_capable else visual_upstream
        url=target+'/'+path+('?' + query if query else '')
        headers={k:v for k,v in request.headers.items() if k.lower() not in HOP}
        try:
            req=app.state.client.build_request(request.method,url,headers=headers,content=await request.body())
            result=await app.state.client.send(req,stream=True)
        except httpx.HTTPError:
            state.update(ready=False,text_ready=False,status='starting',phase='reconnecting_search')
            return JSONResponse({'error':'Không kết nối được tìm kiếm; Nộp DRES vẫn sẵn sàng.'},status_code=503)
        return StreamingResponse(result.aiter_raw(),status_code=result.status_code,
                                 headers={k:v for k,v in result.headers.items() if k.lower() not in HOP},
                                 background=BackgroundTask(result.aclose))
    return app

app=create_app()
