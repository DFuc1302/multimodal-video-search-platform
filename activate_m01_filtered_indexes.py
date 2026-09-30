"""Activate the verified M01 cleanup with backups and a controlled UI restart."""
import argparse
import base64
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import time

import redis
from migrate_to_redis import key, wait_index, wait_for_redis
from remove_m01_duplicate_indexes import read, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--gateway-pid',type=int,required=True)
    args=parser.parse_args()
    work=args.work.resolve()
    ready=read(work/'ready.json')
    source=Path(ready['source']).resolve()
    if source != Path('/mnt/c/PERSONAL_PROJECT/AIC_26/final_search_index'):
        raise RuntimeError('Unexpected activation target')
    stage=work/'staged'
    backup=work/'backup'
    prebacked=(work/'backup_ready.json').is_file()
    if backup.exists() and not prebacked: raise RuntimeError('Backup already exists; inspect state before resuming')
    removed=read(work/'removed.json')
    proc=Path('/proc')/str(args.gateway_pid)
    if proc.exists():
        command=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
        if 'competition_server:app' not in command: raise RuntimeError('Not the expected UI gateway')
        environment=dict(v.decode().split('=',1) for v in (proc/'environ').read_bytes().split(b'\0') if v)
        executable=str((proc/'exe').resolve())
        cwd=str((proc/'cwd').resolve())
    else:
        cwd=str(Path(__file__).resolve().parent)
        executable='/bin/bash'
        command=['bash',str(Path(cwd)/'run_final_hybrid_wsl.sh'),'run']
        environment={**os.environ,'REDIS_URL':'redis://127.0.0.1:16381/0',
                     'REDIS_PREFIX':'aic:final:ssd:text:v1:','OPEN_BROWSER':'0'}
    client=redis.Redis.from_url(environment['REDIS_URL'],socket_timeout=120,protocol=2)
    print('Waiting for Redis data to finish loading',flush=True)
    os.environ.setdefault('REDIS_READY_TIMEOUT','3600')
    wait_for_redis(client)
    prefix=environment['REDIS_PREFIX']
    manifest=json.loads(client.get(prefix+'manifest'))
    if Path(manifest['source_index']).resolve()!=source: raise RuntimeError('Wrong Redis source')
    files=['meta.db','asr_cache.json','serve_stats.json','serve_manifest.json',
           'faiss_index/frames.faiss','faiss_index/frames_meta.json']
    print('Backing up current indexes on D',flush=True)
    signatures=read(work/'backup_ready.json') if prebacked else {}
    for name in files:
        dest=backup/name
        if prebacked:
            stat=(source/name).stat()
            if stat.st_size != signatures[name]['size'] or dest.stat().st_size != stat.st_size:
                raise RuntimeError(f'Backup size mismatch: {name}')
            # Windows and Linux differ slightly in timestamp precision.
            if abs(stat.st_mtime-signatures[name]['mtime'])>0.001:
                raise RuntimeError(f'Source changed since backup: {name}')
            continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        if name=='meta.db':
            with sqlite3.connect((source/name).as_uri()+'?mode=ro',uri=True) as original, sqlite3.connect(dest) as saved:
                saved.execute('PRAGMA synchronous=OFF')
                original.backup(saved,pages=16384)
        else: shutil.copy2(source/name,dest)
    cache=json.loads(client.get(prefix+'asr_cache'))
    delete_keys=[key(prefix,'frame',r['file']) for r in removed['frames']]
    for name in removed['asr_keys']:
        video=Path(name.split('::')[-1]).stem
        delete_keys.extend(key(prefix,'asr',f'{video}:{n}') for n in range(len(cache[name])))
    delete_keys.extend(prefix+'timeline:'+v.upper() for v in removed['videos'])
    backup_keys=delete_keys+[prefix+v for v in ('manifest','stats','files','frame_count','asr_count','asr_cache')]
    with gzip.open(backup/'redis_restore.jsonl.gz','wt',encoding='utf-8') as out:
        for start in range(0,len(backup_keys),500):
            batch=backup_keys[start:start+500]
            pipe=client.pipeline(transaction=False)
            for name in batch: pipe.dump(name)
            for name,value in zip(batch,pipe.execute()):
                out.write(json.dumps([name,base64.b64encode(value).decode() if value else None])+'\n')
    print('Stopping UI to activate consistent indexes',flush=True)
    if proc.exists(): os.kill(args.gateway_pid,signal.SIGTERM)
    for _ in range(60):
        if not proc.exists(): break
        time.sleep(1)
    else: raise RuntimeError('UI did not stop; no indexes replaced')
    for port in (8080,8081,8082):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',port))==0: raise RuntimeError(f'Port {port} still busy')
    write(work/'activation.json',{'phase':'installing'})
    for name in files:
        target=source/name
        temporary=target.with_name(target.name+'.m01-new')
        shutil.copy2(stage/name,temporary)
        temporary.replace(target)
    # No source media is changed. Remove only the explicitly listed search keys.
    for start in range(0,len(delete_keys),500): client.delete(*delete_keys[start:start+500])
    removed_files={r['file'] for r in removed['frames']}
    kept_files=[x for x in client.lrange(prefix+'files',0,-1) if x.decode() not in removed_files]
    if len(kept_files)!=ready['frames']: raise RuntimeError('Redis file count mismatch')
    temp_list=prefix+'files:m01-new'
    client.delete(temp_list)
    for start in range(0,len(kept_files),1000): client.rpush(temp_list,*kept_files[start:start+1000])
    client.rename(temp_list,prefix+'files')
    for name in removed['asr_keys']: del cache[name]
    client.set(prefix+'asr_cache',json.dumps(cache,ensure_ascii=False))
    client.set(prefix+'frame_count',ready['frames'])
    client.set(prefix+'asr_count',ready['asr_segments'])
    client.set(prefix+'stats',json.dumps(read(source/'serve_stats.json'),ensure_ascii=False))
    with sqlite3.connect(source/'meta.db') as db:
        statuses=dict(db.execute('SELECT ocr_status,count(*) FROM frames GROUP BY ocr_status'))
    manifest.update(frame_count=ready['frames'],asr_count=ready['asr_segments'],
                    asr_videos=ready['asr_videos'],video_count=manifest['video_count']-len(removed['videos']),
                    ocr_statuses=statuses,migrated_at=time.time())
    wait_index(client,prefix,'frame',ready['frames'])
    wait_index(client,prefix,'asr',ready['asr_segments'])
    client.set(prefix+'manifest',json.dumps(manifest))
    write(work/'activation.json',{'phase':'installed'})
    log_path=source.parent/'runtime_logs'/'server_m01_cleanup.log'
    # Relaunch the same gateway with its original runtime and environment.
    with log_path.open('ab') as log:
        child=subprocess.Popen([executable,*command[1:]],cwd=cwd,env=environment,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(work/'activation.json',{'phase':'restarted','pid':child.pid,'log':str(log_path)})
    print(json.dumps({'restarted_pid':child.pid,**ready}),flush=True)


if __name__=='__main__': main()
