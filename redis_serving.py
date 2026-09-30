"""Serving helpers backed by Redis; model weights and media remain filesystem assets."""
import threading
import time
import os
from pathlib import Path

from fastapi.responses import JSONResponse
from redis_store import store, RedisCollection


def text_startup():
    """Validate the local Redis text/summary indexes without touching frames."""
    s = store()
    s.r.ping()
    manifest = s.manifest()
    for kind in ("asr", "summary"):
        expected = int(manifest.get(f"{kind}_count", 0))
        if expected <= 0:
            continue
        raw = s.r.execute_command("FT.INFO", s.prefix + kind + "_idx")
        info = raw if isinstance(raw, dict) else dict(zip(raw[::2], raw[1::2]))
        indexed = int(info.get(b"num_docs", info.get("num_docs", 0)))
        if indexed < expected:
            raise RuntimeError(
                f"Redis {kind} index chưa đủ: {indexed}/{expected}"
            )
    return manifest


def startup(readiness):
    started = time.perf_counter()
    s=store();s.r.ping();manifest=s.manifest()
    col=RedisCollection()
    assert col.count()==manifest['frame_count']
    # By default, complete a real CLIP/search probe before accepting requests.
    fast_startup = os.getenv("REDIS_FAST_STARTUP", "0").lower() in {"1", "true", "yes"}
    if not fast_startup:
        from shared import embed_texts
        vector=embed_texts(['server readiness probe'])
        result=col.query([vector.tolist()],1)
        assert s.rows([result['metadatas'][0][0]['file']])
    readiness.update(ready=True,status='ready',phase='redis',frames=col.count(),error=None,
                     backend='redis',timings_ms={},total_ms=round((time.perf_counter()-started)*1000,2),
                     clip_probe=not fast_startup)


def status():
    from shared import ALL_CATEGORIES, CATEGORY_MAP
    s=store();stats=s.json('stats',{}) or {}
    stats.setdefault('objects', {})
    stats.setdefault('categories', {})
    stats.setdefault('colors', {})
    return {**stats,'indexed':s.manifest()['frame_count'],'all_categories':ALL_CATEGORIES,
            'obj_to_cat':{obj: cat for cat,objects in CATEGORY_MAP.items() for obj in objects},'backend':'redis'}


def neighbors(file,radius):
    import main
    timeline=store().timeline(Path(file).parent.name)
    center=next((i for i,x in enumerate(timeline) if x['file']==file),None)
    if center is None:return JSONResponse({'error':'Frame not found'},status_code=404)
    rows=store().rows([x['file'] for x in timeline[max(0,center-radius):center+radius+1]])
    results=main._public_frame_results(rows)
    for item in results:item['is_center']=item['file']==file
    return dict(video=Path(file).parent.name,center=file,results=results)


class RedisSummarySearch:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(os.getenv('SUMMARY_MODEL_PATH') or 'intfloat/multilingual-e5-small',device='cpu',local_files_only=True)
        self.lock=threading.Lock()
        self.collection=RedisCollection('summary',384)

    def search(self,query,top_k=40,category='',video=''):
        s=store()
        docs={str(d['id']):d for d in s.json('summary_docs',[])
              if (not category or d['category']==category) and (not video or d['video']==video)}
        with self.lock:vector=self.model.encode(['query: '+query],normalize_embeddings=True)[0]
        ranking=self.collection.query([vector.tolist()],self.collection.count())
        dense=[x['file'] for x in ranking['metadatas'][0] if x['file'] in docs]
        sparse_scores={}
        for term in query.split():
            for d,score in s.text_search('summary',term):
                key=str(d['id'])
                if key in docs:sparse_scores[key]=sparse_scores.get(key,0)+score
        sparse=sorted(sparse_scores,key=lambda key:(-sparse_scores[key],key))
        scores={}
        for rank_list in (dense,sparse):
            for rank,key in enumerate(rank_list,1):scores[key]=scores.get(key,0)+1/(60+rank)
        results,seen,segments=[],set(),[]
        timelines={}
        for key in sorted(scores,key=lambda key:(-scores[key],key)):
            d=docs[key];segments.append({**d,'rrf_score':scores[key]})
            if d['kind']!='segment':continue
            if d['video'] not in timelines:timelines[d['video']]=s.timeline(d['video'])
            frames=[f for f in timelines[d['video']] if d['start']<=f['timestamp']<d['end']]
            frames.sort(key=lambda f:(abs(f['timestamp']-(d['start']+d['end'])/2),f['file']))
            for f in frames:
                if f['file'] in seen:continue
                seen.add(f['file'])
                results.append({**f,'video':d['video'],'score':scores[key],'summary':d['text'],
                    'category_code':d['category'],'segment_start':d['start'],'segment_end':d['end'],
                    'source':'summary','objects':[],'categories':[],'colors':[]})
        total=len(results)
        return dict(results=results[:top_k],total=total,segments=segments[:20],score_type='rrf')
