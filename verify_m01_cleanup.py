"""Read-only checks for the active M01 cleanup and accented ASR search."""
import json
from pathlib import Path
import sqlite3
import urllib.request

import redis
from shared import normalize_ascii_text
from remove_m01_duplicate_indexes import duplicate


def request(path, body=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request('http://localhost:8080/'+path,data=data,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=90) as response:
        return json.load(response)


def main():
    root=Path('/mnt/c/PERSONAL_PROJECT/AIC_26/final_search_index')
    cache=json.loads((root/'asr_cache.json').read_text())
    assert not any(duplicate(Path(k.split('::')[-1]).stem) for k in cache)
    client=redis.Redis(port=16381,protocol=2)
    prefix='aic:final:ssd:text:v1:'
    assert json.loads(client.get(prefix+'asr_cache'))==cache
    with sqlite3.connect((root/'meta.db').as_uri()+'?mode=ro',uri=True) as db:
        videos=[r[0] for r in db.execute('SELECT DISTINCT video FROM frame_locator')]
        assert not any(duplicate(v) for v in videos)
        assert len([v for v in videos if v.upper().startswith('M01_')])==29
        assert db.execute('SELECT count(*) FROM frames').fetchone()[0]==912383
        assert db.execute('SELECT count(*) FROM asr_segments').fetchone()[0]==414912
    status=request('asr/status')
    assert status['valid_cached_videos']==1188 and status['search_ready']
    assert request('status')['indexed']==912383
    checks={}
    for family in 'LMS':
        key,segments=next((k,v) for k,v in cache.items() if k.split('::')[-1].startswith(family) and v)
        # A long phrase provides a specific regression probe for each family.
        text=max(segments,key=lambda s:len(s['text']))['text']
        accented=request('asr/search',{'phrase':text,'limit':100})
        plain=request('asr/search',{'phrase':normalize_ascii_text(text),'limit':100})
        assert accented['total']>0, (family,'no search matches')
        assert accented['results']==plain['results'], (family,'accent mismatch')
        assert any(r['video_name']==key.split('::')[-1] and r['frames'] for r in accented['results']), (family,'missing frame mapping')
        assert not any(duplicate(r['video_id']) for r in accented['results'])
        checks[family]={'matches':accented['total'],'accent_equivalence':True,'frames':True}
    report={'frames':912383,'asr_videos':1188,'asr_segments':414912,'m01_videos':29,'search_checks':checks}
    print(json.dumps(report))
    work=root.parent/'output'/'m01_cleanup'
    work.mkdir(parents=True,exist_ok=True)
    (work/'verification.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
