"""Competition gateway tests. No live DRES requests or ML model loading."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from competition_server import create_app
from dres_frames import FrameResolver, FrameError, build_answer
from dres_submission import DresService
from submission_maps import SubmissionMaps
from test_dres import FakeDres

class CompetitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        maps=self.root/'map-keyframes';maps.mkdir()
        self.csv=maps/'N001-V001.csv'
        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,25,0\n11,5.521,25,138\n12,9.021,25,200\n')
        self.find=SubmissionMaps(self.root).find
        self.remote=FakeDres()
        self.service=DresService(self.find,self.root/'history.db',self.remote)
        self.app=create_app(start_search=False,data_root=self.root,dres_service=self.service,
                            transport=httpx.MockTransport(lambda r: httpx.Response(503,json={'ready':False})))

    def body(self, **values):
        return dict(task_type='textual_kis',video='N001_V001',frames=[138],live=False,**values)

    def test_ui_and_manual_submission_work_while_search_starting_and_failed(self):
        with TestClient(self.app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertIn('dres-search-query',client.get('/').text)
            self.assertEqual(client.get('/ready').status_code,503)
            self.assertTrue(client.get('/ready').json()['submission_ready'])
            self.assertEqual(client.get('/search?query=test').status_code,503)
            for status in ['starting','failed']:
                self.app.state.readiness.update(status=status,ready=False)
                preview=client.post('/dres/preview',json=self.body())
                self.assertEqual(preview.status_code,200,preview.text)
                submit=client.post('/dres/submit',json={'preview_id':preview.json()['preview_id']})
                self.assertEqual(submit.json()['state'],'simulated')
            self.assertEqual(self.remote.calls,[])

    def test_fake_live_submission_during_search_failure(self):
        with TestClient(self.app) as client:
            self.app.state.readiness.update(status='failed',ready=False)
            token=client.post('/dres/login',json={'username':'team','password':'test-only'}).json()['connection']
            headers={'X-DRES-Connection':token}
            body=self.body();body.update(live=True,evaluation_id='eval-1')
            preview=client.post('/dres/preview',json=body,headers=headers)
            self.assertEqual(preview.status_code,200,preview.text)
            submitted=client.post('/dres/submit',json={'preview_id':preview.json()['preview_id']},headers=headers)
            self.assertEqual(submitted.json()['state'],'received')
            self.assertEqual(sum('/submit/' in p for p,b in self.remote.calls),1)

    def test_ordinal_converts_exactly_to_original_idx(self):
        answer=build_answer(FrameResolver(self.find),'textual_kis','N001_V001',[11],frame_mode='n')
        self.assertEqual(answer['mapping']['frames'][0]['frame_idx'],138)
        self.assertEqual(answer['payload']['answerSets'][0]['answers'][0]['start'],5521)
        with self.assertRaises(FrameError):
            build_answer(FrameResolver(self.find),'textual_kis','N001-V001',[11])
        with self.assertRaises(FrameError):
            build_answer(FrameResolver(self.find),'textual_kis','N001-V001',[13],frame_mode='n')

    def test_ambiguous_ordinal_rejected(self):
        with self.csv.open('a') as file:file.write('11,6,25,150\n')
        with self.assertRaisesRegex(FrameError,'mâu thuẫn'):
            FrameResolver(self.find).resolve('N001-V001',[11],frame_mode='n')

    def test_external_maps_do_not_need_search_database(self):
        external=self.root/'central-maps';external.mkdir()
        p=external/'M01_V001.csv';p.write_text(self.csv.read_text())
        (self.root/'external_storage.json').write_text(json.dumps({'sources':[{'id':'x','data_root':'/unused','map_dir':str(external)}],'video_sources':{'M01-V001':'x'}}))
        self.assertEqual(self.find('m01-v001'),p)

    def test_proxy_preserves_query_headers_and_range_response(self):
        seen=[]
        def transport(request):
            seen.append(request)
            return httpx.Response(206,headers={'content-type':'video/mp4','content-range':'bytes 0-2/10','content-length':'3','X-Search-Id':'receipt'},stream=httpx.ByteStream(b'abc'))
        app=create_app(start_search=False,data_root=self.root,transport=httpx.MockTransport(transport))
        with TestClient(app) as client:
            app.state.readiness.update(status='ready',ready=True)
            response=client.get('/video/test?x=a%20b',headers={'Range':'bytes=0-2','X-Search-Session':'test'})
            self.assertEqual(response.status_code,206)
            self.assertEqual(response.content,b'abc')
            self.assertEqual(response.headers['content-range'],'bytes 0-2/10')
            forwarded=next(r for r in seen if r.url.path=='/video/test')
            self.assertEqual(forwarded.headers['range'],'bytes=0-2')
            self.assertEqual(forwarded.headers['x-search-session'],'test')
            self.assertEqual(forwarded.url.params['x'],'a b')

    def test_wrong_backend_instance_never_becomes_ready(self):
        app=create_app(start_search=False,data_root=self.root,
                       transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'ready':True,'instance':'wrong'})))
        with TestClient(app) as client:
            time.sleep(.05)
            self.assertFalse(client.get('/ready').json()['ready'])
            self.assertEqual(client.get('/search').status_code,503)

    def test_text_ready_gateway_proxies_ocr_asr_but_blocks_visual(self):
        app=None
        def transport(request):
            if request.url.path == '/ready':
                payload={'ready':False,'text_ready':True,'status':'loading_visual',
                         'instance':app.state.readiness['instance']}
                return httpx.Response(503,json=payload)
            return httpx.Response(200,headers={'content-type':'application/json'},
                                  stream=httpx.ByteStream(b'{"total":1,"results":[]}'))
        app=create_app(start_search=False,data_root=self.root,transport=httpx.MockTransport(transport))
        with TestClient(app) as client:
            for _ in range(20):
                if client.get('/ready').json().get('text_ready'): break
                time.sleep(.01)
            self.assertEqual(client.get('/search?ocr_query=test').status_code,200)
            self.assertEqual(client.get('/search?query=test&search_mode=ocr').status_code,200)
            self.assertEqual(client.post('/asr/frames',json={'phrase':'hello'}).status_code,200)
            self.assertEqual(client.get('/search?query=test').status_code,503)
            self.assertEqual(client.get('/video-match/frames?query=test').status_code,503)

    def test_isolated_workers_route_independently_and_visual_failure_keeps_ocr(self):
        app=None
        visual_ready=False
        seen=[]
        def transport(request):
            nonlocal visual_ready
            seen.append((request.url.port,request.url.path))
            if request.url.path == '/ready':
                if request.url.port == 8081:
                    return httpx.Response(503,json={'ready':False,'text_ready':True,
                        'status':'text_ready','instance':app.state.readiness['instance'],'frames':12})
                return httpx.Response(200 if visual_ready else 503,json={'ready':visual_ready,
                    'text_ready':True,'status':'ready' if visual_ready else 'failed',
                    'instance':app.state.readiness['instance'],'frames':34})
            return httpx.Response(200,headers={'content-type':'application/json'},
                                  stream=httpx.ByteStream(b'{"total":0,"results":[]}'))
        with patch.dict('os.environ',{'ISOLATE_VISUAL_WORKER':'1','SEARCH_SERVER_PORT':'8081','VISUAL_SERVER_PORT':'8082'}):
            app=create_app(start_search=False,data_root=self.root,transport=httpx.MockTransport(transport))
        with TestClient(app) as client:
            for _ in range(30):
                if client.get('/ready').json().get('text_ready'): break
                time.sleep(.01)
            self.assertEqual(client.get('/search?ocr_query=hello').status_code,200)
            self.assertIn((8081,'/search'),seen)
            self.assertEqual(client.get('/search?query=street').status_code,503)
            visual_ready=True
            for _ in range(150):
                if client.get('/ready').status_code == 200: break
                time.sleep(.01)
            self.assertEqual(client.get('/search?query=street').status_code,200)
            self.assertIn((8082,'/search'),seen)

    def test_launch_failure_keeps_submission_online(self):
        from unittest.mock import AsyncMock
        app=create_app(start_search=True,data_root=self.root,dres_service=self.service,
                       transport=httpx.MockTransport(lambda r:httpx.Response(503,json={})))
        with patch('competition_server.asyncio.create_subprocess_exec',new=AsyncMock(side_effect=OSError('fixture launch failure'))):
            with TestClient(app) as client:
                self.assertEqual(client.get('/ready').json()['status'],'failed')
                self.assertEqual(client.post('/dres/preview',json=self.body()).status_code,200)

    def test_ready_instance_enables_search_automatically(self):
        app=None
        def transport(request):
            return httpx.Response(200,json={'ready':True,'status':'ready','frames':123,'instance':app.state.readiness['instance']})
        app=create_app(start_search=False,data_root=self.root,transport=httpx.MockTransport(transport))
        with TestClient(app) as client:
            for _ in range(20):
                response=client.get('/ready')
                if response.status_code==200:break
                time.sleep(.01)
            self.assertTrue(response.json()['ready'])
            self.assertEqual(response.json()['frames'],123)
            self.assertIn('search_boot_ms',response.json())

    def test_redis_wait_and_source_validation_after_parallel_loading(self):
        from unittest.mock import MagicMock
        import redis
        import redis_serving
        store=MagicMock()
        store.r.ping.side_effect=[redis.exceptions.BusyLoadingError('loading'),None]
        store.manifest.return_value={'ready':True,'source_index':str(self.root),'frame_count':1}
        store.r.execute_command.return_value={b'num_docs':1,b'indexing':0}
        with patch.dict('os.environ',{'AIC_GATEWAY_CHILD':'1','REQUIRE_REDIS_SOURCE_MATCH':'1','DATA_ROOT':str(self.root)}), patch.object(redis_serving,'store',return_value=store), patch.object(redis_serving.time,'sleep'):
            redis_serving.text_startup()
            self.assertEqual(store.r.ping.call_count,2)
            store.r.ping.side_effect=None
            store.manifest.return_value['source_index']=str(self.root/'wrong')
            with self.assertRaisesRegex(RuntimeError,'source'):
                redis_serving.text_startup()

if __name__=='__main__': unittest.main()
