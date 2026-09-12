import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from PIL import Image, ImageDraw

from vision_gui_agent.models import Element, Observation, ActionDecision
from vision_gui_agent.perception import OmniParserVisualGrounder, GeminiVisualGrounder


def proposal(ident=1, text='Save', **kw):
    return Element(ident,'','button',text,'','','button',20,20,180,40,**kw)


def detector(boxes):
    return SimpleNamespace(predict=lambda **_: [SimpleNamespace(boxes=SimpleNamespace(
        xyxy=SimpleNamespace(tolist=lambda:boxes), conf=SimpleNamespace(tolist=lambda:[.98]*len(boxes))))])


def ocr_record(text, box):
    x,y,w,h=box
    return ([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],(text,.99))


class DiscoveryRegression(unittest.TestCase):
    def test_wide_colored_button_is_not_an_input(self):
        self.assertNotIn(OmniParserVisualGrounder._kind(Image.new('RGB',(300,100),'blue'),(20,20,180,40)), {'input','textarea'})

    def test_price_does_not_make_unrelated_heading_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(500,150),'white').save(path)
            ocr=lambda *a,**k:[ocr_record('Account overview',(10,10,110,20)),ocr_record('$42',(330,60,40,20))]
            elements=asyncio.run(OmniParserVisualGrounder(detector([]),ocr).detect(path))
            self.assertFalse(next(e for e in elements if e.text=='Account overview').actionable)

    def test_detected_row_context_distinguishes_repeated_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(500,200),'white').save(path)
            boxes=[[5,5,400,80],[300,20,380,60],[5,95,400,170],[300,110,380,150]]
            records=[ocr_record('Row A',(20,25,60,20)),ocr_record('Save',(310,25,50,20)),
                     ocr_record('Row B',(20,115,60,20)),ocr_record('Save',(310,115,50,20))]
            elements=asyncio.run(OmniParserVisualGrounder(detector(boxes),lambda *a,**k:records).detect(path))
            saves=[e for e in elements if e.text=='Save']
            self.assertEqual([e.context for e in saves],['Row A Save','Row B Save'])
            self.assertFalse(elements[0].actionable)

    def test_gemini_preserves_observed_checkbox_state_and_disabled_state(self):
        grounder=object.__new__(GeminiVisualGrounder)
        response={'elements':[{'kind':'checkbox','label':'Review','x':10,'y':10,'width':20,'height':20,
                              'checked':True,'enabled':False,'readonly':False,'actionable':True}]}
        grounder._generate=lambda request:SimpleNamespace(text=json.dumps(response))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(100,100),'white').save(path)
            element=asyncio.run(grounder.detect(path))[0]
        self.assertTrue(element.checked)
        self.assertFalse(element.enabled)
        self.assertFalse(element.actionable)

from vision_gui_agent.agent import Agent, AgentConfig

class RefinementRegression(unittest.TestCase):
    def test_high_confidence_ambiguous_target_cannot_bypass_negative_refinement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'screen.png';Image.new('RGB',(300,120),'white').save(path)
            target=replace(proposal(),tag='other',proposal_sources=('detector',),uncertainties=('control_type',),confidence=1)
            obs=Observation(str(path),str(path),[target],'','Screen')
            policy=SimpleNamespace(decide=AsyncMock(return_value=ActionDecision('click',1)))
            grounder=SimpleNamespace(refine=AsyncMock(return_value=None))
            with patch('vision_gui_agent.agent.observe',AsyncMock(return_value=obs)),patch('vision_gui_agent.agent.execute',AsyncMock(return_value=None)) as dispatch:
                asyncio.run(Agent(policy,AgentConfig(root,root/'runs.db',root/'graph.json',max_steps=1,memory_mode='none'),grounder).run(object(),'save record'))
            grounder.refine.assert_awaited_once()
            dispatch.assert_not_awaited()

    def test_negative_crop_coordinates_are_rejected(self):
        grounder=object.__new__(GeminiVisualGrounder)
        grounder._generate=lambda request:SimpleNamespace(text=json.dumps({'found':True,'x':-4,'y':0,'width':30,'height':30,'kind':'button','actionable':True}))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(1000,800),'white').save(path)
            target=replace(proposal(),x=400,y=300,width=40,height=40)
            result=asyncio.run(grounder.refine(path,target))
        self.assertIsNone(result)

    def test_recovery_dispatches_nothing_and_enriches_the_next_policy_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'screen.png';Image.new('RGB',(300,120),'white').save(path)
            initial=Observation(str(path),str(path),[],'','Screen')
            recovered=replace(proposal(),semantic_confirmed=True,proposal_sources=('gemini',))
            seen=[]
            async def choose(goal,obs,*_):
                seen.append(obs)
                return ActionDecision('inspect',text='Save control') if len(seen)==1 else ActionDecision('click',obs.elements[0].id)
            grounder=SimpleNamespace(discover=AsyncMock(return_value=[recovered]))
            with patch('vision_gui_agent.agent.observe',AsyncMock(return_value=initial)),patch('vision_gui_agent.agent.execute',AsyncMock(return_value=None)) as dispatch:
                asyncio.run(Agent(SimpleNamespace(decide=choose),AgentConfig(root,root/'runs.db',root/'graph.json',max_steps=2,memory_mode='none'),grounder).run(object(),'save record'))
            self.assertEqual(len(seen[1].elements),1)
            self.assertEqual(dispatch.await_count,1)

    def test_readable_badges_do_not_overlap_controls_or_each_other(self):
        from vision_gui_agent.perception import badge_layout, selection_tiles
        elements=[replace(proposal(i+1,'Save'),x=(i%6)*25,y=(i//6)*25,width=20,height=20) for i in range(24)]
        badges,size=badge_layout((150,100),elements)
        overlap=lambda a,b:a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
        for i,box in enumerate(badges):
            self.assertGreaterEqual(box[3]-box[1],24)
            self.assertFalse(any(overlap(box,other) for other in badges[:i]))
            self.assertFalse(any(overlap(box,(e.x,e.y,e.x+e.width,e.y+e.height)) for e in elements))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(150,100),'white').save(path)
            tiles=selection_tiles(Observation(str(path),str(path),elements,'',''))
            self.assertEqual(len(tiles),12)
            self.assertTrue(all(metadata['header_height']==28 for metadata,_ in tiles))

from vision_gui_agent.executor import execute

class CapabilityRegression(unittest.TestCase):
    def test_native_bridge_requires_explicit_hybrid_mode_before_input(self):
        page=SimpleNamespace(mouse=SimpleNamespace(click=AsyncMock()),evaluate=AsyncMock(),wait_for_timeout=AsyncMock())
        obs=Observation('','',[replace(proposal(),tag='color',input_type='color')],'','')
        with self.assertRaisesRegex(ValueError,'hybrid'):
            asyncio.run(execute(page,obs,ActionDecision('set_color',1,text='#123456')))
        page.mouse.click.assert_not_awaited()

from vision_gui_agent.perception import observe
from playwright.async_api import async_playwright

class CoordinateRegression(unittest.TestCase):
    def test_url_belongs_to_capture_not_end_of_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Page:
                url='https://test/before'
                async def screenshot(self,path,**kw): Image.new('RGB',(100,100),'white').save(path)
            page=Page()
            async def detect(path):
                page.url='https://test/after'
                return []
            obs=asyncio.run(observe(page,Path(tmp),0,SimpleNamespace(detect=detect)))
            self.assertEqual(obs.url,'https://test/before')

    def test_moving_target_is_refused_before_dispatch(self):
        async def scenario(root):
            async with async_playwright() as p:
                browser=await p.chromium.launch()
                page=await browser.new_page(viewport={'width':500,'height':300})
                await page.set_content('<button style="position:absolute;left:100px;top:100px;width:100px;height:40px;background:red" onclick="window.hit=true">Move</button>')
                grounder=SimpleNamespace(detect=AsyncMock(return_value=[replace(proposal(),x=100,y=100,width=100,height=40)]))
                obs=await observe(page,root,0,grounder)
                await page.evaluate("document.querySelector('button').style.left='300px'")
                with self.assertRaisesRegex(ValueError,'changed|stale'):
                    await execute(page,obs,ActionDecision('click',1))
                self.assertIsNone(await page.evaluate('window.hit'))
                await browser.close()
        with tempfile.TemporaryDirectory() as tmp: asyncio.run(scenario(Path(tmp)))

    def test_dpr_mapping_and_viewport_resize(self):
        import numpy as np
        async def scenario(root):
            async with async_playwright() as p:
                browser=await p.chromium.launch()
                for dpr in (1,2):
                    page=await browser.new_page(viewport={'width':500,'height':300},device_scale_factor=dpr)
                    await page.set_content('<button style="position:absolute;left:100px;top:100px;width:100px;height:40px;background:red;border:0" onclick="window.point=[event.clientX,event.clientY]"></button>')
                    async def detect(path):
                        with Image.open(path).convert('RGB') as image: pixels=np.asarray(image)
                        y,x=np.where((pixels==[255,0,0]).all(axis=2))
                        return [replace(proposal(),x=int(x.min()),y=int(y.min()),width=int(x.max()-x.min()+1),height=int(y.max()-y.min()+1))]
                    obs=await observe(page,root,dpr,SimpleNamespace(detect=detect))
                    info={}
                    await execute(page,obs,ActionDecision('click',1),dispatch_info=info)
                    self.assertEqual(await page.evaluate('window.point'),[150,120])
                    self.assertEqual(tuple(info['delivered_point']),(150,120))
                    self.assertEqual(tuple(info['requested_point']),(150*dpr,120*dpr))
                    self.assertEqual(tuple(obs.capture.image_size),(500*dpr,300*dpr))
                    newer=await observe(page,root,dpr+10,SimpleNamespace(detect=detect))
                    await page.set_viewport_size({'width':600,'height':300})
                    with self.assertRaisesRegex(ValueError,'geometry changed'):
                        await execute(page,newer,ActionDecision('click',1))
                    await page.close()
                await browser.close()
        with tempfile.TemporaryDirectory() as tmp:asyncio.run(scenario(Path(tmp)))

    def test_desktop_two_times_pixels_transform_or_refuse(self):
        from vision_gui_agent.desktop import DesktopPage
        class Backend:
            calls=[]
            def screenshot(self):return Image.new('RGB',(400,200),'white')
            def click(self,x,y):self.calls.append((x,y))
        async def scenario(root):
            target=replace(proposal(),x=180,y=80,width=40,height=40)
            for size in ((200,100),None):
                backend=Backend();backend.calls=[]
                page=DesktopPage(backend,input_size=size,origin=(10,20))
                obs=await observe(page,root,0,SimpleNamespace(detect=AsyncMock(return_value=[target])))
                if size:
                    await execute(page,obs,ActionDecision('click',1))
                    self.assertEqual(backend.calls,[(110,70)])
                else:
                    with self.assertRaisesRegex(ValueError,'mapping'):
                        await execute(page,obs,ActionDecision('click',1))
                    self.assertEqual(backend.calls,[])
        with tempfile.TemporaryDirectory() as tmp:asyncio.run(scenario(Path(tmp)))

class ContextAndRecoveryRegression(unittest.TestCase):
    def test_refinement_request_contains_goal_overview_and_competing_rows(self):
        from io import BytesIO
        requests=[]
        def generate(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(text=json.dumps({'found':True,'instance_confirmed':True,'kind':'button','actionable':True,'x':48,'y':36,'width':30,'height':30,'context':'Row A'}))
        grounder=object.__new__(GeminiVisualGrounder);grounder.model='fixture'
        grounder.client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        grounder.types=SimpleNamespace(Part=SimpleNamespace(from_bytes=lambda **kw:kw),GenerateContentConfig=lambda **kw:kw,AutomaticFunctionCallingConfig=lambda **kw:kw)
        target=replace(proposal(),x=100,y=100,width=30,height=30,context='Row A')
        other=replace(target,id=2,y=200,context='Row B')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(1440,1000),'white').save(path)
            result=asyncio.run(grounder.refine(path,target,goal='Save Row A',decision=ActionDecision('click',1),elements=[target,other]))
        self.assertEqual(result.context,'Row A')
        contents=requests[0]['contents']
        self.assertEqual(Image.open(BytesIO(contents[0]['data'])).size,(1440,1000))
        self.assertIn('Save Row A',contents[2]);self.assertIn('Row B',contents[2])
        self.assertEqual((result.x,result.y),(100,100))

    def test_agent_recaptures_and_reselects_after_a_policy_delay(self):
        import numpy as np
        import sqlite3
        async def scenario(root):
            async with async_playwright() as p:
                browser=await p.chromium.launch();page=await browser.new_page(viewport={'width':500,'height':300})
                await page.set_content('<button style="position:absolute;left:100px;top:100px;width:100px;height:40px;background:red;border:0" onclick="window.points.push([event.clientX,event.clientY]);this.style.background=\'lime\'"></button><script>window.points=[]</script>')
                async def detect(path):
                    with Image.open(path).convert('RGB') as im: pixels=np.asarray(im)
                    if (pixels==[0,255,0]).all(axis=2).any():return [replace(proposal(2,'Finished'),actionable=False)]
                    y,x=np.where((pixels==[255,0,0]).all(axis=2))
                    return [replace(proposal(),x=int(x.min()),y=int(y.min()),width=int(x.max()-x.min()+1),height=int(y.max()-y.min()+1))]
                calls=0
                async def choose(goal,obs,*_):
                    nonlocal calls
                    calls+=1
                    if obs.elements[0].text=='Finished':
                        from vision_gui_agent.models import EvidenceRecord
                        return ActionDecision('done',grounding=(EvidenceRecord('element_text','Finished',2),))
                    if calls==1:await page.evaluate("document.querySelector('button').style.left='300px'")
                    from vision_gui_agent.models import VerificationCondition
                    return ActionDecision('click',1,verify=VerificationCondition('element_visible',pattern='Finished'))
                config=AgentConfig(root,root/'runs.db',root/'graph.json',max_steps=3,max_action_attempts=1,memory_mode='none')
                result=await Agent(SimpleNamespace(decide=choose),config,SimpleNamespace(detect=detect)).run(page,'open panel')
                self.assertTrue(result.completed,result.error)
                self.assertEqual(await page.evaluate('window.points'),[[350,120]])
                with sqlite3.connect(config.database_path) as db:
                    rows=db.execute('SELECT dispatch_status,verification_status,dispatch_info_json FROM transitions ORDER BY step').fetchall()
                self.assertEqual(rows[0][:2],('not_dispatched','unavailable'))
                self.assertEqual(json.loads(rows[1][2])['delivered_point'],[350,120])
                await browser.close()
        with tempfile.TemporaryDirectory() as tmp:asyncio.run(scenario(Path(tmp)))

class SemanticBoundaryRegression(unittest.TestCase):
    def test_invalid_model_state_is_not_observed_editability(self):
        grounder=object.__new__(GeminiVisualGrounder)
        grounder._generate=lambda request:SimpleNamespace(text=json.dumps({'elements':[{
            'kind':'input','label':'Quantity','x':10,'y':10,'width':80,'height':30,
            'enabled':'false','readonly':'false','context_bounds':[0,0,float('nan'),40]}]}))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(100,100),'white').save(path)
            element=asyncio.run(grounder.detect(path))[0]
        self.assertNotIn('enabled',element.state_observed)
        self.assertNotIn('readonly',element.state_observed)
        self.assertIsNone(element.context_bounds)
        with self.assertRaisesRegex(ValueError,'editability'):
            ActionDecision('fill',1,text='3').validate_for(Observation('','',[element],'',''))

    def test_duplicate_instance_and_unavailable_refinement_abstain(self):
        from vision_gui_agent.perception import GroundingAbstention
        grounder=object.__new__(GeminiVisualGrounder)
        target=replace(proposal(),width=40)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(300,120),'white').save(path)
            grounder._generate=lambda request:SimpleNamespace(text=json.dumps({'found':True,'kind':'button','x':20,'y':20,'width':40,'height':40,'actionable':True}))
            self.assertIsNone(asyncio.run(grounder.refine(path,target,elements=[target,replace(target,id=2,x=150)])))
            def unavailable(request):raise RuntimeError('offline')
            grounder._generate=unavailable
            with self.assertRaises(GroundingAbstention):asyncio.run(grounder.refine(path,target))

    def test_corpus_keeps_oracle_safe_points_without_price_false_positive(self):
        from vision_gui_agent.perception_validation import validate
        result=asyncio.run(validate(Path(__file__).parent/'fixtures/perception'))
        gates=result['oracle_fixed_candidate_gates']
        self.assertEqual(sum(r['oracle_needs_refinement'] for r in gates),4)
        self.assertTrue(all(r['new_gate']==r['oracle_needs_refinement'] for r in gates))
        self.assertFalse(any(r['old_gate'] for r in gates))
        fusion=result['totals']['fusion']
        self.assertEqual((fusion['visible_targets'],fusion['actionable_safe_point_hits']),(7,6))
        self.assertEqual(fusion['wrong_typed_proposals'],0)
        # The intentionally disabled proposal still requires semantic rejection.
        self.assertEqual(fusion['false_actionable_proposals'],1)
        heading=next(r for r in result['results'] if r['case']=='unrelated_price' and r['mode']=='fusion')
        self.assertEqual(heading['false_actionable_proposals'],0)

    def test_uncertain_proposal_requires_explicit_actionability_confirmation(self):
        grounder=object.__new__(GeminiVisualGrounder)
        target=replace(proposal(),tag='other',proposal_sources=('detector',),uncertainties=('actionability',))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(300,120),'white').save(path)
            for response in ({'found':True}, {'found':'false','actionable':True}):
                grounder._generate=lambda request:SimpleNamespace(text=json.dumps(dict(x=20,y=20,width=40,height=40,kind='button',**response)))
                with self.subTest(response=response):
                    self.assertIsNone(asyncio.run(grounder.refine(path,target)))
