import unittest
from vision_gui_agent.visual_function_lab import TASKS, TASK_SPLIT, VisualFunctionLabEvaluator

class EvaluationRegression(unittest.TestCase):
    def test_terminal_truth(self):
        for task in TASKS.values():
            with self.subTest(task=task.id):
                evaluator = VisualFunctionLabEvaluator()
                evaluator.reset(task.initial_state)
                self.assertFalse(evaluator.score(task.id))
                for action in task.actions[:-1]:
                    evaluator.act(action)
                self.assertFalse(evaluator.score(task.id))
                evaluator.act(task.actions[-1])
                self.assertTrue(evaluator.score(task.id))
                evaluator.reset(task.initial_state)
                evaluator.act('open_dataset' if task.actions[0] != 'open_dataset' else 'open_document')
                self.assertFalse(evaluator.score(task.id))

    def test_splits_do_not_claim_seen_tasks_are_held_out(self):
        groups = list(TASK_SPLIT.values())
        for index, group in enumerate(groups):
            self.assertTrue(set(group) <= set(TASKS))
            for other in groups[index + 1:]:
                self.assertFalse(set(group) & set(other))

import asyncio
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import AsyncMock, patch
from PIL import Image
from vision_gui_agent.models import ActionDecision, Element, EvidenceRecord, Observation, VerificationCondition, VerificationResult
from vision_gui_agent.logging_store import RunLogger
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.perception import observe

def element(ident=1, text='Open', **kw):
    return Element(ident, '', 'button', text, '', '', 'button', 10, 10, 100, 40, **kw)

def observation(root, name, elements=None):
    path = root / (name + '.png')
    Image.new('RGB', (200, 100), 'white').save(path)
    return Observation(str(path), str(path), elements or [element()], 'https://example.test/' + name, name)

class ProvenanceRegression(unittest.TestCase):
    def test_legacy_post_observations_are_not_training_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); log = RunLogger(root / 'runs.db')
            log.start_run('run', 'goal', 'fake')
            log.log('run', 0, 'a', 'b', ActionDecision('click', 1), True, observation(root, 'b'), {})
            log.finish_run('run', True, 1, 'b', None)
            self.assertEqual(log.training_examples(), [])
            self.assertEqual(log.completed_workflows('goal'), {})
            log.close()

    def test_stateless_run_never_reads_workflows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = Agent(object(), AgentConfig(root, root / 'runs.db', root / 'graph.json', memory_mode='none', max_steps=0))
            with patch.object(RunLogger, 'completed_workflows', return_value={}) as history, patch('vision_gui_agent.agent.observe', AsyncMock(return_value=observation(root, 'a'))):
                asyncio.run(agent.run(object(), 'goal'))
            history.assert_not_called()

    def test_repeated_step_captures_keep_original_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class Page:
                url = ''
                count = 0
                async def screenshot(self, path, **kw):
                    self.count += 1
                    Image.new('RGB', (30, 30), 'red' if self.count == 1 else 'blue').save(path)
            async def captures():
                page = Page(); grounder = type('Grounder', (), {'detect': AsyncMock(return_value=[])})()
                first = await observe(page, root, 1, grounder)
                second = await observe(page, root, 1, grounder)
                return first, second
            first, second = asyncio.run(captures())
            self.assertNotEqual(first.screenshot_path, second.screenshot_path)
            with Image.open(first.screenshot_path) as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_exact_source_target_roundtrip_and_omitted_record(self):
        for omit in (False, True):
            with self.subTest(omit=omit), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); db = root / 'runs.db'; log = RunLogger(db)
                a = observation(root, 'a', [element(7, 'Open')])
                b = observation(root, 'b', [element(2, 'Save')])
                c = observation(root, 'c', [element(9, 'Finished')])
                log.start_run('run', 'goal', 'fake')
                actions = [ActionDecision('click', 7), ActionDecision('click', 2), ActionDecision('done')]
                for step, (before, after, decision) in enumerate(zip((a,b,c), (b,c,c), actions)):
                    if omit and step == 1: continue
                    log.log('run', step, before.title, after.title, decision, True, after, {},
                            verification=VerificationResult('passed', 'fixture'),
                            before_observation=before, after_observation=after, dispatch_status='dispatched')
                log.finish_run('run', True, 2, 'c', None); log.close()
                Image.new('RGB', (200,100), 'blue').save(a.screenshot_path)
                log = RunLogger(db)
                pairs = log.training_examples()
                self.assertEqual(pairs[0]['observation']['title'], 'a')
                self.assertEqual(pairs[0]['action']['element_id'], 7)
                with Image.open(pairs[0]['observation']['screenshot_path']) as image:
                    self.assertEqual(image.getpixel((0,0)), (255,255,255))
                workflows = log.completed_workflows('goal')
                if omit:
                    self.assertEqual(workflows, {})
                else:
                    agent = Agent(object(), AgentConfig(root, db, root / 'missing.json'))
                    agent._hydrate_completed_workflows(workflows, 'goal')
                    edges = [(agent.graph.graph.nodes[s]['label'], agent.graph.graph.nodes[t]['label'], e['action']['element_id'])
                             for s,t,e in agent.graph.graph.edges(data=True) if e['action']['action'] == 'click']
                    self.assertEqual(edges, [('a','b',7), ('b','c',2)])
                log.close()

from vision_gui_agent.state_graph import StateGraph
from vision_gui_agent.verification import verify

class OutcomeRegression(unittest.TestCase):
    def test_unverified_completed_edges_are_not_replayable(self):
        graph = StateGraph(); graph.graph.add_nodes_from(['a','b'])
        graph.add_transition('a','b',ActionDecision('click',1),True,'goal','run')
        graph.add_transition('b','b',ActionDecision('done'),True,'goal','run')
        graph.mark_run_completed('run')
        self.assertIsNone(graph.replay('a','goal'))
        self.assertEqual(graph._reliability('a', ActionDecision('click',1), 'goal'), .5)

    def test_exact_value_not_substring_or_context(self):
        source = Observation('', '', [replace(element(), tag='input', text='Quantity')], '', '')
        for value, context in [('12',''), ('','Quantity 2')]:
            latest = replace(source, elements=[replace(source.elements[0], value=value, context=context)])
            result = asyncio.run(verify(None, source, latest, VerificationCondition('element_value', element_id=1, expected='2'), 6))
            self.assertNotEqual(result.status, 'passed')

    def test_missing_state_and_generic_change_are_not_effect_proof(self):
        source = Observation('', '', [replace(element(), tag='checkbox')], '', '')
        for condition, changed in [(VerificationCondition('element_checked',element_id=1,expected='true'),False),
                                   (VerificationCondition('page_changed'),True)]:
            result = asyncio.run(verify(None, source, source, condition, 6, page_changed=changed))
            self.assertIn(result.status, {'unavailable','ambiguous'})

class ReplayRegression(unittest.TestCase):
    def test_stale_ids_and_duplicate_labels_refuse(self):
        obs = Observation('', '', [element(1, 'Save'), element(2, 'Save')], '', '')
        for decision in (ActionDecision('click',1), ActionDecision('click',1,grounding=(EvidenceRecord('element_text','Save',1),))):
            with self.assertRaises(ValueError):
                Agent._reground(decision, obs)

    def test_context_is_required_alongside_label_after_id_permutation(self):
        obs = Observation('', '', [element(1,'Save',context='Row B'), element(2,'Save',context='Row A')], '', '')
        decision = ActionDecision('click',1,grounding=(EvidenceRecord('element_text','Save',1), EvidenceRecord('context','Row A',1)))
        self.assertEqual(Agent._reground(decision,obs).element_id,2)

    def test_functional_state_pairs_do_not_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original = observation(root,'a',[element(),element(2,'Save')])
            for changed in (replace(original,elements=[replace(original.elements[0],checked=True),original.elements[1]]),
                            replace(original,elements=[replace(original.elements[0],value='12'),original.elements[1]]),
                            replace(original,elements=original.elements+[element(3,'Modal',actionable=False)]),
                            replace(original,elements=[replace(original.elements[0],selected=True),original.elements[1]]),
                            replace(original,elements=original.elements+[element(3,'Save')]),
                            replace(original,url=original.url+'#/other')):
                graph=StateGraph(); left,_=graph.add_observation(original);right,_=graph.add_observation(changed)
                self.assertNotEqual(left,right)

    def test_equivalent_ids_preserve_prototype_and_explicit_query_policy(self):
        from vision_gui_agent.state_graph import normalized_url
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); first=observation(root,'a',[element(1,'Open')]); graph=StateGraph()
            left,_=graph.add_observation(first)
            right,_=graph.add_observation(replace(first,elements=[replace(first.elements[0],id=7,x=35)]))
            self.assertEqual(left,right)
            self.assertEqual(graph.graph.nodes[left]['elements'][0]['id'],1)
            self.assertNotEqual(normalized_url('https://test/#/a'),normalized_url('https://test/#/b'))
            self.assertEqual(normalized_url('https://test/?tracking=1#/a',('tracking',)), 'https://test/#/a')
            self.assertNotEqual(normalized_url('https://test/?record=1'),normalized_url('https://test/?record=2'))

    def test_replay_on_off_resets_verify_the_same_effect_with_fresh_ids(self):
        async def scenario(root, mode, ident):
            before = observation(root, 'start', [element(ident,'Open')])
            after = observation(root, 'end', [element(9,'Finished',actionable=False)])
            click = ActionDecision('click',ident,grounding=(EvidenceRecord('element_text','Open',ident),),
                                   verify=VerificationCondition('element_visible',pattern='Finished'))
            done = ActionDecision('done',grounding=(EvidenceRecord('element_text','Finished',9),))
            async def choose(_goal, obs, *_):
                return done if obs.title == 'end' else click
            policy=type('Policy',(),{'decide':AsyncMock(side_effect=choose)})()
            agent=Agent(policy,AgentConfig(root,root/'runs.db',root/'graph.json',max_steps=2,memory_mode=mode))
            delivered=[]
            async def dispatch(page, obs, decision, *args, **kwargs):
                delivered.append(obs.elements[0].id)
                self.assertEqual(decision.element_id, ident)
            with patch('vision_gui_agent.agent.observe',AsyncMock(side_effect=[before,after])),patch('vision_gui_agent.agent.execute',dispatch):
                result=await agent.run(object(),'open panel')
            self.assertTrue(result.completed,(result.error, result.history))
            self.assertEqual(result.history[0].verification.status,'passed')
            self.assertEqual(delivered,[ident])
            return policy.decide.call_count
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(asyncio.run(scenario(root,'graph',1)),2)
            self.assertEqual(asyncio.run(scenario(root,'graph',7)),1)
            self.assertEqual(asyncio.run(scenario(root,'none',4)),2)

class OutcomeIntegrationRegression(unittest.TestCase):
    def test_noop_then_unrelated_completion_has_no_reusable_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); before=observation(root,'before'); after=observation(root,'after',[element(2,'Finished',actionable=False)])
            policy=type('Policy',(),{'decide':AsyncMock(side_effect=[ActionDecision('click',1),ActionDecision('done',grounding=(EvidenceRecord('element_text','Finished',2),))])})()
            agent=Agent(policy,AgentConfig(root,root/'runs.db',root/'graph.json',max_steps=2))
            with patch('vision_gui_agent.agent.observe',AsyncMock(side_effect=[before,after])),patch('vision_gui_agent.agent.execute',AsyncMock(return_value=None)):
                result=asyncio.run(agent.run(object(),'open panel'))
            self.assertTrue(result.completed,(result.error, result.history))
            click=[(s,e) for s,t,e in agent.graph.graph.edges(data=True) if e['action']['action']=='click'][0]
            self.assertEqual(click[1]['verification_status'],'not_requested')
            self.assertFalse(click[1]['replayable'])
            self.assertEqual(agent.graph._reliability(click[0],ActionDecision('click',1),'open panel'),.5)
            self.assertIsNone(agent.graph.replay(click[0],'open panel'))

    def test_value_matrix_and_delayed_observation(self):
        source=Observation('','',[replace(element(),tag='input',text='Quantity')],'','')
        condition=VerificationCondition('element_value',element_id=1,expected='2')
        statuses=[]
        for value in ('12','','2'):
            latest=replace(source,elements=[replace(source.elements[0],id=7,value=value)])
            statuses.append(asyncio.run(verify(None,source,latest,condition,6)).status)
        self.assertEqual(statuses,['failed','unavailable','passed'])
        duplicate=replace(source,elements=[replace(source.elements[0],id=2,value='2'),replace(source.elements[0],id=3,value='2')])
        self.assertEqual(asyncio.run(verify(None,source,duplicate,condition,6)).status,'unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); before=observation(root,'form',source.elements)
            absent=replace(before,elements=[])
            after=replace(before,elements=[replace(source.elements[0],id=7,value='2')])
            policy=type('Policy',(),{'decide':AsyncMock(return_value=ActionDecision('fill',1,text='2'))})()
            agent=Agent(policy,AgentConfig(root,root/'runs.db',root/'graph.json',memory_mode='passive-action-model',max_steps=1))
            with patch('vision_gui_agent.agent.observe',AsyncMock(side_effect=[before,absent,after])),patch('vision_gui_agent.agent.execute',AsyncMock(return_value=None)):
                result=asyncio.run(agent.run(object(),'set quantity'))
            self.assertEqual(result.history[0].verification.status,'passed')

    def test_hover_change_is_not_verified_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); before=observation(root,'hover-before')
            after=observation(root,'hover-after')
            Image.new('RGB',(200,100),'blue').save(after.screenshot_path)
            result=asyncio.run(verify(None,before,after,VerificationCondition('element_changed',element_id=1),6))
            self.assertEqual(result.status,'ambiguous')

    def test_metadata_is_not_visible_confirmation(self):
        source=Observation('','',[],'','')
        latest=replace(source,elements=[element(1,'Go',context='Saved successfully')])
        for pattern in ('button','Saved successfully'):
            result=asyncio.run(verify(None,source,latest,VerificationCondition('element_visible',pattern=pattern),6))
            self.assertNotEqual(result.status,'passed')

    def test_filename_is_exact_and_missing_detection_cannot_prove_absence(self):
        source=Observation('','',[replace(element(),tag='file',value='other-report.pdf')],'','')
        result=asyncio.run(verify(None,source,source,VerificationCondition('element_filename',element_id=1,expected='report.pdf'),6))
        self.assertNotEqual(result.status,'passed')
        absent=replace(source,elements=[])
        result=asyncio.run(verify(None,source,absent,VerificationCondition('element_absent',pattern='Open'),6))
        self.assertNotEqual(result.status,'passed')

    def test_dispatch_error_without_recapture_has_no_invented_after_state(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); before=observation(root,'error-before')
            policy=type('Policy',(),{'decide':AsyncMock(return_value=ActionDecision('click',1))})()
            config=AgentConfig(root,root/'runs.db',root/'graph.json',max_steps=1,memory_mode='none')
            with patch('vision_gui_agent.agent.observe',AsyncMock(side_effect=[before,RuntimeError('capture unavailable')])),patch('vision_gui_agent.agent.execute',AsyncMock(side_effect=RuntimeError('dispatch error'))):
                asyncio.run(Agent(policy,config).run(object(),'open panel'))
            with sqlite3.connect(config.database_path) as db:
                before_json,after_json,status=db.execute('SELECT before_observation_json,after_observation_json,dispatch_status FROM transitions').fetchone()
            self.assertIsNotNone(before_json)
            self.assertIsNone(after_json)
            self.assertEqual(status,'error')

    def test_replay_abstains_on_incompatible_control_and_bad_manifest(self):
        import json
        from vision_gui_agent.visual_function_lab import load_task_split
        decision=ActionDecision('fill',1,text='Ada',grounding=(EvidenceRecord('element_text','Name',1),))
        with self.assertRaises(ValueError):
            Agent._reground(decision,Observation('','',[element(1,'Name')],'',''))
        with tempfile.TemporaryDirectory() as tmp:
            manifest=json.loads(Path('vision_gui_agent/task_split.json').read_text())
            manifest['held_out']=manifest['exploration'][:1]
            path=Path(tmp)/'split.json';path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_task_split(path)
