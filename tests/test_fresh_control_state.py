import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from PIL import Image
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.models import ActionDecision, Element, Observation, VerificationCondition
from vision_gui_agent.verification import verify


class FreshControlStateTests(unittest.TestCase):
    def test_verification_reads_new_value_without_sending_expected_value(self):
        async def scenario(root):
            path=root/'screen.png';Image.new('RGB',(300,150),'white').save(path)
            source=Element(1,'','input','Name','','','input',20,20,150,30,
                proposal_sources=('gemini',),semantic_confirmed=True,state_observed=('enabled','readonly','value'))
            detected=replace(source,id=4,tag='other',role='other',semantic_confirmed=False,proposal_sources=('detector',),value='',state_observed=())
            before=Observation(str(path),str(path),[source],'https://fixture','Form')
            after=replace(before,elements=[detected])
            grounder=SimpleNamespace(refine=AsyncMock(return_value=replace(source,value='Ada',state_observed=('value',))))
            agent=Agent(SimpleNamespace(),AgentConfig(root,root/'runs.db',root/'graph.json',memory_mode='none'),grounder)
            condition=VerificationCondition('element_value',element_id=1,expected='Ada')
            refreshed=await agent._refresh_control_state(before,after,condition)
            result=await verify(None,before,refreshed,condition,8)
            self.assertEqual(result.status,'passed')
            self.assertNotIn('Ada',str(grounder.refine.call_args))
            self.assertEqual(after.elements,[detected])
            self.assertEqual(refreshed.capture,after.capture)
            grounder.refine.return_value=replace(source,value='Grace',state_observed=('value',))
            wrong=await agent._refresh_control_state(before,after,condition)
            self.assertEqual((await verify(None,before,wrong,condition,8)).status,'failed')
        with tempfile.TemporaryDirectory() as tmp:asyncio.run(scenario(Path(tmp)))

    def test_no_refinement_across_navigation_or_when_value_already_observed(self):
        async def scenario(root):
            source=Element(1,'','input','Name','','','input',20,20,150,30,
                proposal_sources=('gemini',),semantic_confirmed=True)
            before=Observation('','',[source],'https://before','Form')
            grounder=SimpleNamespace(refine=AsyncMock())
            agent=Agent(SimpleNamespace(),AgentConfig(root,root/'runs.db',root/'graph.json'),grounder)
            condition=VerificationCondition('element_value',element_id=1,expected='Ada')
            navigated=replace(before,url='https://after')
            self.assertIs(await agent._refresh_control_state(before,navigated,condition),navigated)
            observed=replace(before,elements=[replace(source,value='Ada')])
            self.assertIs(await agent._refresh_control_state(before,observed,condition),observed)
            grounder.refine.assert_not_awaited()
        with tempfile.TemporaryDirectory() as tmp:asyncio.run(scenario(Path(tmp)))

    def test_evidence_text_cannot_be_promoted_into_an_action_by_refinement(self):
        async def scenario(root):
            path=root/'screen.png';Image.new('RGB',(300,150),'white').save(path)
            text=Element(1,'','text','Article heading','','','text',20,20,150,30,
                         actionable=False,proposal_sources=('ocr',))
            observation=Observation(str(path),str(path),[text],'https://fixture','Article')
            grounder=SimpleNamespace(refine=AsyncMock(return_value=replace(text,tag='button',role='button',
                         actionable=True,semantic_confirmed=True)))
            policy=SimpleNamespace(decide=AsyncMock(return_value=ActionDecision('click',1)))
            with patch('vision_gui_agent.agent.observe',AsyncMock(return_value=observation)),patch('vision_gui_agent.agent.execute',AsyncMock()) as execute:
                await Agent(policy,AgentConfig(root,root/'runs.db',root/'graph.json',memory_mode='none',max_steps=1),grounder).run(object(),'Open article')
            execute.assert_not_awaited()
            grounder.refine.assert_not_awaited()
        with tempfile.TemporaryDirectory() as tmp:asyncio.run(scenario(Path(tmp)))
