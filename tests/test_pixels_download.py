from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from playwright.async_api import async_playwright

from vision_gui_agent.action_model import ActionModel
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.benchmark_agent import CalibrationGrounder
from vision_gui_agent.executor import execute, requires_hybrid
from vision_gui_agent.models import ActionDecision, Element, EvidenceRecord, Observation, VerificationCondition
from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab


class PixelsDownloadTests(unittest.TestCase):
    def test_requires_hybrid_contracts(self):
        download_click = ActionDecision("click", 1, verify=VerificationCondition("download_created"))
        self.assertFalse(requires_hybrid(download_click))
        self.assertTrue(requires_hybrid(ActionDecision("upload", 1, text="file.txt")))
        self.assertTrue(requires_hybrid(ActionDecision("set_color", 1, text="#123456")))

    def test_execute_click_with_download_created_in_pixels_mode(self):
        page = SimpleNamespace(
            mouse=SimpleNamespace(click=AsyncMock()),
            evaluate=AsyncMock(return_value={"size": [1440, 1000], "origin": [0, 0], "space": "viewport"}),
            wait_for_timeout=AsyncMock(),
        )
        element = Element(1, "", "button", "Confirm export", "", "", "button", 10, 10, 80, 30)
        obs = Observation("", "", [element], "", "Page")
        decision = ActionDecision("click", 1, verify=VerificationCondition("download_created"))

        result = asyncio.run(execute(page, obs, decision, execution_mode="pixels"))
        self.assertIsNone(result)
        page.mouse.click.assert_awaited_once()

    def test_fullsuite_export_pdf_in_pixels_mode(self):
        class ExportPolicy:
            model = "test-export-policy"
            async def decide(self, _goal, observation, _context, _history):
                visible = {item.text: item for item in observation.elements}
                if "Launch brief.pdf is ready" in visible:
                    target = visible["Launch brief.pdf is ready"]
                    return [ActionDecision("done", grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "Confirm export" in visible and "PDF document selected" in visible:
                    target = visible["Confirm export"]
                    return [ActionDecision("click", target.id, grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "PDF document" in visible:
                    target = visible["PDF document"]
                    return [ActionDecision("click", target.id,
                        verify=VerificationCondition("element_checked", element_id=target.id, expected="true"),
                        grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                label = "Export options" if "Export options" in visible else "Launch Brief"
                target = visible[label]
                return [ActionDecision("click", target.id,
                    verify=VerificationCondition("download_created") if label == "Export options" else None,
                    grounding=(EvidenceRecord("element_text", target.text, target.id),))]

        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                server = serve_visual_function_lab(0)
                try:
                    server.RequestHandlerClass.evaluator.reset()
                    async with async_playwright() as playwright:
                        browser = await playwright.chromium.launch()
                        try:
                            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                            await page.goto(f"http://127.0.0.1:{server.server_address[1]}/fullsuite")
                            result = await Agent(ExportPolicy(), AgentConfig(root, root / "runs.sqlite3", root / "graph.json",
                                max_steps=6, execution_mode="pixels", memory_mode="none"), CalibrationGrounder()).run(page, "export the Launch Brief as PDF")
                            valid_pdf = bool(result.download_paths and Path(result.download_paths[0]).read_bytes().startswith(b"%PDF-"))
                            return result, valid_pdf
                        finally:
                            await browser.close()
                finally:
                    server.shutdown()
                    server.server_close()

        result, valid_pdf = asyncio.run(scenario())
        self.assertTrue(result.completed, result.error)
        self.assertTrue(valid_pdf)
        self.assertGreaterEqual(len(result.download_paths), 1)

    def test_fullsuite_export_pdf_with_early_confirm_recovery(self):
        tried_early_confirm = False

        class RecoveryExportPolicy:
            model = "test-recovery-export-policy"

            async def decide(self, _goal, observation, _context, _history):
                nonlocal tried_early_confirm
                visible = {item.text: item for item in observation.elements}
                if "Launch brief.pdf is ready" in visible:
                    target = visible["Launch brief.pdf is ready"]
                    return [ActionDecision("done", element_id=target.id,
                        verify=VerificationCondition("element_visible", pattern="Launch brief.pdf is ready"),
                        grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "Confirm export" in visible and "PDF document selected" in visible:
                    target = visible["Confirm export"]
                    return [ActionDecision("click", target.id, grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "Confirm export" in visible and not tried_early_confirm:
                    tried_early_confirm = True
                    target = visible["Confirm export"]
                    return [ActionDecision("click", target.id,
                        verify=VerificationCondition("download_created"),
                        grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "PDF document" in visible:
                    target = visible["PDF document"]
                    return [ActionDecision("click", target.id,
                        verify=None,
                        grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                label = "Export options" if "Export options" in visible else "Launch Brief"
                target = visible[label]
                return [ActionDecision("click", target.id,
                    verify=VerificationCondition("download_created") if label == "Export options" else None,
                    grounding=(EvidenceRecord("element_text", target.text, target.id),))]

        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                server = serve_visual_function_lab(0)
                try:
                    server.RequestHandlerClass.evaluator.reset()
                    async with async_playwright() as playwright:
                        browser = await playwright.chromium.launch()
                        try:
                            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                            await page.goto(f"http://127.0.0.1:{server.server_address[1]}/fullsuite")
                            result = await Agent(RecoveryExportPolicy(), AgentConfig(root, root / "runs.sqlite3", root / "graph.json",
                                max_steps=8, execution_mode="pixels", memory_mode="none"), CalibrationGrounder()).run(page, "export the Launch Brief as PDF")
                            valid_pdf = bool(result.download_paths and Path(result.download_paths[0]).read_bytes().startswith(b"%PDF-"))
                            return result, valid_pdf
                        finally:
                            await browser.close()
                finally:
                    server.shutdown()
                    server.server_close()

        result, valid_pdf = asyncio.run(scenario())
        self.assertTrue(result.completed, result.error)
        self.assertTrue(valid_pdf)
        self.assertGreaterEqual(len(result.download_paths), 1)

    def test_fullsuite_export_pdf_active_action_model_learning_and_reuse(self):
        class StepPolicy:
            model = "test-step-policy"

            async def decide(self, _goal, observation, _context, _history):
                visible = {item.text: item for item in observation.elements}
                if "Launch brief.pdf is ready" in visible:
                    target = visible["Launch brief.pdf is ready"]
                    return [ActionDecision("done", element_id=target.id,
                        verify=VerificationCondition("element_visible", pattern="Launch brief.pdf is ready"),
                        grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "Confirm export" in visible and "PDF document selected" in visible:
                    target = visible["Confirm export"]
                    return [ActionDecision("click", target.id, grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "PDF document" in visible:
                    target = visible["PDF document"]
                    return [ActionDecision("click", target.id, grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                label = "Export options" if "Export options" in visible else "Launch Brief"
                target = visible[label]
                return [ActionDecision("click", target.id, grounding=(EvidenceRecord("element_text", target.text, target.id),))]

        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                server = serve_visual_function_lab(0)
                try:
                    server.RequestHandlerClass.evaluator.reset()
                    async with async_playwright() as playwright:
                        browser = await playwright.chromium.launch()
                        try:
                            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                            await page.goto(f"http://127.0.0.1:{server.server_address[1]}/fullsuite")
                            config = AgentConfig(root, root / "runs.sqlite3", root / "graph.json",
                                max_steps=8, execution_mode="pixels", memory_mode="active-action-model")
                            agent = Agent(StepPolicy(), config, CalibrationGrounder())
                            result = await agent.run(page, "export the Launch Brief as PDF")
                            valid_pdf = bool(result.download_paths and Path(result.download_paths[0]).read_bytes().startswith(b"%PDF-"))

                            # Verify action model captured effects from the harmless actions
                            model = ActionModel.load(root / "action-model-v2.json")
                            self.assertIn("launch_brief.v1", model.schemas)
                            self.assertIn("export_options.v1", model.schemas)
                            self.assertIn("pdf_document.v1", model.schemas)
                            self.assertIn("confirm_export.v1", model.schemas)
                            self.assertTrue(any(e.predicate == "export_format" and e.resulting_value == "pdf" for e in model.schemas["pdf_document.v1"].effects))
                            return result, valid_pdf
                        finally:
                            await browser.close()
                finally:
                    server.shutdown()
                    server.server_close()

        result, valid_pdf = asyncio.run(scenario())
        self.assertTrue(result.completed, result.error)
        self.assertTrue(valid_pdf)


if __name__ == "__main__":
    unittest.main()
