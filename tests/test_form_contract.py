from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from playwright.async_api import async_playwright

from vision_gui_agent.executor import execute
from vision_gui_agent.models import ActionDecision, Element, Observation


def control(ident: int, tag: str, x: int, y: int, *, checked: bool | None = None, input_type: str = "", value: str = "") -> Element:
    return Element(ident, "", tag, tag, "", "", tag, x, y, 20 if tag in {"checkbox", "radio"} else 180, 24, checked=checked, input_type=input_type, value=value)


class FormContractTests(unittest.TestCase):
    def test_browser_actions_cover_supported_form_controls(self) -> None:
        async def scenario() -> None:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch()
                page = await browser.new_page(viewport={"width": 500, "height": 500})
                await page.set_content('''<style>input,textarea,select,button{position:absolute;left:10px;width:180px;height:24px}#text{top:10px}#password{top:35px}#notes{top:60px}#choice{top:90px}#file{top:120px}#off{top:150px;left:10px;width:20px}#on{top:150px;left:35px;width:20px}#radio{top:150px;left:60px;width:20px}#color{top:180px}#date{top:210px}#range{top:240px}#submit{top:270px}</style><form><input id="text"><input id="password" type="password"><textarea id="notes"></textarea>
                    <select id="choice"><option>Choose</option><option>India</option></select><input id="file" type="file">
                    <input id="off" type="checkbox" checked><input id="on" type="checkbox"><input id="radio" type="radio" name="x" checked>
                    <input id="color" type="color" value="#000000"><input id="date" type="date"><input id="range" type="range" min="0" max="20" value="3">
                    <button id="submit" type="submit">Submit</button></form><p id="result"></p><script>document.querySelector('form').onsubmit=e=>{e.preventDefault();result.textContent='Form submitted'}</script>''')
                elements = [
                    control(1, "input", 10, 10), control(2, "input", 10, 35, input_type="password"), control(3, "textarea", 10, 60),
                    control(4, "select", 10, 90, value="Choose"), control(5, "file", 10, 120, input_type="file"), control(6, "checkbox", 10, 150, checked=True),
                    control(7, "checkbox", 35, 150, checked=False), control(8, "radio", 60, 150, checked=True), control(9, "color", 10, 180, input_type="color"),
                    control(10, "input", 10, 210, input_type="date"), control(11, "range", 10, 240, input_type="range"), control(12, "button", 10, 270),
                ]
                observation = Observation("", "", elements, "", "Form")
                with tempfile.NamedTemporaryFile() as upload:
                    upload.write(b"proof"); upload.flush()
                    for decision in (
                        ActionDecision("fill", 1, text="Ada"), ActionDecision("fill", 2, text="secret"), ActionDecision("fill", 3, text="line one\nline two"),
                        ActionDecision("select", 4, text="Choose"), ActionDecision("select", 4, text="India"), ActionDecision("upload", 5, text=upload.name), ActionDecision("set_checked", 6, checked=False),
                        ActionDecision("set_checked", 7, checked=True), ActionDecision("set_color", 9, text="#12ab34"), ActionDecision("set_date", 10, text="2026-09-02"),
                        ActionDecision("set_range", 11, text="12"), ActionDecision("click", 12),
                    ):
                        await execute(page, observation, decision)
                state = await page.evaluate('''() => Object.fromEntries(['text','password','notes','choice','file','off','on','color','date','range','result'].map(id => {
                    const element = document.getElementById(id); return [id, id === 'file' ? element.files[0]?.name : ['off','on'].includes(id) ? element.checked : id === 'result' ? element.textContent : element.value]
                }))''')
                self.assertEqual(state, {"text": "Ada", "password": "secret", "notes": "line one\nline two", "choice": "India",
                                         "file": Path(upload.name).name, "off": False, "on": True, "color": "#12ab34", "date": "2026-09-02", "range": "12", "result": "Form submitted"})
                await browser.close()
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
