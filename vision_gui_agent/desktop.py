from __future__ import annotations

import asyncio
from pathlib import Path


class DesktopMouse:
    def __init__(self, backend) -> None:
        self.backend = backend

    async def click(self, x: float, y: float) -> None:
        await asyncio.to_thread(self.backend.click, x, y)

    async def wheel(self, _x: float, y: float) -> None:
        amount = max(1, round(abs(y) / 100))
        await asyncio.to_thread(self.backend.scroll, -amount if y > 0 else amount)


class DesktopKeyboard:
    KEY_NAMES = {"control": "ctrl", "meta": "command", "arrowup": "up", "arrowdown": "down",
                 "arrowleft": "left", "arrowright": "right", "escape": "esc"}

    def __init__(self, backend) -> None:
        self.backend = backend

    async def press(self, key: str) -> None:
        keys = [self.KEY_NAMES.get(part.casefold(), part.casefold()) for part in key.split("+")]
        function = self.backend.hotkey if len(keys) > 1 else self.backend.press
        await asyncio.to_thread(function, *keys)

    async def type(self, text: str) -> None:
        await asyncio.to_thread(self.backend.write, text, interval=0.01)


class DesktopPage:
    """Small Playwright-shaped adapter over the real OS screen and input devices."""

    def __init__(self, backend=None) -> None:
        if backend is None:
            try:
                import pyautogui as backend
            except Exception as exc:
                raise RuntimeError("Desktop control needs a graphical session and the pyautogui dependency") from exc
        backend.FAILSAFE = True
        self.backend = backend
        self.mouse, self.keyboard = DesktopMouse(backend), DesktopKeyboard(backend)

    async def screenshot(self, path: str, full_page: bool = False) -> None:
        del full_page
        image = await asyncio.to_thread(self.backend.screenshot)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    async def wait_for_timeout(self, milliseconds: float) -> None:
        await asyncio.sleep(milliseconds / 1000)
