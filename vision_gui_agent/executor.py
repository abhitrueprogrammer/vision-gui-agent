from playwright.async_api import Page
from .models import ActionDecision, Observation

async def execute(page: Page, observation: Observation, decision: ActionDecision) -> None:
    if decision.action == "done": return
    if decision.action == "scroll":
        await page.mouse.wheel(0, 650 if decision.direction != "up" else -650); await page.wait_for_timeout(250); return
    element = next((item for item in observation.elements if item.id == decision.element_id), None)
    if element is None: raise ValueError(f"Element {decision.element_id} is not present in this observation")
    locator = page.locator(element.selector)
    if decision.action == "click": await locator.click()
    elif decision.action == "fill": await locator.fill(decision.text or "")
    elif decision.action == "select": await locator.select_option(label=decision.text)
    elif decision.action == "press": await locator.press(decision.key or "Enter")
    await page.wait_for_timeout(350)
