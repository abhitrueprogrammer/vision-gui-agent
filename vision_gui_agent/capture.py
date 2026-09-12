"""Screenshot-to-input geometry and conservative pre-dispatch freshness checks."""
from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.async_api import Page

from .models import CaptureMetadata, Observation


class StaleObservation(ValueError):
    """No input was sent: a fresh observation and selection are required."""


async def page_url(page) -> str:
    value=getattr(page,'url','')
    if callable(value): value=value()
    if hasattr(value,'__await__'): value=await value
    return str(value or '')


async def input_geometry(page) -> tuple[tuple[float,float] | None, tuple[float,float], str]:
    if hasattr(page,'input_geometry'):
        raw=await page.input_geometry()
        return (tuple(raw['size']) if raw.get('size') else None,tuple(raw.get('origin',(0,0))),raw.get('space','unknown'))
    viewport=getattr(page,'viewport_size',None)
    if viewport is None and isinstance(page,Page):
        # Browser viewport metadata only; this never discovers application targets.
        viewport=await page.evaluate('({width:window.innerWidth,height:window.innerHeight})')
    if isinstance(viewport,dict) and viewport.get('width',0)>0 and viewport.get('height',0)>0:
        return (float(viewport['width']),float(viewport['height'])),(0.0,0.0),'browser_css_pixels'
    return None,(0.0,0.0),'unknown'


async def capture_screen(page, path: Path) -> tuple[CaptureMetadata,str]:
    for attempt in range(3):
        url=await page_url(page)
        geometry=await input_geometry(page)
        captured_at=time.time()
        await page.screenshot(path=str(path),full_page=False)
        if url != await page_url(page) or geometry != await input_geometry(page):
            if attempt==2: raise StaleObservation('URL or input geometry changed during capture')
            continue
        with Image.open(path) as image: size=image.size
        return CaptureMetadata(captured_at,size,geometry[0],geometry[1],geometry[2]),url
    raise StaleObservation('Unstable capture')


async def prepare_dispatch(page, observation: Observation, element, info: dict) -> tuple[float,float] | None:
    """Transform native screenshot pixels once, after verifying the source is current."""
    info['status']='not_dispatched'
    current_size,current_origin,current_space=await input_geometry(page)
    metadata=observation.capture
    # Compatibility for direct executor callers that provide screenshot files:
    # infer browser scale from actual image dimensions, not a guessed DPR.
    if metadata is None and current_size and Path(observation.screenshot_path).is_file():
        with Image.open(observation.screenshot_path) as image: size=image.size
        metadata=CaptureMetadata(None,size,current_size,current_origin,current_space)
    if metadata is None:
        # Legacy injected executor contracts have no capture/freshness guarantee.
        if element is None: return None
        point=(element.x+element.width/2,element.y+element.height/2)
        info.update(requested_point=point,delivered_point=point,mapping='legacy_unverified')
        return point
    if not metadata.input_size or not current_size or metadata.input_space=='unknown' or metadata.image_space!='screenshot_pixels':
        raise StaleObservation('Screenshot has no established input coordinate mapping')
    if (tuple(metadata.input_size)!=tuple(current_size) or tuple(metadata.origin)!=tuple(current_origin)
            or metadata.input_space!=current_space):
        raise StaleObservation('Input geometry changed since capture')
    if observation.url and await page_url(page)!=observation.url:
        raise StaleObservation('URL changed since capture')
    fresh=await page.screenshot(full_page=False)
    with Image.open(observation.screenshot_path).convert('RGB') as source, Image.open(BytesIO(fresh)).convert('RGB') as latest:
        if source.size!=tuple(metadata.image_size) or latest.size!=source.size or latest.tobytes()!=source.tobytes():
            raise StaleObservation('Screen changed since capture; reselect the target')
    if (current_size,current_origin,current_space)!=await input_geometry(page) or (observation.url and await page_url(page)!=observation.url):
        raise StaleObservation('Input geometry or URL changed during freshness check')
    scale=(metadata.input_size[0]/metadata.image_size[0],metadata.input_size[1]/metadata.image_size[1])
    info.update(captured_at=metadata.captured_at,freshness_checked_at=time.time(),
                image_size=metadata.image_size,input_size=metadata.input_size,origin=metadata.origin,
                pixel_to_input_scale=scale,input_space=metadata.input_space,mapping='verified')
    if element is None:return None
    point=(element.x+element.width/2,element.y+element.height/2)
    if not (0<=point[0]<metadata.image_size[0] and 0<=point[1]<metadata.image_size[1]):
        raise StaleObservation('Target point is outside the captured image')
    delivered=(metadata.origin[0]+point[0]*scale[0],metadata.origin[1]+point[1]*scale[1])
    info.update(requested_point=point,delivered_point=delivered)
    return delivered
