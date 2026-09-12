"""Render the authored P1 corpus; annotations are evaluator-only, never agent DOM input."""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1] / 'tests/fixtures/perception'
CASES = [
    dict(id='wide_button',html='<button style="left:20px;top:20px;width:180px;height:40px;background:#1769cc;color:white;border:0">Continue</button>',
         detector=[[20,20,200,60]],ocr=[['Continue',[65,30,90,20]]],
         targets=[dict(instance='continue',label='Continue',kind='button',box=[20,20,180,40])]),
    dict(id='unrelated_price',html='<span style="left:10px;top:10px">Account overview</span><span style="left:330px;top:60px">$42</span>',
         detector=[],ocr=[['Account overview',[10,10,130,20]],['$42',[330,60,40,20]]],targets=[]),
    dict(id='duplicate_rows',html='<section style="left:5px;top:5px;width:395px;height:75px;border:1px solid #999"><span style="left:15px;top:20px">Row A</span></section><button style="left:300px;top:20px;width:80px;height:40px">Save</button><section style="left:5px;top:95px;width:395px;height:75px;border:1px solid #999"><span style="left:15px;top:20px">Row B</span></section><button style="left:300px;top:110px;width:80px;height:40px">Save</button>',
         detector=[[5,5,400,80],[300,20,380,60],[5,95,400,170],[300,110,380,150]],
         ocr=[['Row A',[20,25,60,20]],['Save',[310,25,50,20]],['Row B',[20,115,60,20]],['Save',[310,115,50,20]]],
         targets=[dict(instance='row_a_save',label='Save',kind='button',box=[300,20,80,40]),dict(instance='row_b_save',label='Save',kind='button',box=[300,110,80,40])]),
    dict(id='border_recovery',html='<button style="left:20px;top:20px;width:120px;height:40px;border:1px solid #333;background:white">Download</button><span style="left:20px;top:100px">Plain heading</span>',
         detector=[],ocr=[['Download',[40,30,80,20]],['Plain heading',[20,100,120,20]]],
         targets=[dict(instance='download',label='Download',kind='button',box=[20,20,120,40])]),
    dict(id='plain_miss',html='<button style="left:20px;top:20px;width:100px;height:40px;border:0;background:white">Details</button>',
         detector=[],ocr=[['Details',[40,30,60,20]]],targets=[dict(instance='details',label='Details',kind='button',box=[20,20,100,40])]),
    dict(id='fields',html='<label style="left:20px;top:5px">Quantity</label><input value="12" style="left:20px;top:30px;width:180px;height:40px"><input type="checkbox" checked style="left:20px;top:100px;width:20px;height:20px;margin:0"><span style="left:50px;top:100px">Reviewed</span><button disabled style="left:220px;top:30px;width:120px;height:40px">Unavailable</button>',
         detector=[[20,30,200,70],[20,100,40,120],[220,30,340,70]],ocr=[['Quantity',[20,5,80,20]],['12',[28,40,20,20]],['Reviewed',[50,100,100,20]],['Unavailable',[230,40,100,20]]],
         targets=[dict(instance='quantity',label='Quantity',kind='input',box=[20,30,180,40]),dict(instance='reviewed',label='Reviewed',kind='checkbox',box=[20,100,20,20])]),
]

async def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as playwright:
        browser=await playwright.chromium.launch()
        page=await browser.new_page(viewport={'width':500,'height':220},device_scale_factor=1)
        for case in CASES:
            html='<style>body{margin:0;background:white;font:16px Arial}body *{position:absolute;box-sizing:border-box;font:16px Arial}</style>'+case['html']
            (ROOT/(case['id']+'.html')).write_text(html)
            await page.set_content(html)
            await page.screenshot(path=str(ROOT/(case['id']+'.png')))
            for target in case['targets']:
                x,y,w,h=target['box'];target.update(visible=True,safe_region=[x+3,y+3,w-6,h-6])
        await browser.close()
    (ROOT/'manifest.json').write_text(json.dumps({'version':1,'source':'authored Chromium pages, DPR 1; manually specified oracle boxes and proposal/OCR fixtures; not a model accuracy benchmark','cases':CASES},indent=2)+'\n')

if __name__=='__main__':asyncio.run(main())
