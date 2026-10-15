import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import json
from PIL import Image
from vision_gui_agent.models import Element
from vision_gui_agent.perception import GeminiVisualGrounder


class SemanticCoordinateTests(unittest.TestCase):
    def grounder(self, response):
        grounder=object.__new__(GeminiVisualGrounder)
        grounder._generate=lambda request:SimpleNamespace(text=json.dumps(response))
        return grounder

    def test_discovery_decodes_normalized_corners_once(self):
        grounder=self.grounder({'elements':[{'kind':'input','label':'Search','box_2d':[470,330,515,565],
            'context_box_2d':[470,330,515,655],'enabled':True,'readonly':False,'actionable':True}]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(1440,1000),'white').save(path)
            elements=asyncio.run(grounder.detect(path))
        self.assertEqual(len(elements),1)
        target=elements[0]
        for actual,expected in zip((target.x,target.y,target.width,target.height),(475.2,470,338.4,45)):
            self.assertAlmostEqual(actual,expected)
        self.assertAlmostEqual(target.context_bounds[2],468)

    def test_refinement_uses_crop_dimensions_and_one_origin_offset(self):
        grounder=self.grounder({'found':True,'actionable':True,'kind':'button','box_2d':[250,250,750,750]})
        target=Element(1,'','button','Go','','','button',400,300,40,40)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(1440,1000),'white').save(path)
            refined=asyncio.run(grounder.refine(path,target))
        self.assertIsNotNone(refined)
        # Crop [352,260,488,380]: 136x120.
        self.assertEqual((refined.x,refined.y,refined.width,refined.height),(386,290,68,60))

    def test_a_menu_row_around_a_detected_word_is_the_same_control(self):
        # Sidebar link: the detected box is the word, Gemini returns the whole row around it.
        grounder=self.grounder({'found':True,'actionable':True,'kind':'link','box_2d':[326,0,674,662]})
        target=Element(1,'','other','Documents','','','other',400,300,40,20,proposal_sources=('detector','ocr'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(1440,1000),'white').save(path)
            refined=asyncio.run(grounder.refine(path,target))
        self.assertIsNotNone(refined)
        # A box that does not contain the word at all is still a different control.
        grounder=self.grounder({'found':True,'actionable':True,'kind':'link','box_2d':[0,0,200,200]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(1440,1000),'white').save(path)
            self.assertIsNone(asyncio.run(grounder.refine(path,target)))

    def test_refinement_crop_outlines_the_proposed_box_outside_the_control(self):
        # Weaker models picked the highlighted neighbouring menu row; the outline marks which instance to confirm.
        sent=[]
        def generate_content(model,contents,config):
            sent.append(contents);return SimpleNamespace(text=json.dumps({'found':True,'actionable':True,'kind':'button','box_2d':[326,0,674,662]}))
        grounder=object.__new__(GeminiVisualGrounder)
        grounder._generate=lambda request:request(SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
        grounder.types=SimpleNamespace(Part=SimpleNamespace(from_bytes=lambda data,mime_type:data),
                                       GenerateContentConfig=lambda **kwargs:None,AutomaticFunctionCallingConfig=lambda **kwargs:None)
        grounder.model='test'
        target=Element(1,'','other','Save','','','other',400,300,40,20,proposal_sources=('detector','ocr'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(1440,1000),'white').save(path)
            self.assertIsNotNone(asyncio.run(grounder.refine(path,target)))
        from io import BytesIO
        crop=Image.open(BytesIO(sent[0][1])).convert('RGB')
        # Crop origin (352,264): the outline sits 3 px outside the box, the control itself is untouched.
        red,green,blue=crop.getpixel((45,46))  # JPEG softens the 2 px line
        self.assertTrue(red>150 and green<100 and blue>150)
        self.assertTrue(min(crop.getpixel((68,46)))>245)
        self.assertIn('magenta',sent[0][2])

    def test_invalid_normalized_box_never_falls_back_to_legacy_coordinates(self):
        for box in ([0,-2,500,1000],[500,100,100,500],[0,0,1001,1000]):
            grounder=self.grounder({'elements':[{'kind':'input','label':'Search','box_2d':box,
                                               'x':10,'y':10,'width':100,'height':40}]})
            with self.subTest(box=box),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'screen.png';Image.new('RGB',(1440,1000),'white').save(path)
                self.assertEqual(asyncio.run(grounder.detect(path)),[])

    def test_refinement_cannot_switch_to_adjacent_control(self):
        grounder=self.grounder({'found':True,'actionable':True,'kind':'button','box_2d':[250,700,750,950]})
        target=Element(1,'','other','EN','','','other',100,100,40,40,proposal_sources=('detector',))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(400,300),'white').save(path)
            refined=asyncio.run(grounder.refine(path,target))
        self.assertIsNone(refined)
        self.assertIn('selected_instance_changed',grounder.last_refinement_rejection)

    def test_null_value_is_unknown_not_literal_none(self):
        grounder=self.grounder({'elements':[{'kind':'input','label':'Search','box_2d':[100,100,200,500],
                                           'value':None,'actionable':True}]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'screen.png';Image.new('RGB',(1440,1000),'white').save(path)
            element=asyncio.run(grounder.detect(path))[0]
        self.assertEqual(element.value,'')
        self.assertNotIn('value',element.state_observed)
