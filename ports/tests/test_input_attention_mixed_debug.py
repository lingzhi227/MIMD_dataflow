import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'toolchain'))
from input_attention_mixed_debug import inspect
from frontend import Error

class MixedDebug(unittest.TestCase):
    def test_precision_and_missing_observation(self):
        p=ROOT/'tests/fixtures/history/input-attention-codegen-20260907T142401616614Z'
        s=json.loads((p/'schedule.json').read_text());rows=json.loads((p/'results.json').read_text())['cases'];r=dict(diagnostics=rows)
        wide=inspect(s,r,'p0_0',3,10)
        self.assertEqual(wide['storage_dtype'],'f32');self.assertEqual(wide['raw_words'],rows[3]['mixed_z'][0][0]);self.assertTrue(wide['observed'])
        half=inspect(s,r,'p0_0',3,12);self.assertEqual(half['word_bits'],16)
        hidden=inspect(s,r,'p0_0',3,15);self.assertFalse(hidden['observed']);self.assertIsNone(hidden['values'])
        self.assertFalse(inspect(s,None,'p0_0',0,0)['available'])
        with self.assertRaises(Error):inspect(s,r,'p8_0',0,0)
        with self.assertRaises(Error):inspect(s,r,'p0_0',0,19)

if __name__=='__main__':unittest.main()
