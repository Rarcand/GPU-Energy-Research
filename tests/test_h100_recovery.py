import contextlib,io,json,sys,unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'collection'/'h100'))
from diagnose_h100_recovery import inherited_devices,child

class H100Tests(unittest.TestCase):
    def test_torch_probe_logs_device_name_and_reaches_both_dtypes(self):
        # Reproduce the actual readiness path without allocating a CUDA device.
        cuda=MagicMock()
        cuda.get_device_properties.return_value=SimpleNamespace(name='NVIDIA H100 PCIe',major=9,minor=0)
        cuda.mem_get_info.return_value=(1024,2048)
        fake=MagicMock()
        fake.__version__='2.8.0+cu128'
        fake.__file__='mock-torch.py'
        fake.version.cuda='12.8'
        fake.cuda=cuda
        fake.testing=SimpleNamespace(assert_close=MagicMock())
        output=io.StringIO()
        with patch.dict(sys.modules,{'torch':fake}),contextlib.redirect_stdout(output):
            child('torch')
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        initialized=next(e for e in events if e['stage']=='device_initialized')
        self.assertEqual(initialized['name'],'NVIDIA H100 PCIe')
        self.assertEqual([e['dtype'] for e in events if e['stage']=='dtype_pass'],['float16','bfloat16'])
        self.assertIn('torch_probe_pass',[e['stage'] for e in events])
        self.assertEqual(fake.testing.assert_close.call_count,2)

    def test_gpu_selection_respects_inherited_visibility(self):
        devices=[dict(index='0',uuid='GPU-abc'),dict(index='1',uuid='GPU-def')]
        self.assertEqual(inherited_devices(devices,'1'),[devices[1]])
        self.assertEqual(inherited_devices(devices,'GPU-def'),[devices[1]])
        self.assertEqual(inherited_devices(devices,''),[])
        self.assertEqual(inherited_devices(devices,None),devices)

if __name__=='__main__':unittest.main()
