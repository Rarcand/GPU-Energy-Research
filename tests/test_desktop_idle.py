import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collection'/'rtx-3070'))
from collect_desktop_idle import foreign_activity,quiet


class IdleTests(unittest.TestCase):
    def sample(self,name,value):
        return {'active':[{'process':name,'utilization':value}]}

    def test_native_game_and_remote_video_are_busy(self):
        self.assertTrue(foreign_activity(self.sample('RobloxStudioBeta',5)))
        self.assertTrue(foreign_activity(self.sample('parsecd',5)))
        self.assertTrue(foreign_activity(self.sample('unknown',2)))
        self.assertTrue(foreign_activity(self.sample(None,2)))

    def test_only_small_compositor_and_wsl_accounting_are_allowed(self):
        self.assertFalse(foreign_activity(self.sample('dwm',1)))
        self.assertTrue(foreign_activity(self.sample('dwm',10)))
        self.assertFalse(foreign_activity(self.sample('vmmemWSL',90)))

    def test_vm_worker_requires_independent_wsl_identity(self):
        sample=self.sample('vmwp',80)
        self.assertTrue(foreign_activity(sample))
        sample['active'][0]['wsl_vm_id']='vm-guid'
        self.assertTrue(foreign_activity(sample))
        sample['verified_wsl_vm_ids']=['other-vm-guid']
        self.assertTrue(foreign_activity(sample))
        sample['verified_wsl_vm_ids']=['vm-guid']
        self.assertFalse(foreign_activity(sample))
        sample['active'][0]['process']='RobloxStudioBeta'
        self.assertTrue(foreign_activity(sample))

    def test_idle_requires_low_total_utilization_and_memory(self):
        state={'utilization':1,'memory_used_mib':700}
        self.assertTrue(quiet({'active':[]},state))
        self.assertTrue(quiet({'active':[]},dict(state,memory_used_mib=1700)))
        self.assertFalse(quiet(self.sample('parsecd',5),state))
        self.assertFalse(quiet({'active':[]},dict(state,utilization=10)))
        self.assertFalse(quiet({'active':[]},dict(state,utilization=6)))
        self.assertFalse(quiet({'active':[]},dict(state,memory_used_mib=5000)))


if __name__=='__main__':unittest.main()
