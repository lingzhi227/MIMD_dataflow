from pathlib import Path
import sys
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'hls-layout.json').exists())
sys.path.insert(0, str(ROOT/'tools'))
from bootstrap import configure
configure(ROOT)
from event_protocol import Op, Protocol
from protocol_safety import explore
from summa_protocol import from_plan, mismatched_stream_order


class EventProtocols(unittest.TestCase):
    def test_bounded_dag_deadlock_and_capacity_repair(self):
        bad = explore(mismatched_stream_order(1))
        self.assertEqual(bad['status'], 'DEADLOCK')
        self.assertEqual([(s['actor'], s['op']) for s in bad['witness']], [('producer', 'put')])
        self.assertEqual(explore(mismatched_stream_order(2))['status'], 'PASS')

    def test_join_is_required_for_both_arrival_orders(self):
        plan = {'profile': 'mesh_gemm.v1', 'rounds': 2,
                'stages': [{}, {}, {'after': ['broadcast_A', 'broadcast_B']}]}
        self.assertEqual(explore(from_plan(plan))['status'], 'PASS')
        self.assertEqual(explore(from_plan(plan, omit_y_wait=True))['status'], 'UNSAFE')
        self.assertEqual(explore(from_plan(plan, early_reuse=True))['status'], 'UNSAFE')

    def test_fanout_release_one_reader_does_not_permit_reuse(self):
        ops = (Op('write_begin','b',0,'w'), Op('write_end','b',0,'w'),
               Op('borrow','b',0,'r1'), Op('borrow','b',0,'r2'),
               Op('release','b',0,'r1'), Op('write_begin','b',1,'w'))
        bad = explore(Protocol((('p', ops),), ('b',)))
        self.assertEqual(bad['status'], 'UNSAFE')
        self.assertIn('overlaps', bad['reason'])

    def test_completion_must_match_version_and_actor(self):
        p = Protocol((('p', (Op('write_begin','b',0,'w'), Op('write_end','b',1,'w'))),), ('b',))
        self.assertEqual(explore(p)['status'], 'UNSAFE')

    def test_release_must_match_borrow_version(self):
        ops = (Op('write_begin','b',0,'w'), Op('write_end','b',0,'w'),
               Op('borrow','b',0,'r'), Op('release','b',1,'r'))
        self.assertEqual(explore(Protocol((('p', ops),), ('b',)))['status'], 'UNSAFE')

    def test_limit_is_never_pass(self):
        self.assertEqual(explore(mismatched_stream_order(2), max_states=1)['status'], 'INCONCLUSIVE')

    def test_leaked_ownership_is_not_successful_termination(self):
        p = Protocol((('p', (Op('write_begin','b',0,'w'),)),), ('b',))
        self.assertEqual(explore(p)['status'], 'UNSAFE')


if __name__ == '__main__':
    unittest.main()
