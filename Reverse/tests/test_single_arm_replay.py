"""Focused offline checks; pure bookkeeping, no SDK/network/model initialization."""
from copy import deepcopy
import unittest
from Reverse.core import ReverseEpisode, ReverseError

MASK={s:dict(enabled=True,gripper=True) for s in ('left','right')}


class SingleArmReplay(unittest.TestCase):
    def episode(self,side='left',dual=True):
        events=[]
        core=ReverseEpisode(['Aojiru','Cola'],lambda kind,**kw:events.append((kind,kw)))
        core.begin('',MASK,targets={'left':'Aojiru','right':'Cola'} if dual else {side:'Aojiru'})
        core.anchor(dict(left=[0.]*8,right=[.5]*7+[0.],body={},speed={}))
        for chunk in range(12):
            for index in range(3):
                n=chunk*3+index+1
                actual=dict(left=[n*.01+j*.002 for j in range(7)]+[1.5 if chunk>=8 else 0.],
                            right=[.5-n*.004+j*.003 for j in range(7)]+[1.5 if chunk>=8 else .1],body={},speed={})
                # Distinguish SDK targets from unexecuted requested predictions.
                request=deepcopy(actual)
                for arm in ('left','right'):request[arm][:7]=[99.]*7
                core.record(str(chunk),request,{arm:dict(positions=actual[arm][:7],gripper=actual[arm][7]) for arm in ('left','right')})
        return core,events

    def check_side(self,side,dual=True,completed=False):
        core,events=self.episode(side,dual)
        other='right' if side=='left' else 'left'
        # Stage hand can lag the current rejected hand; the WRONG result wins.
        core.active_hand=other if dual else side
        core.last_gate=dict(state='WRONG',hands={side:dict(pass_count=1),**({other:dict(pass_count=10)} if dual else {})})
        frozen=deepcopy(core.last_sent[other]);history=deepcopy(list(core.history))
        expected=[deepcopy(core.history[0]['before'][side])]
        expected += [deepcopy(v['action'][side]) for g in core.history for v in g['steps']]
        plan=core.replay_plan()
        self.assertEqual(core.rollback_hand,side)
        self.assertEqual(len(core.history),8)
        self.assertEqual(list(core.history),history)
        self.assertTrue(all(p[other]==frozen for p in plan))
        self.assertTrue(any(side in p.get('reverse_release',[]) for p in plan))
        self.assertTrue(all(other not in p.get('reverse_release',[]) for p in plan))
        # Joint path is the exact reverse SDK path; release adds a same-pose step.
        actual=[p[side][:7] for p in plan if not p.get('reverse_release')]
        self.assertEqual(actual,[p[:7] for p in reversed(expected)])
        release=next(i for i,p in enumerate(plan) if p.get('reverse_release'))
        self.assertEqual(plan[release][side][:7],plan[release-1][side][:7])
        self.assertEqual(plan[release-1][side][7],1.5)
        self.assertTrue(all(p[side][7]==0. for p in plan[release:]))
        begin=next(v for k,v in events if k=='ROLLBACK_BEGIN')
        self.assertEqual(begin['rollback_hand'],side)
        self.assertEqual(begin['held_hands'],[other])
        self.assertEqual(begin['hold_targets'][other],frozen)
        core.replay_finished(plan[-1]);self.assertIsNone(core.rollback_hand)
        # Normal forward VLA may command that arm again after replay.
        p=deepcopy(core.last_sent);p[other][0]+=.02
        core.record('new-vla',p,{s:dict(positions=p[s][:7],gripper=p[s][7]) for s in ('left','right')})
        self.assertEqual(core.last_sent[other][0],p[other][0])

    def test_left_error_holds_right(self):self.check_side('left')
    def test_right_error_holds_left(self):self.check_side('right')
    def test_single_left_task_still_holds_right(self):self.check_side('left',dual=False)
    def test_single_right_task_still_holds_left(self):self.check_side('right',dual=False)

    def test_both_fail_use_known_active_hand_only(self):
        for side in ('left','right'):
            with self.subTest(side=side):
                core,_=self.episode();core.active_hand=side
                core.last_gate=dict(state='WRONG',hands={s:dict(pass_count=0) for s in ('left','right')})
                other='right' if side=='left' else 'left';frozen=deepcopy(core.last_sent[other])
                self.assertTrue(all(p[other]==frozen for p in core.replay_plan()))
                self.assertEqual(core.rollback_hand,side)

    def test_ambiguous_side_never_silently_replays_both(self):
        core,_=self.episode()
        for group in core.history:
            group['before']['right']=list(group['before']['left'])
            for v in group['steps']:v['action']['right']=list(v['action']['left'])
        core.active_hand=None;core.last_gate=None
        with self.assertRaisesRegex(ReverseError,'无法确定'):core.replay_plan()

    def test_empty_history_retains_bounded_retry(self):
        core,_=self.episode();core.history.clear()
        for _ in range(3):
            self.assertEqual(core.replay_plan(),[]);core.replay_finished(core.last_sent)
        with self.assertRaises(ReverseError):core.replay_plan()

if __name__=='__main__':unittest.main()
