"""Independent hand latching and unified eight-chunk history regressions."""
from copy import deepcopy
import time
from Reverse.core import ReverseEpisode

MASK={s:dict(enabled=True,gripper=True) for s in ('left','right')}


def point(left=0.,right=0.,left_grip=0.,right_grip=0.):
    return dict(left=[left]*7+[left_grip],right=[right]*7+[right_grip],body={},speed={})


def episode():
    core=ReverseEpisode(['Aojiru','Cola']);core.begin('',MASK,targets={'left':'Aojiru','right':'Cola'})
    core.anchor(point());return core


def vote(core,side,state):
    now=time.monotonic_ns();core.votes[side].clear()
    for i in range(10):core.ingest(side,core.targets[side],core.generation,i,now-(10-i)*1000,state)


def send(core,c,p):
    core.record(c,p,{s:dict(positions=p[s][:7],gripper=p[s][7]) for s in ('left','right')})


def test_unified_history_keeps_eight_groups_across_hand_switches():
    core=episode();last=point()
    for n in range(12):
        side='left' if n%2==0 else 'right';candidate=[]
        for i in range(6):
            p=deepcopy(last);p[side][:7]=[last[side][0]+i*.02]*7;candidate.append(p)
        assert core.gate(candidate)['state']=='ORDINARY'
        for p in candidate:send(core,str(n),p)
        last=candidate[-1]
        assert [g['chunk_id'] for g in core.history]==[str(i) for i in range(max(0,n-7),n+1)]
    assert len(core.history)==8


def test_right_close_does_not_end_left_retry_mask_or_clear_left_votes():
    core=episode();core.mask_active=True;core.retry_hand='left'
    vote(core,'right','PASS');vote(core,'left','WRONG')
    action=point(right=.03,right_grip=1.5)
    result=core.gate([action]);assert result['state']=='PASS' and set(result['hands'])=={'right'}
    send(core,'right',action)
    assert core.mask_active and 'right' in core.closed_hands
    assert len(core.votes['left'])==10 and not core.votes['right']
    assert core.gate([action])['state']=='ORDINARY'
    vote(core,'left','PASS');action['left'][7]=1.5
    assert core.gate([action])['state']=='PASS'
    assert core.mask_active  # Predictions and gate checks do not end the mask.
    send(core,'left',action);assert not core.mask_active


def test_reopening_one_hand_preserves_other_hand_latch_and_history():
    core=episode();vote(core,'left','PASS');vote(core,'right','PASS')
    both=point(left_grip=1.5,right_grip=1.5)
    assert core.gate([both])['state']=='PASS';send(core,'closed',both)
    opened=deepcopy(both);opened['left'][7]=0.;send(core,'left-open',opened)
    assert core.closed_hands=={'right'}
    assert core.gate([both])['state']=='WAIT'
    assert [g['chunk_id'] for g in core.history]==['closed','left-open']
