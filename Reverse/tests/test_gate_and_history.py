"""Offline simulations of the real publisher, sync executor and RTC scheduler."""
from copy import deepcopy
from pathlib import Path
import threading
import time
import pytest
from Reverse.core import ReverseEpisode, ReverseError, prompt_targets
from Reverse.runtime import ReverseRuntime
from Reverse.rolling_gate import guard_step

MASK = {s:dict(enabled=True,joints=[True]*7,gripper=True) for s in ('left','right')}
MASK.update(lift=False,waist=False,head=False,speed=False)
NAMES=['Aojiru','Cola']+['fixture'+str(i) for i in range(69)]


def step(value=0.,closed=0.):
    return dict(left=[value]*7+[closed],right=[0.]*8,body={},speed=dict(linear=0.,angular=0.))


def episode():
    core=ReverseEpisode(NAMES);core.begin('Grasp Aojiru with the left hand.',MASK);core.anchor(step());return core


def votes(core,states,*,side='left',age=.02,generation=None):
    now=time.monotonic_ns();core.votes[side].clear()
    floor=core.rearm_ns[side]
    if now-floor < len(states)*1000:
        time.sleep(.0001);now=time.monotonic_ns()
    start=max(floor,now-int(len(states)*age*1e9))
    gap=max(1,(now-start)//max(1,len(states)))
    for i,state in enumerate(states):
        core.ingest(side,core.targets[side],generation or core.generation,i+1,start+i*gap,state)


def test_ordinary_does_not_consult_siglip_or_authorize_future_close():
    core=episode();core.votes=None
    assert core.gate([step(),step(closed=1.5)])['state']=='ORDINARY'


@pytest.mark.parametrize('states,expected',[
    (['PASS']*10,'PASS'),
    (['WRONG']*3+['PASS']*7,'PASS'),
    (['PASS']*5+['WRONG']*3+['PASS']*2,'PASS'),
    (['WRONG']*4+['PASS']*6,'WAIT'),
    (['PASS']*9+['UNCERTAIN'],'WAIT'),
    (['PASS']*8+['WRONG','PASS'],'WAIT'),
    (['UNCERTAIN']*10,'WAIT'),
])
def test_7of10_and_latest_two_are_both_required(states,expected):
    core=episode();votes(core,states)
    assert core.gate([step(closed=1.5)])['state']==expected


def test_incomplete_duplicate_reordered_stale_future_and_old_generation_never_release():
    core=episode();now=time.monotonic_ns()
    for _ in range(10):core.ingest('left','Aojiru',core.generation,1,now,'PASS')
    assert len(core.votes['left'])==1 and core.gate([step(closed=1.5)])['state']=='WAIT'
    votes(core,['PASS']*9);assert core.gate([step(closed=1.5)])['state']=='WAIT'
    votes(core,['PASS']*10,age=1.);assert core.gate([step(closed=1.5)])['state']=='WAIT'
    votes(core,['PASS']*10,generation='old');assert not core.votes['left']
    core.ingest('left','Aojiru',core.generation,10,now+10**12,'PASS');assert not core.votes['left']


def test_mask_ends_only_after_successful_approved_close_and_rearms_on_open():
    core=episode();core.mask_active=True;core.retry_hand='left';votes(core,['PASS']*10)
    action=step(closed=1.5)
    assert core.gate([action])['state']=='PASS' and core.mask_active
    sent={s:dict(positions=action[s][:7],gripper=action[s][7]) for s in ('left','right')}
    core.command_allowed(sent);core.record('a',action,sent)
    assert not core.mask_active and 'left' in core.closed_hands
    votes(core,['WRONG']*10);assert not core.votes['left']
    assert core.gate([action])['state']=='ORDINARY'
    old_ns=time.monotonic_ns();opened=step()
    core.record('b',opened,{'left':dict(positions=[0.]*7,gripper=0.)})
    core.ingest('left','Aojiru',core.generation,90,old_ns,'PASS')
    assert 'left' not in core.closed_hands and not core.votes['left']
    assert core.gate([action])['state']=='WAIT'


def test_final_publisher_boundary_refuses_unapproved_or_expired_close():
    core=episode();cmd={'left':dict(positions=[0.]*7,gripper=1.5)}
    with pytest.raises(ReverseError):core.command_allowed(cmd)
    votes(core,['PASS']*10);core.gate([step(closed=1.5)])
    for v in core.votes['left']:v['received_ns']-=2_000_000_000
    with pytest.raises(ReverseError):core.command_allowed(cmd)


def test_eight_chunks_include_partial_and_use_successful_targets():
    core=episode()
    for i in range(12):
        core.record(str(i),step(99),{'left':dict(positions=[i*.01]*7,gripper=0.),'right':dict(positions=[0.]*7,gripper=0.)})
    assert [g['chunk_id'] for g in core.history]==[str(i) for i in range(4,12)]
    assert [v['left'][0] for v in core.replay_plan()]==pytest.approx([i*.01 for i in range(11,2,-1)])


def test_release_at_historical_close_point_holds_other_hand():
    core=episode()
    for i,grip in enumerate([0.,1.5,1.5]):
        core.record('a',step(i*.01,grip),{'left':dict(positions=[i*.01]*7,gripper=grip),'right':dict(positions=[.1]*7,gripper=1.5)})
    plan=core.replay_plan();idx=next(i for i,p in enumerate(plan) if p.get('reverse_release'))
    assert plan[idx]['left'][:7]==plan[idx-1]['left'][:7]
    assert all(p['left'][7]==0. for p in plan[idx:])
    assert all(p['right']==[.1]*7+[1.5] for p in plan)
    core.replay_finished(plan[-1]);assert core.mask_active and core.retry_hand=='left'


def test_target_validation_and_no_history_retry_bound():
    with pytest.raises(ReverseError):prompt_targets('pick anything',NAMES)
    with pytest.raises(ReverseError):prompt_targets('Grasp unknown with the left hand.',NAMES)
    core=episode()
    for _ in range(3):assert core.replay_plan()==[];core.replay_finished()
    with pytest.raises(ReverseError):core.replay_plan()



@pytest.mark.parametrize('side', ['left', 'right'])
def test_open_wrong_hand_replays_closed_history_without_reclosing(side):
    core=ReverseEpisode(NAMES)
    core.begin('',MASK,targets={'left':'Aojiru','right':'Cola'})
    origin=step();origin['right'][7]=1.5
    core.anchor(origin)
    other='right' if side=='left' else 'left'
    for i,grip in enumerate([0.,1.5,1.5,0.]):
        action=step();action[side]=[.01*i]*7+[grip]
        action[other]=[.2]*7+[1.5]
        core.record(str(i),action,{s:dict(positions=action[s][:7],gripper=action[s][7])
                                   for s in ('left','right')},source='hold' if i==3 else 'vla')
    core.last_gate=dict(state='WRONG',hands={side:dict(passed=False,pass_count=0)})
    frozen=deepcopy(core.last_sent[other])
    expected=[g['before'][side][:7] for g in list(core.history)[:1]]
    expected += [v['action'][side][:7] for g in core.history for v in g['steps']]
    plan=core.replay_plan()
    assert [p[side][:7] for p in plan]==list(reversed(expected))
    assert all(p[side][7]==0. for p in plan)
    assert all(p[other]==frozen for p in plan)
    assert all(not p.get('reverse_release') for p in plan)
