import numpy as np
import pytest

from grain_core import liquid_common_range as cr, liquid_reference_error as population
from grain_core import open_cohort as co, selective_silence as ss, stream_precision as sp
from grain_core import stream_readout as sr
from grain_core.dual_view_increment import assess_precision,compare
from grain_core.liquid_axis_series import encode_bipolar_origin
from grain_core.riccati_mobius import ConstrainedRiccatiCell, stack_transition_contrasts


def test_half_variance_counts_and_frozen_chart_units():
    rng=np.random.default_rng(13)
    n=np.tile([1,10],1000)
    mean=rng.normal(size=(len(n),1))
    a,b=(mean+rng.normal(size=mean.shape)/np.sqrt(n[:,None]) for _ in range(2))
    v=sp.split_event_variance(a,b,n,n)
    np.testing.assert_allclose(v,1,atol=.1)
    ref=sp.eb_fit(a,n,v)
    z=sp.eb_apply([[4],[4]],[1,30],ref)
    assert z[0]<z[1]
    chart=population.fit_chart(a,["a"],population="units",basis="signed",counts=n,event_var=v)
    with pytest.raises(ValueError):
        population.locate(chart,[[4]],axes=["a"],basis="signed")
    _,dist=population.locate(chart,[[4],[4]],axes=["a"],basis="signed",counts=[1,30])
    assert dist[0]<dist[1]
    with pytest.raises(ValueError):
        sp.eb_apply([[1]],[0],ref)


def test_frozen_common_range_uses_same_scale_on_application():
    rng=np.random.default_rng(21)
    x=rng.normal(size=(20,5,2))
    x[:,:,1]+=x[:,:,0]
    n=np.ones((20,5))
    ref=cr.fit(x,["a","b"],1,"test","event",permutations=19,counts=[n,n],event_var=[[.1],[.2]])
    budget=cr.calibrate(ref,x,counts=[n,n])
    out=cr.estimate(ref,[[0],[0]],budget,counts=[1,2])
    assert len(out["canonical_observations"])==2
    with pytest.raises(ValueError):
        cr.estimate(ref,[[0],[0]],budget)


def test_gls_and_transition_errors_include_both_endpoints():
    r=.8
    np.testing.assert_allclose(sp.transition_scale([3,1],r,0),np.hypot(1,3*r))
    H,b=stack_transition_contrasts([[[2]],[[3]],[[4]]],np.ones((2,1,1)),np.zeros((2,1,1,1)),contrast_error=[2,4])
    np.testing.assert_allclose(H.ravel(),[.5,.25]);np.testing.assert_allclose(b,[1.5,1])
    A=np.eye(2); cov=np.array([[4.,1.],[1.,2.]])
    X,_=sp.whiten_system(A,np.ones(2),cov)
    np.testing.assert_allclose(X.T@X,np.linalg.inv(cov))
    with pytest.raises(ValueError):
        sp.path_quantile([[1]],[[1]],.95)


def test_equal_widths_same_comparator_and_sampling_status():
    cfg=dict(view1=[0,1],view2=[0,-1],half_widths=[0,0],rates=[0,0,0],c_bound=20,calibration_id="source")
    a=compare(**cfg,budgets=[0,2,2]);b=compare(**cfg,budgets=[[0],[2],[2]])
    assert a==b
    assert assess_precision(**cfg,structural_budgets=[0,0,0],multiplier=2,se1=[1,1],se2=[1,1])["status"]=="sampling_underpowered"
    assert assess_precision(**cfg,structural_budgets=[0,0,0],multiplier=2,se1=[0,0],se2=[0,0])["status"]=="infeasible_under_declared_bounds"


@pytest.mark.parametrize("clock",["jump","ending_pulse"])
def test_event_clock_scan_suffix_idle_and_negative_symmetry(clock):
    c=ConstrainedRiccatiCell(2,3)
    r=np.random.default_rng(1)
    u=r.normal(size=(100,2))
    t=np.cumsum(r.choice([0,.1,5,100],len(u)))
    cfg=dict(start=0,cutoff=t[-1]+5,pulse=.5,clock=clock)
    x,e=c.event_path(u,t,**cfg)
    y,f=c.event_path(u,t,parallel=False,**cfg)
    np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)
    np.testing.assert_allclose(e,f,atol=1e-12,rtol=0)
    assert (np.abs(x)<=1).all()
    assert (c.counterfactual_window_delta(u,t,window=1,**cfg)==0).all()
    z,e2=c.event_path(np.vstack([u,[999,999]]),np.r_[t,t[-1]+9],**cfg)
    np.testing.assert_array_equal(e,e2)
    p=encode_bipolar_origin(u); q=encode_bipolar_origin(-u)
    np.testing.assert_allclose(p[:,:2],q[:,2:4],atol=1e-15)
    assert not np.allclose(c.event_path(u[:1],[0],start=0,cutoff=0)[1],0)


def test_absence_is_partition_of_expected_mass_not_unseen_states():
    past=[[("u","x",2),("v","x",-1),("w","x",0),("u","y",0)]]
    r=ss.assess(past,[("u","y",9),("w","x",8)],"x")
    assert r["present"]==r["active_elsewhere"]==r["not_observed"]==1/3
    assert r["signed_elsewhere_flux"]==2/3
    assert r["bridge_share_shift"]==-.5
    assert ss.assess(past,[],"z")["present"] is None
    s=np.log1p([1,2,3])+10
    assert abs(ss.adjust_volume(s,[1,2,3],10+np.log1p(9),9))<1e-11


def test_cohort_reweight_and_no_reference_and_prefix():
    d=co.decompose({"u":(9,0),"v":(1,1)},{"u":(1,0),"v":(9,1)})
    assert d["within"]==d["entry"]==d["exit"]==0
    assert d["stayer_reweight"]==pytest.approx(.8)
    assert co.decompose({"u":(1,0)},{"v":(1,2)})["status"]=="no_stayer_reference"
    def run(xs):
        filt=co.PrefixHygiene(max_prior_repeats=1)
        return [filt.observe(i,"u",v) for i,v in enumerate(xs)]
    assert run("aab") == run("aabaaaa")[:3]
    assert not run("aa")[-1]["keep"]


def test_rubric_crossfit_and_unknown_transport():
    x=np.tile([[1,-1],[1,1],[-1,-1],[-1,1]],(16,1))
    f=np.arange(len(x))//4%4
    y=x[:,0]*x[:,1]
    o=sr.crossfit_register(x[:,:1],x[:,1:],y,f,ridge=.1)
    a=sr.crossfit_register(x[:,:1],x[:,1:],y,f,ridge=.1,interact=False)
    assert np.mean((y-o)**2)<.001 and np.mean((y-a)**2)>.9
    with pytest.raises(ValueError):
        sr.crossfit_register(x[:,:1],x[:,1:],y,f,authors=["one"]*len(f))
    ref=sr.act_directions({"promotion":[[1,2]]},{"promotion":[[0,2]]})
    assert "ownership" in ref["missing"] and ref["axes"]==["promotion"]
    labels=["x"]*4+["y"]*4
    gate=sr.transport_gate([1]*8,[0]*4+[2]*4,list(range(4))*2,labels)
    assert gate["x"]["carry"] and not gate["y"]["carry"]
    s=sr.ScopedInnovation(1,gate)
    assert s.observe(0,"u","x",1)["prior"]==s.observe(1,"u","y",1)["prior"]==0
    assert s.observe(1,"u","x",1)["innovation"]==.5
