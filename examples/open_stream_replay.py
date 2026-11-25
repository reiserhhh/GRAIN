"""Reproduce aggregate-only G1-G6 checks; never emits raw dialogue or participant keys."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from threadpoolctl import threadpool_limits

from grain_core import open_cohort as co, selective_silence as ss, stream_precision as sp
from grain_core import stream_readout as sr
from grain_core.riccati_mobius import ConstrainedRiccatiCell,estimate_cross_view_lag1_rates


def simulation():
    rng=np.random.default_rng(1619)
    N,T,K=400,36,1
    hidden=rng.normal(size=(3,N,T,K))
    rho=.72
    for i in range(1,T):
        hidden[:,:,i]=rho*hidden[:,:,i-1]+np.sqrt(1-rho*rho)*hidden[:,:,i]
    n=rng.choice([1,50],size=(2,N,T,K))
    se=2.5/np.sqrt(n)
    a,b=hidden[0]+hidden[1:]+rng.normal(size=se.shape)*se
    raw=np.ravel(estimate_cross_view_lag1_rates(a,b,1))
    adj=sp.lag_rates(a,b,*se).ravel()
    # Reuse one independent noise realization per endpoint (adjacent contrasts correlate).
    ns=rng.choice([1,50],size=(1800,T))
    std=2.5/np.sqrt(ns)
    eps=rng.normal(size=std.shape)*std
    delta=eps[:,1:]-rho*eps[:,:-1]
    scale=sp.transition_scale(std,rho,0.)
    q=sp.path_quantile(delta[:600],scale[:600])
    hit=(np.abs(delta[600:])<=q*scale[600:]).all(1)
    # Nested random subset under iid event errors: Cov(full, subset)=Var(full).
    full,sub=1.,2.
    shared_err=rng.normal(size=a.shape)*full
    total=hidden[0]+hidden[1]+shared_err
    selected=hidden[0]+hidden[2]+shared_err+rng.normal(size=b.shape)*np.sqrt(sub**2-full**2)
    std1,std2=np.full(a.shape,full),np.full(b.shape,sub)
    diag=sp.lag_rates(total,selected,std1,std2)
    joint=sp.lag_rates(total,selected,std1,std2,cross_error_cov=std1**2)
    return dict(true_rate=rho,raw_rates=raw.tolist(),corrected_rates=adj.tolist(),
                nested_diagonal_common=float(diag[0,0]),nested_overlap_common=float(joint[0,0]),
                held_path_coverage=float(hit.mean()),calibration_paths=600,held_paths=len(hit))


def panda(meta,paths):
    with np.load(paths) as f:
        src=f["source_indices"]
    with np.load(meta) as f:
        t,c,l=[f[k][src] for k in ("times","venue_codes","lengths")]
    if t.shape!=(386,36):
        raise ValueError("unexpected SOURCE panel")
    binid=((t-t.min())//(30*86400)).astype(int)
    cut=int(binid.max()+1)//3
    targets=[k for k,n in Counter(c[binid<cut]).most_common(5)]
    center=np.log1p(l[binid<cut]).mean()
    bins=[[] for _ in range(binid.max()+1)]
    for u in range(len(src)):
        for k,venue,z in zip(binid[u],c[u],np.log1p(l[u])-center,strict=True):
            bins[k].append((u,int(venue),float(z)))
    def panel(rows,v):
        return {u:(len(vals),np.mean(vals)) for (u,k),vals in ss.pool(rows).items() if k==v}
    parts=[]
    rows=[]
    skipped=0
    for v in targets:
        for i in range(cut,len(bins)-1):
            old,new=panel(bins[i-1],v),panel(bins[i],v)
            if old and new:
                d=co.decompose(old,new)
                if d["status"]=="decomposed":
                    parts.append(d)
                else:
                    skipped+=1
            r=ss.assess(bins[i-3:i],bins[i],v)
            z=[z for _,k,z in bins[i] if k==v]
            ns=[z for _,k,z in bins[i+1] if k==v]
            if r["eligible"] and z and ns:
                rows.append((i,int(v),np.mean(z),np.log1p(len(z)),r["signed_elsewhere_flux"],
                             np.mean(ns)-np.mean(z),r["active_elsewhere"],r["not_observed"]))
    ar=np.asarray(rows)
    # Fixed time split, embargo one bin: no training label overlaps first held bin.
    boundary=2*(binid.max()+1)//3
    tr=ar[:,0]<boundary-1
    te=ar[:,0]>=boundary
    def mse(cols):
        X=np.column_stack((np.ones(len(ar)),ar[:,cols],*(ar[:,1]==k for k in targets[1:])))
        coef=np.linalg.lstsq(X[tr],ar[tr,5],rcond=None)[0]
        return float(np.mean((ar[te,5]-X[te]@coef)**2))
    return dict(source_authors=len(src),events=t.size,bins=len(bins),eligible_pairs=len(parts),
                no_stayer_pairs=skipped,ranked_venues=len(targets),held_rows=int(te.sum()),fit_rows=int(tr.sum()),
                closure=max(abs(d["closure"]) for d in parts),
                sign_reversals=sum(int(d["change"]*d["within"]<0) for d in parts),
                stayer_reweight_abs_mean=np.mean([abs(d["stayer_reweight"]) for d in parts]),
                held_active_elsewhere=float(ar[te,6].mean()),held_not_observed=float(ar[te,7].mean()),
                next_log_length_change_mse=mse([2,3]),adding_signed_flux_mse=mse([2,3,4]))


def rpc(path):
    cell=ConstrainedRiccatiCell(1,2)
    diffs=[]
    errors=[]
    times=[]
    cold=[]
    own=[]
    groups=[]
    flags=[]
    missing=0
    with zipfile.ZipFile(path) as z:
        names=sorted((n for n in z.namelist() if "/dialogues/" in n and n.endswith(".json")),
                     key=lambda n:sha256(n.encode()).digest())[:512]
        for group,name in enumerate(names):
            turns=json.loads(z.read(name))["utterances"]
            if any(r.get("timestamp") in ("NaT", None, "") for r in turns):
                missing+=1
                continue
            ts=np.array([(datetime.fromisoformat(r["timestamp"])-datetime(1970,1,1)).total_seconds() for r in turns])
            if len(ts)<2 or (np.diff(ts)<0).any():
                continue
            times.extend(np.diff(ts))
            tm=(ts-ts[0])/60
            x=np.array([np.log1p(len(r["text"]))-3 for r in turns])[:,None]
            j,end=cell.event_path(x,tm,start=0,cutoff=tm[-1],pulse=.25)
            _,seq=cell.event_path(x,tm,start=0,cutoff=tm[-1],pulse=.25,parallel=False)
            errors.append(np.max(abs(seq-end)))
            held=cell.rollout_sequential(x,np.diff(tm,prepend=0))
            diffs.append(abs(j[-1]-held[-1]).mean())
            # Test last same-speaker reading within this dialogue vs fixed cold level.
            mem=sr.ScopedInnovation(2.,{"dialogue":{"carry":True}})
            for r,t,v in zip(turns,tm,x.ravel(),strict=True):
                rec=mem.observe(t,r["interlocutor_id"],"dialogue",v)
                if rec["carried"]:
                    cold.append(v*v)
                    own.append(rec["innovation"]**2)
                    groups.append(group)
                    flags.append(group<256)
    train=np.array(flags)
    cold,own,groups=map(np.asarray,(cold,own,groups))
    gate=sr.transport_gate(cold[train],own[train],groups[train],np.full(sum(train),"dialogue"))["dialogue"]
    return dict(sample_dialogues=len(names),missing_clock=missing,ordered_dialogues=len(errors),median_gap_seconds=float(np.median(times)),
                max_scan_error=max(errors),median_abs_endpoint_difference=float(np.median(diffs)),
                gate_cal=gate,held_cold_mse=float(cold[~train].mean()),held_carry_mse=float(own[~train].mean()),
                held_readings=int(sum(~train)))


def esconv(path):
    docs=json.loads(Path(path).read_text())
    text=[]; p=[]; y=[];groups=[]
    axes=["Question","Self-disclosure","Providing Suggestions"]
    for i,d in enumerate(docs):
        for r in d["dialog"]:
            act=r.get("annotation",{}).get("strategy")
            if act:
                line=r["content"]
                text.append(line)
                p.append(sr.surface(line,reply=True,quoted='"' in line))
                y.append([float(act==k) for k in axes]); groups.append(i)
    z=HashingVectorizer(n_features=64,alternate_sign=True,ngram_range=(1,2)).transform(text).toarray()
    f=np.asarray(groups)%5
    y=np.array(y)
    def oof(interact):
        pred=sr.crossfit_register(z,p,y,f,ridge=10,interact=interact,authors=groups)
        return np.mean((pred-y)**2,axis=0).tolist()
    return dict(dialogues=len(docs),annotated_turns=len(y),axes=axes,group="dialogue, not author",
                additive_mse=oof(False),interaction_mse=oof(True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("metadata","indices","rpc","esconv","output"):
        parser.add_argument("--"+k,required=True,type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        result=dict(synthetic=simulation(),pandora_source=panda(args.metadata,args.indices),
                    real_persona_chat=rpc(args.rpc),esconv=esconv(args.esconv))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False,sort_keys=True)+"\n")
