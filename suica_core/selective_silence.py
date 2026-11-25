"""Window-level observable absence and migration. No imputed reading of missing speech."""

from collections import Counter, defaultdict
import numpy as np


def pool(events):
    """Rows are (actor, condition, signed scalar reading); one item is one arrival."""
    bank=defaultdict(list)
    for u,c,x in events:
        if not np.isfinite(x):
            raise ValueError("expected a finite observed reading")
        bank[u,c].append(float(x))
    return bank


def assess(history, current, target, *, decay=1., min_windows=1):
    """History is ordered equal-duration bins BEFORE current, including observed empty bins.

    One actor's target exposure weight is its past bin-level posting fraction.
    Caller must mark observation outages separately, never pass them as empty bins.
    """
    if not history or not 0<decay<=1 or min_windows<1:
        raise ValueError("positive history and valid exposure rule required")
    weights=Counter()
    sums=Counter()
    support=Counter()
    shares=Counter()
    active_weight=Counter()
    denom=sum(decay**i for i in range(len(history)))
    elsewhere=Counter()
    for age,events in enumerate(reversed(history)):
        bank=pool(events)
        actors=Counter()
        for (u,c),vals in bank.items():
            actors[u]+=len(vals)
            if c!=target:
                elsewhere[u]+=1
        for u,total in actors.items():
            w=decay**age
            active_weight[u]+=w
            vals=bank.get((u,target),[])
            shares[u]+=w*len(vals)/total
            if vals:
                weights[u]+=w
                sums[u]+=w*np.mean(vals)
                support[u]+=1
    cur=pool(current)
    totals=Counter()
    for (u,c),vals in cur.items():
        totals[u]+=len(vals)
    eligible=[u for u in weights if support[u]>=min_windows]
    W=sum(weights[u]/denom for u in eligible)
    channels=Counter()
    signed=0.
    bridge=[]
    for u in eligible:
        wt=weights[u]/denom
        nt=len(cur.get((u,target),[]))
        label="present" if nt else "active_elsewhere" if totals[u] else "not_observed"
        channels[label]+=wt
        if label=="active_elsewhere":
            signed+=wt*sums[u]/weights[u]
        if totals[u] and elsewhere[u]:
            bridge.append(nt/totals[u]-shares[u]/active_weight[u])
    return {"eligible":len(eligible),"exposure_mass":W,
            **{k:channels[k]/W if W else None for k in ("present","active_elsewhere","not_observed")},
            "signed_elsewhere_flux":signed/W if W else None,
            "bridge_active":len(bridge),"bridge_share_shift":float(np.mean(bridge)) if bridge else None}


def adjust_volume(source_shifts,source_volumes,shift,volume):
    """Freeze a source linear volume baseline; residuals needn't be orthogonal out of sample."""
    s,n=np.asarray(source_shifts,float),np.asarray(source_volumes,float)
    if (s.ndim!=1 or s.shape!=n.shape or len(s)<3 or np.any(n<0) or volume<0
            or not np.isfinite([s,n]).all() or not np.isfinite([volume,shift]).all()
            or np.ptp(n)==0):
        raise ValueError("distinct finite calibration volumes required")
    a,b=np.linalg.lstsq(np.column_stack([np.ones(len(n)),np.log1p(n)]),s,rcond=None)[0]
    return float(shift-a-b*np.log1p(volume))
