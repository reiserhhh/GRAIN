"""Exact cohort accounting and forward-only event hygiene."""

from collections import Counter, defaultdict, deque
from hashlib import blake2b
import numpy as np


def decompose(before,after):
    """Mapping actor -> (positive exposure mass, observed mean) in each period."""
    def unpack(panel):
        if not panel or any(not np.isfinite([n,z]).all() or n<=0 for n,z in panel.values()):
            raise ValueError("positive observed masses and finite means required")
        N=sum(n for n,z in panel.values())
        return {u:(n/N,z) for u,(n,z) in panel.items()}
    a,b=unpack(before),unpack(after)
    common=a.keys() & b.keys()
    change=sum(w*z for w,z in b.values())-sum(w*z for w,z in a.values())
    if not common:
        return dict(status="no_stayer_reference", change=change)
    def stayer(p):
        s=sum(p[u][0] for u in common)
        w={u:p[u][0]/s for u in common}
        return w,sum(w[u]*p[u][1] for u in common)
    wa,ma=stayer(a)
    wb,mb=stayer(b)
    within=sum((wa[u]+wb[u])/2*(b[u][1]-a[u][1]) for u in common)
    reweight=sum((wb[u]-wa[u])*(a[u][1]+b[u][1])/2 for u in common)
    entry=sum(b[u][0]*(b[u][1]-mb) for u in b.keys()-common)
    exit_=sum(a[u][0]*(ma-a[u][1]) for u in a.keys()-common)
    return dict(status="decomposed",change=change,within=within,stayer_reweight=reweight,
                entry=entry,exit=exit_,closure=change-within-reweight-entry-exit_)


class PrefixHygiene:
    """Before-arrival actor counts; ties follow declared ingestion order. All arrivals count."""
    def __init__(self,span=60.,max_prior_repeats=2,max_prior_burst=10):
        if not np.isfinite(span) or span<=0 or min(max_prior_repeats,max_prior_burst)<1:
            raise ValueError("invalid filter limits")
        self.span,self.repeats,self.burst=span,max_prior_repeats,max_prior_burst
        self.last=-np.inf
        self.by_actor=Counter()
        self.by_text=Counter()
        self.recent=defaultdict(deque)

    def observe(self,time,actor,text):
        if not np.isfinite(time) or time<self.last or not isinstance(text,str):
            raise ValueError("monotone times and text required")
        self.last=time
        key=actor,blake2b(text.encode(),digest_size=16).digest()
        q=self.recent[actor]
        while q and q[0]<=time-self.span:
            q.popleft()
        n,k,b=self.by_actor[actor],self.by_text[key],len(q)
        self.by_actor[actor]+=1
        self.by_text[key]+=1
        q.append(time)
        return dict(prior_count=n,prior_identical=k,prior_burst=b,prior_share=k/n if n else None,
                    keep=k<self.repeats and b<self.burst)
