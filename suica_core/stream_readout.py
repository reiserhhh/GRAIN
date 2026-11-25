"""Register-conditioned frozen readouts and prospective condition-scoped innovations."""

from dataclasses import dataclass
import unicodedata

import numpy as np
from scipy.stats import t as student


ACT_AXES=("orientation","ownership","commitment","promotion","conditionality","retrospective","forward")


def act_directions(positive,negative):
    """Build independent *contrasts* from source exemplars; no forced orthogonality.

    Inputs are per-axis lists of (examples, embedding dim) arrays. Unknown when
    the rubric supplies no anchors. Returns direction plus overlap for auditing.
    """
    rows=[]
    for axis in ACT_AXES:
        if axis not in positive or axis not in negative:
            continue
        a,b=np.asarray(positive[axis],float),np.asarray(negative[axis],float)
        if (a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or not len(a) or not len(b)
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError("paired finite source exemplars needed")
        v=a.mean(0)-b.mean(0)
        if np.linalg.norm(v)>1e-12:
            rows.append((axis,v/np.linalg.norm(v)))
    if not rows:
        raise ValueError("no usable contrast")
    names,v=zip(*rows,strict=True)
    mat=np.array(v)
    return dict(axes=list(names),missing=[k for k in ACT_AXES if k not in names],directions=mat,overlap=mat@mat.T)


def surface(text, *, reply=False, quoted=False):
    """Observable covariates, not psychological labels or inferred language identity."""
    scripts=[unicodedata.name(ch,"").split(" ")[0] for ch in text if ch.isalpha()]
    return np.array([np.log1p(len(text)),float(reply),float(quoted),
                     sum(s in {"HIRAGANA","KATAKANA","CJK"} for s in scripts)/max(1,len(scripts))])


@dataclass
class RegisterProbe:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    interact: bool

    def design(self,z,p):
        z,p=np.asarray(z,float),np.asarray(p,float)
        if (z.ndim!=2 or p.shape!=(len(z),len(self.mean)) or not np.isfinite(z).all()
                or not np.isfinite(p).all()):
            raise ValueError("aligned finite embedding and register matrices required")
        p=(p-self.mean)/self.scale
        parts=[np.ones((len(z),1)),z,p]
        if self.interact:
            parts.append((z[:,:,None]*p[:,None,:]).reshape(len(z),-1))
        return np.column_stack(parts)

    def predict(self,z,p):
        return self.design(z,p)@self.weights

    @classmethod
    def fit(cls,z,p,y,*,ridge=1.,interact=True):
        p,y=np.asarray(p,float),np.asarray(y,float)
        if (p.ndim!=2 or not len(p) or y.ndim not in (1,2) or len(y)!=len(p)
                or not np.isfinite(y).all() or not np.isfinite(ridge) or ridge<=0):
            raise ValueError("source labels and positive regularization required")
        sd=p.std(0)
        model=cls(p.mean(0),np.where(sd>1e-10,sd,1),np.empty(0),interact)
        X=model.design(z,p)[:,1:]
        xm,ym=X.mean(0),y.mean(0)
        X=X-xm
        if X.shape[1]<=len(X):
            coef=np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@(y-ym))
        else:
            coef=X.T@np.linalg.solve(X@X.T+ridge*np.eye(len(X)),y-ym)
        model.weights=np.concatenate([np.expand_dims(ym-xm@coef,0),coef],axis=0)
        return model


def crossfit_register(z,p,y,folds,*,ridge=1.,interact=True,authors=None):
    """Freeze folds by author before scoring; each fold refits all source parameters."""
    z,p,y,f=np.asarray(z),np.asarray(p),np.asarray(y),np.asarray(folds)
    if f.shape!=(len(y),) or min(len(z),len(p))!=len(y) or len(z)!=len(p) or len(np.unique(f))<2:
        raise ValueError("aligned source arrays and >=2 folds required")
    if authors is not None:
        if len(authors)!=len(f) or len(set(zip(authors,f)))!=len(set(authors)):
            raise ValueError("author crosses folds")
    pred=np.empty_like(y,dtype=float)
    for k in np.unique(f):
        tr=f!=k
        pred[~tr]=RegisterProbe.fit(z[tr],p[tr],y[tr],ridge=ridge,interact=interact).predict(z[~tr],p[~tr])
    return pred


def transport_gate(fresh_loss,carry_loss,units,conditions,*,alpha=.05):
    """Source-only, independent-unit paired improvement; simultaneous lower t bounds.

    Approximate inference on mean per-unit gains; never fit the gate on the application fold.
    """
    a,b,u,c=map(np.asarray,(fresh_loss,carry_loss,units,conditions))
    if (a.ndim!=1 or not len(a) or not all(v.shape==a.shape for v in (b,u,c))
            or not np.isfinite([a,b]).all() or not 0<alpha<1):
        raise ValueError("paired source losses and unit/condition IDs required")
    d=a-b
    conds=np.unique(c)
    rows={}
    for k in conds:
        sel=c==k
        gains=np.array([d[sel & (u==i)].mean() for i in np.unique(u[sel])])
        lo=None
        if len(gains)>=3:
            lo=float(gains.mean()-student.ppf(1-alpha/len(conds),len(gains)-1)*gains.std(ddof=1)/np.sqrt(len(gains)))
        rows[k]=dict(units=len(gains),gain=float(gains.mean()),lower=lo,carry=lo is not None and lo>0)
    return rows


class ScopedInnovation:
    """Past-only, (actor, condition)-bound state with explicit idle half-life and cold baseline."""
    def __init__(self,half_life,gate,*,cold=0.):
        if not np.isfinite([half_life,cold]).all() or half_life<=0:
            raise ValueError("invalid state clock or cold start")
        self.half_life,self.gate,self.cold=half_life,dict(gate),float(cold)
        self.memory={}
        self.last=-np.inf

    def observe(self,time,actor,condition,value):
        if not np.isfinite([time,value]).all() or time<self.last:
            raise ValueError("monotone finite stream required")
        key=actor,condition
        prev=self.memory.get(key)
        allowed=prev is not None and self.gate.get(condition,{}).get("carry",False)
        pred=self.cold
        if allowed:
            t,z=prev
            pred+=(z-self.cold)*2**(-(time-t)/self.half_life)
        self.last=time
        self.memory[key]=time,value
        return dict(carried=bool(allowed),prior=float(pred),innovation=float(value-pred))
