from __future__ import annotations

"""Systematic NSGA-II family development.

NSGA-II remains the proposal/survival authority. The frozen landscape augments
only offspring generation. All methods share the same first 16 exact Sobol calls.
After call 16, native NSGA-II uses population size 16, binary tournament, SBX
(p=0.90, eta=20), polynomial mutation (eta=20, p=1/d), feasibility-aware
nondominated sorting and crowding-distance elitist survival.

Lineages:
LEGACY_OPERATOR: historical parent-first NSGA operator pattern. A 768-row
  structural pool may replace at most 2 of 4 inherited native offspring using a
  frozen global-surrogate value/feasibility gate.
TRUE_NATIVE_V111: native 50% guided-offspring mix; 2 of 4 native offspring are
  protected and up to 2 are replaced by structural offspring with no universal
  post-gate.
SYSTEMATIC_A: same protected 50% native contract, advanced GHOST/SPARC/TRAIL
  geometry, and a generic frozen-surrogate preference-noninferiority admission
  check for the guidable slots. Exact NSGA rank/crowding still decides survival.

There are no species or problem-ID branches in this module.
"""
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from runtime import sobol_points, key_for_x

BASE_NO50K='BASE_NSGA2_NO50K'
BASE_50K='BASE_NSGA2_50K'
GHOST_A='DEV_NSGA50K_GHOST_A'
SPARC_LEG='DEV_NSGA50K_SPARC_LEGACY_OPERATOR'
TRAIL_LEG='DEV_NSGA50K_TRAIL_LEGACY_OPERATOR'
METHODS=[BASE_NO50K,BASE_50K,GHOST_A,SPARC_LEG,TRAIL_LEG]
HEURISTIC={GHOST_A:'ghost',SPARC_LEG:'sparc',TRAIL_LEG:'trail'}
LINEAGE={GHOST_A:'SYSTEMATIC_A',SPARC_LEG:'LEGACY_OPERATOR',TRAIL_LEG:'LEGACY_OPERATOR'}
GHOST=GHOST_A
SPARC=SPARC_LEG
TRAIL=TRAIL_LEG

def _bool(s):
    if pd.api.types.is_bool_dtype(s): return s.to_numpy(bool)
    return s.astype(str).str.lower().isin(['true','1','yes','optimal','feasible']).to_numpy(bool)

def zero_outcomes(Y,F):
    Y=np.asarray(Y,float);F=np.asarray(F,bool).reshape(-1);out=np.zeros_like(Y,float);ok=F & np.isfinite(Y).all(axis=1);out[ok]=Y[ok];return out

def _pareto_mask(Y):
    Y=np.asarray(Y,float);n=len(Y);keep=np.ones(n,bool)
    for i in range(n):
        if not keep[i]: continue
        dom=np.all(Y>=Y[i],axis=1)&np.any(Y>Y[i],axis=1)
        if np.any(dom): keep[i]=False
    return keep

def _pareto_representative_ids(Y,good_ids,cap):
    if cap<=0 or len(good_ids)==0:return []
    ids=np.asarray(good_ids,int);Yg=np.asarray(Y,float)[ids];front=ids[_pareto_mask(Yg)]
    if len(front)>cap: front=front[np.linspace(0,len(front)-1,cap,dtype=int)]
    return list(map(int,front))

def representative_prior_ids(X,Y,F,n,seed):
    X=np.asarray(X,float);Y=np.asarray(Y,float);F=np.asarray(F,bool);n=min(int(n),len(X));good=np.where(F & np.isfinite(Y).all(1))[0];mandatory=[]
    if len(good):
        per=max(8,min(32,max(8,n//max(8,4*Y.shape[1]))))
        for j in range(Y.shape[1]):
            order=good[np.argsort(Y[good,j],kind='mergesort')];mandatory.extend(map(int,order[-min(per,len(order)):]))
        mandatory.extend(_pareto_representative_ids(Y,good,min(256,max(32,n//4))))
    mandatory=list(dict.fromkeys(mandatory))[:min(n,384)];used=set(mandatory);need=n-len(mandatory)
    if need>0:
        tree=cKDTree(X);targets=sobol_points(max(512,need*6),X.shape[1],int(seed)+17);k=min(32,len(X));nbr=np.asarray(tree.query(targets,k=k)[1],int)
        if nbr.ndim==1:nbr=nbr[:,None]
        for row in nbr:
            for jj in np.atleast_1d(row):
                j=int(jj)
                if j not in used: used.add(j);mandatory.append(j);break
            if len(mandatory)>=n:break
    if len(mandatory)<n:
        for j in range(len(X)):
            if j not in used:used.add(j);mandatory.append(j)
            if len(mandatory)>=n:break
    ids=np.asarray(mandatory[:n],int)
    if len(ids)!=n or len(np.unique(ids))!=n: raise RuntimeError('representative prior construction failed')
    return ids

def _history_xy_feasible(ctx,method):
    h=ctx.history(method)
    if not len(h):return np.empty((0,len(ctx.features))),np.empty((0,ctx.objective_dim))
    xc=[f'x01__{f}' for f in ctx.features];ok=_bool(h['exact_feasible']);X=h.loc[ok,xc].to_numpy(float);Y=h.loc[ok,ctx.objective_norm_columns].to_numpy(float);g=np.isfinite(X).all(1)&np.isfinite(Y).all(1);return X[g],Y[g]

def _simplex_weight(m,seed):
    if int(m)==1:return np.ones(1,float)
    return np.random.default_rng(int(seed)).dirichlet(np.ones(int(m)))

def _u(Y,w):
    Y=np.asarray(Y,float);w=np.asarray(w,float).reshape(-1);z=w[None,:]*Y
    return np.min(z,axis=1)+0.05*np.sum(z,axis=1)

def _dedup_rows(X,Y,F,decimals=12):
    X=np.asarray(X,float);Y=np.asarray(Y,float);F=np.asarray(F,bool);last={}
    for i,x in enumerate(X):last[key_for_x(x,decimals)]=i
    ids=np.asarray(sorted(last.values()),int);return X[ids],Y[ids],F[ids]

def _rank_and_crowding(Y,F):
    Y=np.asarray(Y,float);F=np.asarray(F,bool);n=len(Y);rank=np.full(n,10**9,int);crowd=np.zeros(n,float)
    feasible=np.where(F & np.isfinite(Y).all(1))[0]
    remaining=list(map(int,feasible));r=0
    while remaining:
        A=np.asarray(remaining,int);front=[]
        for i in A:
            others=A[A!=i]
            dominated=False
            if len(others): dominated=bool(np.any(np.all(Y[others]>=Y[i],axis=1)&np.any(Y[others]>Y[i],axis=1)))
            if not dominated:front.append(int(i))
        if not front:break
        rank[front]=r
        if len(front)<=2:crowd[front]=np.inf
        else:
            ff=np.asarray(front,int);m=Y.shape[1]
            for j in range(m):
                order=ff[np.argsort(Y[ff,j],kind='mergesort')];crowd[order[0]]=crowd[order[-1]]=np.inf
                lo=float(Y[order[0],j]);hi=float(Y[order[-1],j]);den=max(1e-15,hi-lo)
                for k in range(1,len(order)-1):
                    if np.isfinite(crowd[order[k]]):crowd[order[k]]+=(Y[order[k+1],j]-Y[order[k-1],j])/den
        S=set(front);remaining=[i for i in remaining if i not in S];r+=1
    bad=np.where(~(F & np.isfinite(Y).all(1)))[0]
    if len(bad):rank[bad]=r+1;crowd[bad]=0.0
    return rank,crowd

def _crowding_for_front(Y,ids):
    ids=np.asarray(ids,int);c=np.zeros(len(ids),float)
    if len(ids)<=2:return np.full(len(ids),np.inf)
    for j in range(Y.shape[1]):
        oo=np.argsort(Y[ids,j],kind='mergesort');c[oo[0]]=c[oo[-1]]=np.inf;lo=float(Y[ids[oo[0]],j]);hi=float(Y[ids[oo[-1]],j]);den=max(1e-15,hi-lo)
        for k in range(1,len(oo)-1):
            if np.isfinite(c[oo[k]]):c[oo[k]]+=(Y[ids[oo[k+1]],j]-Y[ids[oo[k-1]],j])/den
    return c

def _survival(X,Y,F,pop=16):
    # Efficient elitist selection. For large frozen-prior pools we compute only
    # as many Pareto fronts as are required to fill the 16-member population.
    X,Y,F=_dedup_rows(X,Y,F);pop=min(int(pop),len(X));selected=[];remaining=np.where(F & np.isfinite(Y).all(1))[0]
    while len(selected)<pop and len(remaining):
        A=np.asarray(remaining,int);Ya=Y[A];n=len(A)
        if n<=96:
            front=[]
            for ii,i in enumerate(A):
                oth=np.delete(Ya,ii,axis=0);dom=bool(len(oth) and np.any(np.all(oth>=Y[i],axis=1)&np.any(oth>Y[i],axis=1)))
                if not dom:front.append(int(i))
            front=np.asarray(front,int)
        else:
            # block the pairwise dominance test to avoid an NxNxM memory spike
            dominated=np.zeros(n,bool);block=192
            for a in range(0,n,block):
                Q=Ya[a:a+block];ge=np.all(Ya[None,:,:]>=Q[:,None,:],axis=2);gt=np.any(Ya[None,:,:]>Q[:,None,:],axis=2);dominated[a:a+len(Q)]=np.any(ge&gt,axis=1)
            front=A[~dominated]
        need=pop-len(selected)
        if len(front)<=need:selected.extend(map(int,front))
        else:
            c=_crowding_for_front(Y,front);order=sorted(range(len(front)),key=lambda k:(-float(c[k]) if np.isfinite(c[k]) else -1e300,int(front[k])));selected.extend(int(front[k]) for k in order[:need])
        fs=set(map(int,front));remaining=np.asarray([i for i in A if int(i) not in fs],int)
    if len(selected)<pop:
        bad=[int(i) for i in range(len(X)) if int(i) not in set(selected)];selected.extend(bad[:pop-len(selected)])
    ids=np.asarray(selected[:pop],int);return X[ids],Y[ids],F[ids]

def _initial_population(ctx,method,use_archive):
    h=ctx.history(method);init=int(ctx.cfg['nsga_dev']['initial_calls']);h=h.iloc[:min(init,len(h))].copy();xc=[f'x01__{f}' for f in ctx.features]
    Xl=h[xc].to_numpy(float);Yl=h[ctx.objective_norm_columns].to_numpy(float);Fl=_bool(h['exact_feasible']) & np.isfinite(Yl).all(1)
    if not use_archive:return _survival(Xl,Yl,Fl,ctx.cfg['nsga_dev']['population_size'])
    ids=np.asarray(ctx.subset_local,int);Xa=ctx.X0[ids];Ya=np.asarray(ctx.Y0[ids],float);Fa=np.asarray(ctx.F0[ids],bool)&np.isfinite(Ya).all(1)
    X=np.vstack([Xa,Xl]);Y=np.vstack([Ya,Yl]);F=np.r_[Fa,Fl];return _survival(X,Y,F,ctx.cfg['nsga_dev']['population_size'])

def _parent_population(ctx,method,use_archive):
    Xp,Yp,Fp=_initial_population(ctx,method,use_archive);h=ctx.history(method);init=int(ctx.cfg['nsga_dev']['initial_calls']);pop=int(ctx.cfg['nsga_dev']['population_size']);xc=[f'x01__{f}' for f in ctx.features]
    post=h.iloc[init:].copy();full=len(post)//pop
    for g in range(full):
        z=post.iloc[g*pop:(g+1)*pop];X=z[xc].to_numpy(float);Y=z[ctx.objective_norm_columns].to_numpy(float);F=_bool(z['exact_feasible']) & np.isfinite(Y).all(1);Xp,Yp,Fp=_survival(np.vstack([Xp,X]),np.vstack([Yp,Y]),np.r_[Fp,F],pop)
    return Xp,Yp,Fp,full

def _binary_tournament(rank,crowd,rng):
    a,b=rng.integers(0,len(rank),size=2)
    if rank[a]<rank[b]:return a
    if rank[b]<rank[a]:return b
    ca,cb=crowd[a],crowd[b]
    if ca>cb:return a
    if cb>ca:return b
    return int(a if rng.random()<.5 else b)

def _sbx(p1,p2,rng,eta=20.0):
    p1=np.asarray(p1,float);p2=np.asarray(p2,float);u=rng.random(len(p1));beta=np.where(u<=.5,(2*u)**(1/(eta+1)),(1/(2*(1-u)))**(1/(eta+1)));c=.5*((1+beta)*p1+(1-beta)*p2);return np.clip(c,0,1)

def _poly_mut(x,rng,eta=20.0,pm=None):
    x=np.asarray(x,float).copy();d=len(x);pm=1.0/d if pm is None else float(pm)
    for j in range(d):
        if rng.random()>=pm:continue
        u=rng.random();delta=(2*u)**(1/(eta+1))-1 if u<.5 else 1-(2*(1-u))**(1/(eta+1));x[j]=np.clip(x[j]+delta,0,1)
    return x

def _native_generation(ctx,method,use_archive,gen):
    Xp,Yp,Fp,full=_parent_population(ctx,method,use_archive);rank,crowd=_rank_and_crowding(Yp,Fp);pop=int(ctx.cfg['nsga_dev']['population_size']);d=len(ctx.features)
    # shared family random stream; population differences, not arbitrary method seeds, drive divergence
    rng=np.random.default_rng(ctx.seed+730001+10007*int(gen));out=[]
    for k in range(pop*12):
        i=_binary_tournament(rank,crowd,rng);j=_binary_tournament(rank,crowd,rng);p1,p2=Xp[i],Xp[j]
        child=_sbx(p1,p2,rng,float(ctx.cfg['nsga_dev']['sbx_eta'])) if rng.random()<float(ctx.cfg['nsga_dev']['sbx_probability']) else p1.copy()
        child=_poly_mut(child,rng,float(ctx.cfg['nsga_dev']['mutation_eta']),1.0/d)
        if len(ctx._filter_new([child],method)) and key_for_x(child) not in {key_for_x(z) for z in out}:out.append(child)
        if len(out)>=pop:break
    if len(out)<pop:
        extra=ctx._filter_new(sobol_points(pop*64,d,ctx.seed+740001+10007*int(gen)),method);seen={key_for_x(z) for z in out}
        for z in extra:
            if key_for_x(z) not in seen:out.append(z);seen.add(key_for_x(z))
            if len(out)>=pop:break
    if len(out)<pop:raise RuntimeError(f'{method}: native NSGA generation produced {len(out)}/{pop} novel offspring')
    return np.asarray(out[:pop],float),{'generation':int(gen),'parent_population_rows':len(Xp),'parent_feasible_rows':int(Fp.sum())}

def _repair(ctx,method,x,seed,reserved):
    x=np.asarray(x,float);seen={key_for_x(z) for z in reserved}
    if len(ctx._filter_new([x],method)) and key_for_x(x) not in seen:return x
    cand=ctx._filter_new(sobol_points(512,len(ctx.features),int(seed)),method)
    for z in cand:
        if key_for_x(z) not in seen:return np.asarray(z,float)
    raise RuntimeError(f'{method}: could not repair duplicate offspring')

def _score_pool(ctx,pool,w,penalty=.35):
    if not len(pool):return np.empty(0),np.empty((0,ctx.objective_dim)),np.empty(0)
    pred,pinf=ctx.global_model.predict(pool);pred=np.asarray(pred,float);pinf=np.asarray(pinf,float).reshape(-1);score=_u(pred,w)-float(penalty)*np.nan_to_num(pinf,nan=1.0,posinf=1.0,neginf=0.0);return score,pred,pinf

def _sample_good(ctx,max_rows,seed):
    ids=np.where(ctx.good0 & np.isfinite(ctx.Y0).all(axis=1))[0]
    if len(ids)<=max_rows: return ids
    rng=np.random.default_rng(int(seed)); m=ctx.objective_dim; keep=set()
    for j in range(m):
        order=ids[np.argsort(ctx.Y0[ids,j],kind='mergesort')]
        for x in order[-min(64,len(order)):]: keep.add(int(x))
    rem=np.asarray([x for x in ids if int(x) not in keep],int); need=max(0,int(max_rows)-len(keep))
    chosen=list(sorted(keep));
    if need and len(rem): chosen.extend(rng.choice(rem,size=min(need,len(rem)),replace=False).tolist())
    return np.asarray(chosen[:max_rows],int)


def _u(Y,w):
    """Numpy augmented-Chebyshev preference utility matching qParEGO semantics."""
    Y=np.asarray(Y,float); w=np.asarray(w,float).reshape(-1)
    if Y.ndim!=2 or Y.shape[1]!=len(w): raise RuntimeError(f'preference utility shape mismatch Y={Y.shape}, w={w.shape}')
    wy=Y*w[None,:]
    return np.min(wy,axis=1)+0.05*np.sum(wy,axis=1)


def _robust01(v):
    v=np.asarray(v,float); ok=np.isfinite(v); out=np.zeros(len(v),float)
    if not ok.any(): return out
    lo,hi=np.quantile(v[ok],[0.05,0.95]); out[ok]=0.5 if hi<=lo+1e-15 else np.clip((v[ok]-lo)/(hi-lo),0,1); return out


def _global_guard(ctx,method,rows,seed):
    X=ctx._filter_new(sobol_points(max(int(rows)*2,int(rows)+64),len(ctx.features),int(seed)),method); return X[:int(rows)]


def _ghost_pool(ctx,method,w,seed,max_rows):
    ids=_sample_good(ctx,int(ctx.cfg['nsga_dev']['structure_archive_rows']),seed+11); X=ctx.X0[ids]; Y=ctx.Y0[ids]
    oof=np.asarray(getattr(ctx,'oof_all',np.empty((0,ctx.objective_dim))),float)
    if oof.shape!=ctx.Y0.shape: raise RuntimeError('GHOST requires complete all-row OOF predictions')
    P=oof[ids]; ok=np.isfinite(Y).all(axis=1)&np.isfinite(P).all(axis=1); X,Y,P=X[ok],Y[ok],P[ok]
    ue=_u(Y,w); up=_u(P,w); blind=np.maximum(ue-up,0.0); score=0.65*_robust01(blind)+0.35*_robust01(ue)
    ncent=min(160,len(X)); ci=np.argsort(score,kind='mergesort')[-ncent:][::-1]; centers=X[ci]
    tree=cKDTree(X); k=min(17,len(X)); dists=tree.query(centers,k=k)[0]; local=np.median(dists[:,1:],axis=1) if k>1 else np.full(len(centers),0.04)
    rng=np.random.default_rng(seed+17); rows=[]; target=max(32,int(max_rows*0.85)); factors=np.asarray([0.35,0.7,1.15])
    while len(rows)<target:
        j=int(rng.integers(0,len(centers))); c=centers[j]; radius=float(np.clip(local[j]*factors[int(rng.integers(0,len(factors)))],0.003,0.18)); z=rng.normal(size=len(c)); z/=max(np.linalg.norm(z),1e-12); rows.append(np.clip(c+radius*z,0,1))
    return ctx._filter_new(np.vstack([rows,_global_guard(ctx,method,max_rows-len(rows),seed+29)]),method)[:max_rows]


def _productive_displacements(X,Y,w,seed,max_pairs=400):
    U=_u(Y,w); tree=cKDTree(X); elite=np.argsort(U,kind='mergesort')[-min(256,len(X)):][::-1]; vec=[]; mag=[]; gain=[]; m=Y.shape[1]
    for e in elite:
        _,nbr=tree.query(X[e],k=min(48,len(X))); nbr=np.atleast_1d(nbr).astype(int); dy=Y[e][None,:]-Y[nbr]; du=U[e]-U[nbr]; ok=(du>1e-5)&(np.sum(dy>0,axis=1)>=max(1,math.ceil(m/2)))&(np.min(dy,axis=1)>-0.20)
        if not ok.any(): continue
        ids=nbr[ok]; dui=du[ok]; j=int(ids[int(np.argmax(dui))]); v=X[e]-X[j]; L=float(np.linalg.norm(v))
        if L<=1e-8: continue
        vec.append(v/L); mag.append(L); gain.append(float(np.max(dui)))
        if len(vec)>=max_pairs: break
    return np.asarray(vec,float),np.asarray(mag,float),np.asarray(gain,float),elite









def _legacy_sparc_pool(ctx,method,w,seed,max_rows):
    """Generalized port of the older pairwise productive-direction SPARC."""
    ids=_sample_good(ctx,int(ctx.cfg['nsga_dev']['structure_archive_rows']),seed+331);X=ctx.X0[ids];Y=ctx.Y0[ids]
    if len(X)<8:return _global_guard(ctx,method,max_rows,seed+337)
    U=_u(Y,w);tree=cKDTree(X);elite=np.argsort(U)[-min(160,len(X)):][::-1];vec=[];mag=[];gain=[]
    for e in elite:
        _,nbr=tree.query(X[e],k=min(40,len(X)));nbr=np.atleast_1d(nbr).astype(int);du=U[e]-U[nbr];good=du>1e-5
        if not good.any():continue
        cand=nbr[good];jj=int(cand[int(np.argmax(du[good]))]);v=X[e]-X[jj];L=float(np.linalg.norm(v))
        if L<=1e-8:continue
        vec.append(v/L);mag.append(L);gain.append(float(U[e]-U[jj]))
    if len(vec)<4:return _global_guard(ctx,method,max_rows,seed+347)
    D=np.asarray(vec,float);G=np.asarray(gain,float);mean=np.average(D,axis=0,weights=np.maximum(G,1e-8));mean/=max(np.linalg.norm(mean),1e-12)
    steps=np.quantile(np.asarray(mag,float),[0.35,0.60,0.85]);anchors=X[np.argsort(U)[-min(96,len(X)):]]
    rng=np.random.default_rng(seed+353);rows=[];target=max(24,int(max_rows*0.80));attempts=0;limit=max(256,16*target)
    while len(rows)<target and attempts<limit:
        attempts+=1;a=anchors[int(rng.integers(0,len(anchors)))];noise=rng.normal(size=len(a));noise-=np.dot(noise,mean)*mean;noise/=max(np.linalg.norm(noise),1e-12)
        direction=mean+0.30*noise;direction/=max(np.linalg.norm(direction),1e-12);step=float(steps[int(rng.integers(0,3))]);rows.append(np.clip(a+step*direction,0,1))
    guard=max(0,int(max_rows)-len(rows));stack=np.asarray(rows,float) if rows else np.empty((0,len(ctx.features)))
    if guard:stack=np.vstack([stack,_global_guard(ctx,method,guard,seed+359)])
    return ctx._filter_new(stack,method)[:max_rows]


def _legacy_trail_pool(ctx,method,w,seed,max_rows):
    """Generalized port of the older pairwise TRAIL continuation geometry."""
    ids=_sample_good(ctx,int(ctx.cfg['nsga_dev']['structure_archive_rows']),seed+371);X=ctx.X0[ids];Y=ctx.Y0[ids]
    if len(X)<8:return _global_guard(ctx,method,max_rows,seed+373)
    U=_u(Y,w);tree=cKDTree(X);anchors=np.argsort(U)[-min(144,len(X)):][::-1];segments=[]
    for a in anchors:
        _,nbr=tree.query(X[a],k=min(48,len(X)));nbr=np.atleast_1d(nbr).astype(int);du=U[a]-U[nbr];good=du>1e-5
        if not good.any():continue
        cand=nbr[good];p=int(cand[int(np.argmax(du[good]))]);v=X[a]-X[p];L=float(np.linalg.norm(v))
        if L<=1e-8:continue
        d=v/L;head=np.inf
        for j,z in enumerate(d):
            if z>1e-12:head=min(head,(1-X[a,j])/z)
            elif z<-1e-12:head=min(head,(0-X[a,j])/z)
        if not np.isfinite(head):head=2*L
        segments.append((float(U[a]-U[p]),X[a],d,L,max(0.0,head)))
    if not segments:return _global_guard(ctx,method,max_rows,seed+379)
    segments.sort(key=lambda z:z[0],reverse=True);segments=segments[:96];rng=np.random.default_rng(seed+383);rows=[];target=max(24,int(max_rows*0.80));factors=[0.55,0.90,1.25]
    attempts=0;limit=max(256,16*target)
    while len(rows)<target and attempts<limit:
        attempts+=1;_,a,d,L,head=segments[int(rng.integers(0,len(segments)))];step=min(float(L*factors[int(rng.integers(0,3))]),float(0.85*head))
        if step<=1e-5:continue
        rows.append(np.clip(a+step*d,0,1))
    guard=max(0,int(max_rows)-len(rows));stack=np.asarray(rows,float) if rows else np.empty((0,len(ctx.features)))
    if guard:stack=np.vstack([stack,_global_guard(ctx,method,guard,seed+389)])
    return ctx._filter_new(stack,method)[:max_rows]


def _legacy_structural_pool(ctx,method,mode,w,seed,max_rows):
    if mode=='sparc':return _legacy_sparc_pool(ctx,method,w,seed,max_rows)
    if mode=='trail':return _legacy_trail_pool(ctx,method,w,seed,max_rows)
    raise ValueError(mode)

def _structural_pool(ctx,method,mode,w,seed,max_rows):
    if mode=='ghost': return _ghost_pool(ctx,method,w,seed,max_rows)
    raise ValueError(mode)




def _choose_guided(ctx,method,mode,lineage,parent,w,seed,reserved):
    cfg=ctx.cfg['nsga_dev'];max_rows=int(cfg['legacy_structure_pool_rows'] if lineage=='LEGACY_OPERATOR' else cfg['structure_pool_rows'])
    pool=_legacy_structural_pool(ctx,method,mode,w,seed,max_rows) if lineage in ('LEGACY_OPERATOR','TRUE_NATIVE_V111') else _structural_pool(ctx,method,mode,w,seed,max_rows)
    if len(reserved) and len(pool):
        seen={key_for_x(z) for z in reserved};pool=np.asarray([z for z in pool if key_for_x(z) not in seen],float)
    if not len(pool):return parent,False,0,np.nan,np.nan,'NO_STRUCTURAL_POOL'
    score,_,_= _score_pool(ctx,pool,w,float(cfg['legacy_infeasibility_penalty']))
    good=np.isfinite(score)
    if not good.any():return parent,False,len(pool),np.nan,np.nan,'NONFINITE_STRUCTURAL_SCORE'
    ids=np.where(good)[0];j=int(ids[np.argmax(score[good])]);cand=np.asarray(pool[j],float);cs=float(score[j]);ps=float(_score_pool(ctx,np.asarray([parent]),w,float(cfg['legacy_infeasibility_penalty']))[0][0]);dist=float(np.linalg.norm(cand-parent))
    cand=_repair(ctx,method,cand,seed+777,reserved)
    dist=float(np.linalg.norm(cand-parent))
    if lineage=='LEGACY_OPERATOR':
        accept=cs>ps+float(cfg['legacy_gate_tolerance']);reason='LEGACY_SURROGATE_GATE_ACCEPT' if accept else 'LEGACY_SURROGATE_GATE_REJECT'
    elif lineage=='TRUE_NATIVE_V111':
        accept=True;reason='V111_GUIDED_OFFSPRING_NO_POST_GATE'
    else:
        accept=cs>=ps-float(cfg['systematic_noninferiority_tolerance']);reason='SYSTEMATIC_SURROGATE_NONINFERIOR_ACCEPT' if accept else 'SYSTEMATIC_SURROGATE_NONINFERIOR_REJECT'
    return (cand if accept else parent),bool(accept),len(pool),ps,cs,reason

def _batch(ctx,method,use_archive,mode,lineage,q):
    n=len(ctx.history(method));init=int(ctx.cfg['nsga_dev']['initial_calls']);post=max(0,n-init);pop=int(ctx.cfg['nsga_dev']['population_size']);gen=post//pop;off=_native_generation(ctx,method,use_archive,gen)[0];offset=post%pop;base=np.asarray(off[offset:offset+q],float)
    if len(base)!=q:raise RuntimeError(f'{method}: invalid generation slice {offset}:{offset+q}')
    rows=[];modes=[];guided=[];pools=[];ps=[];cs=[];dists=[];reasons=[];weights=[]
    protected=int(ctx.cfg['nsga_dev']['protected_parent_slots_per_batch'])
    for i,parent in enumerate(base):
        parent=_repair(ctx,method,parent,ctx.seed+760001+gen*1009+offset+i,rows);w=_simplex_weight(ctx.objective_dim,ctx.seed+770001+gen*1009+offset+i);weights.append(w);x=parent;g=False;nr=0;p=np.nan;c=np.nan;reason='NATIVE_NSGA_OFFSPRING'
        if mode is not None and i>=protected:
            x,g,nr,p,c,reason=_choose_guided(ctx,method,mode,lineage,parent,w,ctx.seed+780001+gen*10007+(offset+i)*101,rows)
        rows.append(np.asarray(x,float));modes.append(f'{mode.upper() if mode else "NSGA"}::{lineage}::{reason}');guided.append(int(g));pools.append(nr);ps.append(p);cs.append(c);dists.append(float(np.linalg.norm(np.asarray(x)-parent)));reasons.append(reason)
    md={'family':'nsga2','heuristic':mode or ('prior_N' if use_archive else 'no_prior'),'implementation_lineage':lineage,'algorithm_stage':'adaptive','guided':guided,'parent_mode':['NATIVE_NSGA2_OFFSPRING']*q,'parent_score':ps,'custom_score':cs,'selected_acq_score':[cs[i] if guided[i] else ps[i] for i in range(q)],'guided_minus_parent_acq':[np.nan]*q,'guided_minus_parent_gate':[cs[i]-ps[i] if np.isfinite(cs[i]) and np.isfinite(ps[i]) else np.nan for i in range(q)],'custom_pool_rows':pools,'candidate_distance_from_parent':dists,'guidance_reason':reasons,'admission_score_kind':[('NATIVE_NSGA2' if mode is None else ('GLOBAL_SURROGATE_GATE' if lineage=='LEGACY_OPERATOR' else ('NO_POST_GATE' if lineage=='TRUE_NATIVE_V111' else 'GLOBAL_SURROGATE_NONINFERIOR')))]*q,'nsga_generation':[gen]*q,'nsga_generation_offset':[offset+i for i in range(q)]}
    for j,o in enumerate(ctx.compiled_problem['objectives']):md[f'guidance_weight__{o["slug"]}']=[float(weights[i][j]) for i in range(q)]
    return np.asarray(rows,float),modes,md

def _shared_init(ctx,method,q):
    n=len(ctx.history(method));b=n//ctx.batch_size+1;seed=ctx.seed+310001+1009*b;X=ctx._filter_new(sobol_points(max(256,16*q),len(ctx.features),seed),method)
    if len(X)<q:X=np.vstack([X,ctx._filter_new(sobol_points(max(512,32*q),len(ctx.features),seed+17),method)])
    if len(X)<q:raise RuntimeError(f'{method}: common initialization could only produce {len(X)}/{q} novel points')
    return X[:q],[f'COMMON_SOBOL_INIT::{b}']*q

def run_method(ctx,method):
    use_archive=method!=BASE_NO50K;mode=HEURISTIC.get(method);lineage=LINEAGE.get(method,'BASELINE');initial=int(ctx.cfg['nsga_dev']['initial_calls']);budget=int(ctx.budget)
    while len(ctx.history(method))<budget:
        n=len(ctx.history(method));q=min(ctx.batch_size,budget-n)
        if n<initial:
            X,modes=_shared_init(ctx,method,q);metadata={'family':'nsga2','heuristic':mode or ('prior_N' if use_archive else 'no_prior'),'implementation_lineage':lineage,'algorithm_stage':'warmup','guided':np.zeros(q,int),'admission_score_kind':['COMMON_INIT']*q}
        else:X,modes,metadata=_batch(ctx,method,use_archive,mode,lineage,q)
        ctx.evaluate_batch(method,X,modes=modes,metadata=metadata);h=ctx.history(method);g=int(pd.to_numeric(h.tail(q).get('guided',pd.Series([0]*q)),errors='coerce').fillna(0).sum())
        print(f'[{method}] {len(h):3d}/{budget} | exact ΔHV={ctx.current_delta_hv(method):.9g} | stage={"warmup" if n<initial else "adaptive"} | lineage={lineage} | guided={g}/{q}',flush=True)
    return ctx.history(method)
RUNNERS={m:(lambda ctx,m=m:run_method(ctx,m)) for m in METHODS}
