from __future__ import annotations
import ast
from typing import Any, Callable
import numpy as np
import pandas as pd

ALLOWED_FUNCS={'abs','sqrt','log','log1p','exp','clip','minimum','maximum','safe_divide','col'}
ALLOWED_NODES=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Call,ast.Name,ast.Load,ast.Constant,
               ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.Mod,ast.USub,ast.UAdd,
               ast.keyword)

class ExpressionError(ValueError): pass

def resolve_column(df: pd.DataFrame, token: str) -> str:
    token=str(token)
    aliases=[]
    if ':' in token:
        kind,raw=token.split(':',1)
        mapping={
            'result':[raw],
            'medium':[f'requested_medium__{raw}',raw],
            'requested_medium':[f'requested_medium__{raw}',raw],
            'applied_medium':[f'applied_medium__{raw}',raw],
            'exchange':[f'exchange_flux__{raw}',raw],
            'reaction':[f'reaction_flux__{raw}',raw],
        }
        aliases=mapping.get(kind,[token])
    else:
        aliases=[token]
    for c in aliases:
        if c in df.columns: return c
    lower={str(c).lower():c for c in df.columns}
    for c in aliases:
        if c.lower() in lower: return str(lower[c.lower()])
    raise KeyError(f'Column/alias {token!r} not found. Tried {aliases[:5]}')

def _check(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n,ALLOWED_NODES): raise ExpressionError(f'Disallowed expression syntax: {type(n).__name__}')
        if isinstance(n,ast.Call):
            if not isinstance(n.func,ast.Name) or n.func.id not in ALLOWED_FUNCS:
                raise ExpressionError(f'Disallowed function in expression: {ast.dump(n.func)}')

def evaluate_expression(df: pd.DataFrame, expr: str) -> pd.Series:
    node=ast.parse(expr,mode='eval'); _check(node)
    def col(name): return pd.to_numeric(df[resolve_column(df,str(name))], errors='coerce')
    def safe_divide(a,b):
        aa=np.asarray(a,float); bb=np.asarray(b,float); out=np.full(np.broadcast(aa,bb).shape,np.nan,float)
        np.divide(aa,bb,out=out,where=np.isfinite(bb)&(np.abs(bb)>1e-12)); return pd.Series(out,index=df.index)
    env={
        'abs':np.abs,'sqrt':np.sqrt,'log':np.log,'log1p':np.log1p,'exp':np.exp,
        'clip':np.clip,'minimum':np.minimum,'maximum':np.maximum,'safe_divide':safe_divide,'col':col,
    }
    for c in df.columns:
        if str(c).isidentifier(): env[str(c)]=pd.to_numeric(df[c],errors='coerce')
    return pd.Series(eval(compile(node,'<expression>','eval'),{'__builtins__':{}},env), index=df.index)

def materialize_targets_and_derived(df: pd.DataFrame, spec: dict[str,Any]) -> tuple[pd.DataFrame,list[str]]:
    out=df.copy(); created=[]
    for d in spec.get('derived',[]) or []:
        name=str(d['name']); expr=str(d['expression']); out[name]=evaluate_expression(out,expr); created.append(name)
    for t in spec.get('targets',[]):
        name=str(t['name'])
        if 'expression' in t:
            out[name]=evaluate_expression(out,str(t['expression'])); created.append(name)
        elif 'value' in t or 'column' in t:
            tok=str(t.get('value',t.get('column'))); src=resolve_column(out,tok)
            if name!=src: out[name]=pd.to_numeric(out[src],errors='coerce'); created.append(name)
    return out,created
