"""Safe Boolean GPR interpreter; no eval and no optimizer dependency."""
import re
from functools import lru_cache

@lru_cache(maxsize=50000)
def parse_rule(rule):
    tokens=re.findall(r"\(|\)|[^\s()]+", rule)
    if not tokens: return None
    i=0
    def atom():
        nonlocal i
        if i>=len(tokens): raise ValueError('Incomplete GPR')
        t=tokens[i]; i+=1
        if t=='(':
            x=expr('or')
            if i>=len(tokens) or tokens[i]!=')': raise ValueError('Unclosed GPR')
            i+=1; return x
        if t.lower() in {'and','or'} or t==')': raise ValueError('Invalid GPR token')
        return ('gene',t)
    def expr(op):
        nonlocal i
        child=(lambda:expr('and')) if op=='or' else atom
        children=[child()]
        while i<len(tokens) and tokens[i].lower()==op:
            i+=1; children.append(child())
        return children[0] if len(children)==1 else (op,tuple(children))
    x=expr('or')
    if i!=len(tokens): raise ValueError('Invalid GPR expression')
    return x

def rule_genes(node):
    if node is None: return set()
    if node[0]=='gene': return {node[1]}
    return set().union(*(rule_genes(c) for c in node[1]))

def supported(node, off):
    if node is None: return None
    if node[0]=='gene': return node[1] not in off
    vals=[supported(c,off) for c in node[1]]
    return all(vals) if node[0]=='and' else any(vals)

def reaction_states(model, off=()):
    unknown=set(off)-set(model.genes)
    if unknown: raise ValueError('Unknown gene IDs: '+', '.join(sorted(unknown)))
    return {rid:('unknown' if not r.gene_rule else 'supported' if supported(parse_rule(r.gene_rule),set(off)) else 'disabled') for rid,r in model.reactions.items()}

def gene_details(model, ids):
    return [{'id':gid, **model.genes[gid], 'reactions':[rid for rid,r in model.reactions.items() if gid in rule_genes(parse_rule(r.gene_rule))], 'gpr':{rid:r.gene_rule for rid,r in model.reactions.items() if gid in rule_genes(parse_rule(r.gene_rule))}} for gid in ids]
