# Unit test of the PATCHED metric functions vs the ORIGINAL, on stubs (CPU). Run with the Ada env python.
import ast, sys, pathlib, gc, time, numpy as np, torch, pandas as pd, os
ORIG, PATCHED = sys.argv[1], sys.argv[2]
def grab(path, names):
    src=open(path).read(); tree=ast.parse(src)
    return "\n\n".join(ast.get_source_segment(src,n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
names={'get_hook_acts','encode_sae','streaming_feature_metrics'}  # free_memory (gc/cache trimming) is stubbed: it cannot affect results and is slow on a busy node
class Cfg: d_sae=64
class SAE:
    cfg=Cfg(); W=torch.randn(32,64,generator=torch.Generator().manual_seed(0))*0.5
    def encode(self,x): return torch.relu(x@self.W-0.3)
class Model:
    def __init__(self,noisy): self.noisy=noisy
    def run_with_cache(self,batch,names_filter):
        g=torch.Generator().manual_seed(int(batch.sum())%100000)
        a=torch.randn(batch.shape[0],batch.shape[1],32,generator=g)
        if self.noisy: a=a+0.05*torch.randn(a.shape,generator=torch.Generator().manual_seed(int(batch.sum())%100000+1))
        return None,{names_filter[0]:a}
class FakeWB:
    def __init__(self): self.hb, self.cond = [], []
    def heartbeat(self, condition, done, total, elapsed, mem=None): self.hb.append((condition, done, total))
    def log_condition(self, summary): self.cond.append(summary['condition'])
FW = FakeWB()
OUT=pathlib.Path('/tmp/ut_out'); OUT.mkdir(exist_ok=True,parents=True)
def run(path, bos, snap, width):
    ns=dict(free_memory=lambda: None,_WB=FW,time=time,gpu_mem_gib=lambda t: {},gc=gc,torch=torch,np=np,pd=pd,os=os,tqdm=lambda it,desc=None:it,DEVICE='cpu',BATCH_SIZE=1,FIRING_THRESHOLD=0.001,G0_BOS=bos,SNAP_EVERY=snap,OUTPUT_DIR=OUT)
    exec(grab(path,names),ns)
    toks=torch.randint(0,1000,(250,width),generator=torch.Generator().manual_seed(3))
    return ns['streaming_feature_metrics'](Model(False),Model(True),SAE(),toks,'h','cond',12,batch_size=1)
sm0,pf0=run(ORIG,'none',0,256)
sm1,pf1=run(PATCHED,'none',100,256)
print('1) none-arm: patched summary & per-feature IDENTICAL to ORIGINAL:', sm0==sm1 and pf0.equals(pf1))
S=torch.load(OUT/'snap_cond.pt'); snaps=S['snaps']
print('2) snapshots', tuple(snaps.shape), 'positions', S['n_positions'])
tot=snaps[-1]; blocks=snaps.clone(); blocks[1:]=snaps[1:]-snaps[:-1]
print('3) block sums == totals:', bool(torch.allclose(blocks.sum(0),tot)))
n=S['n_positions'][-1]; num=tot[4]-tot[0]*tot[1]/n; dx=tot[2]-tot[0]**2/n; dy=tot[3]-tot[1]**2/n
r=torch.clamp(num/torch.sqrt(torch.clamp(dx*dy,min=1e-12)),-1,1).numpy()
print('4) corr rebuilt from snapshots == reported corr:', bool(np.allclose(r,pf1['corr'].values,atol=1e-9)))
smm,_=run(PATCHED,'prepend_mask',0,257); smk,_=run(PATCHED,'prepend_keep',0,257)
print('5) stat positions per 250 windows: none/mask/keep =', sm1['n_tokens']//250, smm['n_tokens']//250, smk['n_tokens']//250, '(expect 256/256/257)')

# 6) wandb hooks: heartbeat every SNAP_EVERY windows from inside the streaming loop; add_summary forwards each condition
FW.hb.clear(); FW.cond.clear()
run(PATCHED,'none',100,256)
print('6) heartbeats from streaming loop (expect [("cond",100,250),("cond",200,250)]):', FW.hb)
ns2=dict(_WB=FW,pd=pd,all_summaries=[])
exec(grab(PATCHED,{'add_summary'}),ns2); ns2['add_summary']({'condition':'RTN INT8','bits':8})
print('7) add_summary forwarded to logger (expect ["RTN INT8"]):', FW.cond)
ok = FW.hb==[('cond',100,250),('cond',200,250)] and FW.cond==['RTN INT8']
print('WANDB HOOKS OK' if ok else 'WANDB HOOKS FAILED'); sys.exit(0 if ok else 1)
