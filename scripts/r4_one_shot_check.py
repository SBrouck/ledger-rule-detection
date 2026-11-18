import os, numpy as np, pandas as pd
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"]="true"; os.environ["TABPFN_NO_BROWSER"]="1"
from dotenv import load_dotenv; load_dotenv(".env")
from pathlib import Path
from sklearn.metrics import average_precision_score
from ledger.supervised import features, columns, task_mask, split, context
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
x=features(Path("data/r0.05_s0")); tr_mask=split(x); m=task_mask(x,"R4")
tr,te=np.flatnonzero(m&tr_mask),np.flatnonzero(m&~tr_mask)
y=x["violated"].to_numpy(); ytr,yte=y[tr],y[te]
cols=columns(x,"presence")
ctx=context(ytr,np.random.default_rng(1001),3000,1)
Xtr,Xte=x.iloc[tr[ctx]][cols].to_numpy(float),x.iloc[te][cols].to_numpy(float)
print("ctx", len(ctx), "pos", ytr[ctx].sum())
pos=x.iloc[tr[ctx]][ytr[ctx]]
print("the one positive:", {c:v for c,v in pos[cols].iloc[0].items() if v and not c.startswith('log')})
def fit(yc, seed=0):
    clf=TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed); clf.fit(Xtr,yc); return clf.predict_proba(Xte)[:,1]
s=fit(ytr[ctx]); print("AUPRC real labels", round(average_precision_score(yte,s),3))
te_df=x.iloc[te].assign(score=s, h4452=x.iloc[te]["acc_4452"], h44566=x.iloc[te]["acc_44566"])
print(te_df.groupby(["violated","h4452","h44566"]).score.describe()[["count","mean","min","max"]].round(3))
rng=np.random.default_rng(0); yperm=ytr[ctx].copy(); rng.shuffle(yperm)
print("shuffled ctx pos", yperm.sum(), "AUPRC", round(average_precision_score(yte,fit(yperm)),3), "prevalence", round(yte.mean(),3))
