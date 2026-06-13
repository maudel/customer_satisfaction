"""
run_real.py · Sprint 3 sobre DATA REAL (data/master) — RESUMIBLE, 1 modelo a la vez.
Cada modelo: Optuna -> trial CSV -> CV 3-fold OOF -> fit final -> métricas val/bt ->
guarda fila + OOF + proba_val incrementalmente. Reanuda saltando modelos ya hechos.
Fase final (--finalize): stacking + importancia + persistencia + metrics.
Uso:
    python run_real.py --trials 30          # procesa el próximo modelo pendiente y sale
    python run_real.py --finalize           # stacking + importancia (cuando todos listos)
"""
from __future__ import annotations
import argparse, json, time, os
import numpy as np, pandas as pd, joblib
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from src import config as C3, optuna_panel as OP, importancia

R=C3.DIR_REPORTES; PARC=R/"_parciales"; PARC.mkdir(exist_ok=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trials",type=int,default=C3.N_TRIALS_OPTUNA)
    ap.add_argument("--finalize",action="store_true")
    ap.add_argument("--only",type=str,default=None)
    a=ap.parse_args()
    Xtr,Xva,Xbt,ytr,yva,ybt,spw,cols=OP.cargar_splits()
    # subsample para BÚSQUEDA y CV (memoria/tiempo); el fit FINAL usa TRAIN completo
    ss=min(len(Xtr),12000)
    idx=Xtr.sample(n=ss,random_state=C3.SEMILLA).index
    Xs,ys=Xtr.loc[idx].reset_index(drop=True),ytr.loc[idx].reset_index(drop=True)
    objs=OP.espacios(Xs,ys,Xva,yva,spw)

    if not a.finalize:
        # próximo modelo pendiente
        pend=[n for n in objs if not (PARC/f"row_{n.replace(' ','_')}.json").exists()]
        if a.only: pend=[n for n in [a.only] if n in objs]
        if not pend: print("Todos los modelos ya procesados. Usá --finalize."); return
        nombre=pend[0]; key=nombre.replace(' ','_'); t0=time.time()
        print(f"▶ {nombre} ({a.trials} trials)")
        s=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=C3.SEMILLA),study_name=nombre)
        s.optimize(objs[nombre],n_trials=a.trials,show_progress_bar=False)
        pd.DataFrame([{"modelo":nombre,"trial":t.number,"f1_cls0_val":t.value,**t.params}
                      for t in s.trials if t.value is not None]).to_csv(R/f"optuna_trials_{key}.csv",index=False)
        bp=s.best_params.copy()
        skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=C3.SEMILLA); cvs=[]
        for tr,vl in skf.split(Xs,ys):
            fm=OP.construir(nombre,bp); fm.fit(Xs.iloc[tr],ys.iloc[tr])
            pp=OP._proba(fm,Xs.iloc[vl]); cvs.append(OP._f1_0(ys.iloc[vl],pp))
        fm=OP.construir(nombre,bp); fm.fit(Xtr,ytr)
        pv=OP._proba(fm,Xva); pb=OP._proba(fm,Xbt)
        mv=OP.metricas(yva,pv); mb=OP.metricas(ybt,pb); caida=mv["f1_0"]-mb["f1_0"]
        row={"modelo":nombre,"cv_f1_mean":round(float(np.mean(cvs)),4),"cv_f1_std":round(float(np.std(cvs)),4),
            "f1_cls0_val":round(mv["f1_0"],4),"f1_cls0_backtest":round(mb["f1_0"],4),
            "caida":round(caida,4),"score_estabilidad":round(mv["f1_0"]-C3.PENALIZACION_ESTABILIDAD*max(0,caida),4),
            "estable":"sí" if caida<=C3.UMBRAL_INESTABILIDAD else "NO",
            "auc_val":round(mv["auc"],4),"gini_val":round(mv["gini"],4),"brier_val":round(mv["brier"],4),
            "recall_cls0_val":round(mv["recall_0"],4),"prec_cls0_val":round(mv["prec_0"],4),
            "best_params":json.dumps(bp)}
        json.dump(row,open(PARC/f"row_{key}.json","w"))
        np.save(PARC/f"val_{key}.npy",pv)
        # OOF sobre el subsample para stacking
        oofp=np.zeros(len(Xs))
        for tr,vl in StratifiedKFold(n_splits=3,shuffle=True,random_state=C3.SEMILLA).split(Xs,ys):
            fm2=OP.construir(nombre,bp); fm2.fit(Xs.iloc[tr],ys.iloc[tr]); oofp[vl]=OP._proba(fm2,Xs.iloc[vl])
        np.save(PARC/f"oof_{key}.npy",oofp); np.save(PARC/"_ys.npy",ys.values)
        joblib.dump(fm,PARC/f"model_{key}.pkl")
        print(f"  cv={np.mean(cvs):.4f} val={mv['f1_0']:.4f} bt={mb['f1_0']:.4f} caída={caida:+.4f} t={time.time()-t0:.0f}s")
        rest=[n for n in objs if not (PARC/f'row_{n.replace(chr(32),chr(95))}.json').exists()]
        print(f"  pendientes: {rest if rest else 'ninguno (corré --finalize)'}")
        return

    # ── FINALIZE ──────────────────────────────────────────────────────────
    rows=[json.load(open(PARC/f)) for f in sorted(os.listdir(PARC)) if f.startswith("row_")]
    nombres=[r["modelo"] for r in rows]
    oof=pd.DataFrame({n:np.load(PARC/f"oof_{n.replace(' ','_')}.npy") for n in nombres})
    valp=pd.DataFrame({n:np.load(PARC/f"val_{n.replace(' ','_')}.npy") for n in nombres})
    print("[stacking] meta-learner LogReg sobre OOF de",len(nombres),"modelos")
    meta=LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,random_state=C3.SEMILLA)
    ys_oof=np.load(PARC/"_ys.npy")
    meta.fit(oof,ys_oof); sp=meta.predict_proba(valp)[:,1]; ms=OP.metricas(yva,sp)
    rows.append({"modelo":"STACKING","cv_f1_mean":np.nan,"cv_f1_std":np.nan,
        "f1_cls0_val":round(ms["f1_0"],4),"f1_cls0_backtest":np.nan,"caida":np.nan,
        "score_estabilidad":round(ms["f1_0"],4),"estable":"—","auc_val":round(ms["auc"],4),
        "gini_val":round(ms["gini"],4),"brier_val":round(ms["brier"],4),
        "recall_cls0_val":round(ms["recall_0"],4),"prec_cls0_val":round(ms["prec_0"],4),"best_params":"{}"})
    tabla=pd.DataFrame(rows); tabla.to_csv(R/"comparacion_modelos.csv",index=False)
    cand=tabla[tabla["modelo"]!="STACKING"]
    gan=cand.sort_values("score_estabilidad",ascending=False).iloc[0]
    nombre_g=gan["modelo"]; key_g=nombre_g.replace(' ','_'); params_g=json.loads(gan["best_params"])
    print(f"[ganador] {nombre_g} (score {gan['score_estabilidad']}, caída {gan['caida']})")
    pg=joblib.load(PARC/f"model_{key_g}.pkl")
    r=permutation_importance(pg,Xva,yva,scoring=lambda e,X,y:OP._f1_0(y,OP._proba(e,X)),
                             n_repeats=5,random_state=C3.SEMILLA,n_jobs=1)
    imp=np.clip(r.importances_mean,0,None); imp=100*imp/(imp.sum() or 1)
    timp=pd.DataFrame({"variable":cols,"importancia_pct":imp}).sort_values("importancia_pct",ascending=False).reset_index(drop=True)
    timp.to_csv(R/"importancia_variables.csv",index=False)
    audit=importancia.auditar_cap(timp)
    print(f"[15%] máx={audit['max_pct']}% · {'CUMPLE' if audit['cumple'] else 'VIOLA'}")
    modelos={n:joblib.load(PARC/f"model_{n.replace(' ','_')}.pkl") for n in nombres}
    joblib.dump(modelos,R/"modelos_optuna.pkl"); joblib.dump(meta,R/"meta_learner.pkl")
    joblib.dump(oof,R/"oof_probas.pkl"); joblib.dump(valp,R/"val_probas.pkl")
    json.dump({r_["modelo"]:json.loads(r_["best_params"]) for r_ in rows if r_["best_params"] not in ("{}","")},
              open(R/"mejores_params.json","w"),indent=2)
    mb_g=OP.metricas(ybt,OP._proba(pg,Xbt))
    estado={"fuente_datos":"data/master (REAL Olist)","n_train":int(len(Xtr)),"n_val":int(len(Xva)),
        "n_backtest":int(len(Xbt)),"insatisfechos_train_pct":round(float((ytr==0).mean()),4),
        "scale_pos_weight":spw,"seed":C3.SEMILLA,"modelo_ganador":nombre_g,
        "seleccion":"F1(cls0) penalizado por estabilidad","metricas_backtest_ganador":{k:round(float(v),4) for k,v in mb_g.items()},
        "stacking_f1_cls0_val":round(ms["f1_0"],4),"stacking_auc_val":round(ms["auc"],4),
        "auditoria_15pct":audit,"hiperparametros_ganador":params_g,"version":C3.VERSION_PIPELINE}
    json.dump(estado,open(R/"metrics_sprint3.json","w"),indent=2,ensure_ascii=False)
    print("✔ finalize completo.")

if __name__=="__main__": main()
