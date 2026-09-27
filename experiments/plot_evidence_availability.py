"""Scientific figures for a completed, strictly validated v5 diagnostic."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(summary_path, output):
    s=json.loads(summary_path.read_text(encoding="utf-8"))
    assert s["status"]=="complete" and s["integrity"]["passed"]
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colors={"direct":"#64748b","native":"#007b73","degraded":"#bf652d"}
    budgets=[256,512,1024]
    fig,axes=plt.subplots(1,3,figsize=(14.8,4.4),layout="constrained")
    for ax,metric,title in zip(axes[:2],["official_em","conservative_text_em"],["Official span EM (primary)","Strict text EM (secondary)"]):
        for kind,color in colors.items():
            values=[s["actions"][f"{kind}_{b}"]["metrics"][metric]*100 for b in budgets]
            ax.plot(range(3),values,"o-",label=kind,color=color,lw=2)
        ax.axhline(s["actions"]["highres"]["metrics"][metric]*100,color="#6f50a0",ls="--",label="full page 4096")
        ax.set(xticks=range(3),xticklabels=budgets,xlabel="Overview visual-token cap",ylabel="Accuracy (%)",title=title)
        ax.grid(axis="y",alpha=.18)
    axes[0].legend(frameon=False,fontsize=8,loc="best")
    x=np.arange(3)
    for offset,metric,color,label in [(-.07,"official_em","#007b73","Official EM"),(.07,"conservative_text_em","#64748b","Strict EM")]:
        stats=[s["detail_effects"][str(b)][metric] for b in budgets]
        means=np.array([r["difference_pp"] for r in stats])
        ci=np.array([r["ci95_pp"] for r in stats])
        axes[2].errorbar(x+offset,means,yerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt="o",capsize=4,color=color,label=label)
    axes[2].axhline(0,color="#9ca3af",lw=1)
    axes[2].set(xticks=x,xticklabels=budgets,xlabel="Overview visual-token cap",ylabel="Native minus degraded (pp)",title="Paired detail advantage; 95% CI")
    axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle("LookAgain v5 · 85 reused reports · reference-privileged region",fontsize=14,fontweight="bold")
    for suffix in ("png","svg"):
        fig.savefig(output/f"evidence_quality.{suffix}",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8),layout="constrained")
    for kind,color in colors.items():
        rows=[s["actions"][f"{kind}_{b}"] for b in budgets]
        xs=[r["standalone_latency_s"]["mean"] for r in rows]
        ys=[r["metrics"]["official_em"]*100 for r in rows]
        ax.plot(xs,ys,"o-",color=color,label=kind,lw=2)
        offsets={"direct":[(5,4),(5,-12),(-36,-18)],"native":[(-27,-12),(-34,0),(8,4)],"degraded":[(5,-12),(0,12),(8,-4)]}
        for b,xv,yv,offset in zip(budgets,xs,ys,offsets[kind]):
            ax.annotate(str(b),(xv,yv),xytext=offset,textcoords="offset points",fontsize=8,color=color,
                arrowprops=dict(arrowstyle="-",color=color,lw=.5))
    high=s["actions"]["highres"]
    ax.scatter(high["standalone_latency_s"]["mean"],high["metrics"]["official_em"]*100,marker="D",color="#6f50a0",label="full page 4096")
    ax.set(title="Quality versus measured standalone invocation cost",xlabel="Mean synchronized latency (seconds)",ylabel="Official EM (%)")
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.18)
    fig.text(.5,-.01,"Localization supplied by reference annotation; excludes a deployable selector. One active-desktop run.",ha="center",fontsize=8)
    for suffix in ("png","svg"):fig.savefig(output/f"evidence_cost.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();plot(a.summary,a.output)
