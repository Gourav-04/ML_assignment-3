"""Regenerate the CVPR PDF after results, repository URL, or Git revision change."""
import os
import shutil
import tempfile
import re
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from figures import rows

R = Path("results")
P = Path("report")
NAMES = "airplane automobile bird cat deer dog frog horse ship truck".split()


def git(*args):
    if not shutil.which("git"):
        return ""
    cmd = subprocess.run(["git", *args], capture_output=True, text=True)
    return cmd.stdout.strip() if cmd.returncode == 0 else ""


def percent(value):
    return f"{100*float(value):.2f}" + r"\%"


def collect(repo):
    rev = git("rev-parse","HEAD")
    if not rev:
        rev = Path("COMMIT.txt").read_text().strip() if Path("COMMIT.txt").exists() else "unavailable"
    repo = repo or git("remote","get-url","origin")
    if repo.startswith("git@github.com:"):
        repo = "https://github.com/" + repo.split(":",1)[1].removesuffix(".git")
    repo_label = r"\url{"+repo+"}" if re.fullmatch(r"https?://[\w./~-]+", repo) else "pending GitHub push"
    m = {"Repo":repo_label,"Commit":r"\texttt{"+rev+"}"}
    cls = rows("classifiers_test")
    d = {(r["model"],r["condition"]):float(r["acc"]) for r in cls}
    def acc(kind,cond): return d[kind,cond]
    clean, shuffled = acc("patch","clean"),acc("patch","pixel")
    m["LocalResult"] = (f"The clean patch network scored {percent(clean)} and random grouping scored "
        f"{percent(shuffled)}. The signed clean-minus-shuffled difference was {100*(clean-shuffled):+.1f} points. "
        f"The ordinary dense MLP scored {percent(acc('mlp','clean'))}. The paired patch difference cannot be attributed to a change in parameter count. " + ("Contiguous neighbourhoods helped the shared first layer." if clean>shuffled else "The paired result gives no advantage to contiguous neighbourhoods in this setting."))
    mem=rows("memory")
    def crossing(key):
        return next((int(r["epoch"]) for r in mem if float(r[key])>0.20),None)
    a,b=crossing("val_acc"),crossing("flipped_assigned_acc")
    m["MemoryParams"]=f'{int(mem[0]["parameters"]):,}'
    text = f"The clean validation threshold was crossed at epoch {a}." if a else "Clean validation never crossed its threshold."
    text += f" Assigned-label accuracy crossed at epoch {b}." if b else " Assigned-label accuracy never crossed its threshold."
    if a and b:
        text += (" Under this definition, transferable learning preceded substantial random-label fitting." if a < b else
                 " The threshold ordering does not establish that structure was learned first.")
    text += " At epoch 65 the MLP predicted the assigned label on "+percent(mem[-1]["flipped_assigned_acc"])+" of flipped examples."
    peak=max(mem,key=lambda r:float(r["val_acc"]))
    text += (f" Clean validation peaked at {percent(peak['val_acc'])} in epoch {peak['epoch']} and ended at {percent(mem[-1]['val_acc'])}. " 
             f"On the identical flipped images, original-label accuracy ended at {percent(mem[-1]['flipped_original_acc'])}.")
    m["MemoryResult"]=text
    cnnloss=acc("cnn","clean")-acc("cnn","pixel")
    mlploss=acc("mlp","clean")-acc("mlp","pixel")
    recovered=acc("cnn","tile")-acc("cnn","pixel")
    m["SpatialResult"]=(f"The clean-minus-shuffled gap was {100*cnnloss:+.1f} percentage points for the CNN and "
                        f"{100*mlploss:+.1f} for the dense MLP. Holding tiles intact changed CNN accuracy by "
                        f"{100*recovered:+.1f} points relative to full shuffling. " +
                        ("This pattern supports sensitivity to local neighbourhoods." if cnnloss>mlploss and recovered>0
                         else "The two interventions give mixed evidence for the locality prediction.")+
                        " Tile boundaries and changed global positions limit finer attribution.")
    m["ConfidenceResult"]="CNN mean confidence on errors was "+", ".join(
        percent(r["wrong_conf"])+" on "+r["condition"]+" images" for r in cls if r["model"]=="cnn")+"."
    cnn={r["condition"]:r for r in cls if r["model"]=="cnn"}
    error_delta=int(cnn["pixel"]["errors"])-int(cnn["clean"]["errors"])
    conf_delta=float(cnn["pixel"]["wrong_conf"])-float(cnn["clean"]["wrong_conf"])
    m["ConfidenceResult"]+=f" Full shuffling changed the error count by {error_delta:+d} out of 10,000 and mean error confidence by {100*conf_delta:+.1f} points."
    rep=rows("representation")
    win=json.loads((R/"selected.json").read_text())["bottleneck"]
    m["ReconstructionResult"]=f"Validation MSE selected width {win}; test reconstruction errors also appear for the representation comparison required in Q3(c)."
    den=rows("denoising")
    gain=float(den[2]["acc"])-float(den[0]["acc"])
    paired=json.loads((R/"paired_denoising.json").read_text())
    m["DenoiseResult"]="Frozen-CNN test accuracy was "+", ".join(percent(r["acc"])+" for "+r["input"] for r in den)+f". Denoising changed accuracy by {100*gain:+.2f} percentage points relative to the corrupted image. "
    m["DenoiseResult"]+=(f'It corrected {paired["gains"]} previously wrong predictions and spoiled {paired["losses"]} previously correct ones '
                         f'(paired exact McNemar $p={paired["exact_mcnemar_p"]:.3f}$). ')
    if paired["exact_mcnemar_p"]>0.05:
        m["DenoiseResult"]+="The observed net change gives little evidence of a reliable improvement. "
    m["DenoiseResult"]+=f'The original clean-input CNN accuracy of {percent(acc("cnn","clean"))} shows how much discriminative information remains unrecovered.'
    mse_order=sorted(rep,key=lambda r:float(r["test_mse"]))
    probe_order=sorted(rep,key=lambda r:-float(r["probe_test"]))
    m["ProbeResult"]="In order of increasing test MSE the widths were "+", ".join(r["size"] for r in mse_order)+". In descending probe accuracy the widths were "+", ".join(r["size"] for r in probe_order)+". "+(
        "The rankings agree in this sweep. Width is a shared explanatory variable, so this does not establish that improving MSE causes better classification." if mse_order==probe_order else
        "The rankings disagree. Pixel fidelity and linearly accessible class information reward different features.")
    sweep=rows("svm_sweep")
    bonus=json.loads((R/"bonus.json").read_text())
    m["SvmResult"]=f'Validation selected {bonus["kernel"]} with $C={bonus["C"]}$; the frozen SVM achieved {percent(bonus["test_acc"])} on test. '
    if bonus["C"]==max(float(r["C"]) for r in sweep):
        m["SvmResult"]+="The winner lies at the upper end of the evaluated grid; validation does not establish that it is a global optimum. "
    for kernel in ("linear","rbf","poly"):
        group=[r for r in sweep if r["kernel"]==kernel]
        best=max(group,key=lambda r:float(r["val"]))
        m["SvmResult"]+=f'For {kernel}, validation peaked at $C={best["C"]}$ ({percent(best["val"])}); raising $C$ from 0.0001 to 1 moved training accuracy from {percent(group[0]["train"])} to {percent(group[-1]["train"])}. '
        high=group[-1]
        gap=float(high["train"])-float(high["val"])
        m["SvmResult"]+=f'At $C=1$ its train--validation gap was {100*gap:.2f} points. '
    m["SvmResult"]+="A wide positive gap at high $C$ indicates greater sensitivity to the training sample. The support-vector fraction in the plot counts each training point once across the binary subproblems."
    cm=np.loadtxt(R/"svm_confusion.csv",delimiter=",",dtype=int)
    pairs=rows("pairs")[::4]
    m["PairResult"]="The combined error counts were "+", ".join(
        f'{NAMES[int(r["class_i"])]}/{NAMES[int(r["class_j"])]} ({int(r["errors_ij"])+int(r["errors_ji"])})' for r in pairs)+"."
    examples=[r for r in rows("pairs") if r["source"]=="test"]
    wrong=sum(r["predicted"]!=r["true"] for r in examples)
    outside=sum(r["predicted"] not in (r["class_i"],r["class_j"]) for r in examples)
    m["PairResult"]+=f" Of the six near-boundary test examples, {wrong} received an incorrect multiclass prediction, including {outside} predictions outside the inspected pair."
    cont=np.loadtxt(R/"contingency.csv",delimiter=",",dtype=int,ndmin=2)
    options=rows("cluster_selection")
    k=bonus["k"]
    m["ClusterResult"]="Validation silhouette by $k$ was "+", ".join(f'{r["k"]} ({float(r["silhouette"]):.3f})' for r in options)+f'. The selected $k={k}$ gave test ARI {bonus["ARI"]:.3f} and NMI {bonus["NMI"]:.3f}. '
    # Modal-cluster coassignment versus SVM confusion for a pair.
    candidates=[]
    for i in range(10):
        for j in range(i+1,10):
            a=cont[:,i]/cont[:,i].sum()
            b=cont[:,j]/cont[:,j].sum()
            overlap=np.minimum(a,b).sum()
            errors=(cm[i,j]+cm[j,i])/(cm[i].sum()+cm[j].sum())
            candidates.append((overlap-errors,i,j,overlap,errors))
    _,i,j,overlap,errors=max(candidates)
    m["ClusterResult"]+=(f"The {NAMES[i]} and {NAMES[j]} cluster histograms overlap by {percent(overlap)} "
                         f"(sum of the smaller class-conditional mass in each cluster), while only {percent(errors)} "
                         "of the two classes' images were confused with one another by the SVM. Similar unsupervised "
                         "assignments can coexist with a separating supervised boundary.")
    m["Headline"]=f"The CNN's clean-to-shuffled accuracy gap was {100*cnnloss:.2f} points; the dense MLP's gap was {100*mlploss:.2f} points."

    def tab(records):
        return "\n".join(" & ".join(map(str, record)) + r"\\" for record in records)

    def pct(val):
        return f"{float(val)*100:.2f}"

    def best_epoch(kind, condition):
        return max(rows(kind+"_"+condition), key=lambda r:float(r["val_acc"]))

    m["Repository"]=m["Repo"]
    m["Revision"]=rev
    m["ChosenWidth"]=str(win)
    m["CleanAccuracy"]=percent(acc("cnn","clean"))
    with np.load(R/"split.npz") as split:
        m["PopulationRows"]=tab((name, f"{len(split[k]):,}",f"{len(split[k])//10:,}") for name,k in [
            ("Q1(a), Q2, Q3","train"),("Q1(b) memorisation","memory"),("Q4 SVM and clustering","svm")])
    local=[]
    for kind, condition, name in [("mlp","clean","Dense, clean"),("patch","clean","Patch, contiguous"),("patch","pixel","Patch, shuffled")]:
        best=best_epoch(kind,condition)
        local.append((name, f"{int(best['parameters']):,}",pct(best["val_acc"]),pct(acc(kind,condition))))
    m["LocalRows"]=tab(local)
    epochs={1,15,30,33,50,int(mem[-1]["epoch"])}
    m["MemoryRows"]=tab((r["epoch"],*[pct(r[k]) for k in ("unchanged_acc","flipped_original_acc","flipped_assigned_acc","val_acc")])
        for r in mem if int(r["epoch"]) in epochs)
    m["SpatialRows"]=tab((r["model"].upper(),r["condition"],f"{int(best_epoch(r['model'],r['condition'])['parameters']):,}",pct(r["acc"]),f"{int(r['errors']):,}",pct(r["wrong_conf"])) for r in cls if r["model"]!="patch")
    m["EncoderRows"]=tab((r["size"],f"{int(rows('ae_'+r['size'])[0]['parameters']):,}",f"{float(r['val_mse']):.5f}",f"{float(r['test_mse']):.5f}",pct(r["probe_test"])) for r in rep)
    clean_row=next(r for r in cls if r["model"]=="cnn" and r["condition"]=="clean")
    m["DenoiseRows"]=tab((label,pct(r["acc"]),f"{int(r['errors']):,}",pct(r["wrong_conf"])) for label,r in
        [("Clean image (Q2)",clean_row)]+[(r["input"],r) for r in den])
    m["SweepRows"]=tab((r["kernel"],r["C"],pct(r["train"]),pct(r["val"]),pct(r["sv_fraction"])) for r in sweep)
    m["PairRows"]=tab((NAMES[int(r["class_i"])]+", "+NAMES[int(r["class_j"])],r["errors_ij"],r["errors_ji"],
        int(r["errors_ij"])+int(r["errors_ji"]),percent((int(r["errors_ij"])+int(r["errors_ji"]))/(cm[int(r["class_i"])].sum()+cm[int(r["class_j"])].sum()))) for r in pairs)
    m["ClusterRows"]=tab((idx,*row) for idx,row in enumerate(cont))
    m["LocalFindings"]=m["LocalResult"]+" The dense-to-patch difference also includes changes in connectivity and sharing."
    m["MemoryFindings"]=m["MemoryResult"]+" The threshold describes substantial random-label fitting and cannot date the first memorised example."
    m["CleanFindings"]=(f"The clean CNN attained {percent(acc('cnn','clean'))} on the official test images, and the dense reference attained {percent(acc('mlp','clean'))}. "
                        f"Validation selected epochs {best_epoch('cnn','clean')['epoch']} and {best_epoch('mlp','clean')['epoch']}, respectively. "
                        "This establishes the clean baseline for testing the locality hypothesis.")
    m["PixelFindings"]=f"Pixel shuffling changed CNN accuracy by {-100*cnnloss:+.2f} points and MLP accuracy by {-100*mlploss:+.2f} points. The fixed permutation preserves every input value while changing the neighbours presented to each convolution."
    m["TileFindings"]=m["SpatialResult"]
    m["ConfidenceFindings"]=m["ConfidenceResult"]+" These conditional averages describe a different set of mistaken images in each condition."
    m["EncoderSelection"]=m["ReconstructionResult"]
    m["ReconstructionFindings"]=f"The selected width represents 3,072 image coordinates using {win} latent coordinates. This ratio compares dimensions; the code and the input image use different numerical precision."
    m["DenoiseFindings"]=m["DenoiseResult"]+f" In total, {paired['gains']+paired['losses']:,} images changed correctness. The net change conceals how often the classifier reversed individual decisions."
    m["ProbeFindings"]=m["ProbeResult"]+f" The strongest frozen readout attained {percent(probe_order[0]['probe_test'])}. The experiment measures information accessible to a linear classifier."
    m["SweepFindings"]="At small $C$, classification violations incur a relatively small penalty. Jointly low training and validation scores suggest underfitting. The table also shows when additional fitting on training data has a smaller return on validation."
    m["SvmFindings"]=m["SvmResult"]
    m["PairExamples"]=m["PairResult"]+" "+" ".join(f"Test image {r['image_id']} is a {NAMES[int(r['true'])]} with score {float(r['score']):+.6f} and predicted class {NAMES[int(r['predicted'])]}." for r in examples)
    m["ClusterSelection"]="K-means uses ten initializations per candidate on the training subset. Validation silhouette on the fixed sample of 2,000 by candidate count was "+", ".join(f"{r['k']} ({float(r['silhouette']):.4f})" for r in options)+f". The selected count was {k}. We assigned official test images to centroids before inspecting their labels."
    m["ClusterFindings"]=f"Test ARI was {bonus['ARI']:.4f} and test NMI was {bonus['NMI']:.4f}. ARI compares pairwise assignment agreements after adjustment for chance. NMI measures the information shared by cluster identity and class."
    m["OverlapFindings"]=(f"The class distributions for {NAMES[i]} and {NAMES[j]} overlap by {percent(overlap)} across clusters. "
        f"The SVM directly swaps these two classes on {percent(errors)} of their test examples. Overlap sums the smaller class-conditional mass in each cluster; the swap rate counts supervised errors into the other class. Similar coarse cluster memberships can coexist with a separating supervised boundary.")
    m["AbstractFindings"]=("Which image properties support recognition on CIFAR10? We intervene on neighbourhood structure, labels and reconstruction targets in one stratified split. "
        +m["Headline"]+f" Denoising corrected {paired['gains']:,} previously incorrect predictions and introduced {paired['losses']:,} errors. "
        f"On a frozen autoencoder representation the validation-selected SVM attained {percent(bonus['test_acc'])} test accuracy. "
        "The paired comparisons distinguish pixel fidelity, linear class readout and unsupervised cluster membership.")
    m["Conclusions"]=(f"Preserving intact tiles changed CNN accuracy by {100*recovered:+.2f} points relative to full pixel shuffling. "
        +m["ProbeResult"]+f" The paired counts in the denoising experiment yield a net change of {paired['gains']-paired['losses']:+,} successful predictions. "
        "These results concern the selected readouts and fixed training budgets. The supervised and unsupervised class-pair comparison further separates global cluster membership from decision-boundary geometry.")
    return m


def build(repo="",sources_only=False):
    root=Path(__file__).resolve().parent
    os.chdir(root)
    compiler=shutil.which("pdflatex")
    if not sources_only and not compiler:
        raise SystemExit("PDF compilation needs pdflatex on PATH. Use --sources-only and compile report/main.tex in Overleaf, or install a LaTeX distribution and reopen PowerShell.")
    from figures import run as refresh_plots
    from report.make_visuals import draw
    measures=collect(repo)
    refresh_plots()
    dest=P/"figures"
    dest.mkdir(parents=True,exist_ok=True)
    for name in ("q1_curves","q2_curves","memory","svm","pairs"):
        shutil.copy2(R/f"{name}.pdf",dest/f"{name}.pdf")
    draw(R,root/"data"/"cifar-10-python.tar.gz",root)
    content="\n".join(r"\newcommand{\%s}{%s}" % (key,value) for key,value in measures.items())
    (P/"measurements.tex").write_text(content+"\n",encoding="utf-8")
    if sources_only:
        print("Ready for Overleaf: report/main.tex, measurements.tex, cvpr.sty, and figures/")
        return
    # A failed compile must leave the last successful report intact.
    with tempfile.TemporaryDirectory(dir=P.resolve()) as tmp:
        for _ in range(2):
            proc=subprocess.run([compiler,"-interaction=nonstopmode","-halt-on-error","-output-directory="+tmp,"main.tex"],
                cwd=P,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace")
            if proc.returncode:
                raise SystemExit("LaTeX compilation failed:\n"+proc.stdout[-3000:])
        os.replace(Path(tmp)/"main.pdf",P/"main.pdf")
    print("Updated report/main.pdf")


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--repo",default="",help="Published HTTPS repository URL")
    arg.add_argument("--sources-only",action="store_true",help="Refresh Overleaf sources without local LaTeX")
    build(**vars(arg.parse_args()))
