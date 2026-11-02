"""Print the numbers of Tables 5 to 10 of the paper directly from figs/p3/results.json.

No simulation is run: the script only reads the stored result file, so every number quoted
in the paper can be checked in a few seconds.   Usage:  python3 make_tables.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(os.path.dirname(HERE), "figs", "p3", "results.json")))
SCHEMES = [("raw", "No action mask"), ("step1", "One-step exact filter"),
           ("online", "Backup-orbit filter"), ("mask", "Certified mask, proposed")]


def pct(x, d=2):
    return ("%." + str(d) + "f") % (100.0 * x)


def head(s):
    print("\n" + s + "\n" + "-" * len(s))


g, inv, lc = R["grid"], R["inv"], R["lemma_check"]
head("Table 5. Output of the verifier on the reference quantizer")
print("Cells                                      ", g["ncell"])
print("Edges of the cell graph                    ", g["edges"])
print("Graph construction time (s)                ", round(g["build"], 4))
print("Fixed-point sweeps                         ", g["it"])
print("Fixed-point time (s)                       ", round(g["fix"], 3))
print("Certified cells                            ", inv["cells"])
print("Certified fraction (%)                     ", pct(inv["frac"]))
print("Admissible actions per certified cell      ", round(inv["mean_actions"], 2))
print("Certified cells reachable from the start   ", inv["reach"], "(%s%%)" % pct(inv["reach_frac"], 1))
print("Sampled state-action pairs                 ", lc["pairs"])
print("Violations of Lemma 1, Lemma 2, Theorem 1  ", lc["viol_lemma1"], lc["viol_lemma2"], lc["viol_thm1"])
print("Listed successor cells reached by a sample ", lc["succ_observed"], "of", lc["succ_listed"],
      "(%d%%)" % round(100 * lc["tight"]))

head("Table 6. Training outcome and certified fraction of each learned table")
print("%-28s %8s %9s %8s %9s %10s %9s" % ("Scheme", "Succ(%)", "Coll(%)", "Collis.", "Dead-end", "Unvis.(%)", "Cert.(%)"))
for k, name in SCHEMES:
    t, v = R["train"][k], R["ver"][k]
    print("%-28s %8s %9s %8d %9d %10s %9s" % (name, pct(t["succ"], 1), pct(t["coll"], 1), t["coll_total"], t["dead"],
                                              pct(v["unvisited_in_S"], 1), pct(v["frac"])))
print("%-28s %8s %9s %8s %9s %10s %9s" % ("Repaired table", "-", "-", "-", "-", "-", pct(R["repair"]["frac"])))

head("Table 7. Monte Carlo cross-check of the certificate")
print("Free starts inside the certified set :", R["mc"]["inside"][1], "draws,", R["mc"]["inside"][0], "violations")
print("Free starts outside the certified set:", R["mc"]["outside"][1], "draws,", R["mc"]["outside"][0], "violations")
for n, v in R["mc"]["curve"]:
    print("Uniform sampling, backup-orbit table :", n, "draws,", v, "violations")

head("Table 8. Sound and unsound abstractions of the same map")
names = {"center": "Cell center only", "sample27": "27 sample points", "open": "Half-open cells, no margins",
         "sound": "Closed cells with margins, proposed"}
for u in R["unsound"]:
    print("%-30s edges %8d  certified %6s%%  violations %5d of %d  wrong successor %6d of %d" % (
        names.get(u["tag"], u["tag"]), u["edges"], pct(u["frac"]), u["viol"], u["draws"], u["lat_bad"], u["lat_tot"]))

head("Table 9. Certified fraction and verification cost against quantizer resolution")
print("%9s %8s %9s %10s %7s %9s" % ("cell (m)", "heading", "cells", "cert.(%)", "sweeps", "time (s)"))
for r in R["res"]:
    print("%9s %8d %9d %10s %7d %9.3f" % (r["cell"], r["nth"], r["ncell"], pct(r["frac"]), r["it"], r["time"]))

head("Table 10. Results in three-dimensional environments")
E3 = R["threeD"]
for key, lab in [("ncyl", "Cylinders"), ("nblk", "Blocks"), ("ncell", "Cells"), ("edges", "Edges"), ("it", "Fixed-point sweeps")]:
    print("%-34s" % lab, *["%12s" % E3[e][key] for e in ("E1", "E2", "E3")])
print("%-34s" % "Graph construction time (s)", *["%12.3f" % E3[e]["build"] for e in ("E1", "E2", "E3")])
print("%-34s" % "Fixed-point time (s)", *["%12.3f" % E3[e]["fix"] for e in ("E1", "E2", "E3")])
print("%-34s" % "Certified fraction (%)", *["%12s" % pct(E3[e]["frac"]) for e in ("E1", "E2", "E3")])
print("%-34s" % "Admissible actions, mean", *["%12.2f" % E3[e]["mean_act"] for e in ("E1", "E2", "E3")])
for k, lab in [("off", "No mask"), ("step1", "One-step filter"), ("mask", "Certified mask")]:
    print("%-34s" % ("Collisions in training: " + lab), *["%12d" % E3[e]["train"][k]["coll_total"] for e in ("E1", "E2", "E3")])
for k, lab in [("off", "No mask"), ("step1", "One-step filter"), ("mask", "Certified mask")]:
    print("%-34s" % ("Success, last 500 (%): " + lab), *["%12s" % pct(E3[e]["train"][k]["succ"], 1) for e in ("E1", "E2", "E3")])
for k, lab in [("off", "No mask"), ("step1", "One-step filter"), ("mask", "Certified mask")]:
    print("%-34s" % ("Certified fraction of table: " + lab), *["%12s" % pct(E3[e]["train"][k]["cert"]) for e in ("E1", "E2", "E3")])
print("%-34s" % "Violations, starts inside", *["%12d" % E3[e]["mc"]["inside"][0] for e in ("E1", "E2", "E3")])
print("%-34s" % "Violations, starts outside", *["%12d" % E3[e]["mc"]["outside"][0] for e in ("E1", "E2", "E3")])
print("%-34s" % "Greedy track length of the mask", *["%12.1f" % E3[e]["train"]["mask"]["length"] for e in ("E1", "E2", "E3")])
