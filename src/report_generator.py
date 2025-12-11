import json, os
from datetime import datetime

OUT = "reports/final_report.md"

def _load_json(p):
    if os.path.exists(p):
        return json.load(open(p))
    return None

def generate_final_report(out_path=OUT):
    offline = _load_json("reports/offline_metrics.json")
    ab = _load_json("reports/ab_results.json")
    rtb = _load_json("reports/rtb_report.json")

    lines = []
    lines.append(f"# Final Ads ML Report\nGenerated: {datetime.utcnow().isoformat()}Z\n")
    lines.append("## 1) Offline Model Comparison\n")
    if offline:
        lines.append("```\n" + json.dumps(offline, indent=2) + "\n```\n")
    else:
        lines.append("_offline metrics missing_\n")

    lines.append("## 2) A/B Test Summary\n")
    if ab:
        lines.append("```\n" + json.dumps(ab, indent=2) + "\n```\n")
    else:
        lines.append("_ab test results missing_\n")

    lines.append("## 3) RTB Simulation Summary\n")
    if rtb:
        lines.append("```\n" + json.dumps(rtb, indent=2) + "\n```\n")
    else:
        lines.append("_rtb results missing_\n")

    lines.append("## 4) Recommendations\n")
    # naive recommender: if RPM bootstrap CI positive and z-test p < 0.05, recommend deploy
    deploy_reco = "insufficient data"
    try:
        if ab and ab.get("rpm_bootstrap") and ab["rpm_bootstrap"]["ci"][0] > 0:
            deploy_reco = "Recommend deploy candidate model (RPM uplift +ve CI)"
        elif ab and ab.get("ctr_ztest") and ab["ctr_ztest"]["p_value"] < 0.05:
            deploy_reco = "Recommend deploy candidate model (CTR stat. sig.)"
        else:
            deploy_reco = "Do not deploy automatically; run longer experiment / check SRE issues"
    except Exception:
        deploy_reco = "ERROR in recommendation logic"
    lines.append(deploy_reco + "\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("Final report saved to", out_path)
    return out_path

if __name__ == "__main__":
    generate_final_report()
