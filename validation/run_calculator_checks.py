"""Check the calculator against the command line, and compile the appendix.

    ./.venv/bin/python validation/run_calculator_checks.py

Exit status is zero only when every check passes. The app check needs the
calculator's dev server running; start it with ``npm run dev`` in ``calculator``
or pass ``--base-url``. If it is not reachable the check is recorded as skipped
and the script still checks the Python API against the CLI.

What is compared: every method's estimate, standard error, interval bounds and
width, plus each design projection. They must agree exactly, because the two
paths run the same command on the same bytes. A tolerance is used only to state
the observed difference, not to excuse one.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppict import __version__  # noqa: E402
from ppict.report import build_report  # noqa: E402

DATA = ROOT / "calculator" / "data" / "simulated_trial.csv"
RESULTS = ROOT / "validation" / "results"

METHOD_FIELDS = ("estimate", "se", "ci_low", "ci_high", "ci_width", "width_ratio")
PLAN_FIELDS = (
    "variance_unadjusted",
    "variance_adjusted",
    "n_required_unadjusted",
    "n_required_adjusted",
    "patients_saved",
    "cost_saved",
    "months_saved",
    "width_ratio",
    "ci_narrowing",
    "equivalent_unadjusted_n",
)
EXACT = 1e-12


def run_cli(cost: float, power: float, rate: float) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "ppict.cli",
        "analyze",
        "--csv",
        str(DATA),
        "--cost",
        f"{cost:g}",
        "--power",
        f"{power:g}",
        "--patients-per-month",
        f"{rate:g}",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"the CLI failed with exit {completed.returncode}:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def post_app(
    base_url: str,
    cost: float,
    power: float,
    rate: float,
    file_path: Optional[Path] = None,
) -> Dict[str, Any]:
    boundary = "----ppictcalculatorcheck"
    fields = {
        "cost": f"{cost:g}",
        "power": f"{power:g}",
        "patientsPerMonth": f"{rate:g}",
    }
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    if file_path is not None:
        chunks.append(
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="trial.csv"\r\n'
                "Content-Type: text/csv\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(file_path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/analyze",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def max_difference(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_methods = {row["method"]: row for row in left["analysis"]["methods"]}
    right_methods = {row["method"]: row for row in right["analysis"]["methods"]}
    if set(left_methods) != set(right_methods):
        raise SystemExit(
            f"method sets differ: {sorted(left_methods)} versus {sorted(right_methods)}"
        )
    worst = 0.0
    for name in left_methods:
        for field in METHOD_FIELDS:
            a = left_methods[name][field]
            b = right_methods[name][field]
            if a is None or b is None:
                if a is not b:
                    raise SystemExit(f"{name}.{field} is missing on one side")
                continue
            worst = max(worst, abs(float(a) - float(b)))
    for field in PLAN_FIELDS:
        a, b = left["plan"][field], right["plan"][field]
        if a is None or b is None:
            if a is not b:
                raise SystemExit(f"plan.{field} is missing on one side")
            continue
        worst = max(worst, abs(float(a) - float(b)))
    return float(worst)


def compile_sap(tex: str, out_dir: Path) -> Tuple[bool, str, Optional[Path]]:
    """Compile the appendix with latexmk. Returns (ok, note, pdf path)."""
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        return False, "latexmk was not found on PATH", None

    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "sap_appendix.tex"
    tex_path.write_text(tex, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="ppict-sap-") as tmp:
        work = Path(tmp)
        shutil.copy2(tex_path, work / "appendix.tex")
        completed = subprocess.run(
            [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "appendix.tex"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
        pdf = work / "appendix.pdf"
        log = work / "appendix.log"
        log_text = log.read_text(errors="replace") if log.exists() else ""
        overfull = len(re.findall(r"^Overfull \\hbox", log_text, flags=re.MULTILINE))
        page_match = re.search(r"Output written on .*?\((\d+) pages?", log_text)
        pages = int(page_match.group(1)) if page_match else 0
        if completed.returncode != 0 or not pdf.exists():
            tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-20:])
            return False, f"latexmk failed with exit {completed.returncode}\n{tail}", None
        shutil.copy2(pdf, out_dir / "sap_appendix.pdf")
        note = f"{pages} pages, {pdf.stat().st_size} bytes, {overfull} overfull boxes"
        if overfull:
            return False, note, out_dir / "sap_appendix.pdf"
        return True, note, out_dir / "sap_appendix.pdf"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3040")
    parser.add_argument("--skip-app", action="store_true")
    parser.add_argument("--cost", type=float, default=4200.0)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--rate", type=float, default=25.0)
    args = parser.parse_args(argv)

    if not DATA.exists():
        raise SystemExit(
            f"missing {DATA}. Run: ./.venv/bin/python tools/make_demo_trial.py"
        )

    command_text = (
        "python -m ppict.cli analyze --csv calculator/data/simulated_trial.csv "
        f"--cost {args.cost:g} --power {args.power:g} --patients-per-month {args.rate:g}"
    )
    # Relative so the appendix records the dataset path as a reader would type it.
    direct = build_report(
        DATA.relative_to(ROOT),
        cost_per_patient=args.cost,
        power=args.power,
        patients_per_month=args.rate,
        command=command_text,
    ).as_dict()
    cli = run_cli(args.cost, args.power, args.rate)

    app: Optional[Dict[str, Any]] = None
    app_note = "skipped with --skip-app"
    if not args.skip_app:
        try:
            app = post_app(args.base_url, args.cost, args.power, args.rate)
            if app.get("contract") != direct.get("contract"):
                raise SystemExit(
                    f"contract mismatch: app {app.get('contract')} versus package {direct.get('contract')}"
                )
            app_note = f"checked at {args.base_url}, with and without an uploaded copy of the file"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            app_note = f"not reachable at {args.base_url} ({exc}); server not running"
            app = None

    diffs = {
        "Python API to CLI": max_difference(direct, cli),
    }
    if app is not None:
        diffs["CLI to app route"] = max_difference(cli, app)
        diffs["Python API to app route"] = max_difference(direct, app)
        # The same file through the upload path, which is the path a user takes.
        uploaded = post_app(args.base_url, args.cost, args.power, args.rate, file_path=DATA)
        diffs["CLI to app route, file uploaded"] = max_difference(cli, uploaded)

    worst = max(diffs.values())
    agreement_ok = worst <= EXACT

    sap_ok, sap_note, pdf_path = compile_sap(direct["sap_tex"], RESULTS)
    dash_ok = ("\u2014" not in direct["sap_tex"]) and ("\u2013" not in direct["sap_tex"])

    methods = {row["method"]: row for row in cli["analysis"]["methods"]}
    plan = cli["plan"]

    lines = [
        "# Calculator check",
        "",
        "Produced by `validation/run_calculator_checks.py`. Every number below was",
        "measured in that run; nothing here is typed by hand.",
        "",
        f"- ppict version: `{__version__}`",
        f"- Dataset: `calculator/data/simulated_trial.csv`",
        f"- SHA-256: `{cli['analysis']['dataset']['sha256']}`",
        f"- Trial rows: {cli['analysis']['dataset']['n_trial']} "
        f"({cli['analysis']['dataset']['n_treated']} treated, "
        f"{cli['analysis']['dataset']['n_control']} control), "
        f"{cli['analysis']['dataset']['n_external']} external",
        f"- Inputs: cost per patient {args.cost:g}, power {args.power:g}, "
        f"recruitment {args.rate:g} per month",
        f"- App check: {app_note}",
        "",
        "## Agreement on identical input",
        "",
        "Each row is the largest absolute difference over all four method estimates,",
        "standard errors, interval bounds, widths and width ratios, plus every design",
        "projection.",
        "",
        "| comparison | max absolute difference |",
        "| --- | --- |",
    ]
    for name, value in diffs.items():
        lines.append(f"| {name} | {value:.1e} |")
    lines += [
        "",
        f"Result: {'exact' if worst == 0 else f'within {worst:.1e}'}, "
        f"against a tolerance of {EXACT:.0e}.",
        "",
        "## Estimates from the shared estimator",
        "",
        "| method | estimate | 95% interval | width | width vs unadjusted |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ("unadjusted", "ancova", "ppi_single", "ppi_per_arm"):
        if name not in methods:
            lines.append(f"| {name} | not available | | | |")
            continue
        row = methods[name]
        ratio = row["width_ratio"]
        lines.append(
            f"| {name} | {row['estimate']:.4f} | "
            f"{row['ci_low']:.4f} to {row['ci_high']:.4f} | "
            f"{row['ci_width']:.4f} | {ratio:.3f} |"
        )

    lines += [
        "",
        "## Design projection",
        "",
        f"- Adjusted interval narrowing measured on the file: "
        f"{100 * (1 - methods['ppi_single']['width_ratio']):.1f}%",
        f"- Required total at {args.power:.0%} power: "
        f"{plan['n_required_unadjusted']:.0f} unadjusted, "
        f"{plan['n_required_adjusted']:.0f} adjusted",
        f"- Patients saved: {plan['patients_saved']:.0f}",
        f"- Cost avoided: ${plan['cost_saved']:,.0f}",
        f"- Recruitment time saved: {plan['months_saved']:.1f} months",
        "",
        "## Statistical analysis plan appendix",
        "",
        f"- Written to `validation/results/sap_appendix.tex`",
        f"- Compilation: {'passed' if sap_ok else 'failed'}, {sap_note}",
        "- PDF: `validation/results/sap_appendix.pdf`"
        if pdf_path
        else "- PDF: not produced",
        f"- Em and en dashes in the appendix: {'none' if dash_ok else 'found'}",
        "",
        "## Commands run",
        "",
        "```sh",
        "./.venv/bin/python validation/run_calculator_checks.py",
        command_text,
        "```",
        "",
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS / "CALCULATOR_CHECK.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    ok = agreement_ok and sap_ok and dash_ok
    print(f"agreement: max difference {worst:.1e} ({'pass' if agreement_ok else 'FAIL'})")
    for name, value in diffs.items():
        print(f"  {name}: {value:.1e}")
    print(f"app: {app_note}")
    print(f"sap: {'pass' if sap_ok else 'FAIL'}, {sap_note}")
    print(f"dashes: {'none' if dash_ok else 'FOUND'}")
    print(f"wrote {report_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
