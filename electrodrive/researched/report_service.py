# electrodrive/researched/report_service.py
from __future__ import annotations

"""
ResearchED ReportService: render a portable audit report for a run directory.

Design Doc requirements satisfied here:
- FR-10: generate report.html (required) and report.pdf (optional if backend available),
  embedding key plots and links to raw artifacts.
- FR-3: report is written to run_dir/report.html; plots under run_dir/plots/ and paths are relative/portable.
- FR-7: report includes workflow-relevant dashboards (solve/learn/discover/fmm), viz gallery if present.
- FR-9.6: include warnings + log coverage summary.

Repo grounding:
- Contract ensure_run_dir creates a stub report.html; ReportService overwrites it. :contentReference[oaicite:45]{index=45}
- iter_viz defines deterministic frame ordering: prefer viz_*.png, else viz.png. :contentReference[oaicite:46]{index=46}
- ResearchED manifests: prefer manifest.researched.json (UI-owned) over manifest.json (workflow-owned). :contentReference[oaicite:47]{index=47}
"""

import html
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from electrodrive.researched.contracts.run_dir import ensure_run_dir
from electrodrive.researched.contracts.manifest_schema import MANIFEST_JSON_NAME, RESEARCHED_MANIFEST_NAME
from electrodrive.researched.plot_service import PlotService

log = logging.getLogger(__name__)


def _safe_json_load(path: Path) -> Any:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _load_manifest_any(run_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    # Mirrors researched/api.py manifest preference order. :contentReference[oaicite:48]{index=48}
    for name in (RESEARCHED_MANIFEST_NAME, MANIFEST_JSON_NAME):
        obj = _safe_json_load(run_dir / name)
        if isinstance(obj, dict):
            return obj, name
    return None, ""


_VIZ_RE = re.compile(r"^viz_(\d+)$")


def _viz_sort_key(p: Path) -> tuple[int, int, str]:
    """
    Prefer numeric sort for viz_####.png; fallback to lexicographic.
    """
    m = _VIZ_RE.match(p.stem)
    if m:
        try:
            return (0, int(m.group(1)), p.name)
        except Exception:
            pass
    return (1, 0, p.name)


def _list_viz_frames(viz_dir: Path) -> List[str]:
    """
    Deterministic frame listing aligned with iter_viz._list_frame_paths:
    prefer viz_*.png; else viz.png (if present). :contentReference[oaicite:49]{index=49}

    Enhancement (FR-7): if per-frame overlay images exist (viz_####_overlay.png),
    prefer those for display in post-run reports.
    """
    try:
        if not viz_dir.is_dir():
            return []

        base = [p for p in viz_dir.glob("viz_*.png") if not p.name.endswith("_overlay.png")]
        base.sort(key=_viz_sort_key)

        if base:
            out: List[str] = []
            for p in base:
                overlay = p.with_name(p.stem + "_overlay.png")
                out.append(overlay.name if overlay.is_file() else p.name)
            return out

        single = viz_dir / "viz.png"
        if single.is_file():
            for cand in (viz_dir / "viz_overlay.png", viz_dir / "viz.png_overlay.png"):
                if cand.is_file():
                    return [cand.name]
            return [single.name]

        return []
    except Exception:
        return []


def _list_dir_shallow(dir_path: Path, *, max_items: int = 200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        if not dir_path.is_dir():
            return out
        for i, p in enumerate(sorted(dir_path.iterdir(), key=lambda x: x.name)):
            if i >= max_items:
                out.append({"path": "...", "note": "truncated"})
                break
            try:
                st = p.stat()
                out.append(
                    {
                        "path": p.name,
                        "is_dir": p.is_dir(),
                        "size": int(st.st_size) if p.is_file() else 0,
                        "mtime": float(st.st_mtime),
                    }
                )
            except Exception:
                out.append({"path": p.name})
        return out
    except Exception:
        return out


class ReportService:
    def __init__(self, *, template_dir: Path | None = None) -> None:
        # Templates live in electrodrive/researched/templates/.
        self.template_dir = Path(template_dir) if template_dir is not None else (Path(__file__).resolve().parent / "templates")
        self.plot_service = PlotService()

    def generate(
        self,
        run_dir: Path,
        *,
        pdf: bool = False,
        manifest: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        run_dir = Path(run_dir)
        out: Dict[str, Any] = {"ok": True, "run_dir": str(run_dir), "warnings": []}

        try:
            ensure_run_dir(run_dir)  # ensures plots/, report.html stub exists (FR-3). :contentReference[oaicite:50]{index=50}
        except Exception as exc:
            return {"ok": False, "run_dir": str(run_dir), "warnings": [f"ensure_run_dir failed: {exc!r}"]}

        # Load manifest.
        man = dict(manifest) if isinstance(manifest, Mapping) else None
        man_src = ""
        if man is None:
            man, man_src = _load_manifest_any(run_dir)
        if man_src:
            out["manifest_file"] = man_src

        # Generate plots (FR-7/FR-9/FR-10).
        plot_result = self.plot_service.generate_all(run_dir, manifest=man)
        out["plot_result"] = plot_result

        # Compose report context.
        plots_dir = run_dir / "plots"
        artifacts_dir = run_dir / "artifacts"
        viz_dir = run_dir / "viz"

        plots_listing = _list_dir_shallow(plots_dir)
        artifacts_listing = _list_dir_shallow(artifacts_dir)
        viz_frames = _list_viz_frames(viz_dir)

        gate_dashboard = _safe_json_load(plots_dir / "gate_dashboard.json")
        if not isinstance(gate_dashboard, dict):
            gate_dashboard = None

        # CSS (inline for portability).
        css_path = self.template_dir / "report.css"
        css = ""
        try:
            css = css_path.read_text(encoding="utf-8")
        except Exception:
            css = ""

        context: Dict[str, Any] = {
            "title": "ResearchED Run Report",
            "css": css,
            "run_dir_name": run_dir.name,
            "context_run_dir": str(run_dir),
            "manifest": man or {},
            "manifest_file": man_src,
            "plots": plots_listing,
            "artifacts": artifacts_listing,
            "viz_frames": viz_frames,
            "warnings": (plot_result.get("warnings") or []) + out["warnings"],
            "coverage": plot_result.get("coverage") or {},
            "gate_dashboard": gate_dashboard or {},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
        }

        # Render HTML from Jinja2 template (optional dependency).
        html_text = self._render_html(context=context)
        report_path = run_dir / "report.html"
        _atomic_write_text(report_path, html_text)
        out["report_html"] = str(report_path)

        # Optional PDF via WeasyPrint (best-effort).
        if pdf:
            pdf_path = run_dir / "report.pdf"
            ok_pdf = self._render_pdf(html_text, pdf_path, warnings=out["warnings"])
            if ok_pdf:
                out["report_pdf"] = str(pdf_path)
            else:
                out["report_pdf"] = None

        return out

    def _render_html(self, *, context: Dict[str, Any]) -> str:
        tpl_path = self.template_dir / "report.html.j2"
        tpl_src = ""
        try:
            tpl_src = tpl_path.read_text(encoding="utf-8")
        except Exception:
            tpl_src = ""

        # Try Jinja2 if available.
        try:
            import jinja2  # type: ignore

            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.template_dir)),
                autoescape=jinja2.select_autoescape(["html", "xml"]),
            )
            tpl = env.get_template("report.html.j2")
            return tpl.render(**context)
        except Exception:
            # Fallback: simple built-in HTML builder (no templating deps).
            return self._render_fallback(context=context, template_text=tpl_src)

    def _render_fallback(self, *, context: Dict[str, Any], template_text: str) -> str:
        """
        Minimal fallback renderer if Jinja2 isn't installed.
        Produces a functional report even without templates.
        """
        man = context.get("manifest") or {}
        cov = context.get("coverage") or {}
        plots = context.get("plots") or []
        artifacts = context.get("artifacts") or []
        viz_frames = context.get("viz_frames") or []
        warnings = context.get("warnings") or []

        run_dir_path = Path(context.get("context_run_dir", "."))

        def esc(x: Any) -> str:
            return html.escape(str(x))

        # Inline CSS.
        css = context.get("css") or ""

        def link_if_exists(rel: str, label: str | None = None) -> str:
            return f'<a href="{esc(rel)}">{esc(label or rel)}</a>'

        parts = []
        parts.append("<!doctype html><html><head><meta charset='utf-8'/>")
        parts.append(f"<title>{esc(context.get('title'))}</title>")
        parts.append("<style>")
        parts.append(css)
        parts.append("</style>")
        parts.append("</head><body>")
        parts.append("<header class='topbar'>")
        parts.append(f"<h1>{esc(context.get('title'))}</h1>")
        parts.append(f"<div class='meta'>Generated: {esc(context.get('generated_at'))}</div>")
        parts.append("</header>")

        parts.append("<section><h2>Run summary</h2>")
        parts.append("<div class='grid2'>")
        parts.append(f"<div><b>run_id</b>: {esc(man.get('run_id') or man.get('id') or '')}</div>")
        parts.append(f"<div><b>workflow</b>: {esc(man.get('workflow') or 'unknown')}</div>")
        parts.append(f"<div><b>status</b>: {esc(man.get('status') or man.get('run_status') or '')}</div>")
        parts.append(f"<div><b>started_at</b>: {esc(man.get('started_at') or '')}</div>")
        parts.append(f"<div><b>ended_at</b>: {esc(man.get('ended_at') or '')}</div>")
        git = man.get("git") if isinstance(man.get("git"), dict) else {}
        parts.append(f"<div><b>git</b>: {esc(git.get('sha'))} ({esc(git.get('branch'))})</div>")
        parts.append("</div>")
        parts.append("</section>")

        parts.append("<section><h2>Links</h2><ul>")
        for rel in ("command.txt", RESEARCHED_MANIFEST_NAME, MANIFEST_JSON_NAME, "metrics.json", "events.jsonl", "evidence_log.jsonl", "stdout.log"):
            try:
                if (run_dir_path / rel).is_file():
                    parts.append(f"<li>{link_if_exists(rel)}</li>")
            except Exception:
                pass
        parts.append("</ul></section>")

        if warnings:
            parts.append("<section><h2>Warnings</h2><ul>")
            for w in warnings:
                parts.append(f"<li>{esc(w)}</li>")
            parts.append("</ul></section>")

        # Plots gallery
        parts.append("<section><h2>Plots</h2>")
        for item in plots:
            rel = f"plots/{item.get('path')}"
            if str(item.get("path", "")).lower().endswith(".png"):
                parts.append(f"<figure class='plot'><img src='{esc(rel)}' alt='{esc(item.get('path'))}'/></figure>")
        parts.append("</section>")

        # Viz gallery
        if viz_frames:
            parts.append("<section><h2>Visualization frames</h2><div class='vizgrid'>")
            for fn in viz_frames[:60]:
                parts.append(f"<img class='vizframe' src='{esc('viz/' + fn)}' alt='{esc(fn)}'/>")
            parts.append("</div></section>")

        # Artifacts
        parts.append("<section><h2>Artifacts</h2><ul>")
        for item in artifacts:
            p = str(item.get("path", ""))
            parts.append(f"<li>{link_if_exists('artifacts/' + p, p)}</li>")
        parts.append("</ul></section>")

        # Coverage
        parts.append("<section><h2>Log coverage (FR-9.6)</h2>")
        parts.append("<pre class='code'>")
        parts.append(esc(json.dumps(cov, indent=2, sort_keys=True)))
        parts.append("</pre></section>")

        parts.append("</body></html>")
        return "\n".join(parts)

    def _render_pdf(self, html_text: str, pdf_path: Path, *, warnings: List[str]) -> bool:
        """
        Optional PDF generation. Best-effort: try WeasyPrint; otherwise skip gracefully (FR-10).
        """
        try:
            from weasyprint import HTML  # type: ignore
        except Exception:
            warnings.append("PDF backend not available (install weasyprint to enable report.pdf)")
            return False
        try:
            HTML(string=html_text, base_url=str(pdf_path.parent)).write_pdf(str(pdf_path))
            return True
        except Exception as exc:
            warnings.append(f"PDF render failed: {exc!r}")
            return False
