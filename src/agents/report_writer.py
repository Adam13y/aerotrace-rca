"""Report writer agent: assembles and renders the Root Cause Analysis report.

Outputs (all under ``data/reports/`` with a ``YYYYmmdd_HHMMSS`` stamp):

* ``rca_report_<stamp>.md``  - full aerospace RCA report.
* ``rca_summary_<stamp>.md`` - one-page quick-look summary.
* ``rca_report_<stamp>.json`` - machine-readable report (when JSON output is requested).
* ``rca_plots_<stamp>.png``   - per-sensor telemetry plot with anomaly spans (if matplotlib is available).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.models.telemetry_schema import (
    AnomalyEvent,
    Hypothesis,
    RCAReport,
    Severity,
    TelemetryAnalysis,
    TelemetryReading,
    anomaly_labels,
)
from src.utils.helpers import LLMRouter, LLMUnavailableError, format_met, readings_to_frame
from src.utils.logger import get_logger

logger = get_logger("agents.report_writer")

EXEC_SUMMARY_SYSTEM = """You are the chair of a launch-vehicle anomaly review board writing the executive summary of a
root cause analysis report for programme management. Be factual, concise (120-200 words), use engineering
language, state confidence levels and do not invent data beyond what is provided. Plain prose, no headings."""

_SEVERITY_ICON = {Severity.CRITICAL: "🔴", Severity.WARNING: "🟠", Severity.INFO: "🔵"}
_SEVERITY_COLOR = {Severity.CRITICAL: "#d62728", Severity.WARNING: "#ff7f0e", Severity.INFO: "#1f77b4"}


@dataclass(slots=True)
class ReportConfig:
    """Configuration for :class:`ReportWriter`."""

    mission_name: str = "Unnamed mission"
    vehicle: str = "Unspecified vehicle"
    include_plots: bool = True
    use_llm_executive_summary: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ReportConfig:
        """Build from the ``report_writer`` section of ``agents.yaml``."""
        data = data or {}
        return cls(
            mission_name=str(data.get("mission_name", "Unnamed mission")),
            vehicle=str(data.get("vehicle", "Unspecified vehicle")),
            include_plots=bool(data.get("include_plots", True)),
            use_llm_executive_summary=bool(data.get("use_llm_executive_summary", True)),
        )


def _cell(text: Any) -> str:
    """Escape a value for use inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


class ReportWriter:
    """Build :class:`RCAReport` objects and render them to Markdown / JSON."""

    def __init__(
        self,
        report_dir: str | Path,
        router: LLMRouter | None = None,
        config: ReportConfig | None = None,
        use_llm: bool = True,
    ) -> None:
        """Create the agent.

        Args:
            report_dir: Output directory for reports.
            router: Optional LLM router for the executive summary narrative.
            config: Report configuration.
            use_llm: Disable LLM narrative even if a router is supplied.
        """
        self.report_dir = Path(report_dir)
        self.router = router
        self.config = config or ReportConfig()
        self.use_llm = use_llm and router is not None and self.config.use_llm_executive_summary

    # ------------------------------------------------------------------ #
    # Report assembly
    # ------------------------------------------------------------------ #
    def build_report(
        self,
        anomalies: list[AnomalyEvent],
        hypotheses: list[Hypothesis],
        analysis: TelemetryAnalysis | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RCAReport:
        """Assemble the report content (summary, conclusion, recommendations)."""
        actionable = [a for a in anomalies if not a.expected]
        report = RCAReport(
            title=f"Root Cause Analysis - {self.config.mission_name}",
            summary=self._deterministic_summary(anomalies, actionable, hypotheses, analysis),
            anomalies=sorted(anomalies, key=lambda a: a.timestamp),
            hypotheses=hypotheses,
            conclusion=self._conclusion(actionable, hypotheses),
            recommendations=self._recommendations(actionable, hypotheses),
            analysis=analysis,
            metadata={"mission": self.config.mission_name, "vehicle": self.config.vehicle, **(metadata or {})},
        )
        if self.use_llm and actionable:
            narrative = self._llm_summary(report)
            if narrative:
                report = report.model_copy(update={"summary": narrative, "metadata": {**report.metadata, "summary_source": "llm"}})
        return report

    def _deterministic_summary(
        self,
        anomalies: list[AnomalyEvent],
        actionable: list[AnomalyEvent],
        hypotheses: list[Hypothesis],
        analysis: TelemetryAnalysis | None,
    ) -> str:
        counts = {s: sum(a.severity == s for a in actionable) for s in Severity}
        expected = len(anomalies) - len(actionable)
        span = f" over {analysis.duration_s:.0f}s of telemetry from {len(analysis.sensors)} sensors" if analysis else ""
        parts = [
            f"Automated analysis{span} identified {len(actionable)} actionable anomalies "
            f"({counts[Severity.CRITICAL]} critical, {counts[Severity.WARNING]} warning, {counts[Severity.INFO]} info)"
            f" and {expected} expected transients associated with planned vehicle events."
        ]
        if actionable:
            first = min(actionable, key=lambda a: a.timestamp)
            parts.append(
                f"The first actionable deviation was the {first.label}."
            )
        if hypotheses:
            lead = hypotheses[0]
            parts.append(f"The leading hypothesis is '{lead.title}' (confidence {lead.confidence:.0%}, {lead.subsystem}).")
        else:
            parts.append("No root-cause hypotheses were required.")
        return " ".join(parts)

    @staticmethod
    def _conclusion(actionable: list[AnomalyEvent], hypotheses: list[Hypothesis]) -> str:
        if not actionable:
            return "No actionable anomalies were detected; all deviations are attributable to planned vehicle events."
        if not hypotheses:
            return "Anomalies were detected but no root-cause hypothesis could be formed; manual review is required."
        lead = hypotheses[0]
        covered = {aid for h in hypotheses for aid in h.related_anomalies}
        uncovered = [a for a in actionable if a.id not in covered]
        text = (
            f"The most probable root cause is **{lead.title}** ({lead.confidence:.0%} confidence), "
            f"explaining {len(lead.related_anomalies)} of {len(actionable)} actionable anomalies. "
        )
        others = [h for h in hypotheses[1:] if set(h.related_anomalies) - set(lead.related_anomalies)]
        if others:
            text += "Independent events are best explained by: " + "; ".join(
                f"{h.title} ({h.confidence:.0%})" for h in others[:3]
            ) + ". "
        if uncovered:
            text += f"{len(uncovered)} anomalies are not explained by any hypothesis and require manual review. "
        text += "Findings are preliminary until confirmed by the responsible subsystem engineers."
        return text

    @staticmethod
    def _recommendations(actionable: list[AnomalyEvent], hypotheses: list[Hypothesis]) -> list[str]:
        severity_by_id = {a.id: a.severity for a in actionable}
        recommendations: list[str] = []
        seen: set[str] = set()
        for hypothesis in hypotheses:
            severities = [severity_by_id[i] for i in hypothesis.related_anomalies if i in severity_by_id]
            top = max(severities, key=lambda s: s.rank) if severities else Severity.INFO
            priority = {Severity.CRITICAL: "P1", Severity.WARNING: "P2", Severity.INFO: "P3"}[top]
            for action in hypothesis.recommended_actions:
                key = action.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    recommendations.append(f"[{priority}] {action.strip()} (re: {hypothesis.title})")
        if any(a.severity == Severity.CRITICAL for a in actionable):
            recommendations.append("[P1] Convene an anomaly review board and place affected hardware on hold pending closure.")
        recommendations.append("[P3] Add confirmed findings to the AeroTrace knowledge base to improve future triage.")
        order = {"[P1]": 0, "[P2]": 1, "[P3]": 2}
        return sorted(recommendations, key=lambda r: order.get(r[:4], 3))[:15]

    def _llm_summary(self, report: RCAReport) -> str | None:
        assert self.router is not None
        labels = anomaly_labels(report.anomalies)
        payload = {
            "mission": self.config.mission_name,
            "vehicle": self.config.vehicle,
            "draft_summary": report.summary,
            "anomalies": [a.to_prompt_dict(labels) for a in report.anomalies if not a.expected],
            "hypotheses": [
                {
                    **h.model_dump(include={"title", "confidence", "subsystem", "description"}),
                    "related_anomalies": [labels.get(i, i) for i in h.related_anomalies],
                }
                for h in report.hypotheses
            ],
            "conclusion": report.conclusion,
        }
        try:
            response = self.router.complete(
                EXEC_SUMMARY_SYSTEM,
                "Write the executive summary for this RCA:\n" + json.dumps(payload, indent=1, default=str),
                prefer="cloud",
            )
        except LLMUnavailableError as exc:
            logger.info("Executive summary LLM unavailable, using deterministic summary: %s", exc)
            return None
        text = response.text.strip()
        return text or None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render_markdown(self, report: RCAReport, plot_file: str | None = None) -> str:
        """Render the full RCA report as Markdown."""
        actionable = [a for a in report.anomalies if not a.expected]
        expected = [a for a in report.anomalies if a.expected]
        labels = anomaly_labels(report.anomalies)
        counts = report.severity_counts
        meta = report.metadata
        lines: list[str] = [
            f"# {report.title}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Mission | {_cell(meta.get('mission', '-'))} |",
            f"| Vehicle | {_cell(meta.get('vehicle', '-'))} |",
            f"| Data source | {_cell(meta.get('source_file', report.analysis.source_file if report.analysis else '-'))} |",
            f"| Report generated | {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')} |",
            f"| Pipeline run ID | {_cell(meta.get('run_id', '-'))} |",
            f"| Models used | {_cell(', '.join(meta.get('providers', [])) or 'rule-based only')} |",
            f"| Severity counts | 🔴 {counts['CRITICAL']} critical · 🟠 {counts['WARNING']} warning · 🔵 {counts['INFO']} info |",
            "",
            "## 1. Executive Summary",
            "",
            report.summary,
            "",
            "## 2. Conclusion",
            "",
            report.conclusion,
            "",
        ]
        lines += self._render_data_overview(report.analysis)
        lines += ["## 4. Event Timeline", ""]
        if report.anomalies:
            lines += [
                "| MET | Anomaly | Duration | Baseline → Peak | Category | Flight phase |",
                "|---|---|---|---|---|---|",
            ]
            for a in report.anomalies:
                tag = " (expected)" if a.expected else ""
                lines.append(
                    f"| {format_met(a.mission_elapsed_s)} | {_SEVERITY_ICON[a.severity]} {_cell(a.label)}{tag} "
                    f"| {a.duration_s:.1f}s | {_fmt(a.baseline_value)} → {_fmt(a.peak_value)} {_cell(a.unit)} "
                    f"| {_cell(a.classification.get('category', '-'))} | {_cell(a.flight_phase or '-')} |"
                )
        else:
            lines.append("No anomalies detected.")
        lines.append("")
        if plot_file:
            lines += ["### Telemetry overview", "", f"![Telemetry with anomaly spans]({plot_file})", ""]

        lines += ["## 5. Detailed Findings", ""]
        if not actionable:
            lines += ["No actionable anomalies.", ""]
        for i, a in enumerate(actionable, 1):
            cls = a.classification
            lines += [
                f"### 5.{i} {_SEVERITY_ICON[a.severity]} {a.label}",
                "",
                f"- **Window:** {a.timestamp.isoformat(timespec='milliseconds')} → "
                f"{(a.end_timestamp or a.timestamp).isoformat(timespec='milliseconds')} ({a.duration_s:.1f}s)",
                f"- **Flight phase:** {a.flight_phase or 'unknown'}",
                f"- **Deviation score:** {a.score:.1f}σ (methods: {', '.join(a.detection_methods)})",
                f"- **Triage:** {cls.get('category', '-')} / {cls.get('subsystem', '-')} "
                f"(confidence {float(cls.get('confidence', 0)):.0%}, by {cls.get('source', '-')})",
            ]
            if cls.get("fault_class"):
                retagged = f" (triage had labelled it '{cls['triage_category']}')" if cls.get("triage_category") else ""
                lines.append(f"- **Signal check:** {cls['fault_class']} - {cls.get('fault_class_reason', '')}{retagged}")
            if cls.get("original_severity"):
                lines.append(
                    f"- **Severity adjusted:** {cls['original_severity']} → {a.severity.value} "
                    f"by {cls.get('severity_adjusted_by', 'triage')}"
                )
            if a.correlated_with:
                lines.append(f"- **Correlated anomalies:** {'; '.join(labels.get(c, c) for c in a.correlated_with)}")
            lines += ["", a.description, ""]
            if cls.get("assessment"):
                lines += [f"> Triage assessment: {cls['assessment']}", ""]

        lines += ["## 6. Root Cause Hypotheses", ""]
        if not report.hypotheses:
            lines += ["No hypotheses generated.", ""]
        else:
            lines += ["| Rank | Hypothesis | Confidence | Subsystem | Anomalies | Source |", "|---|---|---|---|---|---|"]
            for rank, h in enumerate(report.hypotheses, 1):
                lines.append(
                    f"| {rank} | {_cell(h.title)} | {_confidence_bar(h.confidence)} | {_cell(h.subsystem)} "
                    f"| {len(h.related_anomalies)} | {_cell(h.source)} |"
                )
            lines.append("")
            for rank, h in enumerate(report.hypotheses, 1):
                lines += [f"### H{rank}. {h.title} - {h.confidence:.0%}", "", h.description, ""]
                if h.related_anomalies:
                    lines += ["**Related anomalies:**", ""] + [f"- {labels.get(x, x)}" for x in h.related_anomalies] + [""]
                lines += ["**Supporting evidence**", ""] + [f"- {e}" for e in h.supporting_evidence or ["None listed."]] + [""]
                lines += ["**Counter-evidence / open questions**", ""] + [f"- {e}" for e in h.counter_evidence or ["None listed."]] + [""]

        lines += ["## 7. Recommendations", ""]
        lines += [f"{i}. {r}" for i, r in enumerate(report.recommendations, 1)] or ["None."]
        lines.append("")

        if expected:
            lines += ["## 8. Expected Transients (for reference)", ""]
            lines += [f"- **{a.label}** - {a.description}" for a in expected]
            lines.append("")

        lines += [
            "## Appendix A - Methodology",
            "",
            "- **Detection:** per-sensor robust z-score of residuals against a centred rolling median, trailing "
            "moving-average deviation, robust derivative analysis and hard engineering limits. Flagged samples "
            "within the merge gap are combined into events and graded by peak deviation (σ).",
            "- **Context:** events are tagged with flight phase; anomalies on different sensors within the "
            "correlation window are linked. Overlaps with planned events are graded INFO and marked expected.",
            "- **Triage:** local LLM (Ollama) classification with Claude and rule-based fallbacks.",
            "- **Hypotheses:** Claude-led reasoning over anomalies, correlation groups and knowledge-base matches, "
            "with local-LLM and knowledge-base fallbacks.",
            "",
            "*This report was generated automatically by AeroTrace-RCA and must be reviewed by qualified engineers "
            "before any flight-safety decision.*",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_data_overview(analysis: TelemetryAnalysis | None) -> list[str]:
        lines = ["## 3. Data Overview", ""]
        if analysis is None:
            return lines + ["No dataset overview available.", ""]
        lines += [
            f"- **Record:** {analysis.start.isoformat(timespec='seconds')} → {analysis.end.isoformat(timespec='seconds')} "
            f"({analysis.duration_s:.1f}s, {analysis.total_readings:,} samples)",
            "",
            "| Sensor | Unit | Samples | Rate (Hz) | Min | Max | Mean | Std | Imputed |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in analysis.sensors:
            lines.append(
                f"| {_cell(s.sensor_id)} | {_cell(s.unit)} | {s.samples} | {s.sample_rate_hz:g} | {_fmt(s.minimum)} "
                f"| {_fmt(s.maximum)} | {_fmt(s.mean)} | {_fmt(s.std)} | {s.imputed_samples} |"
            )
        lines.append("")
        if analysis.data_quality_notes:
            lines += ["**Data quality notes**", ""] + [f"- {n}" for n in analysis.data_quality_notes] + [""]
        return lines

    def render_summary(self, report: RCAReport, report_file: str | None = None) -> str:
        """Render a one-page quick-look summary."""
        actionable = [a for a in report.anomalies if not a.expected]
        counts = report.severity_counts
        lines = [
            f"# Quick-Look: {report.title}",
            "",
            f"*Generated {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}*"
            + (f" · full report: `{report_file}`" if report_file else ""),
            "",
            f"**Status:** {'🔴 CRITICAL anomalies present' if counts['CRITICAL'] else '🟠 Warnings present' if counts['WARNING'] else '🟢 Nominal'}",
            "",
            f"**Anomalies:** {len(actionable)} actionable ({counts['CRITICAL']} critical / {counts['WARNING']} warning) "
            f"+ {len(report.anomalies) - len(actionable)} expected transients",
            "",
            "## Top findings",
            "",
        ]
        top = sorted(actionable, key=lambda a: (-a.severity.rank, a.timestamp))[:5]
        lines += [
            f"- {_SEVERITY_ICON[a.severity]} **{a.label}** - "
            f"{a.classification.get('category', 'unclassified')}; {_fmt(a.baseline_value)} → {_fmt(a.peak_value)} {a.unit}"
            for a in top
        ] or ["- None"]
        lines += ["", "## Leading hypotheses", ""]
        lines += [f"{i}. **{h.title}** - {h.confidence:.0%}" for i, h in enumerate(report.hypotheses[:3], 1)] or ["- None"]
        lines += ["", "## Immediate actions", ""]
        lines += [f"- {r}" for r in report.recommendations if r.startswith("[P1]")][:5] or ["- None"]
        lines += ["", "## Conclusion", "", report.conclusion, ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def write(
        self,
        report: RCAReport,
        readings: Sequence[TelemetryReading] | None = None,
        formats: Sequence[str] = ("markdown",),
        timestamp: datetime | None = None,
    ) -> dict[str, Path]:
        """Write the report files.

        Args:
            report: Report to render.
            readings: Original telemetry, used for the overview plot.
            formats: Any of ``"markdown"`` and ``"json"``.
            timestamp: Override for the filename stamp (defaults to now, UTC).

        Returns:
            Mapping of output kind (``report``, ``summary``, ``json``, ``plot``) to path.
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stamp = (timestamp or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
        paths: dict[str, Path] = {}

        plot_path: Path | None = None
        if self.config.include_plots and readings and "markdown" in formats:
            plot_path = self._render_plot(readings, report.anomalies, self.report_dir / f"rca_plots_{stamp}.png")
            if plot_path:
                paths["plot"] = plot_path

        if "markdown" in formats:
            report_path = self.report_dir / f"rca_report_{stamp}.md"
            report_path.write_text(self.render_markdown(report, plot_path.name if plot_path else None), encoding="utf-8")
            summary_path = self.report_dir / f"rca_summary_{stamp}.md"
            summary_path.write_text(self.render_summary(report, report_path.name), encoding="utf-8")
            paths["report"], paths["summary"] = report_path, summary_path
        if "json" in formats:
            json_path = self.report_dir / f"rca_report_{stamp}.json"
            json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            paths["json"] = json_path
        for kind, path in paths.items():
            logger.info("Wrote %s: %s", kind, path)
        return paths

    @staticmethod
    def _render_plot(readings: Sequence[TelemetryReading], anomalies: list[AnomalyEvent], path: Path) -> Path | None:
        """Plot every sensor with shaded anomaly intervals. Returns ``None`` if plotting is unavailable."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.info("matplotlib not installed - skipping plots")
            return None
        frame = readings_to_frame(readings)
        if frame.empty:
            return None
        t0 = frame["timestamp"].min()
        sensors = sorted(frame["sensor_id"].unique())
        fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 2.1 * len(sensors)), sharex=True, squeeze=False)
        try:
            for ax, sensor in zip(axes[:, 0], sensors):
                group = frame[frame["sensor_id"] == sensor]
                met = (group["timestamp"] - t0).dt.total_seconds()
                ax.plot(met, group["value"], lw=0.8, color="#333333")
                ax.set_ylabel(f"{sensor}\n[{group['unit'].iloc[0]}]", fontsize=8)
                ax.grid(alpha=0.3)
                for a in anomalies:
                    if a.sensor_id != sensor or a.mission_elapsed_s is None:
                        continue
                    ax.axvspan(
                        a.mission_elapsed_s - 0.5, a.mission_elapsed_s + a.duration_s + 0.5,
                        color=_SEVERITY_COLOR[a.severity], alpha=0.12 if a.expected else 0.35, lw=0,
                    )
            axes[-1, 0].set_xlabel("Mission elapsed time [s]")
            fig.suptitle("Telemetry overview (red=critical, orange=warning, blue=info/expected)", fontsize=10)
            fig.tight_layout()
            fig.savefig(path, dpi=110)
        finally:
            plt.close(fig)
        return path


def _fmt(value: float | None) -> str:
    """Format a number compactly for tables."""
    if value is None:
        return "-"
    return f"{value:.4g}"


def _confidence_bar(confidence: float) -> str:
    """Render a confidence value as a small text bar."""
    filled = round(confidence * 10)
    return f"{'█' * filled}{'░' * (10 - filled)} {confidence:.0%}"
