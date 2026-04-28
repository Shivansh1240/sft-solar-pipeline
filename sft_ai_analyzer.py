"""
sft_ai_analyzer.py
==================
AI-powered SFT simulation analysis using the Anthropic API (Claude Opus).

This module sends simulation results to Claude Opus for:
1. Automated physics interpretation of diagnostics
2. Anomaly detection in flux transport
3. Solar cycle forecast narrative generation
4. Comparison with published literature benchmarks
5. Parameter recommendation for next simulation

Usage:
    from sft_ai_analyzer import SFTAnalyzer
    analyzer = SFTAnalyzer()
    report = analyzer.analyze_simulation(diag_results, sim_params)
    print(report)
"""

import json
import numpy as np

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("  [Warning] anthropic package not found. Install with: pip install anthropic")


def _numpy_to_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_serializable(i) for i in obj]
    return obj


class SFTAnalyzer:
    """
    AI-powered analyzer for Surface Flux Transport simulations.
    Uses Claude Opus via the Anthropic API.
    """

    # Latest Claude model
    MODEL = "claude-opus-4-5"

    SYSTEM_PROMPT = """You are an expert solar physicist specializing in Surface Flux Transport (SFT) modeling 
and solar cycle dynamics. You analyze numerical simulation results of the SFT equation:

∂⟨Br⟩/∂t = -1/(R cosλ) ∂/∂λ [u(λ)⟨Br⟩ cosλ] + 1/(R cosλ) ∂/∂λ [η cosλ ∂⟨Br⟩/∂λ] - ⟨Br⟩/τ

You provide:
1. Physical interpretation of simulation diagnostics
2. Identification of which processes dominate (advection/diffusion/decay)
3. Comparison with published literature values
4. Solar cycle forecasting implications
5. Concrete recommendations for follow-up simulations

Be quantitative, cite specific values, and connect to the broader solar physics context.
Reference key papers: Athalathil et al. 2024, Petrovay & Talafha 2019, Yeates et al. 2023, 
Baumann et al. 2004, Cameron et al. 2010, Hathaway et al. 2003.

Format your response with clear section headers using markdown."""

    VERIFICATION_SYSTEM_PROMPT = """You are a numerical PDE solver verifier acting as a scientific referee.

You are NOT doing a qualitative interpretation. You are VERIFYING correctness of an SFT (Surface Flux Transport)
solver with analytic/ground-truth tests.

Given a JSON report of verification tests, you must:
1) For each test: judge PASS/FAIL consistency with the stated expectation and tolerance.
2) If FAIL: identify the most likely cause category:
   - physics mismatch (wrong PDE term or units)
   - advection discretization / limiter / geometry factor issue
   - diffusion discretization / boundary condition issue
   - decay (tau) handling issue
   - time stepping / IMEX consistency issue
   - grid / interpolation / indexing issue
3) Propose concrete debugging steps and code-level fixes (which file/function to inspect first).
4) Quantify: cite the key error numbers from the report.

Output format (markdown):
## Overall verdict
PASS/FAIL + 1 sentence.

## Per-test verdicts
- Test: PASS/FAIL — key numbers — likely cause (if fail)

## Most likely root causes (ranked)

## Recommended code inspection order

## Suggested parameter tweaks (only if relevant)
"""

    def __init__(self):
        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic()
        else:
            self.client = None

        # Optional model override without editing code
        # (useful because Anthropic model names change over time)
        import os
        self.model = os.environ.get("ANTHROPIC_MODEL", self.MODEL)

    def _format_diagnostics_for_prompt(self, diag_results, sim_params, label=""):
        """Format simulation results into a structured prompt."""

        flux_d   = diag_results.get('flux', {})
        cent_d   = diag_results.get('centroid', {})
        prof_d   = diag_results.get('profile', {})
        dip_d    = diag_results.get('dipole', {})

        # Extract key scalar metrics
        flux_error  = float(flux_d.get('max_error_pct', 0))
        t_final     = float(flux_d['t_yr'][-1]) if 'flux' in diag_results else 0

        peak_init   = float(prof_d.get('peak_pos', [0])[0])  if 'profile' in diag_results else 0
        peak_final  = float(prof_d.get('peak_pos', [0])[-1]) if 'profile' in diag_results else 0
        fwhm_init   = float(prof_d.get('fwhm_pos_deg', [0])[0])  if 'profile' in diag_results else 0
        fwhm_final  = float(prof_d.get('fwhm_pos_deg', [0])[-1]) if 'profile' in diag_results else 0

        pos_cent_init  = float(cent_d.get('pos_centroid_deg', [0])[0])   if 'centroid' in diag_results else 0
        pos_cent_final = float(cent_d.get('pos_centroid_deg', [0])[-1])  if 'centroid' in diag_results else 0
        neg_cent_init  = float(cent_d.get('neg_centroid_deg', [0])[0])   if 'centroid' in diag_results else 0
        neg_cent_final = float(cent_d.get('neg_centroid_deg', [0])[-1])  if 'centroid' in diag_results else 0
        avg_rate       = float(cent_d.get('avg_pos_rate', 0))             if 'centroid' in diag_results else 0

        dip_init  = float(dip_d.get('dipole_G', [0])[0])   if 'dipole' in diag_results else 0
        dip_final = float(dip_d.get('dipole_G', [0])[-1])  if 'dipole' in diag_results else 0

        prompt = f"""
# SFT Simulation Analysis Request
{'Label: ' + label if label else ''}

## Simulation Parameters
- Turbulent diffusivity η: {sim_params.get('eta', 500e6) / 1e6:.0f} km²/s
- Peak meridional flow u₀: {sim_params.get('u0', 12.5):.1f} m/s
- Decay timescale τ: {sim_params.get('tau', 5*3.156e7) / (365.25*86400):.1f} years
- BMR latitude λ₀: {sim_params.get('lat0_deg', 0):.0f}°
- Simulation duration: {t_final:.1f} years
- Grid resolution: {sim_params.get('dlat', 1.0):.1f}°
- Time step: {sim_params.get('dt', 1800):.0f} s
- Numerical scheme: {sim_params.get('scheme', 'RK-IMEX + van Leer')}

## Key Diagnostic Results

### Flux Conservation
- Maximum relative flux error: {flux_error:.4f}%
- Expected decay: Φ(t) = Φ₀ exp(-t/τ)

### Profile Evolution
- Initial peak amplitude: {peak_init:.3f} G
- Final peak amplitude: {peak_final:.5f} G
- Amplitude reduction: {(1 - peak_final/max(peak_init, 1e-10))*100:.1f}%
- Initial FWHM: {fwhm_init:.1f}°
- Final FWHM: {fwhm_final:.1f}°
- FWHM broadening: {fwhm_final - fwhm_init:.1f}°

### Centroid Migration
- Positive polarity: {pos_cent_init:.1f}° → {pos_cent_final:.1f}°
- Negative polarity: {neg_cent_init:.1f}° → {neg_cent_final:.1f}°
- Average migration rate: {avg_rate:.1f} °/yr

### Axial Dipole Moment
- Initial: {dip_init:.4f} G
- Final: {dip_final:.4f} G

## Analysis Requested
1. What do these results tell us about the relative importance of the three transport mechanisms?
2. Is the numerical performance (flux conservation, stability) satisfactory?
3. How do these results compare to published literature benchmarks?
4. What are the implications for solar cycle forecasting?
5. What specific follow-up simulations or parameter variations would be most scientifically valuable?
"""
        return prompt

    def analyze_simulation(self, diag_results, sim_params, label=""):
        """
        Send simulation results to Claude Opus for analysis.

        Parameters
        ----------
        diag_results : dict from sft_diagnostics.print_diagnostic_summary
        sim_params   : dict with eta, u0, tau, lat0_deg, dt, dlat
        label        : descriptive label for the simulation case

        Returns
        -------
        str : AI-generated analysis report (markdown)
        """
        if not ANTHROPIC_AVAILABLE or self.client is None:
            return self._fallback_analysis(diag_results, sim_params)

        prompt = self._format_diagnostics_for_prompt(diag_results, sim_params, label)

        print(f"  [AI Analyzer] Sending to Claude Opus for analysis...")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def compare_cases(self, case_results_list, case_labels, sim_params_list):
        """
        Compare multiple simulation cases (e.g., Case 1 vs Case 2).

        Parameters
        ----------
        case_results_list : list of diag_result dicts
        case_labels       : list of strings (case names)
        sim_params_list   : list of param dicts

        Returns
        -------
        str : comparative analysis
        """
        if not ANTHROPIC_AVAILABLE or self.client is None:
            return "AI comparison unavailable (anthropic package not installed)."

        summaries = []
        for diag, label, params in zip(case_results_list, case_labels, sim_params_list):
            summaries.append(
                self._format_diagnostics_for_prompt(diag, params, label)
            )

        full_prompt = (
            "# Multi-Case SFT Comparison Analysis\n\n"
            + "\n\n---\n\n".join(summaries)
            + "\n\n## Comparison Task\n"
            "Please compare these simulation cases, highlighting:\n"
            "1. Key differences in transport behavior between cases\n"
            "2. The effect of different initial conditions (equatorial vs mid-latitude BMR)\n"
            "3. Which case is more representative of real solar conditions\n"
            "4. Combined implications for solar cycle prediction\n"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": full_prompt}]
        )

        return response.content[0].text

    def sensitivity_interpretation(self, sens_results, lat_deg):
        """
        Ask Claude to interpret parameter sensitivity results.
        """
        if not ANTHROPIC_AVAILABLE or self.client is None:
            return "AI interpretation unavailable."

        # Summarize sensitivity quantitatively
        eta_summary = []
        for r in sens_results.get('eta_sweep', []):
            eta_summary.append({
                'eta_km2_s': r['eta'] / 1e6,
                'peak_G': round(r['peak'], 5),
                'fwhm_deg': round(r['fwhm'], 2) if not np.isnan(r['fwhm']) else None,
                'centroid_deg': round(r['centroid_pos'], 1) if not np.isnan(r['centroid_pos']) else None,
            })

        u0_summary = []
        for r in sens_results.get('u0_sweep', []):
            u0_summary.append({
                'u0_m_s': r['u0'],
                'peak_G': round(r['peak'], 5),
                'fwhm_deg': round(r['fwhm'], 2) if not np.isnan(r['fwhm']) else None,
                'centroid_deg': round(r['centroid_pos'], 1) if not np.isnan(r['centroid_pos']) else None,
            })

        tau_summary = []
        for r in sens_results.get('tau_sweep', []):
            tau_summary.append({
                'tau_yr': r['tau_yr'],
                'peak_G': round(r['peak'], 5),
            })

        prompt = f"""
# SFT Parameter Sensitivity Interpretation

## Results at t=2 years

### Varying η (diffusivity), u₀=12.5 m/s, τ=5 yr:
{json.dumps(eta_summary, indent=2)}

### Varying u₀ (flow speed), η=500 km²/s, τ=5 yr:
{json.dumps(u0_summary, indent=2)}

### Varying τ (decay timescale), η=500 km²/s, u₀=12.5 m/s:
{json.dumps(tau_summary, indent=2)}

## Analysis Requested
1. Quantify the logarithmic sensitivities (partial derivatives) of peak amplitude to each parameter.
2. Which parameter has the largest effect on polar field buildup (centroid migration)?
3. What do these sensitivities imply for the uncertainty in solar cycle forecasts given observational uncertainties in η and u₀?
4. Are these sensitivities consistent with Petrovay & Talafha 2019 and Whitbread et al. 2017?
5. Recommend an optimal observational strategy to reduce forecast uncertainty.
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_forecast_narrative(self, dipole_t_yr, dipole_G, current_cycle=25):
        """
        Generate a solar cycle forecast narrative based on dipole evolution.
        """
        if not ANTHROPIC_AVAILABLE or self.client is None:
            return "AI forecast unavailable."

        final_dipole = float(dipole_G[-1])
        peak_dipole  = float(np.max(np.abs(dipole_G)))
        t_peak       = float(dipole_t_yr[np.argmax(np.abs(dipole_G))])

        prompt = f"""
# Solar Cycle Forecast Based on SFT Axial Dipole Evolution

## Dipole Evolution Data
- Simulation covers {dipole_t_yr[-1]:.1f} years
- Initial dipole moment: {dipole_G[0]:.4f} G
- Peak dipole moment: {peak_dipole:.4f} G (at t={t_peak:.1f} yr)
- Final dipole moment: {final_dipole:.4f} G
- Current solar cycle being modeled: Cycle {current_cycle}

## Forecast Request
Based on this axial dipole evolution from our SFT simulation:
1. Estimate the expected amplitude of Solar Cycle {current_cycle + 1}
   using the established polar-field precursor method.
2. Compare to observational benchmarks from past cycles (Ohl's rule, Schatten et al.)
3. Provide a confidence interval accounting for parameter uncertainty.
4. Discuss what aspects of this single-BMR simulation limit real forecast applicability.
5. What data (HMI magnetograms, helioseismic inversions) would most improve this forecast?
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _fallback_analysis(self, diag_results, sim_params):
        """Basic text report if Anthropic API is unavailable."""
        prof = diag_results.get('profile', {})
        cent = diag_results.get('centroid', {})
        flux = diag_results.get('flux', {})

        lines = [
            "=" * 60,
            "  SFT SIMULATION ANALYSIS (local fallback — no API)",
            "=" * 60,
            f"  η = {sim_params.get('eta', 500e6)/1e6:.0f} km²/s",
            f"  u₀ = {sim_params.get('u0', 12.5):.1f} m/s",
            f"  τ = {sim_params.get('tau', 5*3.156e7)/(365.25*86400):.1f} yr",
            "",
            "  KEY FINDINGS:",
            f"  • Flux conservation error: {flux.get('max_error_pct', 0):.3f}%",
            f"  • Avg poleward migration: {cent.get('avg_pos_rate', 0):.1f} °/yr",
            "",
            "  Install 'anthropic' package for full AI analysis.",
            "=" * 60,
        ]
        return "\n".join(lines)

    def verify_simulator(self, verification_report):
        """
        Referee mode: consume a structured verification report (dict) and return a
        PASS/FAIL + debugging suggestions report (markdown).
        """
        report = _numpy_to_serializable(verification_report)

        if not ANTHROPIC_AVAILABLE or self.client is None:
            # Minimal local fallback: deterministic summary
            summary = report.get("summary", {})
            lines = [
                "## Overall verdict",
                f"{'PASS' if summary.get('failed', 0) == 0 else 'FAIL'} — "
                f"{summary.get('passed', 0)}/{summary.get('total_tests', 0)} tests passed.",
                "",
                "## Per-test verdicts",
            ]
            for t in report.get("tests", []):
                status = "PASS" if t.get("passed") else "FAIL"
                lines.append(f"- {t.get('test_name', 'Unnamed')}: {status} — {t.get('details', '')}")
            lines.append("")
            lines.append("## Next steps")
            lines.append("- Install `anthropic` and re-run with `--verify --verify-ai` for detailed root-cause ranking.")
            return "\n".join(lines)

        prompt = (
            "# SFT Verification Report\n\n"
            "You are given JSON from a verification test suite.\n"
            "Return the referee verdict following the required output format.\n\n"
            "```json\n"
            + json.dumps(report, indent=2)
            + "\n```\n"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.VERIFICATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text
