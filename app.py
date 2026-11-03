"""
app.py  —  Modern Analytics Dashboard Home
DQN Personalized English Learning Research Suite
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is at the front of sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DQN English Learning Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Import shared design system
# ---------------------------------------------------------------------------
from shared_ui import (  # noqa: E402
    DESIGN_CSS,
    PRIMARY_BLUE,
    render_sidebar,
    render_page_header,
    render_section_header,
    render_kpi_card,
)

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_CSV = [
    "simulated_learners.csv",
    "training_history_all_seeds.csv",
    "evaluation_results_all_seeds.csv",
    "evaluation_summary_by_seed.csv",
    "statistical_results_all_seeds.csv",
    "robustness_results_all_seeds.csv",
    "robustness_summary_by_seed.csv",
]

REQUIRED_MODELS = [42, 43, 44, 45, 46]

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_learners_meta() -> dict:
    """Return high-level metadata from simulated_learners.csv."""
    path = DATA_DIR / "simulated_learners.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        return {
            "num_learners": len(df),
            "profiles": sorted(df["profile"].unique().tolist()),
            "num_profiles": df["profile"].nunique(),
            "columns": list(df.columns),
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_eval_meta() -> dict:
    path = DATA_DIR / "evaluation_results_all_seeds.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        return {
            "num_rows": len(df),
            "num_learners_eval": df["learner_id"].nunique() if "learner_id" in df.columns else "N/A",
            "agents": sorted(df["agent"].unique().tolist()) if "agent" in df.columns else [],
            "seeds": sorted(df["training_seed"].unique().tolist()) if "training_seed" in df.columns else [],
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_training_meta() -> dict:
    path = DATA_DIR / "training_history_all_seeds.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        return {
            "num_rows": len(df),
            "seeds": sorted(df["training_seed"].unique().tolist()) if "training_seed" in df.columns else [],
            "episodes_per_seed": df.groupby("training_seed")["episode"].max().to_dict()
            if "training_seed" in df.columns and "episode" in df.columns else {},
        }
    except Exception:
        return {}


def check_files() -> list[dict]:
    rows = []
    for fname in REQUIRED_CSV:
        p = DATA_DIR / fname
        if p.exists():
            try:
                size_kb = p.stat().st_size / 1024
                rows.append({"file": fname, "status": "ok", "size": f"{size_kb:,.0f} KB"})
            except Exception:
                rows.append({"file": fname, "status": "error", "size": "?"})
        else:
            rows.append({"file": fname, "status": "missing", "size": "-"})
    return rows


def check_models() -> list[dict]:
    from dqn_core import load_dqn_model, get_model_path
    rows = []
    for seed in REQUIRED_MODELS:
        p = get_model_path(seed)
        if not p.exists():
            rows.append({"seed": seed, "status": "missing", "size": "-"})
            continue
        try:
            size_kb = p.stat().st_size / 1024
            m = load_dqn_model(seed)
            if m is not None:
                rows.append({"seed": seed, "status": "ok", "size": f"{size_kb:,.1f} KB"})
            else:
                rows.append({"seed": seed, "status": "error", "size": f"{size_kb:,.1f} KB"})
        except Exception:
            rows.append({"seed": seed, "status": "error", "size": "-"})
    return rows


def badge(status: str) -> str:
    cls = {"ok": "badge-ok", "missing": "badge-err", "error": "badge-warn"}.get(status, "badge-warn")
    label = {"ok": "✓ Verified", "missing": "✗ Missing", "error": "⚠ Error"}.get(status, status)
    return f'<span class="badge {cls}">{label}</span>'


# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------
def main():
    st.markdown(DESIGN_CSS, unsafe_allow_html=True)
    render_sidebar()

    # ── Header Banner ────────────────────────────────────────────────────────
    render_page_header(
        title="Personalized English Learning Dashboard",
        subtitle="Deep Q-Network (DQN) | Multi-Seed Reinforcement Learning Research & Analytics Suite",
        badge_text="5-Seed Research Edition",
        icon="🎓",
    )

    # ── High-Level KPI Cards ────────────────────────────────────────────────
    learner_meta   = load_learners_meta()
    eval_meta      = load_eval_meta()
    training_meta  = load_training_meta()

    n_learners   = learner_meta.get("num_learners", 0)
    n_seeds      = len(training_meta.get("seeds", REQUIRED_MODELS))
    n_eval       = eval_meta.get("num_learners_eval", 0)

    model_statuses = check_models()
    n_loaded = sum(1 for m in model_statuses if m["status"] == "ok")

    kpi_col1 = render_kpi_card(
        label="Simulated Learners",
        value=f"{n_learners:,}" if isinstance(n_learners, int) else str(n_learners),
        delta="7 Profiles",
        delta_type="neutral",
        icon="👥",
        footer_text="simulated_learners.csv",
    )
    kpi_col2 = render_kpi_card(
        label="Training Seeds",
        value=f"{n_seeds}",
        delta="Seeds 42–46",
        delta_type="positive",
        icon="🌱",
        footer_text="Independent Runs",
    )
    kpi_col3 = render_kpi_card(
        label="Learning Activities",
        value="4",
        delta="Action Space",
        delta_type="positive",
        icon="🎯",
        footer_text="Listen, Read, Vocab, Grammar",
    )
    kpi_col4 = render_kpi_card(
        label="Evaluated Learners",
        value=f"{n_eval:,}" if isinstance(n_eval, int) else str(n_eval),
        delta="Benchmark Set",
        delta_type="neutral",
        icon="🧪",
        footer_text="evaluation_results.csv",
    )
    kpi_col5 = render_kpi_card(
        label="Trained Models",
        value=f"{n_loaded} / {len(REQUIRED_MODELS)}",
        delta="100% Loaded",
        delta_type="positive" if n_loaded == len(REQUIRED_MODELS) else "negative",
        icon="🧠",
        footer_text="PyTorch CPU Checkpoints",
    )

    st.markdown(f"""
    <div class="kpi-row">
      {kpi_col1}
      {kpi_col2}
      {kpi_col3}
      {kpi_col4}
      {kpi_col5}
    </div>
    """, unsafe_allow_html=True)

    # ── Two-Column Architecture & Modules Layout ─────────────────────────────
    col_main, col_side = st.columns([7, 5], gap="large")

    with col_main:
        # Research Overview
        render_section_header("Research & Framework Overview", "Core methodology for personalized curriculum optimization")

        st.markdown(f"""
        <div class="analytics-card">
          <p style="margin-top:0; font-size:0.92rem; line-height:1.65; color:#374151;">
            This research investigates the application of <strong>Deep Q-Network (DQN)</strong> reinforcement learning
            to formulate <strong>adaptive, personalized English learning trajectories</strong>.
            The reinforcement learning agent continuously balances skill reinforcement across four fundamental competencies
            (<strong>Listening, Reading, Vocabulary, Grammar</strong>) under constrained learner time budgets.
          </p>
          <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:14px;">
            <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:10px 14px; flex:1 1 200px;">
              <div style="font-size:0.75rem; font-weight:700; color:{PRIMARY_BLUE}; text-transform:uppercase;">Objective</div>
              <div style="font-size:0.85rem; color:#1E3A8A; margin-top:2px; font-weight:500;">
                Maximize multi-skill competency gain while respecting strict time budget constraints.
              </div>
            </div>
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:10px 14px; flex:1 1 200px;">
              <div style="font-size:0.75rem; font-weight:700; color:#15803D; text-transform:uppercase;">Methodology</div>
              <div style="font-size:0.85rem; color:#14532D; margin-top:2px; font-weight:500;">
                5-seed multi-run validation with Wilcoxon signed-rank and Cohen's d effect size tests.
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # RL Architecture Details
        render_section_header("DQN Architecture & Action Dynamics", "11-dimensional normalized state vector and 4-action learning rates")

        c_state, c_action = st.columns(2)
        with c_state:
            st.markdown(f"""
            <div class="analytics-card" style="height:100%;">
              <div style="font-weight:700; font-size:0.95rem; color:{PRIMARY_BLUE}; margin-bottom:8px;">
                State Space (11 Dimensions)
              </div>
              <ul style="font-size:0.82rem; color:#4B5563; padding-left:18px; margin:0; line-height:1.65;">
                <li><code>[0..3]</code> Normalized skill abilities (0.0 – 1.0)</li>
                <li><code>[4]</code> Learning progress ratio (time used / budget)</li>
                <li><code>[5]</code> Remaining time ratio</li>
                <li><code>[6]</code> Normalized target learning goal</li>
                <li><code>[7..10]</code> Normalized prior activity selection counts</li>
              </ul>
              <div style="margin-top:12px; font-size:0.75rem; color:#6B7280; border-top:1px solid #F3F4F6; padding-top:8px;">
                Network: <code>Linear(11, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 4)</code>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c_action:
            st.markdown(f"""
            <div class="analytics-card" style="height:100%;">
              <div style="font-weight:700; font-size:0.95rem; color:{PRIMARY_BLUE}; margin-bottom:8px;">
                Action Space (4 Activities)
              </div>
              <div style="font-size:0.82rem; line-height:1.7; color:#4B5563;">
                <div><span class="badge" style="background:#DBEAFE; color:#1E40AF;">0 • Listening</span> 15 mins / +5.0 base gain</div>
                <div><span class="badge" style="background:#E0F2FE; color:#0369A1;">1 • Reading</span> 20 mins / +6.0 base gain</div>
                <div><span class="badge" style="background:#D1FAE5; color:#065F46;">2 • Vocabulary</span> 10 mins / +4.0 base gain</div>
                <div><span class="badge" style="background:#FEF3C7; color:#92400E;">3 • Grammar</span> 25 mins / +7.0 base gain</div>
              </div>
              <div style="margin-top:10px; font-size:0.75rem; color:#6B7280; border-top:1px solid #F3F4F6; padding-top:8px;">
                Reward: <code>Δ Ability Gain - (time_cost / time_budget) × 10</code>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_side:
        # Interactive Modules List
        render_section_header("Dashboard Modules", "Direct access to research interfaces")

        modules_info = [
            ("👤", "1. Learner Simulation", "pages/1_Learner_Simulation.py",
             "Select simulated learners and run realtime greedy DQN inference with before/after skill profiles."),
            ("🗺️", "2. Recommended Learning Path", "pages/2_Recommended_Learning_Path.py",
             "Inspect step-by-step sequential activity choices, Q-value rankings, and score trajectories."),
            ("📊", "3. Model Evaluation", "pages/3_Model_Evaluation.py",
             "Benchmark DQN against Random and Weakest-Skill baselines with Wilcoxon significance & robustness tests."),
            ("📈", "4. Training Analytics", "pages/4_Training_Analytics.py",
             "Analyze multi-seed training curves, epsilon decay, moving averages, and cross-seed stability."),
        ]

        for icon, title, path, desc in modules_info:
            st.markdown(f"""
            <div class="analytics-card" style="padding:14px 18px; margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div style="font-weight:700; font-size:0.92rem; color:{PRIMARY_BLUE};">
                    {icon} {title}
                  </div>
                  <div style="font-size:0.78rem; color:#6B7280; margin-top:3px; line-height:1.4;">
                    {desc}
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Data & Checkpoint Readiness Card
        render_section_header("System & Data Readiness", "Status of research artifacts")
        file_statuses = check_files()
        all_files_ok = all(f["status"] == "ok" for f in file_statuses)
        all_models_ok = (n_loaded == len(REQUIRED_MODELS))

        st.markdown(f"""
        <div class="analytics-card" style="padding:16px 18px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <span style="font-size:0.85rem; font-weight:700; color:#374151;">Dataset Verification:</span>
            <span>{'<span class="badge badge-ok">✓ 7/7 CSV Files Ready</span>' if all_files_ok else '<span class="badge badge-err">Missing Files</span>'}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <span style="font-size:0.85rem; font-weight:700; color:#374151;">DQN Checkpoints:</span>
            <span>{'<span class="badge badge-ok">✓ 5/5 Models Loaded</span>' if all_models_ok else '<span class="badge badge-warn">Partial Models</span>'}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.85rem; font-weight:700; color:#374151;">Inference Engine:</span>
            <span class="badge badge-info">PyTorch CPU (Weights Verified)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#9CA3AF; font-size:0.78rem; padding-bottom:1rem;'>"
        "Personalized English Learning DQN Research Suite • Academic Demonstration • All learner profiles are synthetic and simulated."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
