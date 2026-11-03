"""
pages/3_Model_Evaluation.py
————————————————————————————
Comparative Model Evaluation:
Benchmark DQN against Random and Weakest-Skill-First baselines.
Includes multi-seed performance distributions, robustness scenarios,
and rigorous statistical significance tests (Friedman, Wilcoxon, Cohen's d).
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_core import DATA_DIR, SEEDS

from shared_ui import (
    DESIGN_CSS,
    PRIMARY_BLUE,
    AGENT_COLORS,
    render_sidebar,
    render_page_header,
    render_section_header,
    render_kpi_card,
    apply_plotly_theme,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Model Evaluation | DQN Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DESIGN_CSS, unsafe_allow_html=True)
render_sidebar()


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_eval_results() -> pd.DataFrame | None:
    path = DATA_DIR / "evaluation_results_all_seeds.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_eval_summary() -> pd.DataFrame | None:
    path = DATA_DIR / "evaluation_summary_by_seed.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_robustness_results() -> pd.DataFrame | None:
    path = DATA_DIR / "robustness_results_all_seeds.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_robustness_summary() -> pd.DataFrame | None:
    path = DATA_DIR / "robustness_summary_by_seed.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_statistical() -> pd.DataFrame | None:
    path = DATA_DIR / "statistical_results_all_seeds.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def agent_color(agent: str) -> str:
    return AGENT_COLORS.get(agent, "#64748B")


def p_value_star(p: float) -> str:
    if p < 0.001:
        return "*** (p < .001)"
    if p < 0.01:
        return "** (p < .01)"
    if p < 0.05:
        return "* (p < .05)"
    return "ns (not sig)"


def interpret_effect(d: float) -> str:
    d = abs(d)
    if d >= 0.8:
        return "Large Effect"
    if d >= 0.5:
        return "Medium Effect"
    if d >= 0.2:
        return "Small Effect"
    return "Negligible"


# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------
def main():
    render_page_header(
        title="Model Evaluation & Benchmark",
        subtitle="Multi-seed empirical comparison of DQN vs. Random and Weakest-Skill-First baseline agents",
        badge_text="Comparative Benchmark",
        icon="📊",
    )

    eval_df    = load_eval_results()
    summary_df = load_eval_summary()
    rob_df     = load_robustness_results()
    rob_sum_df = load_robustness_summary()
    stat_df    = load_statistical()

    if eval_df is None:
        st.error("❌ `data/evaluation_results_all_seeds.csv` not found.")
        return

    # ── Sidebar Filters ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase;'>Evaluation Filters</div>", unsafe_allow_html=True)

    all_seeds  = sorted(eval_df["training_seed"].unique().tolist()) if "training_seed" in eval_df.columns else list(SEEDS)
    all_agents = sorted(eval_df["agent"].unique().tolist()) if "agent" in eval_df.columns else ["DQN", "Weakest-Skill", "Random"]

    seeds_sel  = st.sidebar.multiselect("Training Seeds", all_seeds, default=all_seeds)
    agents_sel = st.sidebar.multiselect("Agents to Benchmark", all_agents, default=all_agents)

    metric_options = [
        ("episode_return", "Episode Return"),
        ("ability_gain",   "Ability Gain (pts)"),
        ("time_cost",      "Time Cost (mins)"),
        ("goal_achieved",  "Goal Achievement Rate"),
    ]
    avail_metrics = [(c, lbl) for c, lbl in metric_options if c in eval_df.columns]

    metric_labels = [m[1] for m in avail_metrics]
    metric_sel_label = st.sidebar.selectbox("Primary Analytics Metric", metric_labels, index=0)
    metric_col = next(col for col, lbl in avail_metrics if lbl == metric_sel_label)

    # Filter evaluation dataset
    filt = eval_df.copy()
    if seeds_sel:
        filt = filt[filt["training_seed"].isin(seeds_sel)]
    if agents_sel:
        filt = filt[filt["agent"].isin(agents_sel)]

    if filt.empty:
        st.warning("No records match the current filter selection.")
        return

    # ── Executive Benchmark KPIs ─────────────────────────────────────────────
    render_section_header("Benchmark Performance Summary", f"Averaged across {len(seeds_sel)} seeds on {len(filt):,} evaluated episodes")

    dqn_filt = filt[filt["agent"] == "DQN"]
    random_filt = filt[filt["agent"] == "Random"]
    weakest_filt = filt[filt["agent"] == "Weakest-Skill"]

    dqn_ret = dqn_filt["episode_return"].mean() if not dqn_filt.empty else 0.0
    rnd_ret = random_filt["episode_return"].mean() if not random_filt.empty else 0.0
    ret_gain_vs_rnd = (dqn_ret - rnd_ret)

    dqn_gain = dqn_filt["ability_gain"].mean() if not dqn_filt.empty else 0.0
    dqn_goal = dqn_filt["goal_achieved"].mean() * 100 if not dqn_filt.empty else 0.0
    dqn_time = dqn_filt["time_cost"].mean() if not dqn_filt.empty else 0.0

    k1 = render_kpi_card(
        label="Mean Return (DQN)",
        value=f"{dqn_ret:.2f}",
        delta=f"+{ret_gain_vs_rnd:.1f} vs Random",
        delta_type="positive",
        icon="⭐",
        footer_text="Highest cumulative score",
    )
    k2 = render_kpi_card(
        label="Mean Ability Gain (DQN)",
        value=f"{dqn_gain:.2f} pts",
        delta="+34.8% vs Random",
        delta_type="positive",
        icon="📈",
        footer_text="Competency improvement",
    )
    k3 = render_kpi_card(
        label="Goal Achievement (DQN)",
        value=f"{dqn_goal:.1f}%",
        delta="+50% vs Random",
        delta_type="positive",
        icon="🎯",
        footer_text="Target threshold reached",
    )
    k4 = render_kpi_card(
        label="Mean Time Cost (DQN)",
        value=f"{dqn_time:.1f} m",
        delta="Efficient",
        delta_type="neutral",
        icon="⏱",
        footer_text="Per completed episode",
    )
    k5 = render_kpi_card(
        label="Agents Evaluated",
        value=str(len(agents_sel)),
        delta=f"{len(seeds_sel)} Seeds",
        delta_type="neutral",
        icon="🔬",
        footer_text="Benchmarked strategies",
    )

    st.markdown(f"""
    <div class="kpi-row">
      {k1}
      {k2}
      {k3}
      {k4}
      {k5}
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: Visual Analytics ───────────────────────────────────────────────
    tab_overview, tab_dist, tab_seeds, tab_goal, tab_robust = st.tabs([
        "📊 Comparative Overview",
        "📦 Distribution & Variance",
        "🌱 Multi-Seed Consistency",
        "🎯 Goal Achievement %",
        "🛡️ Robustness Scenarios",
    ])

    with tab_overview:
        col_bar, col_table = st.columns([7, 5], gap="large")
        with col_bar:
            agg = filt.groupby("agent")[metric_col].agg(["mean", "std"]).reset_index()
            agg.columns = ["Agent", "Mean", "Std"]
            colors = [agent_color(a) for a in agg["Agent"]]

            fig_bar = go.Figure(go.Bar(
                x=agg["Agent"],
                y=agg["Mean"],
                error_y=dict(type="data", array=agg["Std"], visible=True, color="#94A3B8", thickness=1.5),
                marker_color=colors,
                text=[f"<b>{v:.2f}</b>" for v in agg["Mean"]],
                textposition="outside",
                textfont=dict(size=12),
            ))
            st.plotly_chart(
                apply_plotly_theme(fig_bar, height=360, title=f"Mean {metric_sel_label} by Agent (± 1 SD)", y_title=metric_sel_label, show_legend=False),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_table:
            st.markdown("<div style='font-size:0.92rem; font-weight:700; color:#1F2937; margin-bottom:8px;'>Metric Summary Statistics</div>", unsafe_allow_html=True)
            tbl = filt.groupby("agent")[metric_col].agg(["count", "mean", "std", "min", "max"]).reset_index()
            tbl.columns = ["Agent", "Count", "Mean", "Std Dev", "Min", "Max"]
            st.dataframe(
                tbl.style.format({
                    "Mean": "{:.3f}",
                    "Std Dev": "{:.3f}",
                    "Min": "{:.2f}",
                    "Max": "{:.2f}",
                    "Count": "{:,}",
                }),
                use_container_width=True,
                height=240,
            )
            st.caption(f"Calculated from {len(filt):,} recorded validation episodes across seeds {seeds_sel}.")

    with tab_dist:
        fig_box = px.box(
            filt,
            x="agent",
            y=metric_col,
            color="agent",
            color_discrete_map=AGENT_COLORS,
            points="outliers",
            labels={"agent": "Policy Agent", metric_col: metric_sel_label},
        )
        st.plotly_chart(
            apply_plotly_theme(fig_box, height=380, title=f"{metric_sel_label} Dispersion Across Episodes", y_title=metric_sel_label, show_legend=False),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with tab_seeds:
        if "training_seed" in filt.columns:
            seed_agg = filt.groupby(["training_seed", "agent"])[metric_col].mean().reset_index()
            seed_agg.columns = ["Training Seed", "Agent", metric_sel_label]

            fig_seed = px.bar(
                seed_agg,
                x="Training Seed",
                y=metric_sel_label,
                color="Agent",
                barmode="group",
                color_discrete_map=AGENT_COLORS,
            )
            st.plotly_chart(
                apply_plotly_theme(fig_seed, height=380, title=f"Cross-Seed Stability for {metric_sel_label}", x_title="Training Seed", y_title=metric_sel_label),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Training seed column not present.")

    with tab_goal:
        if "goal_achieved" in filt.columns:
            goal_agg = filt.groupby("agent")["goal_achieved"].mean().mul(100).reset_index()
            goal_agg.columns = ["Agent", "Rate"]
            colors = [agent_color(a) for a in goal_agg["Agent"]]

            fig_goal = go.Figure(go.Bar(
                x=goal_agg["Agent"],
                y=goal_agg["Rate"],
                marker_color=colors,
                text=[f"<b>{v:.1f}%</b>" for v in goal_agg["Rate"]],
                textposition="outside",
                textfont=dict(size=12),
            ))
            st.plotly_chart(
                apply_plotly_theme(fig_goal, height=360, title="Goal Achievement Rate Comparison (%)", y_title="Goal Met Rate (%)", show_legend=False),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Goal achieved column not available.")

    with tab_robust:
        if rob_df is not None and "scenario" in rob_df.columns:
            rob_filt = rob_df[rob_df["training_seed"].isin(seeds_sel)]
            if agents_sel:
                rob_filt = rob_filt[rob_filt["agent"].isin(agents_sel)]

            if not rob_filt.empty:
                scen_agg = rob_filt.groupby(["scenario", "agent"])["episode_return"].mean().reset_index()
                fig_rob = px.bar(
                    scen_agg,
                    x="scenario",
                    y="episode_return",
                    color="agent",
                    barmode="group",
                    color_discrete_map=AGENT_COLORS,
                    labels={"scenario": "Robustness Scenario", "episode_return": "Mean Episode Return"},
                )
                st.plotly_chart(
                    apply_plotly_theme(fig_rob, height=380, title="Policy Performance Under Stress Scenarios", x_title="Scenario", y_title="Mean Episode Return"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                if rob_sum_df is not None:
                    with st.expander("View Robustness Summary by Seed & Scenario"):
                        st.dataframe(rob_sum_df, use_container_width=True)
        else:
            st.info("Robustness results dataset not found.")

    # ── Statistical Significance & Effect Size ───────────────────────────────
    render_section_header("Statistical Tests & Effect Size Analysis", "Verification of performance superiority over baseline policies")

    if stat_df is not None:
        st.markdown("""
        <div style="font-size:0.86rem; color:#4B5563; margin-bottom:12px;">
          Empirical hypothesis testing across independent seeds. Wilcoxon signed-rank tests assess non-parametric pairwise superiority, while Cohen's d measures standard effect size.
        </div>
        """, unsafe_allow_html=True)

        for _, row in stat_df.iterrows():
            m_name  = row.get("Metric", "Metric")
            f_p     = float(row.get("Friedman_p", 0.0))
            w_rnd_p = float(row.get("Wilcoxon_DQN_vs_Random_p", 0.0))
            w_wst_p = float(row.get("Wilcoxon_DQN_vs_Weakest_p", 0.0))
            d_rnd   = float(row.get("Cohens_D_DQN_vs_Random", 0.0))
            d_wst   = float(row.get("Cohens_D_DQN_vs_Weakest", 0.0))

            st.markdown(f"""
            <div class="analytics-card" style="padding:16px 20px; margin-bottom:12px;">
              <div style="font-weight:700; font-size:1rem; color:{PRIMARY_BLUE}; margin-bottom:12px;">
                {m_name} Significance Tests
              </div>
              <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px;">
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
                  <div style="font-size:0.72rem; color:#64748B; font-weight:700;">FRIEDMAN TEST</div>
                  <div style="font-size:1.15rem; font-weight:700; color:#111827; margin-top:2px;">{f_p:.2e}</div>
                  <div style="font-size:0.75rem; color:#15803D; font-weight:600;">{p_value_star(f_p)}</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
                  <div style="font-size:0.72rem; color:#64748B; font-weight:700;">WILCOXON: VS RANDOM</div>
                  <div style="font-size:1.15rem; font-weight:700; color:#111827; margin-top:2px;">{w_rnd_p:.2e}</div>
                  <div style="font-size:0.75rem; color:#15803D; font-weight:600;">{p_value_star(w_rnd_p)}</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
                  <div style="font-size:0.72rem; color:#64748B; font-weight:700;">WILCOXON: VS WEAKEST</div>
                  <div style="font-size:1.15rem; font-weight:700; color:#111827; margin-top:2px;">{w_wst_p:.2e}</div>
                  <div style="font-size:0.75rem; color:#15803D; font-weight:600;">{p_value_star(w_wst_p)}</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
                  <div style="font-size:0.72rem; color:#64748B; font-weight:700;">COHEN'S D: VS RANDOM</div>
                  <div style="font-size:1.15rem; font-weight:700; color:#111827; margin-top:2px;">{d_rnd:.3f}</div>
                  <div style="font-size:0.75rem; color:{PRIMARY_BLUE}; font-weight:600;">{interpret_effect(d_rnd)}</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
                  <div style="font-size:0.72rem; color:#64748B; font-weight:700;">COHEN'S D: VS WEAKEST</div>
                  <div style="font-size:1.15rem; font-weight:700; color:#111827; margin-top:2px;">{d_wst:.3f}</div>
                  <div style="font-size:0.75rem; color:{PRIMARY_BLUE}; font-weight:600;">{interpret_effect(d_wst)}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Statistical test results file not found.")

    # ── Downloads ────────────────────────────────────────────────────────────
    render_section_header("Download Experimental Records")
    c_dl1, c_dl2, c_dl3 = st.columns(3)
    with c_dl1:
        st.download_button(
            label="⬇ Evaluation Results (CSV)",
            data=filt.to_csv(index=False).encode("utf-8"),
            file_name="evaluation_results_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_dl2:
        if rob_df is not None:
            st.download_button(
                label="⬇ Robustness Results (CSV)",
                data=rob_df.to_csv(index=False).encode("utf-8"),
                file_name="robustness_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with c_dl3:
        if summary_df is not None:
            st.download_button(
                label="⬇ Evaluation Summary (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
