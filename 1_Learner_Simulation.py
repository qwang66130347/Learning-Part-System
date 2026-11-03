"""
pages/1_Learner_Simulation.py
——————————————————————————————
Interactive Learner Simulation:
Select a simulated learner profile, choose a trained DQN model seed (42-46),
run real greedy DQN inference, and analyze skill development.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_core import (
    ACTIONS,
    ACTION_COLORS,
    DATA_DIR,
    get_available_seeds,
    get_learner_by_id,
    get_profiles,
    get_learner_ids_by_profile,
    load_dqn_model,
    run_episode,
)

from shared_ui import (
    DESIGN_CSS,
    PRIMARY_BLUE,
    ACCENT_BLUE,
    render_sidebar,
    render_page_header,
    render_section_header,
    render_kpi_card,
    apply_plotly_theme,
)

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Learner Simulation | DQN Dashboard",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DESIGN_CSS, unsafe_allow_html=True)
render_sidebar()

# ---------------------------------------------------------------------------
# Data Caching
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_learners_cached() -> pd.DataFrame | None:
    path = DATA_DIR / "simulated_learners.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model_cached(seed: int):
    return load_dqn_model(seed)


# ---------------------------------------------------------------------------
# Visual Chart Generators
# ---------------------------------------------------------------------------
def make_radar(before: dict, after: dict) -> go.Figure:
    categories = list(ACTIONS.values())
    b_vals = [before[k] for k in ["ability_listening", "ability_reading", "ability_vocabulary", "ability_grammar"]]
    a_vals = [after[k]  for k in ["ability_listening", "ability_reading", "ability_vocabulary", "ability_grammar"]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=b_vals + [b_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Baseline (Before)",
        line=dict(color="#94A3B8", width=2),
        fillcolor="rgba(148, 163, 184, 0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=a_vals + [a_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Post-DQN (After)",
        line=dict(color=PRIMARY_BLUE, width=2.5),
        fillcolor="rgba(21, 101, 192, 0.25)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=False,
                gridcolor="#E5E7EB",
                tickfont=dict(size=10, color="#6B7280"),
            ),
            angularaxis=dict(
                gridcolor="#E5E7EB",
                tickfont=dict(size=11, family="Inter, sans-serif", color="#374151", weight=600),
            ),
            bgcolor="#FFFFFF",
        ),
        showlegend=True,
    )
    return apply_plotly_theme(fig, height=340, title="Skill Competency Radar")


def make_ability_bar(learner_info: dict, final_step: dict) -> go.Figure:
    skills = ["Listening", "Reading", "Vocabulary", "Grammar"]
    keys_b = ["ability_listening", "ability_reading", "ability_vocabulary", "ability_grammar"]
    before = [learner_info[k] for k in keys_b]
    after  = [final_step[k]   for k in keys_b]

    fig = go.Figure()
    fig.add_bar(
        name="Initial Score",
        x=skills,
        y=before,
        marker_color="#CBD5E1",
        text=[f"{v:.1f}" for v in before],
        textposition="outside",
        textfont=dict(size=11, color="#64748B"),
    )
    fig.add_bar(
        name="Final Score",
        x=skills,
        y=after,
        marker_color=PRIMARY_BLUE,
        text=[f"{v:.1f}" for v in after],
        textposition="outside",
        textfont=dict(size=11, color=PRIMARY_BLUE, weight="bold"),
    )

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        yaxis=dict(range=[0, 115], title="Score (0-100)"),
        showlegend=True,
    )
    return apply_plotly_theme(fig, height=340, title="Before vs. After Skill Comparison")


def make_progress_curve(steps: list[dict], goal: float) -> go.Figure:
    xs = [s["step"] for s in steps]
    ys = [s["avg_ability"] for s in steps]

    fig = go.Figure()
    # Shaded area under progress curve
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers",
        name="Mean Ability",
        line=dict(color=PRIMARY_BLUE, width=3, shape="spline"),
        marker=dict(size=6, color=PRIMARY_BLUE, symbol="circle"),
        fill="tozeroy",
        fillcolor="rgba(21, 101, 192, 0.08)",
    ))
    # Target goal reference line
    fig.add_trace(go.Scatter(
        x=[xs[0], xs[-1]] if xs else [0, 1],
        y=[goal, goal],
        mode="lines",
        name=f"Target Goal ({goal:.0f})",
        line=dict(color="#EF4444", width=2, dash="dash"),
    ))

    fig.update_layout(showlegend=True)
    return apply_plotly_theme(fig, height=300, title="Cumulative Learning Progress Curve", x_title="Simulation Step", y_title="Average Ability")


def make_activity_bar(action_counts: dict) -> go.Figure:
    actions = list(action_counts.keys())
    counts  = list(action_counts.values())
    colors  = [ACTION_COLORS.get(a, PRIMARY_BLUE) for a in actions]

    fig = go.Figure(go.Bar(
        x=actions,
        y=counts,
        marker_color=colors,
        text=[f"<b>{c}</b>" for c in counts],
        textposition="outside",
        textfont=dict(size=12),
    ))
    return apply_plotly_theme(fig, height=300, title="Recommended Activity Distribution", x_title="Activity", y_title="Times Selected", show_legend=False)


# ---------------------------------------------------------------------------
# Main Page Execution
# ---------------------------------------------------------------------------
def main():
    render_page_header(
        title="Learner Simulation",
        subtitle="Simulate personalized adaptive learning episodes using trained Deep Q-Network models",
        badge_text="Realtime Inference",
        icon="👤",
    )

    df = load_learners_cached()
    if df is None:
        st.error("❌ Cannot load `data/simulated_learners.csv`. Please verify the dataset exists.")
        return

    available_seeds = get_available_seeds()
    if not available_seeds:
        st.error("❌ No DQN model checkpoints located in `models/`.")
        return

    # ── Simulation Sidebar Controls ──────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase;'>Simulation Controls</div>", unsafe_allow_html=True)

    profiles = get_profiles(df)
    profile_sel = st.sidebar.selectbox("1. Learner Profile", profiles)

    learner_ids = get_learner_ids_by_profile(df, profile_sel)
    display_ids = learner_ids[:300]
    learner_id_sel = st.sidebar.selectbox(
        f"2. Learner ID (Top {len(display_ids)})",
        display_ids,
        help="Select a specific simulated learner ID from the chosen profile cluster.",
    )

    seed_sel = st.sidebar.selectbox(
        "3. Model Seed (Checkpoint)",
        available_seeds,
        index=0,
        help="Select the trained DQN network checkpoint (Seeds 42 to 46).",
    )

    sim_seed_sel = st.sidebar.number_input(
        "4. Simulation Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="Ensures exact environment reproducibility (action transition noise).",
    )

    st.sidebar.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    run_btn   = st.sidebar.button("▶ Run DQN Simulation", use_container_width=True)
    reset_btn = st.sidebar.button("↺ Reset Episode",     use_container_width=True)

    if reset_btn:
        for k in ["sim_result", "sim_learner_id", "sim_seed"]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Initial Learner Profile Display ──────────────────────────────────────
    learner = get_learner_by_id(df, learner_id_sel)
    if learner is None:
        st.warning(f"Learner ID {learner_id_sel} not found.")
        return

    init_avg = (
        float(learner["ability_listening"]) + float(learner["ability_reading"]) +
        float(learner["ability_vocabulary"]) + float(learner["ability_grammar"])
    ) / 4.0

    render_section_header("Selected Learner Profile Baseline", f"ID: {learner_id_sel} • Category: {learner['profile']}")

    st.markdown(f"""
    <div class="analytics-card" style="margin-bottom:1.5rem; padding:18px 22px;">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:16px;">
        <div>
          <span style="font-size:1.1rem; font-weight:700; color:#111827;">Learner #{learner_id_sel}</span>
          <span class="badge badge-info" style="margin-left:8px;">{learner['profile']}</span>
        </div>
        <div style="display:flex; gap:16px; font-size:0.85rem; color:#4B5563;">
          <div>🎯 Target Goal: <strong style="color:#111827;">{learner['learning_goal']:.0f} pts</strong></div>
          <div>⏱ Time Budget: <strong style="color:#111827;">{learner['time_budget']:.0f} mins</strong></div>
          <div>📊 Initial Average: <strong style="color:{PRIMARY_BLUE};">{init_avg:.2f} pts</strong></div>
        </div>
      </div>
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px;">
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#64748B; font-weight:600;">
            <span>🎧 LISTENING</span>
            <span style="color:#111827;">{learner['ability_listening']:.1f} / 100</span>
          </div>
          <div style="background:#E2E8F0; height:6px; border-radius:3px; margin-top:6px; overflow:hidden;">
            <div style="background:#1565C0; height:100%; width:{learner['ability_listening']}%;"></div>
          </div>
        </div>
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#64748B; font-weight:600;">
            <span>📖 READING</span>
            <span style="color:#111827;">{learner['ability_reading']:.1f} / 100</span>
          </div>
          <div style="background:#E2E8F0; height:6px; border-radius:3px; margin-top:6px; overflow:hidden;">
            <div style="background:#42A5F5; height:100%; width:{learner['ability_reading']}%;"></div>
          </div>
        </div>
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#64748B; font-weight:600;">
            <span>📝 VOCABULARY</span>
            <span style="color:#111827;">{learner['ability_vocabulary']:.1f} / 100</span>
          </div>
          <div style="background:#E2E8F0; height:6px; border-radius:3px; margin-top:6px; overflow:hidden;">
            <div style="background:#10B981; height:100%; width:{learner['ability_vocabulary']}%;"></div>
          </div>
        </div>
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#64748B; font-weight:600;">
            <span>📐 GRAMMAR</span>
            <span style="color:#111827;">{learner['ability_grammar']:.1f} / 100</span>
          </div>
          <div style="background:#E2E8F0; height:6px; border-radius:3px; margin-top:6px; overflow:hidden;">
            <div style="background:#F59E0B; height:100%; width:{learner['ability_grammar']}%;"></div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Simulation Trigger ───────────────────────────────────────────────────
    if run_btn:
        with st.spinner(f"Executing greedy DQN policy inference using Model Seed {seed_sel}…"):
            model = load_model_cached(seed_sel)
            if model is None:
                st.error(f"❌ Failed to load DQN checkpoint `dqn_model_seed_{seed_sel}.pth`.")
                return
            result = run_episode(model, learner, sim_seed=int(sim_seed_sel))
            st.session_state["sim_result"]     = result
            st.session_state["sim_learner_id"] = learner_id_sel
            st.session_state["sim_seed"]       = seed_sel

    # ── Render Simulation Results ────────────────────────────────────────────
    result = st.session_state.get("sim_result")
    if result is None:
        st.info("💡 Click **▶ Run DQN Simulation** in the left sidebar to execute the recommendation trajectory.")
        return

    summary      = result["summary"]
    steps        = result["steps"]
    learner_info = result["learner_info"]
    final_step   = steps[-1] if steps else {}

    render_section_header("Simulation Episode Results", f"Model: Seed {st.session_state.get('sim_seed')} • Simulation Seed: {summary['sim_seed']}")

    # KPI row
    is_goal_met = summary["goal_achieved"]
    gain_pct = ((summary['final_avg_ability'] - summary['init_avg_ability']) / summary['init_avg_ability'] * 100) if summary['init_avg_ability'] > 0 else 0

    c_kpi1 = render_kpi_card(
        label="Initial Ability",
        value=f"{summary['init_avg_ability']:.1f}",
        delta=None,
        icon="📊",
        footer_text="Baseline score",
    )
    c_kpi2 = render_kpi_card(
        label="Final Ability",
        value=f"{summary['final_avg_ability']:.1f}",
        delta=f"+{summary['ability_gain']:.2f} pts",
        delta_type="positive",
        icon="🚀",
        footer_text=f"Gain: +{gain_pct:.1f}%",
    )
    c_kpi3 = render_kpi_card(
        label="Episode Return",
        value=f"{summary['episode_return']:.2f}",
        delta="Cumulative",
        delta_type="positive" if summary['episode_return'] > 0 else "negative",
        icon="⭐",
        footer_text="DQN reward sum",
    )
    c_kpi4 = render_kpi_card(
        label="Time Budget Used",
        value=f"{summary['total_time_used']:.0f} m",
        delta=f"{summary['total_time_used']/summary['time_budget']*100:.0f}% used",
        delta_type="neutral",
        icon="⏱",
        footer_text=f"Budget: {summary['time_budget']:.0f} mins",
    )
    c_kpi5 = render_kpi_card(
        label="Goal Achieved",
        value="Achieved" if is_goal_met else "Incomplete",
        delta="Goal Met" if is_goal_met else "Time Exhausted",
        delta_type="positive" if is_goal_met else "negative",
        icon="🎯" if is_goal_met else "⏳",
        footer_text=f"Target: {summary['learning_goal']:.0f} pts",
    )
    c_kpi6 = render_kpi_card(
        label="Learning Steps",
        value=str(summary["total_steps"]),
        delta="Steps",
        delta_type="neutral",
        icon="🪜",
        footer_text="Completed actions",
    )

    st.markdown(f"""
    <div class="kpi-row">
      {c_kpi1}
      {c_kpi2}
      {c_kpi3}
      {c_kpi4}
      {c_kpi5}
      {c_kpi6}
    </div>
    """, unsafe_allow_html=True)

    # ── Interactive Charts Grid ──────────────────────────────────────────────
    col_c1, col_c2 = st.columns(2, gap="medium")
    with col_c1:
        st.plotly_chart(make_ability_bar(learner_info, final_step), use_container_width=True, config={"displayModeBar": False})

    with col_c2:
        before_dict = {
            "ability_listening":  learner_info["ability_listening"],
            "ability_reading":    learner_info["ability_reading"],
            "ability_vocabulary": learner_info["ability_vocabulary"],
            "ability_grammar":    learner_info["ability_grammar"],
        }
        after_dict = {
            "ability_listening":  final_step.get("ability_listening", learner_info["ability_listening"]),
            "ability_reading":    final_step.get("ability_reading", learner_info["ability_reading"]),
            "ability_vocabulary": final_step.get("ability_vocabulary", learner_info["ability_vocabulary"]),
            "ability_grammar":    final_step.get("ability_grammar", learner_info["ability_grammar"]),
        }
        st.plotly_chart(make_radar(before_dict, after_dict), use_container_width=True, config={"displayModeBar": False})

    col_c3, col_c4 = st.columns(2, gap="medium")
    with col_c3:
        st.plotly_chart(make_progress_curve(steps, summary["learning_goal"]), use_container_width=True, config={"displayModeBar": False})

    with col_c4:
        st.plotly_chart(make_activity_bar(summary["action_counts"]), use_container_width=True, config={"displayModeBar": False})

    # ── Episode Step Table ───────────────────────────────────────────────────
    render_section_header("Episode Transition Details", "Step-by-step Q-values, immediate rewards, and skill adjustments")

    steps_df = pd.DataFrame(steps)
    display_cols = [
        "step", "action_name", "q_listening", "q_reading", "q_vocabulary", "q_grammar",
        "reward", "time_cost", "time_remaining",
        "avg_ability", "ability_listening", "ability_reading", "ability_vocabulary", "ability_grammar",
    ]
    avail_cols = [c for c in display_cols if c in steps_df.columns]

    st.dataframe(
        steps_df[avail_cols].style.format({
            c: "{:.2f}" for c in avail_cols if steps_df[c].dtype in [float, "float64"]
        }),
        use_container_width=True,
        height=280,
    )

    # ── Export ───────────────────────────────────────────────────────────────
    csv_bytes = steps_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export Simulation Episode Results (CSV)",
        data=csv_bytes,
        file_name=f"simulation_learner_{learner_id_sel}_seed_{seed_sel}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
