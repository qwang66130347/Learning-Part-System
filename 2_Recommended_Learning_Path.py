"""
pages/2_Recommended_Learning_Path.py
——————————————————————————————————————
Modern Enterprise Learning Path:
Visualize sequential learning activities chosen by the DQN policy,
inspect step-level Q-values and immediate rewards, and trace multi-skill development.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_core import ACTIONS, ACTION_COLORS

from shared_ui import (
    DESIGN_CSS,
    PRIMARY_BLUE,
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
    page_title="Learning Path | DQN Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DESIGN_CSS, unsafe_allow_html=True)
render_sidebar()


# ---------------------------------------------------------------------------
# Visual Chart Generators
# ---------------------------------------------------------------------------
def make_q_bar(step_data: dict, selected_action: int) -> go.Figure:
    action_names = list(ACTIONS.values())
    q_vals = [step_data[f"q_{a.lower()}"] for a in action_names]

    # Selected action highlighted in primary blue, others in neutral gray-blue
    colors = [PRIMARY_BLUE if i == selected_action else "#CBD5E1" for i in range(4)]

    fig = go.Figure(go.Bar(
        x=action_names,
        y=q_vals,
        marker_color=colors,
        text=[f"<b>{v:.2f}</b>" if i == selected_action else f"{v:.2f}" for i, v in enumerate(q_vals)],
        textposition="outside",
        textfont=dict(size=11),
    ))
    return apply_plotly_theme(
        fig,
        height=280,
        title=f"Step {step_data['step']} Q-Value Ranking (Argmax = {step_data['action_name']})",
        x_title="Candidate Activity",
        y_title="Estimated Q-Value",
        show_legend=False,
    )


def make_ability_timeline(steps: list[dict]) -> go.Figure:
    xs = [s["step"] for s in steps]
    fig = go.Figure()
    skill_map = {
        "Listening":  ("ability_listening",  "#1565C0"),
        "Reading":    ("ability_reading",    "#42A5F5"),
        "Vocabulary": ("ability_vocabulary", "#10B981"),
        "Grammar":    ("ability_grammar",    "#F59E0B"),
    }
    for name, (col, color) in skill_map.items():
        fig.add_trace(go.Scatter(
            x=xs,
            y=[s[col] for s in steps],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=5, symbol="circle"),
        ))

    return apply_plotly_theme(
        fig,
        height=320,
        title="Multi-Skill Progression Trajectory",
        x_title="Step Number",
        y_title="Ability Score (0-100)",
        show_legend=True,
    )


def make_reward_bar(steps: list[dict]) -> go.Figure:
    xs = [s["step"] for s in steps]
    ys = [s["reward"] for s in steps]
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in ys]

    fig = go.Figure(go.Bar(
        x=xs,
        y=ys,
        marker_color=colors,
        text=[f"{v:.1f}" for v in ys],
        textposition="outside",
        textfont=dict(size=10),
    ))
    return apply_plotly_theme(
        fig,
        height=260,
        title="Immediate Step Reward",
        x_title="Step",
        y_title="Step Reward",
        show_legend=False,
    )


def make_action_dist(steps: list[dict]) -> go.Figure:
    counts = Counter(s["action_name"] for s in steps)
    actions = list(ACTIONS.values())
    vals = [counts.get(a, 0) for a in actions]
    colors = [ACTION_COLORS.get(a, PRIMARY_BLUE) for a in actions]

    fig = go.Figure(go.Bar(
        x=actions,
        y=vals,
        marker_color=colors,
        text=[f"<b>{v}</b>" for v in vals],
        textposition="outside",
        textfont=dict(size=11),
    ))
    return apply_plotly_theme(
        fig,
        height=260,
        title="Activity Selection Frequency",
        x_title="Activity",
        y_title="Selections",
        show_legend=False,
    )


def activity_icon(name: str) -> str:
    icons = {"Listening": "🎧", "Reading": "📖", "Vocabulary": "📝", "Grammar": "📐"}
    return icons.get(name, "📚")


def render_timeline_step(step: dict, is_selected: bool = False) -> str:
    action_name = step["action_name"]
    color = ACTION_COLORS.get(action_name, PRIMARY_BLUE)
    icon  = activity_icon(action_name)
    q_sel = step[f"q_{action_name.lower()}"]
    highlight_style = "border: 2px solid " + PRIMARY_BLUE + ";" if is_selected else "border: 1px solid #E5E7EB;"

    return f"""
    <div style="display:flex; align-items:flex-start; margin-bottom:12px;">
      <div style="background:{color}; color:#fff; border-radius:50%; width:34px; height:34px;
                  display:flex; align-items:center; justify-content:center; font-weight:700;
                  font-size:0.85rem; flex-shrink:0; margin-right:12px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
        {step['step']}
      </div>
      <div style="background:#FFFFFF; {highlight_style} border-radius:12px; padding:12px 16px;
                  flex:1; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:700; color:{color}; font-size:0.95rem;">{icon} {action_name}</span>
          <span style="font-size:0.75rem; background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:6px; font-weight:600;">
            Q: {q_sel:.2f} | R: {step['reward']:.2f}
          </span>
        </div>
        <div style="font-size:0.78rem; color:#6B7280; margin-top:5px;">
          ⏱ <strong>{step['time_cost']:.0f}m</strong> spent • Remaining: {step['time_remaining']:.0f}m • Mean Ability: <strong>{step['avg_ability']:.2f}</strong>
        </div>
        <div style="font-size:0.74rem; color:#9CA3AF; margin-top:3px;">
          L: {step['ability_listening']:.1f} • R: {step['ability_reading']:.1f} • V: {step['ability_vocabulary']:.1f} • G: {step['ability_grammar']:.1f}
        </div>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------
def main():
    render_page_header(
        title="Recommended Learning Path",
        subtitle="Step-by-step activity sequence selected by the DQN policy with Q-values and immediate rewards",
        badge_text="Sequential Policy",
        icon="🗺️",
    )

    result = st.session_state.get("sim_result")

    if result is None:
        st.markdown("""
        <div class="analytics-card" style="text-align:center; padding:3rem 2rem;">
          <div style="font-size:3rem; margin-bottom:12px;">👤</div>
          <div style="font-size:1.2rem; font-weight:700; color:#1F2937;">No Active Simulation Session Found</div>
          <div style="font-size:0.88rem; color:#6B7280; max-width:480px; margin:8px auto 20px auto;">
            Please navigate to the <strong>Learner Simulation</strong> page first, select a simulated learner profile, and click <strong>Run DQN Simulation</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    steps        = result["steps"]
    summary      = result["summary"]
    learner_info = result["learner_info"]

    # ── Executive KPI Summary Row ────────────────────────────────────────────
    k1 = render_kpi_card(
        label="Learner ID",
        value=f"#{summary['learner_id']}",
        delta=summary['profile'],
        delta_type="neutral",
        icon="👤",
        footer_text="Target profile",
    )
    k2 = render_kpi_card(
        label="Total Steps",
        value=str(summary["total_steps"]),
        delta=f"{summary['total_time_used']:.0f} mins",
        delta_type="positive",
        icon="🪜",
        footer_text=f"of {summary['time_budget']:.0f} min budget",
    )
    k3 = render_kpi_card(
        label="Cumulative Return",
        value=f"{summary['episode_return']:.2f}",
        delta="DQN Return",
        delta_type="positive" if summary['episode_return'] > 0 else "negative",
        icon="⭐",
        footer_text="Optimal policy sum",
    )
    k4 = render_kpi_card(
        label="Ability Improvement",
        value=f"+{summary['ability_gain']:.2f}",
        delta=f"{summary['init_avg_ability']:.1f} → {summary['final_avg_ability']:.1f}",
        delta_type="positive",
        icon="📈",
        footer_text="Net average gain",
    )
    k5 = render_kpi_card(
        label="Goal Achievement",
        value="Achieved" if summary["goal_achieved"] else "Incomplete",
        delta="Goal Met" if summary["goal_achieved"] else "Time Cap",
        delta_type="positive" if summary["goal_achieved"] else "negative",
        icon="🎯" if summary["goal_achieved"] else "⏳",
        footer_text=f"Target: {summary['learning_goal']:.0f} pts",
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

    # ── Layout: Timeline on Left, Charts & Step Inspector on Right ───────────
    col_timeline, col_analytics = st.columns([5, 7], gap="large")

    with col_timeline:
        render_section_header("Sequential Activity Timeline", f"Showing {len(steps)} recommended learning actions")

        max_show = st.slider("Display depth (steps)", 1, len(steps), min(15, len(steps)), key="slider_depth")
        for s in steps[:max_show]:
            st.markdown(render_timeline_step(s), unsafe_allow_html=True)
        if max_show < len(steps):
            st.caption(f"Showing first {max_show} of {len(steps)} steps. Adjust slider above to view remaining.")

    with col_analytics:
        # Step Inspector & Q-Value Analysis
        render_section_header("Step-by-Step Q-Value Inspector", "Compare estimated expected return across all 4 actions")

        step_inspect = st.slider("Select step to inspect:", 1, len(steps), 1, key="step_inspector_slider")
        current_step = steps[step_inspect - 1]

        st.markdown(f"""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:10px 14px; margin-bottom:12px; font-size:0.84rem; color:#166534;">
          <strong>Step {step_inspect} Decision:</strong> Agent chose <strong>{current_step['action_name']}</strong>
          (Highest Q-value: <strong>{current_step['q_' + current_step['action_name'].lower()]:.3f}</strong>).
          <span style="color:#14532D; font-size:0.75rem;">Q-values represent estimated cumulative future discounted rewards under policy π*.</span>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(
            make_q_bar(current_step, current_step["action"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Multi-Skill Trajectory
        render_section_header("Competency Trajectory Over Steps")
        st.plotly_chart(make_ability_timeline(steps), use_container_width=True, config={"displayModeBar": False})

        # Dual Row: Immediate Reward & Action Distribution
        col_r1, col_r2 = st.columns(2, gap="medium")
        with col_r1:
            st.plotly_chart(make_reward_bar(steps), use_container_width=True, config={"displayModeBar": False})
        with col_r2:
            st.plotly_chart(make_action_dist(steps), use_container_width=True, config={"displayModeBar": False})

    # ── Full Learning Path Table ─────────────────────────────────────────────
    render_section_header("Exportable Learning Sequence Log", "Full transition parameters and state vector recordings")

    steps_df = pd.DataFrame(steps)
    display_cols = [
        "step", "action_name", "q_listening", "q_reading", "q_vocabulary", "q_grammar",
        "reward", "time_cost", "time_used", "time_remaining",
        "avg_ability", "ability_gain",
        "ability_listening", "ability_reading", "ability_vocabulary", "ability_grammar",
        "goal_achieved",
    ]
    avail = [c for c in display_cols if c in steps_df.columns]
    st.dataframe(
        steps_df[avail].style.format({
            c: "{:.2f}" for c in avail if steps_df[c].dtype in [float, "float64"]
        }),
        use_container_width=True,
        height=260,
    )

    csv_bytes = steps_df[avail].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Complete Learning Path Sequence (CSV)",
        data=csv_bytes,
        file_name=f"learning_path_learner_{summary['learner_id']}_seed_{summary['sim_seed']}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
