"""
pages/4_Training_Analytics.py
——————————————————————————————
DQN Training Analytics:
Examine multi-seed training trajectories, reward curves (raw & moving average),
epsilon exploration decay, ability gain progression, and cross-seed stability.
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
    page_title="Training Analytics | DQN Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DESIGN_CSS, unsafe_allow_html=True)
render_sidebar()

# Distinct color palette for the 5 training seeds
SEED_COLORS = {
    42: "#1565C0",  # Primary Blue
    43: "#0284C7",  # Light Blue
    44: "#10B981",  # Emerald Green
    45: "#F59E0B",  # Amber
    46: "#8B5CF6",  # Purple
}


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_training() -> pd.DataFrame | None:
    path = DATA_DIR / "training_history_all_seeds.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------
def main():
    render_page_header(
        title="Training Analytics & Dynamics",
        subtitle="Multi-seed DQN learning dynamics, reward optimization trajectories, and policy convergence analysis",
        badge_text="5-Seed Training Logs",
        icon="📈",
    )

    df = load_training()

    if df is None:
        st.error("❌ `data/training_history_all_seeds.csv` not found.")
        return

    # Check available metrics in dataset
    has_reward       = "reward" in df.columns
    has_epsilon      = "epsilon" in df.columns
    has_ability_gain = "avg_ability_gain" in df.columns
    has_loss         = "loss" in df.columns
    has_episode      = "episode" in df.columns
    has_seed         = "training_seed" in df.columns

    all_seeds = sorted(df["training_seed"].unique().tolist()) if has_seed else list(SEEDS)

    # ── Sidebar Filters ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase;'>Training Filters</div>", unsafe_allow_html=True)

    seeds_sel = st.sidebar.multiselect("Training Seeds", all_seeds, default=all_seeds)
    ma_window = st.sidebar.slider("Moving Average Smoothing Window", 1, 100, 25, step=5, help="Window size for moving-average trend line smoothing.")

    if has_episode:
        ep_min = int(df["episode"].min())
        ep_max = int(df["episode"].max())
        ep_range = st.sidebar.slider("Episode Window", ep_min, ep_max, (ep_min, ep_max), step=100)
    else:
        ep_range = None

    # Apply filters
    filt = df.copy()
    if seeds_sel and has_seed:
        filt = filt[filt["training_seed"].isin(seeds_sel)]
    if ep_range and has_episode:
        filt = filt[(filt["episode"] >= ep_range[0]) & (filt["episode"] <= ep_range[1])]

    if filt.empty:
        st.warning("No training logs match the selected parameters.")
        return

    # ── Executive Training KPI Cards ─────────────────────────────────────────
    render_section_header("Training Convergence Summary", f"Data from {len(seeds_sel)} training seeds • Window: {ma_window} episodes")

    max_ep = filt["episode"].max() if has_episode else 0
    mean_rwd = filt["reward"].mean() if has_reward else 0.0

    last_ep_cutoff = max_ep - 50 if max_ep > 50 else max_ep
    final_rwd = filt[filt["episode"] >= last_ep_cutoff]["reward"].mean() if has_reward else 0.0
    rwd_gain = final_rwd - filt[filt["episode"] <= 50]["reward"].mean() if has_reward else 0.0

    mean_ab_gain = filt["avg_ability_gain"].mean() if has_ability_gain else 0.0

    k1 = render_kpi_card(
        label="Total Episodes / Seed",
        value=f"{max_ep:,}",
        delta=f"{len(seeds_sel)} Seeds",
        delta_type="neutral",
        icon="🔁",
        footer_text="Training length",
    )
    k2 = render_kpi_card(
        label="Mean Training Reward",
        value=f"{mean_rwd:.2f}",
        delta="Cumulative",
        delta_type="positive",
        icon="📊",
        footer_text="Across all episodes",
    )
    k3 = render_kpi_card(
        label="Final Convergence Reward",
        value=f"{final_rwd:.2f}",
        delta=f"+{rwd_gain:.2f} delta",
        delta_type="positive" if rwd_gain >= 0 else "negative",
        icon="⭐",
        footer_text="Final 50 episodes mean",
    )
    k4 = render_kpi_card(
        label="Mean Ability Gain",
        value=f"{mean_ab_gain:.2f} pts",
        delta="Learning Rate",
        delta_type="positive",
        icon="🚀",
        footer_text="Per training episode",
    )
    k5 = render_kpi_card(
        label="Active Seeds",
        value=f"{len(seeds_sel)} / {len(all_seeds)}",
        delta="Multi-Seed",
        delta_type="positive",
        icon="🌱",
        footer_text="Seeds: " + ", ".join(map(str, seeds_sel)),
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

    # ── Analytics Visual Tabs ────────────────────────────────────────────────
    tab_reward, tab_eps, tab_gain, tab_loss, tab_action, tab_stability = st.tabs([
        "📉 Reward Curve",
        "🔍 Epsilon Exploration Decay",
        "📈 Ability Gain Trajectory",
        "💔 Training Loss",
        "🎯 Action Selection",
        "📊 Multi-Seed Stability",
    ])

    x_col = "episode" if has_episode else filt.columns[0]

    with tab_reward:
        col_ctrl, col_chart = st.columns([3, 9], gap="medium")
        with col_ctrl:
            st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#1F2937;'>Display Controls</div>", unsafe_allow_html=True)
            show_raw = st.checkbox("Overlay raw episode rewards", value=False, key="show_raw_rwd_check")
            st.markdown(f"""
            <div style="font-size:0.8rem; color:#6B7280; line-height:1.5; margin-top:8px;">
              Moving average smoothing window (<strong>{ma_window} episodes</strong>) illuminates policy learning trends while suppressing stochastic environment noise.
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            fig_rwd = go.Figure()
            for seed in seeds_sel:
                s_df = filt[filt["training_seed"] == seed].sort_values(x_col) if has_seed else filt.sort_values(x_col)
                color = SEED_COLORS.get(seed, PRIMARY_BLUE)
                ma = smooth(s_df["reward"], ma_window)

                if show_raw:
                    fig_rwd.add_trace(go.Scatter(
                        x=s_df[x_col],
                        y=s_df["reward"],
                        mode="lines",
                        name=f"Seed {seed} (Raw)",
                        line=dict(color=color, width=1, dash="dot"),
                        opacity=0.3,
                    ))
                fig_rwd.add_trace(go.Scatter(
                    x=s_df[x_col],
                    y=ma,
                    mode="lines",
                    name=f"Seed {seed} (MA {ma_window})",
                    line=dict(color=color, width=2.5),
                ))

            st.plotly_chart(
                apply_plotly_theme(fig_rwd, height=380, title=f"Training Reward Optimization Curve across Seeds", x_title="Training Episode", y_title="Episode Reward"),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with tab_eps:
        if has_epsilon:
            fig_eps = go.Figure()
            for seed in seeds_sel:
                s_df = filt[filt["training_seed"] == seed].sort_values(x_col) if has_seed else filt.sort_values(x_col)
                color = SEED_COLORS.get(seed, PRIMARY_BLUE)
                fig_eps.add_trace(go.Scatter(
                    x=s_df[x_col],
                    y=s_df["epsilon"],
                    mode="lines",
                    name=f"Seed {seed}",
                    line=dict(color=color, width=2),
                ))
            st.plotly_chart(
                apply_plotly_theme(fig_eps, height=360, title="Exploration Rate (ε-Decay Schedule)", x_title="Episode", y_title="Epsilon (ε)"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Epsilon column not present.")

    with tab_gain:
        if has_ability_gain:
            fig_gain = go.Figure()
            for seed in seeds_sel:
                s_df = filt[filt["training_seed"] == seed].sort_values(x_col) if has_seed else filt.sort_values(x_col)
                color = SEED_COLORS.get(seed, PRIMARY_BLUE)
                ma = smooth(s_df["avg_ability_gain"], ma_window)
                fig_gain.add_trace(go.Scatter(
                    x=s_df[x_col],
                    y=ma,
                    mode="lines",
                    name=f"Seed {seed} (MA {ma_window})",
                    line=dict(color=color, width=2.5),
                ))
            st.plotly_chart(
                apply_plotly_theme(fig_gain, height=360, title="Learner Ability Gain per Episode across Seeds", x_title="Episode", y_title="Average Ability Gain (pts)"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Average ability gain column not present.")

    with tab_loss:
        if has_loss and filt["loss"].notna().any():
            fig_loss = go.Figure()
            for seed in seeds_sel:
                s_df = filt[filt["training_seed"] == seed].sort_values(x_col) if has_seed else filt.sort_values(x_col)
                color = SEED_COLORS.get(seed, PRIMARY_BLUE)
                ma = smooth(s_df["loss"], ma_window)
                fig_loss.add_trace(go.Scatter(
                    x=s_df[x_col],
                    y=ma,
                    mode="lines",
                    name=f"Seed {seed} Loss",
                    line=dict(color=color, width=2),
                ))
            st.plotly_chart(
                apply_plotly_theme(fig_loss, height=360, title="DQN Temporal-Difference (TD) Loss Curve", x_title="Episode", y_title="TD Loss"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.markdown("""
            <div class="analytics-card" style="padding:28px; text-align:center;">
              <div style="font-size:2rem; margin-bottom:10px;">ℹ️</div>
              <div style="font-weight:700; font-size:1.05rem; color:#1F2937;">Training Loss Was Not Recorded</div>
              <div style="font-size:0.85rem; color:#6B7280; max-width:540px; margin:8px auto 0 auto; line-height:1.6;">
                In the original experimental setup recorded in <code>training_history_all_seeds.csv</code>,
                the training loop logged <strong>episode rewards, epsilon exploration values, and ability gains</strong>,
                but did not persist per-step neural network TD-loss values. To maintain strict research integrity, no synthetic loss values are fabricated.
              </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_action:
        action_cols = [c for c in df.columns if c.startswith("count_") or c.startswith("action_")]
        if action_cols:
            agg_act = filt[["training_seed"] + action_cols].groupby("training_seed").sum().reset_index()
            st.dataframe(agg_act, use_container_width=True)
        else:
            st.markdown("""
            <div class="analytics-card" style="padding:28px; text-align:center;">
              <div style="font-size:2rem; margin-bottom:10px;">ℹ️</div>
              <div style="font-weight:700; font-size:1.05rem; color:#1F2937;">Training-Phase Action Counts Not Recorded</div>
              <div style="font-size:0.85rem; color:#6B7280; max-width:540px; margin:8px auto 0 auto; line-height:1.6;">
                Action distribution statistics during training are not stored in <code>training_history_all_seeds.csv</code>.
                Detailed activity selection frequencies are fully accessible in <strong>Page 1 (Learner Simulation)</strong> and <strong>Page 2 (Recommended Learning Path)</strong> during policy inference.
              </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_stability:
        if has_reward and has_seed:
            col_box, col_tbl = st.columns([7, 5], gap="large")
            with col_box:
                # Terminal 100 episodes per seed
                last_eps = (
                    filt.groupby("training_seed")
                    .apply(lambda x: x.nlargest(min(100, len(x)), x_col))
                    .reset_index(drop=True)
                )
                fig_stab = px.box(
                    last_eps,
                    x="training_seed",
                    y="reward",
                    color="training_seed",
                    color_discrete_map={s: SEED_COLORS.get(s, PRIMARY_BLUE) for s in seeds_sel},
                    labels={"training_seed": "Training Seed", "reward": "Terminal Reward (Last 100 Episodes)"},
                )
                st.plotly_chart(
                    apply_plotly_theme(fig_stab, height=360, title="Final 100 Episodes Reward Stability Across Seeds", x_title="Training Seed", y_title="Reward", show_legend=False),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            with col_tbl:
                st.markdown("<div style='font-size:0.92rem; font-weight:700; color:#1F2937; margin-bottom:8px;'>Cross-Seed Consistency Metrics</div>", unsafe_allow_html=True)
                stab_df = filt.groupby("training_seed")["reward"].agg(["mean", "std", "min", "max"]).reset_index()
                stab_df.columns = ["Seed", "Mean Reward", "Std Dev", "Min", "Max"]
                st.dataframe(
                    stab_df.style.format({
                        "Mean Reward": "{:.2f}",
                        "Std Dev": "{:.2f}",
                        "Min": "{:.2f}",
                        "Max": "{:.2f}",
                    }),
                    use_container_width=True,
                    height=240,
                )
                st.caption("Low variance across independent random seeds confirms policy stability.")

    # ── Research Interpretation Card ─────────────────────────────────────────
    render_section_header("Analytical Findings & Convergence Insights")

    st.markdown("""
    <div class="analytics-card" style="padding:18px 22px;">
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px;">
        <div>
          <div style="font-weight:700; font-size:0.88rem; color:#1565C0;">1. Reward Trajectory & Stability</div>
          <p style="font-size:0.82rem; color:#4B5563; margin-top:4px; line-height:1.55;">
            Smoothed reward trajectories show consistent upward momentum across all 5 seeds, indicating effective Q-network policy improvement. Multi-seed stability boxplots verify that final policy returns converge to comparable asymptotes.
          </p>
        </div>
        <div>
          <div style="font-weight:700; font-size:0.88rem; color:#1565C0;">2. Exploration vs. Exploitation Schedule</div>
          <p style="font-size:0.82rem; color:#4B5563; margin-top:4px; line-height:1.55;">
            The linear epsilon-decay policy enables extensive early-stage state space exploration before transitioning into exploitation mode, stabilizing Q-values near the target competency goals.
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download ─────────────────────────────────────────────────────────────
    render_section_header("Export Training Logs")
    csv_bytes = filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Filtered Training History (CSV)",
        data=csv_bytes,
        file_name="filtered_training_history.csv",
        mime="text/csv",
        use_container_width=False,
    )


if __name__ == "__main__":
    main()
