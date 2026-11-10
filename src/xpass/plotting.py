
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

MATCH_REPORTS_DIR = (
    Path.home() / "Desktop" / "Football Analyses" / "Match Reports"
)

sys.path.insert(0, str(MATCH_REPORTS_DIR))

from match_report_utils import add_plotly_pitch


def plot_xpass_map(
    df,
    match_id,
    player_id = None,
    player_name = "player",
    opposition = "opposition",
    title=None,
    successful_only=False,
    save_path=None
):

    # --------------------------------------------------
    # 1. Validate columns
    # --------------------------------------------------

    required = [
        "game_id", "player", "x", "y", "end_x", "end_y",
        "xPass", "PAx", "outcome"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if player_id is None and player_name is None:
        raise ValueError("Supply either player_id or player_name.")

    # --------------------------------------------------
    # 2. Extract player and match
    # --------------------------------------------------

    passes = df[
        df["game_id"].astype(str) == str(match_id)
    ].copy()

    if player_id is not None:
        passes = passes[
            passes["player_id"].astype(str) == str(player_id)
        ].copy()

    else:
        passes = passes[
            passes["player"].str.casefold()
            == player_name.casefold()
        ].copy()

    if passes.empty:
        raise ValueError(
            "No passes found for this player in this match."
        )

    if player_name is None:
        player_name = passes["player"].iloc[0]

    # --------------------------------------------------
    # 3. Prepare plotting data
    # --------------------------------------------------

    passes["xPass"] = pd.to_numeric(
        passes["xPass"], errors="coerce"
    )

    passes["PAx"] = pd.to_numeric(
        passes["PAx"], errors="coerce"
    )

    passes = passes.dropna(
        subset=["x", "y", "end_x", "end_y", "xPass", "PAx"]
    ).copy()

    # WhoScored coordinates -> 105 x 68 metre pitch
    passes["plot_x"] = passes["x"] * 1.05
    passes["plot_y"] = passes["y"] * 0.68

    passes["plot_end_x"] = passes["end_x"] * 1.05
    passes["plot_end_y"] = passes["end_y"] * 0.68

    # Actual outcome can be recovered from:
    # PAx = actual outcome - xPass
    passes["actual_completion"] = (
        passes["xPass"] + passes["PAx"]
    ).round().astype(int)

    passes["actual_completion"] = (
        passes["actual_completion"].clip(0, 1)
    )

    passes["completion_label"] = np.where(
        passes["actual_completion"].eq(1),
        "Successful",
        "Unsuccessful"
    )

    # Match minute
    if "minute" in passes.columns:
        passes["display_minute"] = (
            passes["minute"].astype(str) + "'"
        )
    else:
        passes["display_minute"] = "Unknown"

    # Optional filtering
    if successful_only:
        plotted = passes[
            passes["actual_completion"].eq(1)
        ].copy()
    else:
        plotted = passes.copy()

    # --------------------------------------------------
    # 4. Summary statistics
    # --------------------------------------------------

    attempted = len(passes)

    completed = passes["actual_completion"].sum()

    expected_completions = passes["xPass"].sum()

    total_pax = passes["PAx"].sum()

    accuracy = (
        completed / attempted * 100
        if attempted else np.nan
    )

    avg_xpass = (passes["xPass"].sum()) / attempted

    print(f"\n{player_name} vs {opposition}")
    print("-" * 40)
    print(f"Passes attempted: {attempted}")
    print(f"Passes completed: {completed}")
    print(f"Passing accuracy: {accuracy:.1f}%")
    print(f"Expected completions: {expected_completions:.2f}")
    print(f"Total PAx: {total_pax:+.2f}")
    print(f"Average xPass: {avg_xpass:.2f}")

    # --------------------------------------------------
    # 5. Set up Plotly pitch
    # --------------------------------------------------

    fig = go.Figure()

    add_plotly_pitch(fig)

    # Low xPass = difficult
    # High xPass = straightforward

    colours = [
        [0.0, "#3562DB"],
        [0.6, "#7493E7"],
        [1.0, "#CCCCCC"]
    ]

    # --------------------------------------------------
    # 6. Plot passes and hover information
    # --------------------------------------------------

    hover_x = []
    hover_y = []
    hover_text = []

    for _, row in plotted.iterrows():

        x = row["plot_x"]
        y = row["plot_y"]

        ex = row["plot_end_x"]
        ey = row["plot_end_y"]

        xp = row["xPass"]
        pax = row["PAx"]

        norm = np.clip(xp, 0, 1)

        colour = sample_colorscale(
            colours, [norm]
        )[0]

        # Directional arrow
        fig.add_annotation(
            x=ex,
            y=ey,
            ax=x,
            ay=y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=colour
        )

        # Hover target at midpoint
        hover_x.append((x + ex) / 2)
        hover_y.append((y + ey) / 2)

        hover_text.append(
            f"<b>{row['completion_label']}</b><br>"
            f"Minute: {row['display_minute']}<br>"
            f"xPass: {xp:.3f}<br>"
            f"PAx: {pax:+.3f}"
            "<extra></extra>"
        )

    # Hover layer
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(
                size=12,
                color="rgba(0,0,0,0.01)"
            ),
            hovertext=hover_text,
            hovertemplate="%{hovertext}",
            showlegend=False
        )
    )

    # --------------------------------------------------
    # 7. Colour bar
    # --------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=0,
                color=[0],
                colorscale=colours,
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="xPass",
                        side="right"
                    ),
                    tickmode="array",
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=[
                        "0.00", "0.25", "0.50",
                        "0.75", "1.00"
                    ],
                    tickfont=dict(color="#CCCCCC"),
                    title_font=dict(color="#CCCCCC"),
                    thickness=15,
                    len=0.75,
                    x=1.05
                )
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # --------------------------------------------------
    # 8. Layout
    # --------------------------------------------------

    chart_title = title or (
        f"{player_name} Passing Difficulty vs {opposition}"
    )

    fig.update_layout(
        title=dict(
            text=chart_title,
            x=0.5,
            y=0.95,
            xanchor="center"
        ),
        paper_bgcolor="#404040",
        plot_bgcolor="#404040",
        font=dict(
            family="DIN Alternate",
            color="#FFFFFF",
            size=12
        ),
        margin=dict(l=15, r=15, t=55, b=30),
        hoverlabel=dict(
            bgcolor="#3F434A",
            bordercolor="#CCCCCC",
            font=dict(
                color="#CCCCCC",
                size=12,
                family="DIN Alternate"
            ),
            align="auto"
        ),
        autosize=True,
        width=None,
        height=None
    )

    # --------------------------------------------------
    # 9. Export
    # --------------------------------------------------

    if save_path is not None:

        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.write_html(
            save_path,
            config={
                "responsive": True,
                "displayModeBar": False
            }
        )

        print(f"\nSaved interactive map: {save_path}")

    return fig, passes
