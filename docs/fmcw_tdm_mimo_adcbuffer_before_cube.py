"""
FMCW Radar MIMO ADC Buffer Visualisation
=========================================
Illustrates how ADC samples in a FIFO-like buffer are tagged by TX
antenna (fill colour) and RX antenna (hatch pattern) for TDM MIMO
radar configurations.

Waveform modes implemented
---------------------------
TDM (Time-Division Multiplexing)
    One TX is active per chirp.  Row ordering::

        chirp 0  → TX0·RX0
        chirp 1  → TX0·RX1
        ...
        chirp N  → TX1·RX0
        ...

Usage
-----
Run directly; no CLI arguments::

    python fmcw_tdm_mimo_adc.py
"""

import matplotlib  # noqa: E402  (rcParams must be set before figure)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

# ---------------------------------------------------------------------------
# Module-level defaults — edit here to change the look globally
# ---------------------------------------------------------------------------

#: One hex colour string per TX index.
DEFAULT_TX_COLORS: dict[int, str] = {
    0: "#4C9BE8",   # TX0 — blue
    1: "#E8764C",   # TX1 — orange
    2: "#4CE87A",   # TX2 — green
    3: "#C84CE8",   # TX3 — purple
}

#: One matplotlib hatch string per RX index.
#: Valid tokens: ``''`` ``'/'`` ``'\\'`` ``'|'`` ``'-'`` ``'+'``
#: ``'x'`` ``'o'`` ``'O'`` ``'.'`` ``'*'``  (repeat for density).
DEFAULT_RX_HATCHES: dict[int, str] = {
    0: "",        # RX0 — solid fill
    1: "////",    # RX1 — forward diagonal
    2: "xxxx",    # RX2 — cross-hatch
    3: "....",    # RX3 — dots
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_cell(
    ax: plt.Axes,
    x: int,
    y: int,
    color: str,
    hatch: str,
    cell_alpha: float,
    cell_linewidth: float,
    hatch_color: str,
) -> None:
    """Draw one FIFO cell (fill patch + hatch overlay) onto *ax*.

    Two ``FancyBboxPatch`` objects are stacked so that the hatch line
    colour is independent of the fill colour (matplotlib ties hatch
    colour to ``edgecolor`` when ``facecolor`` is also set, requiring
    the two-patch workaround).

    Parameters
    ----------
    ax :
        Axes to draw on.
    x :
        Left edge of the cell in data coordinates.
    y :
        Bottom edge of the cell in data coordinates.
    color :
        CSS hex colour for the cell fill.
    hatch :
        Matplotlib hatch string (``''`` for solid, ``'////'`` etc.).
    cell_alpha :
        Opacity of the fill patch (0–1).
    cell_linewidth :
        Width of the white cell border in points.
    hatch_color :
        Colour of the hatch lines.

    Raises
    ------
    ValueError
        Propagated from ``matplotlib`` if *color* or *hatch_color* is
        not a valid colour specification.
    """
    rgba = to_rgba(color, alpha=cell_alpha)
    box_style = "square,pad=0"

    fill_patch = mpatches.FancyBboxPatch(
        (x, y), 1, 1,
        boxstyle=box_style,
        linewidth=cell_linewidth,
        edgecolor="white",
        facecolor=rgba,
        hatch=hatch,
    )
    ax.add_patch(fill_patch)

    hatch_patch = mpatches.FancyBboxPatch(
        (x, y), 1, 1,
        boxstyle=box_style,
        linewidth=0,
        edgecolor=hatch_color,
        facecolor="none",
        hatch=hatch,
    )
    ax.add_patch(hatch_patch)


def _add_legends(
    ax: plt.Axes,
    num_tx: int,
    num_rx: int,
    tx_colors: dict[int, str],
    rx_hatches: dict[int, str],
    cell_alpha: float,
    hatch_color: str,
) -> None:
    """Attach TX-colour and RX-pattern legends outside the plot area.

    Parameters
    ----------
    ax :
        Axes that owns the legends.
    num_tx :
        Number of TX antennas.
    num_rx :
        Number of RX antennas.
    tx_colors :
        Mapping ``{tx_index: hex_color}``.
    rx_hatches :
        Mapping ``{rx_index: hatch_string}``.
    cell_alpha :
        Alpha used for fill patches in the legend.
    hatch_color :
        Colour of hatch lines in the legend.
    """
    tx_handles = [
        mpatches.Patch(
            facecolor=tx_colors[i],
            edgecolor="grey",
            linewidth=0.5,
            label=f"TX{i}",
            alpha=cell_alpha,
        )
        for i in range(num_tx)
    ]
    rx_handles = [
        mpatches.Patch(
            facecolor="white",
            edgecolor=hatch_color,
            hatch=rx_hatches[i],
            label=f"RX{i}",
            linewidth=0.5,
        )
        for i in range(num_rx)
    ]

    leg_tx = ax.legend(
        handles=tx_handles,
        title="TX (color)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg_tx)
    ax.legend(
        handles=rx_handles,
        title="RX (pattern)",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.60),
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
    )


def _draw_grid(
    ax: plt.Axes,
    num_adc_samples: int,
    num_chirps: int,
    tx_block_size: int,
) -> None:
    """Draw cell borders and TX-block separator lines.

    Parameters
    ----------
    ax :
        Target axes.
    num_adc_samples :
        Number of ADC samples (x-axis extent).
    num_chirps :
        Total number of chirp rows (y-axis extent).
    tx_block_size :
        Number of consecutive rows belonging to the same TX; a thicker
        line is drawn at every multiple of this value.
    """
    for x in range(num_adc_samples + 1):
        ax.axvline(x, color="white", linewidth=0.3, zorder=3)
    for y in range(num_chirps + 1):
        lw = 1.2 if y % tx_block_size == 0 else 0.3
        ax.axhline(y, color="white", linewidth=lw, zorder=3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tdm_mimo_adc_buffer(
    num_tx: int = 4,
    num_rx: int = 4,
    num_adc_samples: int = 16,
    tx_colors: dict[int, str] | None = None,
    rx_hatches: dict[int, str] | None = None,
    cell_alpha: float = 0.85,
    hatch_color: str = "black",
    hatch_linewidth: float = 0.6,
    cell_linewidth: float = 0.4,
    dpi: int = 180,
    output_file: str = "fmcw_tdm_mimo_adc_buffer.png",
) -> plt.Figure:
    """Generate a FIFO-style ADC buffer diagram for TDM MIMO radar.

    In Time-Division Multiplexing MIMO, exactly one TX antenna is active
    per chirp.  All RX antennas sample simultaneously.  The virtual
    channels are filled in the order::

        chirp k  →  TX = k // num_rx,  RX = k % num_rx

    Each cell in the diagram represents one 16-bit ADC word.  The fill
    *colour* encodes the active TX antenna; the *hatch pattern* encodes
    the RX antenna that captured the sample.

    Parameters
    ----------
    num_tx :
        Number of TX antennas.
    num_rx :
        Number of RX antennas.
    num_adc_samples :
        Number of ADC samples per chirp (x-axis width).
    tx_colors :
        Mapping ``{tx_index: css_color}`` overriding
        ``DEFAULT_TX_COLORS``.  Must contain an entry for every TX index
        in ``range(num_tx)``.
    rx_hatches :
        Mapping ``{rx_index: hatch_string}`` overriding
        ``DEFAULT_RX_HATCHES``.  Must contain an entry for every RX
        index in ``range(num_rx)``.
    cell_alpha :
        Opacity of the fill colour (0 = transparent, 1 = opaque).
    hatch_color :
        CSS colour for hatch lines.
    hatch_linewidth :
        Stroke width of hatch lines in points.
    cell_linewidth :
        Stroke width of cell borders in points.
    dpi :
        Resolution of the saved PNG.
    output_file :
        Destination path for the PNG (relative or absolute).

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (caller may call ``plt.show()`` or further
        customise before displaying).

    Raises
    ------
    KeyError
        If *tx_colors* is missing an entry for any TX index in
        ``range(num_tx)``, or *rx_hatches* is missing an entry for any
        RX index in ``range(num_rx)``.
    ValueError
        If *num_tx* or *num_rx* is less than 1, or *num_adc_samples*
        is less than 1.

    Side Effects
    ------------
    Writes a PNG file to *output_file*
    (default: ``fmcw_tdm_mimo_adc_buffer.png``).
    Mutates ``matplotlib.rcParams["hatch.linewidth"]``.

    Examples
    --------
    >>> fig = tdm_mimo_adc_buffer(num_tx=2, num_rx=2, num_adc_samples=4,
    ...                           dpi=72,
    ...                           output_file="/tmp/test_tdm.png")
    ... # doctest: +ELLIPSIS
    Saved ...
    >>> import os; os.path.exists("/tmp/test_tdm.png")
    True
    """
    if num_tx < 1 or num_rx < 1 or num_adc_samples < 1:
        raise ValueError(
            "num_tx, num_rx, and num_adc_samples must each be >= 1."
        )

    colors = tx_colors if tx_colors is not None else DEFAULT_TX_COLORS
    hatches = rx_hatches if rx_hatches is not None else DEFAULT_RX_HATCHES

    num_chirps = num_tx * num_rx
    matplotlib.rcParams["hatch.linewidth"] = hatch_linewidth

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_aspect("equal")
    ax.set_xlim(0, num_adc_samples)
    ax.set_ylim(0, num_chirps)
    ax.invert_yaxis()   # row 0 at top → FIFO "oldest" at top

    for chirp_idx in range(num_chirps):
        tx_idx = chirp_idx // num_rx
        rx_idx = chirp_idx % num_rx
        color = colors[tx_idx]
        hatch = hatches[rx_idx]

        for sample_idx in range(num_adc_samples):
            _draw_cell(
                ax, sample_idx, chirp_idx,
                color, hatch,
                cell_alpha, cell_linewidth, hatch_color,
            )

    # x-axis
    bit_w = (num_adc_samples - 1).bit_length()
    ax.set_xticks(np.arange(0.5, num_adc_samples, 1))
    ax.set_xticklabels(
        [f"0b{i:0{bit_w}b}" for i in range(num_adc_samples)],
        fontsize=7.5,
    )
    ax.set_xlabel(
        "ADC sample index (16-bit words per chirp)",
        fontsize=10,
    )

    # y-axis
    y_labels = [
        f"Chirp {k // num_rx +1:02d}  TX{k // num_rx}\u00b7RX{k % num_rx}"
        for k in range(num_chirps)
    ]
    ax.set_yticks(np.arange(0.5, num_chirps, 1))
    ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.set_ylabel("FIFO row (chirp index)", fontsize=10)

    ax.set_title(
        "FMCW TDM MIMO \u2014 ADC Buffer\n"
        "Color = active TX antenna   |   Pattern = RX channel",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    _add_legends(
        ax, num_tx, num_rx, colors, hatches, cell_alpha, hatch_color,
    )
    _draw_grid(ax, num_adc_samples, num_chirps, tx_block_size=num_rx)

    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    print(f"Saved \u2192 {output_file}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tdm_mimo_adc_buffer()
