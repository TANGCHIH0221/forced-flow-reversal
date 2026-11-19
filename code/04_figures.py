"""README 用的總覽圖：三個策略各一格（各自獨立的 y 軸，不共用）。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from common import FIG, RES

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
BLUE, INK, MUTED, SURF, GRID, NEG = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0", "#86b6ef"


def style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURF)

d = pl.read_csv(RES / "02_short_cover_daily.csv", try_parse_dates=True)
ax = axes[0]
ax.plot(d["date"], d["pnl"].cum_sum() / 1e4, color=BLUE, linewidth=2)
style(ax, "① 融券回補結束後空\n每檔 30 萬，累積損益", "萬元")
ax.annotate(f"{d['pnl'].sum() / 1e4:,.0f} 萬", (d["date"][-1], d["pnl"].sum() / 1e4), xytext=(-6, 6),
            textcoords="offset points", ha="right", fontsize=10, color=INK)

ax = axes[1]
t = pl.read_csv(RES / "03_margin_call_trades.csv", try_parse_dates=True)
cal = t.sort("date").with_columns(g=(pl.col("date").diff().dt.total_days().fill_null(99) > 20).cum_sum())
e = cal.group_by("g").agg(起=pl.col("date").min(), r=pl.col("r5").mean() * 100, n=pl.len()).sort("起")
labels = [f"{x:%Y-%m}\n{n} 檔" for x, n in zip(e["起"], e["n"])]
bars = ax.bar(range(len(e)), e["r"], width=0.62, color=[BLUE if v >= 0 else NEG for v in e["r"]])
ax.set_xticks(range(len(e)), labels)
for i, v in enumerate(e["r"]):
    ax.text(i, v + (0.4 if v >= 0 else -0.4), f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
style(ax, "② 急殺時融資斷頭後做多\n每次急殺：抱 5 日每檔平均淨報酬", "%")
ax.set_ylim(min(e["r"].min() - 3, -4), e["r"].max() + 3)

d = pl.read_csv(RES / "01_surge_fade_daily.csv", try_parse_dates=True)
ax = axes[2]
ax.plot(d["date"], d["raw"].cum_sum() / 1e4, color=BLUE, linewidth=2)
style(ax, "③ 急漲隔日空（當沖）\n每檔 30 萬，累積損益，未對沖", "萬元")
ax.annotate(f"{d['raw'].sum() / 1e4:,.0f} 萬", (d["date"][-1], d["raw"].sum() / 1e4), xytext=(-6, 6),
            textcoords="offset points", ha="right", fontsize=10, color=INK)

fig.tight_layout(w_pad=3)
fig.savefig(FIG / "0_overview.png", dpi=160, facecolor=SURF)
print("saved")
