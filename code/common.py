"""共用：讀 panel、績效統計。"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

# 成本（單邊手續費 0.1425% × 2 折）
FEE_RT = 0.001425 * 0.2 * 2          # 來回手續費 5.7 bp
COST_DAYTRADE = FEE_RT + 0.0015      # 現股當沖：+ 當沖證交稅 0.15% → 20.7 bp
COST_OVERNIGHT = FEE_RT + 0.003      # 非當沖：+ 證交稅 0.3% → 35.7 bp
COST_HEDGE = 0.0002                  # 指數期貨對沖來回（稅、手續費、1 點滑價），假設值

pl.Config.set_tbl_rows(80).set_tbl_cols(24).set_tbl_width_chars(240).set_float_precision(2)


def load_panel(start_year: int = 2020) -> pl.DataFrame:
    return pl.read_parquet(ROOT / "panel.parquet").filter(pl.col("date").dt.year() >= start_year - 1)


def sharpe(s: pl.Series) -> float:
    sd = s.std()
    return float("nan") if not sd else s.mean() / sd * 252 ** 0.5


def daily_stats(d: pl.DataFrame, col: str, label: str) -> dict:
    """d：每個交易日一列，col = 當日報酬（當天所有部位等權平均；沒部位的日子 = 0）。"""
    eq = d[col].cum_sum()
    return {
        "組": label, "交易日": len(d), "日均bp": d[col].mean() * 1e4, "sharpe": sharpe(d[col]),
        "累積%": d[col].sum() * 100, "最大回撤%": (eq - eq.cum_max()).min() * 100,
        "最差一日%": d[col].min() * 100, "賺錢日%": (d[col] > 0).mean() * 100,
    }


def by_year(d: pl.DataFrame, col: str) -> pl.DataFrame:
    rows = [daily_stats(d.filter(pl.col("date").dt.year() == y), col, str(y))
            for y in sorted(d["date"].dt.year().unique())]
    rows.append(daily_stats(d, col, "全期"))
    return pl.DataFrame(rows)
