"""台指期每日 09:00 → 13:30 報酬（給對沖用）。

逐筆資料取當日日盤成交量最大的單一月份合約，09:00:00 後第一筆 → 13:30:00 前最後一筆。
輸出 results/tx_0900_1330.csv。資料自 2021-01 起。
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import polars as pl

RAW = Path(os.environ.get("QUANT_RAW", "data/raw"))
OUT = Path(__file__).resolve().parents[1] / "results" / "tx_0900_1330.csv"

rows = []
for f in sorted(glob.glob(str(RAW / "finmind_futopt/tick/TX/TX_*.parquet"))):
    d = pl.read_parquet(f).filter(~pl.col("contract_date").str.contains("/")).with_columns(
        t=pl.col("date").str.slice(11, 8), day=pl.col("date").str.slice(0, 10))
    d = d.filter((pl.col("t") >= "09:00:00") & (pl.col("t") <= "13:30:00"))
    if d.is_empty():
        continue
    c = d.group_by("contract_date").agg(pl.col("volume").sum()).sort("volume", descending=True)["contract_date"][0]
    d = d.filter(pl.col("contract_date") == c)
    rows.append({"date": d["day"][0], "contract": c, "p0900": d["price"][0], "p1330": d["price"][-1]})
out = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date(), tx_oc=pl.col("p1330") / pl.col("p0900") - 1)
out.write_csv(OUT)
print(out.shape, out["date"].min(), out["date"].max(), out["tx_oc"].mean() * 1e4, out["tx_oc"].std() * 1e4)
