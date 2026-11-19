"""策略 1：融券強制回補結束後空。

制度：股東會、除權息前，融券必須在停券起日（R）前回補完。實測餘額從 R−6 起逐日下降、R 當天歸零，
      R+1~R+3 不能融券賣出，R+4 恢復。回補買盤是被迫的，買完就沒了。
規則（T = R+1 = 進場日）
  選股      R−6 收盤：融券餘額 ≥ 100 張，且 融券餘額 ÷ 20 日均量 ≥ 0.2 天（R−6 就能列名單去借券）
  可交易    R−6 的 20 日均成交額 ≥ 1,000 萬、T 不在處置期間、T 開盤不是漲停
  進出場    T 開盤賣、收盤買回；收盤鎖漲停 → 補在之後第一個沒鎖的收盤
  券源      T 當天融券與當沖先賣後買都被暫停，只能借券；借不借得到是這個策略最大的未知
  成本      手續費 5.7 bp + 證交稅 30 bp + 借券費（假設年化 6%、借 7 天 ≈ 11.5 bp）= 47.2 bp
  部位      每檔 30 萬
"""
from __future__ import annotations

import sys

import polars as pl

from common import COST_OVERNIGHT, RES, UNIT, by_year, load_panel, sharpe

BORROW = 0.06 * 7 / 365
COST = COST_OVERNIGHT + BORROW


def main() -> None:
    out = open(RES / "02_short_cover.txt", "w")

    def p(*a):
        for f in (sys.stdout, out):
            print(*a, file=f)

    px = load_panel().sort(["stock_id", "date"])
    o = "stock_id"
    sh = lambda c, k: pl.col(c).shift(k).over(o)  # noqa: E731
    chain = pl.lit(None, dtype=pl.Float64)
    for k in range(10, 0, -1):
        chain = pl.when(~pl.col(f"lock{k}")).then(pl.col(f"ac{k}") / pl.col("adj_open") - 1).otherwise(chain)
    px = px.with_columns(
        is_T=sh("cover_day", 1),
        bal=sh("short_bal", 7), dtc=sh("short_bal", 7) / sh("vol20", 7), tv=sh("tv20", 7),
        run=sh("adj_close", 1) / sh("adj_close", 7) - 1,
        weak=sh("adj_close", 7) < sh("ma60", 7),
        r=pl.when(pl.col("locked_up")).then(-chain).otherwise(-pl.col("oc")) - COST,
        r3=-(pl.col("ac2") / pl.col("adj_open") - 1) - COST,   # 抱到 R+3 收盤（融券恢復前一天）
        carried=pl.col("locked_up"),
    ).filter(pl.col("date").dt.year() >= 2020)
    ev = px.filter(
        pl.col("is_T") & pl.col("bal").is_not_null() & (pl.col("tv") >= 1e7) & ~pl.col("in_disposition")
        & (pl.col("limit_up") < 9000) & (pl.col("open") < pl.col("limit_up")) & pl.col("r").is_not_null()
    ).with_columns(
        dtc_bin=pl.when(pl.col("bal") < 100).then(pl.lit("0_餘額<100張")).when(pl.col("dtc") < 0.2).then(pl.lit("1_<0.2天"))
        .when(pl.col("dtc") < 0.5).then(pl.lit("2_0.2~0.5天")).when(pl.col("dtc") < 1).then(pl.lit("3_0.5~1天"))
        .otherwise(pl.lit("4_≥1天")),
        run_bin=pl.when(pl.col("run") < 0).then(pl.lit("1_回補期下跌")).when(pl.col("run") < 0.10)
        .then(pl.lit("2_漲0~10%")).otherwise(pl.lit("3_漲≥10%")),
    )
    days = px.select("date").unique().sort("date")

    def tab(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
        return df.group_by(by).agg(
            筆數=pl.len(), 天數=pl.col("date").n_unique(), 盤中毛bp=pl.col("oc").mean() * 1e4,
            淨bp=pl.col("r").mean() * 1e4, 勝率=(pl.col("r") > 0).mean() * 100, 留倉=pl.col("carried").mean() * 100,
            抱到R3淨bp=pl.col("r3").mean() * 1e4).sort(by)

    p(f"全部回補事件（進場日可交易）{len(ev)} 筆；成本 {COST * 1e4:.1f} bp")
    p("\n=== 融券餘額 ÷ 20 日均量（R−6）===");  p(tab(ev, ["dtc_bin"]))
    sel = ev.filter((pl.col("bal") >= 100) & (pl.col("dtc") >= 0.2))
    sel.select("date", "stock_id", "dtc", "run", "oc", "r", "carried").write_csv(RES / "02_short_cover_trades.csv")
    p(f"\n主版本：餘額≥100 張且 ≥0.2 天 → {len(sel)} 筆、{sel['date'].n_unique()} 個交易日，"
      f"每筆淨 {sel['r'].mean() * 1e4:.1f} bp，勝率 {(sel['r'] > 0).mean() * 100:.1f}%")
    p("\n=== 主版本 × 回補期（R−6→R）漲幅 ===");  p(tab(sel, ["run_bin"]))
    p("\n=== 主版本 × R−6 是否在 60 日線下 ===")
    p(tab(sel.with_columns(w=pl.when(pl.col("weak")).then(pl.lit("60日線下")).otherwise(pl.lit("60日線上"))), ["w"]))
    p("\n=== 主版本 × 月份（事件集中在股東會與除息旺季）===")
    p(tab(sel.with_columns(月=pl.col("date").dt.month()), ["月"]))
    d = days.join(sel.group_by("date").agg(n=pl.len(), pnl=(pl.col("r") * UNIT).sum()), on="date", how="left") \
        .with_columns(pl.col("n").fill_null(0), pl.col("pnl").fill_null(0.0))
    d.write_csv(RES / "02_short_cover_daily.csv")
    p(f"\n=== 主版本逐年（每檔 30 萬；有訊號 {int((d['n'] > 0).sum())}/{len(d)} 天，單日最多 {d['n'].max()} 檔）===")
    p(by_year(d, "pnl"))
    p("\n=== 成本敏感度：借券費從 0 到年化 20%（借 7 天）===")
    rows = []
    for rate in (0.0, 0.03, 0.06, 0.10, 0.20):
        adj = sel["r"] + BORROW - rate * 7 / 365
        dd = days.join(sel.with_columns(a=adj).group_by("date").agg(pnl=(pl.col("a") * UNIT).sum()), on="date", how="left") \
            .with_columns(pl.col("pnl").fill_null(0.0))
        rows.append({"借券年化": f"{rate:.0%}", "每筆淨bp": adj.mean() * 1e4, "sharpe": sharpe(dd["pnl"]), "累積_萬": dd["pnl"].sum() / 1e4})
    p(pl.DataFrame(rows))
    out.close()


if __name__ == "__main__":
    main()
