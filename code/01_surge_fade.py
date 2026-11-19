"""策略 3：弱勢股急漲隔日空（當沖）。

規則（T = 進場日，全部只用 T 開盤當下已知的資訊）
  弱勢      T−2 收盤在 60 日均線下（觸發日之前就弱）；另列「不要求弱勢」對照
  觸發      T−1 個股報酬 − 加權指數報酬 ≥ 2%，且 T−1 收盤不是漲跌停
  可交易    T−1 成交額 ≥ 5,000 萬、T−1 與 T 都不在處置期間、T 可現股當沖先賣後買
  開盤      T 開盤漲幅在 −1% ~ +5% 之間（開低的不追空；開高 ≥5% 容易被鎖漲停）
  進出場    開盤賣、收盤買回，不設停損
  留倉      收盤鎖漲停買不回 → 補在之後第一個沒鎖漲停的收盤（改算非當沖成本）
  部位      當天所有訊號等權；日報酬 = 當天各筆淨報酬的平均
  對沖      （對照用，非主版本）同金額做多台指期 09:00 → 13:30（逐筆成交價，2021 起；見 00b_tx_intraday.py）
            加權指數的「開盤價」有一部分是昨收，不能拿來當對沖腿（跟台指期盤中報酬相關只有 0.80）
"""
from __future__ import annotations

import sys

import polars as pl

from common import COST_DAYTRADE, COST_HEDGE, COST_OVERNIGHT, RES, by_year, load_panel


def prepare(px: pl.DataFrame) -> pl.DataFrame:
    o = "stock_id"
    sh = lambda c, k=1: pl.col(c).shift(k).over(o)  # noqa: E731
    px = px.sort([o, "date"]).with_columns(
        x1=sh("ret") - sh("ix_ret"),
        lim1=sh("locked_up") | sh("locked_dn"),
        tv1=sh("Trading_money"),
        disp1=sh("in_disposition"),
        weak=sh("adj_close", 2) < sh("ma60", 2),
        hist=sh("ma60", 2).is_not_null(),
    )
    # 留倉：補在第一個沒鎖漲停的收盤
    chain = pl.lit(None, dtype=pl.Float64)
    for k in range(10, 0, -1):
        chain = pl.when(~pl.col(f"lock{k}")).then(pl.col(f"ac{k}") / pl.col("adj_open") - 1).otherwise(chain)
    px = px.with_columns(
        r_short=pl.when(pl.col("locked_up")).then(-chain - COST_OVERNIGHT).otherwise(-pl.col("oc") - COST_DAYTRADE),
        carried=pl.col("locked_up"),
    )
    return px.filter(pl.col("date").dt.year() >= 2020)


def tradable() -> pl.Expr:
    return (
        pl.col("hist") & pl.col("x1").is_not_null() & (pl.col("tv1") >= 5e7) & ~pl.col("lim1")
        & ~pl.col("disp1") & ~pl.col("in_disposition") & pl.col("daytrade_short_ok")
        & (pl.col("limit_up") < 9000) & (pl.col("open") < pl.col("limit_up")) & pl.col("r_short").is_not_null()
    )


def signal(x_min: float = 0.02, need_weak: bool = True, gap_lo: float = -0.01, gap_hi: float = 0.05) -> pl.Expr:
    s = tradable() & (pl.col("x1") >= x_min) & (pl.col("gap_o") > gap_lo) & (pl.col("gap_o") < gap_hi)
    return s & pl.col("weak") if need_weak else s


def daily(px: pl.DataFrame, sig: pl.Expr) -> pl.DataFrame:
    tx = pl.read_csv(RES / "tx_0900_1330.csv", try_parse_dates=True).select("date", "tx_oc")
    days = px.select("date").unique().sort("date").join(tx, on="date", how="left")
    d = px.filter(sig).group_by("date").agg(n=pl.len(), raw=pl.col("r_short").mean())
    return days.join(d, on="date", how="left").with_columns(pl.col("n").fill_null(0), pl.col("raw").fill_null(0.0)).with_columns(
        # 沒有台指期逐筆的日子（2020）對沖欄為空
        hedged=pl.when(pl.col("tx_oc").is_null()).then(None).otherwise(
            pl.when(pl.col("n") > 0).then(pl.col("raw") + pl.col("tx_oc") - COST_HEDGE).otherwise(0.0)))


def main() -> None:
    px = prepare(load_panel())
    out = open(RES / "01_surge_fade.txt", "w")

    def p(*a):
        for f in (sys.stdout, out):
            print(*a, file=f)

    base = signal()
    tr = px.filter(base)
    d = daily(px, base)
    d.write_csv(RES / "01_surge_fade_daily.csv")

    p(f"筆數 {len(tr)}，日均 {d['n'].mean():.1f} 檔（中位 {d['n'].median():.0f}、p95 {d['n'].quantile(0.95):.0f}、最多 {d['n'].max()}）")
    p(f"每筆淨 {tr['r_short'].mean() * 1e4:.1f} bp，勝率 {(tr['r_short'] > 0).mean() * 100:.1f}%，"
      f"收盤鎖漲停留倉 {tr['carried'].mean() * 100:.2f}%（留倉那些每筆 {tr.filter(pl.col('carried'))['r_short'].mean() * 100:.2f}%）")
    p(f"策略日損益 與 台指期盤中報酬 相關係數 {d.filter(pl.col('n') > 0).select(pl.corr('raw', 'tx_oc')).item():.2f}")
    p("\n=== 未對沖：逐年（當日等權）===");  p(by_year(d, "raw"))
    dh = d.filter(pl.col("hedged").is_not_null())
    p("\n=== 台指期對沖後：逐年（2021 起）===");  p(by_year(dh, "hedged"))
    p("\n=== 同期間未對沖（2021 起，對照用）===");  p(by_year(dh, "raw").tail(1))
    dw = daily(px, signal(need_weak=False)); dwh = dw.filter(pl.col("hedged").is_not_null())
    tw = px.filter(signal(need_weak=False))
    p(f"\n=== 對照：不要求弱勢（{len(tw)} 筆、日均 {dw['n'].mean():.1f} 檔、每筆淨 {tw['r_short'].mean() * 1e4:.1f} bp、"
      f"留倉 {tw['carried'].mean() * 100:.2f}%）：未對沖逐年 ===")
    p(by_year(dw, "raw"))
    p("對沖後（2021 起）：");  p(by_year(dwh, "hedged").tail(1))
    p("\n=== 訊號檔數與當日報酬：檔數多的日子比較好還是比較差 ===")
    for lab, dd in [("弱勢", d), ("不要求弱勢", dw)]:
        q = dd.filter(pl.col("n") > 0).with_columns(b=pl.when(pl.col("n") < 10).then(pl.lit("1_<10檔")).when(pl.col("n") < 30)
            .then(pl.lit("2_10~29")).when(pl.col("n") < 100).then(pl.lit("3_30~99")).otherwise(pl.lit("4_≥100")))
        p(lab, q.group_by("b").agg(天數=pl.len(), 日均bp=pl.col("raw").mean() * 1e4, 日標準差bp=pl.col("raw").std() * 1e4,
                                   最差一日=pl.col("raw").min() * 100).sort("b"))

    p("\n=== 盤中普遍下跌：同樣可交易條件下的全部股票（不看訊號）===")
    u = px.filter(tradable() & (pl.col("gap_o") > -0.01) & (pl.col("gap_o") < 0.05))
    p(u.select(筆數=pl.len(), 盤中毛bp=pl.col("oc").mean() * 1e4, 空單每筆淨bp=pl.col("r_short").mean() * 1e4))
    u0 = px.filter(tradable())
    p("不限開盤位置：", u0.select(筆數=pl.len(), 盤中毛bp=pl.col("oc").mean() * 1e4, 隔夜毛bp=pl.col("gap_o").mean() * 1e4))

    p("\n=== 拆解：弱勢 × 昨日超額（空單每筆淨 bp / 留倉% / 筆數）===")
    p(u.with_columns(
        w=pl.when(pl.col("weak")).then(pl.lit("60日線下")).otherwise(pl.lit("60日線上")),
        t=pl.when(pl.col("x1") >= 0.02).then(pl.lit("3_昨日超額≥2%")).when(pl.col("x1") <= -0.02)
        .then(pl.lit("1_≤−2%")).otherwise(pl.lit("2_−2~2%")),
    ).group_by("w", "t").agg(筆數=pl.len(), 空單淨bp=pl.col("r_short").mean() * 1e4,
                             勝率=(pl.col("r_short") > 0).mean() * 100, 留倉=pl.col("carried").mean() * 100).sort("w", "t"))

    p("\n=== 敏感度：一次改一個條件（全期）===")
    rows = []
    for lab, s in [
        ("基準", signal()),
        ("昨日超額≥1%", signal(x_min=0.01)), ("昨日超額≥3%", signal(x_min=0.03)), ("昨日超額≥5%", signal(x_min=0.05)),
        ("不要求弱勢", signal(need_weak=False)),
        ("不看開盤位置（<漲停即可）", signal(gap_lo=-1, gap_hi=1)),
        ("開盤 −1%~+3%", signal(gap_hi=0.03)), ("開盤 −1%~+7%", signal(gap_hi=0.07)),
        ("開盤 0%~+5%", signal(gap_lo=0.0)),
    ]:
        dd = daily(px, s); t = px.filter(s)
        from common import sharpe
        rows.append({"版本": lab, "筆數": len(t), "日均檔": dd["n"].mean(), "每筆淨bp": t["r_short"].mean() * 1e4,
                     "留倉%": t["carried"].mean() * 100, "sharpe": sharpe(dd["raw"]), "對沖後sharpe": sharpe(dd["hedged"].drop_nulls()),
                     "累積%": dd["raw"].sum() * 100,
                     "2025對沖後sharpe": sharpe(dd.filter(pl.col("date").dt.year() == 2025)["hedged"])})
    p(pl.DataFrame(rows))
    out.close()


if __name__ == "__main__":
    main()
