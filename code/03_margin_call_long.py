"""策略 2：大盤急殺時，融資斷頭後做多。

制度：融資維持率跌破 130% 收到追繳，兩個營業日內沒補，第三天開盤被券商賣出。賣盤是被迫的，賣完就沒了。
規則（D = 事件日，收盤後資料公布才知道；D+1 開盤進場）
  急殺      全市場等權報酬 3 日累計 ≤ −5%（D 日為止）
  斷頭      D 收盤逐檔融資維持率 < 130%、融資餘額單日減少 ≥ 5%、D 當日下跌
  可交易    D−1 融資餘額 ≥ 1,000 張、20 日均成交額 ≥ 2,000 萬、不在處置期間
  第一天    同一次急殺（相隔 ≤10 個交易日算同一次）的第一個急殺日不進場：追繳到真正被賣要兩三天
  進出場    D+1 開盤買、抱 5 個交易日、收盤賣；同一次急殺每檔只進一次
  成本      手續費 5.7 bp + 證交稅 30 bp = 35.7 bp；同一次急殺內各檔等權
樣本只有十次左右的急殺，這裡不報 Sharpe，逐次列出。
"""
from __future__ import annotations

import sys

import polars as pl

from common import COST_OVERNIGHT, RES, load_panel


def main() -> None:
    out = open(RES / "03_margin_call_long.txt", "w")

    def p(*a):
        for f in (sys.stdout, out):
            print(*a, file=f)

    px = load_panel().sort(["stock_id", "date"])
    o = "stock_id"
    cal = px.select("date", "mkt_ew").unique("date").sort("date").with_columns(
        di=pl.int_range(pl.len()), mkt3=pl.col("mkt_ew").rolling_sum(3))
    crash = cal.filter(pl.col("mkt3") <= -0.05).with_columns(
        ep=((pl.col("di") - pl.col("di").shift(1)).fill_null(99) > 10).cum_sum()
    ).with_columns(day_no=pl.int_range(pl.len()).over("ep") + 1)
    px = px.with_columns(
        bal_chg=pl.col("margin_bal") / pl.col("margin_bal").shift(1).over(o) - 1,
        bal_prev=pl.col("margin_bal").shift(1).over(o),
        strong=pl.col("ret240").shift(1).over(o) >= 0.30,
        tv=pl.col("tv20").shift(1).over(o),
        r5=pl.col("ac5") / pl.col("ao1") - 1 - COST_OVERNIGHT,
        r10=pl.col("ac10") / pl.col("ao1") - 1 - COST_OVERNIGHT,
        dd5=pl.min_horizontal([pl.col(f"ac{k}") / pl.col("ao1") - 1 for k in range(1, 6)]),
    ).filter(pl.col("date").dt.year() >= 2020)
    ev = px.join(crash.select("date", "ep", "day_no", "mkt3"), on="date").filter(
        (pl.col("margin_maintenance") < 130) & (pl.col("bal_chg") <= -0.05) & (pl.col("ret") < 0)
        & (pl.col("bal_prev") >= 1000) & (pl.col("tv") >= 2e7) & ~pl.col("in_disposition") & pl.col("r5").is_not_null()
    )
    names = crash.group_by("ep").agg(急殺起=pl.col("date").min(), 急殺迄=pl.col("date").max(), 急殺日數=pl.len(),
                                     最深3日=pl.col("mkt3").min() * 100)

    def per_ep(df: pl.DataFrame) -> pl.DataFrame:
        t = df.sort("date").unique(["ep", "stock_id"], keep="first")
        g = t.group_by("ep").agg(
            檔數=pl.len(), 抱5日淨=pl.col("r5").mean() * 100, 勝率=(pl.col("r5") > 0).mean() * 100,
            抱10日淨=pl.col("r10").mean() * 100, 五日內最深=pl.col("dd5").mean() * 100,
            最深p10=pl.col("dd5").quantile(0.1) * 100)
        return names.join(g, on="ep", how="left").sort("急殺起").drop("ep")

    def total(df: pl.DataFrame, lab: str) -> dict:
        t = df.sort("date").unique(["ep", "stock_id"], keep="first")
        e = t.group_by("ep").agg(m=pl.col("r5").mean())
        return {"版本": lab, "檔數": len(t), "有進場次數": len(e), "正的次數": int((e["m"] > 0).sum()),
                "每檔淨%": t["r5"].mean() * 100, "每次平均淨%": e["m"].mean() * 100, "最差一次%": e["m"].min() * 100,
                "勝率": (t["r5"] > 0).mean() * 100}

    main_ev = ev.filter(pl.col("day_no") >= 2)
    main_ev.sort("date").unique(["ep", "stock_id"], keep="first").select(
        "date", "stock_id", "margin_maintenance", "bal_chg", "strong", "r5", "r10", "dd5"
    ).sort("date").write_csv(RES / "03_margin_call_trades.csv")
    p(f"急殺日 {len(crash)} 天、{crash['ep'].n_unique()} 次")
    p("\n=== 主版本（第一個急殺日不進場）：逐次 ===");  p(per_ep(main_ev))
    p("\n=== 對照：第一個急殺日也進場：逐次 ===");      p(per_ep(ev))
    p("\n=== 彙總 ===")
    p(pl.DataFrame([
        total(main_ev, "主版本"),
        total(main_ev.filter(pl.col("strong")), "主版本・只做長期強勢（240日漲≥30%）"),
        total(main_ev.filter(~pl.col("strong").fill_null(False)), "主版本・其餘"),
        total(ev, "第一天也進場"),
        total(ev.filter(pl.col("day_no") == 1), "只在第一天進場"),
    ]))
    # 對照：平常日的同樣事件
    norm = px.join(crash.select("date"), on="date", how="anti").filter(
        (pl.col("margin_maintenance") < 130) & (pl.col("bal_chg") <= -0.05) & (pl.col("ret") < 0)
        & (pl.col("bal_prev") >= 1000) & (pl.col("tv") >= 2e7) & ~pl.col("in_disposition") & pl.col("r5").is_not_null())
    p("\n=== 對照：不是急殺日的同樣事件 ===")
    p(norm.select(筆數=pl.len(), 天數=pl.col("date").n_unique(), 抱5日淨=pl.col("r5").mean() * 100,
                  勝率=(pl.col("r5") > 0).mean() * 100, 抱10日淨=pl.col("r10").mean() * 100))
    # 對照：急殺日（第 2 天起）但沒有斷頭跡象的股票
    ctl = px.join(crash.filter(pl.col("day_no") >= 2).select("date"), on="date").filter(
        (pl.col("margin_maintenance") >= 140) & (pl.col("ret") < 0) & (pl.col("bal_prev") >= 1000)
        & (pl.col("tv") >= 2e7) & pl.col("r5").is_not_null())
    p("\n=== 對照：同樣急殺日（第 2 天起）、當日下跌、但維持率 ≥140% ===")
    p(ctl.select(筆數=pl.len(), 抱5日淨=pl.col("r5").mean() * 100, 勝率=(pl.col("r5") > 0).mean() * 100,
                 抱10日淨=pl.col("r10").mean() * 100))
    out.close()


if __name__ == "__main__":
    main()
