"""建立日頻 panel（三個策略共用）。

輸入：FinMind 日資料（路徑由環境變數 QUANT_RAW 指定，預設 ./data/raw）。
輸出：panel.parquet，一列 = 一檔上市櫃普通股一個交易日。

重點欄位
  ret            還原日報酬 = close / reference_price − 1（reference_price 已處理除權息）
  adj_close/open 還原價指數（由 ret 連乘）
  locked_up      收盤鎖漲停
  ao{k}/ac{k}    往後第 k 個交易日的還原開盤/收盤；lock{k} 往後第 k 日是否收盤鎖漲停
  daytrade_short_ok  當日可現股當沖先賣後買
  short_suspended    當日在融券停券期間
  in_disposition     當日在處置期間
  short_bal / margin_bal / margin_maintenance  融券餘額、融資餘額（張）、逐檔融資維持率
  mkt_ew         當日全市場等權報酬；ix_ret / ix_oc 加權指數日報酬 / 開盤→收盤報酬
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import polars as pl

RAW = Path(os.environ.get("QUANT_RAW", "data/raw"))
OUT = Path(__file__).resolve().parents[1] / "panel.parquet"
HORIZON = 10


def daily_dir(name: str, cols: list[str] | None = None) -> pl.DataFrame:
    fs = sorted(glob.glob(str(RAW / f"finmind_sponsor/{name}/*.parquet")))
    df = pl.read_parquet(fs).with_columns(pl.col("date").str.to_date())
    return df.select(cols) if cols else df


def load_price() -> pl.DataFrame:
    cols = ["date", "stock_id", "Trading_Volume", "Trading_money", "open", "max", "min", "close"]
    old = pl.read_parquet(sorted(glob.glob(str(RAW / "finmind_daily/price/year=*/data.parquet")))).select(cols)
    new = daily_dir("TaiwanStockPrice", cols)
    return pl.concat([old, new]).unique(["date", "stock_id"], keep="last")


def load_index() -> pl.DataFrame:
    old = pl.read_parquet(RAW / "finmind_daily/taiex.parquet").with_columns(pl.col("date").str.to_date())
    new = daily_dir("TaiwanStockPrice").filter(pl.col("stock_id") == "TAIEX")
    return (
        pl.concat([old.select("date", "open", "close"), new.select("date", "open", "close")])
        .unique("date", keep="last").sort("date")
        .select("date", ix_ret=pl.col("close") / pl.col("close").shift(1) - 1,
                ix_oc=pl.col("close") / pl.col("open") - 1)
    )


def load_margin() -> pl.DataFrame:
    cols = ["date", "stock_id", "ShortSaleTodayBalance", "MarginPurchaseTodayBalance"]
    old = pl.read_parquet(sorted(glob.glob(str(RAW / "finmind_daily/margin/year=*/data.parquet")))).select(cols)
    new = daily_dir("TaiwanStockMarginPurchaseShortSale", cols)
    return pl.concat([old, new]).unique(["date", "stock_id"], keep="last").rename(
        {"ShortSaleTodayBalance": "short_bal", "MarginPurchaseTodayBalance": "margin_bal"})


def expand(df: pl.DataFrame, start: str, end: str, flag: str) -> pl.DataFrame:
    return (
        df.select("stock_id", pl.date_ranges(pl.col(start), pl.col(end)).alias("date"))
        .explode("date").unique().with_columns(pl.lit(True).alias(flag))
    )


def main() -> None:
    info = pl.read_parquet(RAW / "finmind_sponsor/TaiwanStockInfo/all.parquet")
    common = (
        info.filter(pl.col("type").is_in(["twse", "tpex"]) & pl.col("stock_id").str.contains(r"^[1-9]\d{3}$"))
        .group_by("stock_id").agg(pl.col("type").first())
    )
    px = load_price().join(common, on="stock_id").join(
        daily_dir("TaiwanStockPriceLimit"), on=["date", "stock_id"], how="left")
    # limit_up == 0 是興櫃時期（無漲跌幅、價格為均價），整段排除
    px = px.filter((pl.col("close") > 0) & (pl.col("open") > 0) & (pl.col("reference_price") > 0)
                   & (pl.col("limit_up") > 0)).sort(["stock_id", "date"])
    o = "stock_id"
    px = px.with_columns(
        ret=pl.col("close") / pl.col("reference_price") - 1,
        gap_o=pl.col("open") / pl.col("reference_price") - 1,
        oc=pl.col("close") / pl.col("open") - 1,
        # 新上市前五日無漲跌幅，limit_up 為 9999.95
        locked_up=(pl.col("close") >= pl.col("limit_up")) & (pl.col("limit_up") < 9000),
        locked_dn=(pl.col("close") <= pl.col("limit_down")) & (pl.col("limit_up") < 9000),
    ).with_columns(adj_close=(1 + pl.col("ret")).cum_prod().over(o)).with_columns(
        adj_open=pl.col("adj_close") * pl.col("open") / pl.col("close")
    ).with_columns(
        ma60=pl.col("adj_close").rolling_mean(60).over(o),
        tv20=pl.col("Trading_money").rolling_mean(20).over(o),
        vol20=pl.col("Trading_Volume").rolling_mean(20).over(o) / 1000,
        ret240=pl.col("adj_close") / pl.col("adj_close").shift(240).over(o) - 1,
    )
    fwd = []
    for k in range(1, HORIZON + 1):
        fwd += [pl.col("adj_open").shift(-k).over(o).alias(f"ao{k}"),
                pl.col("adj_close").shift(-k).over(o).alias(f"ac{k}"),
                pl.col("locked_up").shift(-k).over(o).alias(f"lock{k}")]
    px = px.with_columns(fwd)

    dt = daily_dir("TaiwanStockDayTrading").select(
        "date", "stock_id", (pl.col("BuyAfterSale").fill_null("") == "").alias("daytrade_short_ok"))
    px = px.join(dt, on=["date", "stock_id"], how="left").with_columns(pl.col("daytrade_short_ok").fill_null(False))

    sus = pl.read_parquet(RAW / "finmind_sponsor/TaiwanStockMarginShortSaleSuspension/all.parquet").with_columns(
        pl.col("date").str.to_date(), pl.col("end_date").str.to_date(strict=False))
    sus_ok = sus.filter(pl.col("end_date").is_not_null() & (pl.col("end_date") >= pl.col("date")))
    px = px.join(expand(sus_ok, "date", "end_date", "short_suspended"), on=["date", "stock_id"], how="left")
    # 停券起日（融券最後回補日）
    px = px.join(sus.filter(~pl.col("reason").str.contains("減資")).select("stock_id", "date").unique()
                 .with_columns(cover_day=pl.lit(True)), on=["date", "stock_id"], how="left")

    disp = pl.read_parquet(RAW / "finmind_sponsor/TaiwanStockDispositionSecuritiesPeriod/all.parquet").with_columns(
        pl.col("period_start").str.to_date(strict=False), pl.col("period_end").str.to_date(strict=False)
    ).filter(pl.col("period_end").is_not_null() & (pl.col("period_end") >= pl.col("period_start")))
    px = px.join(expand(disp, "period_start", "period_end", "in_disposition"), on=["date", "stock_id"], how="left")

    mm = daily_dir("TaiwanStockMarginMaintenance", ["date", "stock_id", "margin_maintenance"])
    px = px.join(load_margin(), on=["date", "stock_id"], how="left").join(mm, on=["date", "stock_id"], how="left")
    px = px.join(px.group_by("date").agg(mkt_ew=pl.col("ret").mean()), on="date").join(load_index(), on="date", how="left")
    px = px.with_columns(
        pl.col("short_suspended").fill_null(False), pl.col("in_disposition").fill_null(False),
        pl.col("cover_day").fill_null(False),
    ).sort(["stock_id", "date"])
    px.write_parquet(OUT)
    print(px.shape, px["date"].min(), px["date"].max(), px["stock_id"].n_unique())


if __name__ == "__main__":
    main()
