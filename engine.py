"""
추천 엔진: 코스피/코스닥 종목의 차트(기술적 지표)와 가치(PER/PBR)를 보고
단타(1~5일), 중타(2~6주) 후보를 골라 매수가·목표가·손절가를 계산합니다.

※ 참고용 신호일 뿐이며 수익을 보장하지 않습니다.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 1. 가격 단위(호가) 맞추기 — 한국거래소 호가단위
# ─────────────────────────────────────────────
def tick_size(price: float) -> int:
    if price < 2000: return 1
    if price < 5000: return 5
    if price < 20000: return 10
    if price < 50000: return 50
    if price < 200000: return 100
    if price < 500000: return 500
    return 1000


def round_tick(price: float, mode: str = "near") -> int:
    t = tick_size(price)
    if mode == "down":
        return int(np.floor(price / t) * t)
    if mode == "up":
        return int(np.ceil(price / t) * t)
    return int(round(price / t) * t)


# ─────────────────────────────────────────────
# 2. 기술적 지표 계산
# ─────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df: Open, High, Low, Close, Volume 컬럼을 가진 일봉 데이터"""
    d = df.copy()
    c = d["Close"]
    for n in (5, 20, 60, 120):
        d[f"MA{n}"] = c.rolling(n).mean()

    # RSI(14)
    diff = c.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    # MACD(12,26,9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIG"]

    # 볼린저밴드(20, 2)
    std = c.rolling(20).std()
    d["BB_UP"] = d["MA20"] + 2 * std
    d["BB_LOW"] = d["MA20"] - 2 * std

    # ATR(14) — 하루 평균 움직임 폭
    prev = c.shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev).abs(), (d["Low"] - prev).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # 거래량 비율(오늘 / 20일 평균)
    d["VOL_RATIO"] = d["Volume"] / d["Volume"].rolling(20).mean()
    d["HIGH20"] = d["High"].rolling(20).max()
    d["HIGH60"] = d["High"].rolling(60).max()
    return d


# ─────────────────────────────────────────────
# 3. 추천 결과 형식
# ─────────────────────────────────────────────
@dataclass
class Pick:
    style: str          # "단타" 또는 "중타"
    code: str
    name: str
    market: str
    close: int          # 최근 종가
    buy: int            # 추천 매수가
    target: int         # 추천 매도가(목표가)
    stop: int           # 손절가
    score: float        # 점수(높을수록 조건이 좋음)
    reasons: list
    per: float | None = None
    pbr: float | None = None
    asof: str = ""

    @property
    def upside(self) -> float:
        return (self.target / self.buy - 1) * 100

    @property
    def downside(self) -> float:
        return (self.stop / self.buy - 1) * 100

    def to_dict(self):
        x = asdict(self)
        x["upside"] = round(self.upside, 1)
        x["downside"] = round(self.downside, 1)
        return x


# ─────────────────────────────────────────────
# 4. 단타(1~5일) 점수
#    상승 추세 + 거래량 급증 + 모멘텀 막 살아나는 종목
# ─────────────────────────────────────────────
def score_short(d: pd.DataFrame):
    if len(d) < 70:
        return None
    t, y = d.iloc[-1], d.iloc[-2]
    reasons, s = [], 0.0

    # 필수 조건
    if not (t.Close > t.MA20 and t.MA5 > t.MA20):
        return None
    if not (45 <= t.RSI <= 72):          # 과열(72↑)은 제외
        return None
    if t.Close > t.BB_UP * 1.02:         # 볼린저 상단 크게 돌파 = 추격매수 위험
        return None

    if t.VOL_RATIO >= 1.5:
        s += min(t.VOL_RATIO, 4) * 10
        reasons.append(f"거래량 평소의 {t.VOL_RATIO:.1f}배")
    if t.MACD_HIST > 0 and t.MACD_HIST > y.MACD_HIST:
        s += 15; reasons.append("MACD 상승 전환/확대")
    if y.MA5 <= y.MA20 * 1.005 and t.MA5 > t.MA20:
        s += 15; reasons.append("5일선이 20일선 위로 골든크로스")
    if t.Close >= t.HIGH20 * 0.98:
        s += 10; reasons.append("20일 고점 근처(돌파 시도)")
    if 55 <= t.RSI <= 68:
        s += 10; reasons.append(f"RSI {t.RSI:.0f} (적당한 상승 힘)")
    chg = (t.Close / y.Close - 1) * 100
    if chg > 12:                          # 하루 너무 급등한 종목은 감점
        s -= 20
    if s < 30:
        return None

    atr = t.ATR
    buy = round_tick(max(t.MA5, t.Close - 0.5 * atr), "down")   # 살짝 눌릴 때 매수
    target = round_tick(buy + 1.5 * atr, "down")
    stop = round_tick(buy - 1.0 * atr, "up")
    return s, buy, target, stop, reasons


# ─────────────────────────────────────────────
# 5. 중타(2~6주) 점수
#    중기 상승추세 속 20일선 근처 눌림목 + 저평가 가점
# ─────────────────────────────────────────────
def score_mid(d: pd.DataFrame, per=None, pbr=None):
    if len(d) < 130:
        return None
    t, prev5 = d.iloc[-1], d.iloc[-6]
    reasons, s = [], 0.0

    if not (t.MA20 > t.MA60 > t.MA120 * 0.98):     # 정배열(중기 상승추세)
        return None
    if not (t.MA60 > prev5.MA60):                   # 60일선 우상향
        return None
    gap = (t.Close / t.MA20 - 1) * 100
    if not (-3 <= gap <= 4):                        # 20일선 근처까지 눌림
        return None
    if not (38 <= t.RSI <= 62):
        return None

    s += 25; reasons.append("20·60·120일선 정배열 (중기 상승추세)")
    s += 15 - abs(gap) * 2; reasons.append(f"20일선 근처 눌림목 ({gap:+.1f}%)")
    if t.VOL_RATIO < 1.0:
        s += 8; reasons.append("눌림 구간 거래량 감소(매도세 약함)")
    if t.MACD > 0:
        s += 7; reasons.append("MACD 0선 위")

    # 가치 가점 (데이터가 있을 때만)
    if per is not None and per > 0:
        if per < 10: s += 12; reasons.append(f"PER {per:.1f} (저평가)")
        elif per < 20: s += 6; reasons.append(f"PER {per:.1f}")
        elif per > 60: s -= 10
    elif per is not None and per <= 0:
        s -= 10                                     # 적자 기업 감점
    if pbr is not None and 0 < pbr < 1.2:
        s += 6; reasons.append(f"PBR {pbr:.2f} (자산 대비 저평가)")

    if s < 35:
        return None
    atr = t.ATR
    buy = round_tick(min(t.Close, t.MA20 + 0.3 * atr), "down")
    target = round_tick(max(buy + 3.0 * atr, t.HIGH60), "down")
    target = min(target, round_tick(buy * 1.20, "down"))       # 목표 최대 +20%
    stop = round_tick(max(t.MA60, buy - 2.0 * atr) * 0.99, "up")
    if stop >= buy:
        stop = round_tick(buy - 2.0 * atr, "up")
    return s, buy, target, stop, reasons


# ─────────────────────────────────────────────
# 6. 데이터 가져오기
#    1순위 FinanceDataReader/pykrx, 실패하면 네이버 금융에서 직접
#    (해외 서버에서 한국거래소 사이트가 막히는 경우 대비)
# ─────────────────────────────────────────────
import re
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
EXCLUDE = r"스팩|우$|우B$|우C$|\(전환\)|리츠|ETN|ETF|KODEX|TIGER|KBSTAR|ACE |SOL |HANARO|ARIRANG|RISE "


def _num(s):
    s = re.sub(r"<[^>]+>", "", s).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_naver_market_sum(html: str, market: str) -> pd.DataFrame:
    heads = [re.sub(r"<[^>]+>", "", h).strip()
             for h in re.findall(r'<th scope="col"[^>]*>(.*?)</th>', html, re.S)]
    num_heads = [h for h in heads if h not in ("N", "종목명", "토론실", "")]
    rows = []
    for tr in re.split(r"<tr[ >]", html):
        m = re.search(r'code=(\d{6})"[^>]*class="tltle"[^>]*>(.*?)</a>', tr)
        if not m:
            continue
        nums = [_num(x) for x in re.findall(r'<td class="number[^"]*"[^>]*>(.*?)</td>', tr, re.S)]
        row = {"Code": m.group(1), "Name": m.group(2).strip(), "Market": market}
        row.update(dict(zip(num_heads, nums)))
        rows.append(row)
    return pd.DataFrame(rows)


def naver_listing(pages: int = 6) -> pd.DataFrame:
    out = []
    for sosok, market in ((0, "KOSPI"), (1, "KOSDAQ")):
        for page in range(1, pages + 1):
            r = requests.get("https://finance.naver.com/sise/sise_market_sum.naver",
                             params={"sosok": sosok, "page": page}, headers=UA, timeout=15)
            r.encoding = "euc-kr"
            out.append(parse_naver_market_sum(r.text, market))
    return pd.concat(out, ignore_index=True)


def load_universe(top_n: int = 300, min_amount_eok: float = 30) -> pd.DataFrame:
    """시가총액 상위 + 거래대금 충분한 종목만 (너무 작은 종목은 위험해서 제외)"""
    try:
        import FinanceDataReader as fdr
        lst = fdr.StockListing("KRX")
        lst = lst[lst["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
        lst = lst[~lst["Name"].str.contains(EXCLUDE, regex=True)]
        if "Amount" in lst.columns:
            lst = lst[lst["Amount"] >= min_amount_eok * 1e8]
        lst = lst.sort_values("Marcap", ascending=False)
        if len(lst) >= 50:
            return lst.head(top_n)[["Code", "Name", "Market"]].reset_index(drop=True)
    except Exception as e:
        print("FDR 종목목록 실패 → 네이버 사용:", e)
    lst = naver_listing()
    lst = lst[~lst["Name"].str.contains(EXCLUDE, regex=True)]
    if "현재가" in lst and "거래량" in lst:
        lst = lst[lst["현재가"] * lst["거래량"] >= min_amount_eok * 1e8]
    if "시가총액" in lst:
        lst = lst.sort_values("시가총액", ascending=False)
    return lst.head(top_n)[["Code", "Name", "Market"]].reset_index(drop=True)


def parse_naver_fchart(xml: str) -> pd.DataFrame:
    rows = [x.split("|") for x in re.findall(r'<item data="([^"]+)"', xml)]
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").astype(float)
    return df[df["Close"] > 0]


def load_prices(code: str, days: int = 260) -> pd.DataFrame:
    try:
        import FinanceDataReader as fdr
        start = (dt.date.today() - dt.timedelta(days=int(days * 1.6))).isoformat()
        df = fdr.DataReader(code, start)
        if df is not None and len(df) >= min(days, 60):
            return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    r = requests.get("https://fchart.stock.naver.com/sise.nhn",
                     params={"symbol": code, "timeframe": "day", "count": days, "requestType": 0},
                     headers=UA, timeout=15)
    return parse_naver_fchart(r.text)


def naver_value(code: str):
    """종목 하나의 PER/PBR (네이버 모바일). 실패하면 (None, None)"""
    try:
        r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/integration", headers=UA, timeout=10)
        per = pbr = None
        for it in r.json().get("totalInfos", []):
            k = str(it.get("code", it.get("key", ""))).lower()
            v = _num(str(it.get("value", "")).replace("배", ""))
            if k == "per":
                per = None if np.isnan(v) else v
            elif k == "pbr":
                pbr = None if np.isnan(v) else v
        return per, pbr
    except Exception:
        return None, None


def load_fundamentals() -> pd.DataFrame | None:
    """PER/PBR. pykrx 실패 시 네이버 시가총액 페이지의 PER 사용, 그것도 실패하면 None"""
    try:
        from pykrx import stock
        for back in range(0, 7):
            day = (dt.date.today() - dt.timedelta(days=back)).strftime("%Y%m%d")
            f = stock.get_market_fundamental(day, market="ALL")
            if f is not None and len(f) and f["PER"].sum() > 0:
                return f[["PER", "PBR"]]
    except Exception as e:
        print("pykrx 실패:", e)
    try:
        lst = naver_listing()
        f = lst.set_index("Code")[["PER"]].copy()
        f["PER"] = f["PER"].fillna(-1)       # 네이버는 적자면 N/A
        f["PBR"] = np.nan
        return f
    except Exception as e:
        print("네이버 PER 실패:", e)
    return None


# ─────────────────────────────────────────────
# 7. 전체 스캔 → 단타·중타 각 1~2개 추천
# ─────────────────────────────────────────────
LAST_SOURCE = {}


def scan(top_n: int = 300, picks_each: int = 2, progress=None,
         price_loader=None, universe=None, fundamentals="auto"):
    price_loader = price_loader or load_prices
    uni = universe if universe is not None else load_universe(top_n)
    fund = load_fundamentals() if isinstance(fundamentals, str) else fundamentals
    global LAST_SOURCE
    LAST_SOURCE = {"fundamentals": None if fund is None else len(fund), "universe": len(uni)}

    shorts, mids = [], []
    total = len(uni)
    for i, row in uni.iterrows():
        if progress:
            progress(i + 1, total, row["Name"])
        try:
            raw = price_loader(row["Code"])
            if raw is None or len(raw) < 70:
                continue
            d = add_indicators(raw)
            last = d.iloc[-1]
            if last.Volume == 0:              # 거래정지
                continue
            asof = str(d.index[-1])[:10]
            per = pbr = None
            if fund is not None and row["Code"] in fund.index:
                per = float(fund.loc[row["Code"], "PER"]); pbr = float(fund.loc[row["Code"], "PBR"])
                pbr = None if np.isnan(pbr) else pbr; per = None if np.isnan(per) else per

            r = score_short(d)
            if r:
                s, b, tg, st, rs = r
                shorts.append(Pick("단타", row["Code"], row["Name"], row["Market"], int(last.Close),
                                   b, tg, st, round(s, 1), rs, per, pbr, asof))
            if per is None and score_mid(d, 5, 1):          # 가치 점수가 필요한 후보만 개별 조회
                per, pbr = naver_value(row["Code"])
                if per is None:                              # 조회 실패 시 가치 점수 없이
                    per = pbr = None
            r = score_mid(d, per, pbr)
            if r:
                s, b, tg, st, rs = r
                mids.append(Pick("중타", row["Code"], row["Name"], row["Market"], int(last.Close),
                                 b, tg, st, round(s, 1), rs, per, pbr, asof))
        except Exception:
            continue

    shorts.sort(key=lambda p: p.score, reverse=True)
    mids.sort(key=lambda p: p.score, reverse=True)
    return shorts[:picks_each], mids[:picks_each]
