"""
클라우드(GitHub Actions)에서 실행되는 스크립트

  python run.py analyze   → 전체 분석, 추천 저장, 폰 알림
  python run.py watch     → 추천 종목 현재가 확인, 매수/목표/손절가 도달 시 알림

결과는 docs/data/latest.json 에 저장되고 폰 앱이 그 파일을 읽어 화면에 보여줍니다.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import requests

import engine
from engine import scan, load_prices, add_indicators

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "docs", "data", "latest.json")
HIST = os.path.join(BASE, "docs", "data", "history.json")
KST = dt.timezone(dt.timedelta(hours=9))


def cfg():
    with open(os.path.join(BASE, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    c["ntfy_topic"] = os.environ.get("NTFY_TOPIC") or c["ntfy_topic"]
    return c


def push(title, message, tags="chart_with_upwards_trend", priority=4, click=None):
    topic = cfg()["ntfy_topic"]
    headers = {"Title": title.encode("utf-8"), "Tags": tags, "Priority": str(priority)}
    if click:
        headers["Click"] = click
    try:
        r = requests.post(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"), headers=headers, timeout=10)
        print("알림:", title, r.status_code)
    except Exception as e:
        print("알림 실패:", e)


def app_url():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo:
        return None
    owner, name = repo.split("/")
    return f"https://{owner.lower()}.github.io/{name}/"


def chart_rows(code, n=90):
    d = add_indicators(load_prices(code, days=220)).tail(n)
    rows = []
    for ts, r in d.iterrows():
        rows.append([str(ts)[:10], round(r.Open), round(r.High), round(r.Low), round(r.Close),
                     None if np.isnan(r.MA5) else round(r.MA5),
                     None if np.isnan(r.MA20) else round(r.MA20),
                     None if np.isnan(r.MA60) else round(r.MA60)])
    return rows


def fmt(p):
    return (f"[{p['style']}] {p['name']}\n"
            f"매수 {p['buy']:,} → 목표 {p['target']:,} ({p['upside']:+.1f}%) / 손절 {p['stop']:,}")


def save(data):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load():
    if not os.path.exists(OUT):
        return None
    with open(OUT, encoding="utf-8") as f:
        return json.load(f)


def analyze():
    c = cfg()
    shorts, mids = scan(top_n=c["top_n"], picks_each=c["picks_each"],
                        progress=lambda i, t, n: i % 50 == 0 and print(f"{i}/{t} {n}"))
    picks = []
    for p in shorts + mids:
        d = p.to_dict()
        try:
            d["chart"] = chart_rows(p.code)
        except Exception:
            d["chart"] = []
        d["status"] = "대기"          # 대기 → 매수가 도달 → 목표가 도달 / 손절
        d["alerted"] = []
        picks.append(d)
    data = {"created": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "picks": picks, "source": engine.LAST_SOURCE}
    save(data)

    # 지난 추천 기록 (최근 60건)
    hist = json.load(open(HIST, encoding="utf-8")) if os.path.exists(HIST) else []
    for d in picks:
        hist.insert(0, {k: d[k] for k in ("created", "style", "code", "name", "buy", "target", "stop")
                        if k in d} | {"created": data["created"]})
    with open(HIST, "w", encoding="utf-8") as f:
        json.dump(hist[:60], f, ensure_ascii=False)

    if picks:
        push(f"📈 오늘의 추천 {len(picks)}종목", "\n\n".join(fmt(p) for p in picks), click=app_url())
    else:
        push("오늘은 추천 없음", "조건에 맞는 종목이 없어요. 쉬는 것도 투자예요.", tags="zzz", priority=3)


def watch():
    if not cfg().get("watch_alert", True):
        return
    data = load()
    if not data:
        return
    for p in data["picks"]:
        try:
            df = load_prices(p["code"], days=10)
            last = df.iloc[-1]
        except Exception:
            continue
        if str(df.index[-1])[:10] <= p.get("asof", ""):   # 분석한 날 이후 가격만 봄
            continue
        price, hi, lo = float(last.Close), float(last.High), float(last.Low)
        p["now"] = round(price)
        done = p.setdefault("alerted", [])
        if lo <= p["buy"] and "buy" not in done:
            push(f"🛒 매수가 도달: {p['name']}", f"[{p['style']}] 현재 {price:,.0f}원 · 추천 매수가 {p['buy']:,}원",
                 tags="shopping_cart", click=app_url())
            done.append("buy"); p["status"] = "매수가 도달"
        if "buy" in done and hi >= p["target"] and "target" not in done and "stop" not in done:
            push(f"🎉 목표가 도달: {p['name']}", f"현재 {price:,.0f}원 · 목표가 {p['target']:,}원 — 매도 고려",
                 tags="tada", priority=5, click=app_url())
            done.append("target"); p["status"] = "목표가 도달"
        if "buy" in done and lo <= p["stop"] and "stop" not in done and "target" not in done:
            push(f"⚠️ 손절가 도달: {p['name']}", f"현재 {price:,.0f}원 · 손절가 {p['stop']:,}원 — 손절 고려",
                 tags="warning", priority=5, click=app_url())
            done.append("stop"); p["status"] = "손절가 도달"
    data["checked"] = dt.datetime.now(KST).strftime("%H:%M")
    save(data)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if mode == "test":
        push("🔔 테스트 알림", "알림이 잘 와요! 이제 평일마다 추천을 받을 수 있어요.", tags="bell", click=app_url())
    elif mode == "watch":
        watch()
    else:
        analyze()
