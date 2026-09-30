# ═════════════════════════════════════════════════════════════════════
#  coin-forward · L1(1H 구조판) 전향 기록기 — GitHub Actions가 매시간 실행
#  과거 630일 분석(coin_quiet_alone.py)과 같은 규칙·같은 태그로, 앞으로 생기는 신호와 결과를 자동으로 쌓는다
#
#  매 실행:
#   1) 새로 완성된 1H 봉들(키)을 찾는다 (놓친 시간은 최대 14일까지 소급 처리)
#   2) 그 주의 우주 B(BingX 선물 · 직전 7일 거래대금 상위 30)를 월요일 기준으로 계산·저장
#   3) 우주 종목마다 1H·30분 데이터로 ▲now 모사 → 신호 기록 (data/signals.csv)
#   4) 종목당 1포지션으로 거래 생성, 열린 거래는 매번 진입부터 다시 계산해 청산 확인 (data/trades.csv)
#   5) 요약 report.md 갱신
#  태그: btc48(조용한 장 |btc48|≤3) · nsig(같은 1H 봉 동시 신호 수, 혼자=1 · 묶음≥4) · rs48 · reb60
# ═════════════════════════════════════════════════════════════════════
import os, sys, json, time, traceback
import numpy as np, pandas as pd
from core import (L1, UNIV_N, STABLE, EXCL_PREFIX, QUIET_BTC, ALONE_N, CROWD_N, COST, STEP,
                  week_of, mg_arrays, align, sim_path, sim_open, now_js_p, sig_detail, reb60_1h, ret48_series)

ROOT   = os.path.dirname(os.path.abspath(__file__))
F_STATE= os.path.join(ROOT,"state","state.json")
F_SIG  = os.path.join(ROOT,"data","signals.csv")
F_TRD  = os.path.join(ROOT,"data","trades.csv")
F_RUN  = os.path.join(ROOT,"data","runs.csv")
F_REP  = os.path.join(ROOT,"report.md")

H        = pd.Timedelta(hours=1)
M30      = pd.Timedelta(minutes=30)
KST      = pd.Timedelta(hours=9)
N1H      = 1500          # 1H 봉 (reb60 1440봉 + 여유)
WARM30   = 300           # 30분 봉 워밍업 (ATR·스윙)
BACKFILL_H   = 72        # 첫 실행 때 소급 시간 (backfill=1 로 표시 · 성적 집계에서 분리)
MAX_CATCHUP_H= 24*14     # 멈췄다 재개할 때 최대 소급
POOL_N   = 120           # 우주 후보 풀 (현재 24h 거래대금 상위)
FAIL_MAX = 0.2           # 우주 종목 20% 넘게 실패하면 이번 실행은 저장하지 않고 다음 실행에서 다시
HOLD_BARS= L1["hold_h"]*60//STEP

def log(*a): print(*a, flush=True)
def ms(t): return int(pd.Timestamp(t).value//1_000_000)

# ───────────── 데이터 소스 (BingX 선물 · 과거 분석과 같은 소스) ─────────────
class BingX:
    def __init__(self):
        import ccxt
        self.ex=ccxt.bingx({"enableRateLimit":True,"options":{"defaultType":"swap"}})
        self._retry(self.ex.load_markets)
        self.sm={}
        for s,m in self.ex.markets.items():
            if m.get("swap") and m.get("quote")=="USDT" and m.get("linear") is not False and m.get("active",True) is not False:
                self.sm.setdefault(m["base"],s)

    def _retry(self,f,*a,**k):
        err=None
        for t in range(4):
            try: return f(*a,**k)
            except Exception as e: err=e; time.sleep(2*(t+1))
        raise err

    def pool(self,n):
        try: t=self._retry(self.ex.fetch_tickers,params={"type":"swap"})
        except Exception: t=self._retry(self.ex.fetch_tickers)
        rows=[]
        for s,x in t.items():
            m=self.ex.markets.get(s)
            if not m or not m.get("swap") or m.get("quote")!="USDT" or m.get("linear") is False: continue
            b=m["base"]
            if b in STABLE or b.startswith(EXCL_PREFIX): continue
            qv=x.get("quoteVolume") or (x.get("baseVolume") or 0)*(x.get("last") or 0)
            rows.append((b,float(qv or 0)))
        rows.sort(key=lambda r:-r[1]); out=[]
        for b,_ in rows:
            if b not in out: out.append(b)
            if len(out)>=n: break
        return out

    def ohlcv(self,base,tf,since,end):
        """[since, end) 구간의 완성된 봉만 · dt = 봉 시작(UTC)"""
        sym=self.sm[base]; dur=self.ex.parse_timeframe(tf)*1000
        now=self.ex.milliseconds(); s=ms(since); e=min(ms(end),now); out=[]
        while s<e:
            to=min(s+dur*1000,e)
            b=self._retry(self.ex.fetch_ohlcv,sym,tf,since=s,limit=1000,params={"until":to-1}) or []
            out+=[x for x in b if s<=x[0]<to]
            s=to
        return to_df(out,dur,now)

def to_df(rows,dur,now_ms):
    cols=["dt","open","high","low","close","volume"]
    if not rows: return pd.DataFrame(columns=cols)
    df=pd.DataFrame(rows,columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts").sort_values("ts")
    df=df[df["ts"]+dur<=now_ms]
    df["dt"]=pd.to_datetime(df["ts"],unit="ms")
    return df[cols].astype({c:float for c in cols[1:]}).reset_index(drop=True)

# ───────────── 상태·파일 ─────────────
def load_state():
    if os.path.exists(F_STATE):
        with open(F_STATE) as f: return json.load(f)
    return {"last_key":None,"univ":{},"runs":0}

def save_state(st):
    os.makedirs(os.path.dirname(F_STATE),exist_ok=True)
    with open(F_STATE,"w") as f: json.dump(st,f,ensure_ascii=False,indent=1,sort_keys=True)

def read_csv(p):
    return pd.read_csv(p,encoding="utf-8-sig") if os.path.exists(p) and os.path.getsize(p)>0 else pd.DataFrame()

# ───────────── 우주 B (월요일 · 직전 7일 거래대금 상위 30 · 7일 봉 ≥120) ─────────────
def compute_universe(src, monday, pool):
    d0=monday-pd.Timedelta(days=7); qv={}
    for b in pool:
        try:
            d=src.ohlcv(b,"1h",d0,monday)
            if len(d)>=120: qv[b]=float((d["volume"]*d["close"]).sum())
        except Exception as e: log(f"  우주 {b} 실패 {str(e)[:60]}")
    v=sorted(((x,b) for b,x in qv.items() if x>0),reverse=True)
    return [b for _,b in v[:UNIV_N]] if len(v)>=UNIV_N else []

# ───────────── 거래 시뮬 (진입 30분봉부터 · 매 실행 다시 계산) ─────────────
def sim_trade(src, sym, entry_bar, cache, now):
    k=(sym,entry_bar)
    if k not in cache:
        cache[k]=src.ohlcv(sym,"30m",pd.Timestamp(entry_bar)-WARM30*M30,now+H)
    d=cache[k]
    pdt=d["dt"].values.astype("datetime64[ns]")
    i=np.searchsorted(pdt,np.datetime64(pd.Timestamp(entry_bar)))
    if i>=len(pdt) or pdt[i]!=np.datetime64(pd.Timestamp(entry_bar)): return None,None
    P=dict(o=d["open"].values,h=d["high"].values,l=d["low"].values,c=d["close"].values)
    atr_at,sw_e=align(pdt,STEP,mg_arrays(d),30)
    r=sim_path(P,int(i),HOLD_BARS,atr_at,sw_e)
    if r is not None:
        r["exit_bar"]=pd.Timestamp(pdt[r["xi"]]); return "closed",r
    return "open",sim_open(P,int(i),HOLD_BARS,atr_at,sw_e)

def fill_closed(t, r):
    ep=t["ep"]; px=ep*(1+r["ret"]/100)
    t.update(status="closed", exit_bar=str(r["exit_bar"]), exit_dt=str(r["exit_bar"]+M30), exit_kst=str(r["exit_bar"]+M30+KST),
             exit_px=px, reason=r["reason"], ret_raw=round(r["ret"],4), ret=round(r["ret"]-COST,4),
             R=round((px-ep)/(ep-t["stop0"]),3), bars=int(r["bars"]), hold_h=round(r["bars"]*STEP/60,1),
             mae=round(r["mae"],3), mfe=round(r["mfe"],3), stop_now=np.nan, unreal=np.nan, hiR=np.nan)

def fill_open(t, s):
    t.update(status="open", stop_now=s["stop_now"], unreal=round(s["unreal"],3), hiR=round(s["hiR"],3),
             bars=int(s["bars"]), hold_h=round(s["bars"]*STEP/60,1))

# ───────────── 메인 ─────────────
def main(src=None, now=None):
    t0=time.time()
    now=(pd.Timestamp.now("UTC").tz_localize(None) if now is None else pd.Timestamp(now)).floor("min")
    st=load_state(); src=src or BingX()
    last_done=now.floor("1h")-H                       # 마지막 완성 1H 봉의 시작 시각
    first=st["last_key"] is None
    start=(last_done-(BACKFILL_H-1)*H) if first else pd.Timestamp(st["last_key"])+H
    start=max(start,last_done-(MAX_CATCHUP_H-1)*H)
    keys=pd.date_range(start,last_done,freq="1h") if start<=last_done else pd.DatetimeIndex([])
    backfill_until=pd.Timestamp(st.get("live_from")) if st.get("live_from") else (last_done+H if first else None)
    if first: st["live_from"]=str(last_done+H)
    log(f"실행 {now} UTC · 처리할 1H 봉 {len(keys)}개 ({keys[0] if len(keys) else '-'} ~ {keys[-1] if len(keys) else '-'})")

    trades=read_csv(F_TRD).to_dict("records"); sigs_new=[]; errors=[]; cache={}

    # 1) 우주
    wk_of_key={k:pd.Timestamp(w) for k,w in zip(keys,week_of(keys))}
    pool=None
    for w in sorted(set(wk_of_key.values())):
        if str(w.date()) in st["univ"]: continue
        pool=pool or src.pool(POOL_N)
        U=compute_universe(src,w,pool); st["univ"][str(w.date())]=U
        log(f"  우주 {w.date()} → {len(U)}종목 {' '.join(U)}")
    scan=sorted(set(s for w in set(wk_of_key.values()) for s in st["univ"].get(str(w.date()),[])))

    # 2) BTC
    btc=src.ohlcv("BTC","1h",start-N1H*H,last_done+H) if len(keys) else pd.DataFrame()
    btc48=ret48_series(btc) if len(btc) else pd.Series(dtype=float)

    # 3) 신호 (▲now 모사 · 완성된 1H 키만)
    kset=set(keys); fails=0
    for s in scan:
        try:
            d1=src.ohlcv(s,"1h",start-N1H*H,last_done+H)
            d30=src.ohlcv(s,"30m",start-WARM30*M30,last_done+H)
            if len(d1)<200 or len(d30)<WARM30//2: continue
            js=now_js_p(d1,d30,L1)
            if len(js)==0: continue
            pdt=d30["dt"].values.astype("datetime64[ns]")
            atr_at,sw_e=align(pdt,STEP,mg_arrays(d30),30)
            r60=reb60_1h(d1); d1dt=d1["dt"].values.astype("datetime64[ns]"); r48=ret48_series(d1)
            for j in js:
                j=int(j); bar=pd.Timestamp(pdt[j]); key=bar.floor("1h")
                if key not in kset or s not in st["univ"].get(str(wk_of_key[key].date()),[]): continue
                ep=float(d30["close"].iloc[j]); stop=float(sw_e[j]) if not np.isnan(sw_e[j]) else np.nan
                k1=np.searchsorted(d1dt,np.datetime64(key),side="right")-1
                det=sig_detail(d1,d30,L1,j)
                b48=btc48.get(key,np.nan); a48=r48.get(key,np.nan)
                sigs_new.append(dict(key=str(key),key_kst=str(key+KST),sym=s,bar=str(bar),entry_dt=str(bar+M30),entry_kst=str(bar+M30+KST),
                    ep=ep,stop0=stop,stop_pct=round((1-stop/ep)*100,3) if stop==stop else np.nan,
                    valid=int(stop==stop and ep>stop), sl1h=round(det["sl1h"],3),vmult=round(det["vmult"],3),reb30=round(det["reb30"],3),
                    reb60=round(float(r60[k1]),3) if k1>=0 else np.nan, btc48=round(b48,3), r48=round(a48,3), rs48=round(a48-b48,3),
                    week=str(wk_of_key[key].date()), backfill=int(backfill_until is not None and key<backfill_until)))
        except Exception as e:
            fails+=1; errors.append(f"{s}: {str(e)[:80]}"); log(f"  {s} 실패 {str(e)[:80]}")
    if scan and fails>FAIL_MAX*len(scan):
        log(f"실패 {fails}/{len(scan)} — 이번 실행은 저장하지 않음 (다음 실행에서 같은 구간 다시 처리)")
        append_run(now,len(keys),0,fails,errors,t0,"abort"); sys.exit(1)

    # nsig = 같은 1H 키에서 유효 신호가 선 우주 종목 수
    S=pd.DataFrame(sigs_new)
    if len(S):
        ns=S[S.valid==1].groupby("key")["sym"].nunique()
        S["nsig"]=S["key"].map(ns).fillna(0).astype(int)
        S["quiet"]=(S.btc48.abs()<=QUIET_BTC).astype(int); S["alone"]=(S.nsig<=ALONE_N).astype(int)
        S["crowd"]=(S.nsig>=CROWD_N).astype(int); S["T"]=(S.quiet&S.alone).astype(int)
        S=S.sort_values(["bar","sym"]).reset_index(drop=True)

    # 4) 열린 거래 갱신
    for t in trades:
        if t.get("status")!="open": continue
        try:
            stt,r=sim_trade(src,t["sym"],t["bar"],cache,now)
            if stt=="closed": fill_closed(t,r); log(f"  청산 {t['sym']} {t['reason']} {t['ret']:+.2f}%")
            elif stt=="open": fill_open(t,r)
        except Exception as e: errors.append(f"open {t['sym']}: {str(e)[:80]}")

    # 5) 새 신호 → 종목당 1포지션 (앞 거래의 청산봉보다 뒤의 봉에서만 새 진입)
    taken=[]; skip=[]
    for _,q in (S.iterrows() if len(S) else []):
        if not q.valid: taken.append(0); skip.append("손절선 무효"); continue
        prev=[t for t in trades if t["sym"]==q.sym]
        busy=any(t["status"]=="open" or pd.Timestamp(t["exit_bar"])>=pd.Timestamp(q.bar) for t in prev)
        if busy: taken.append(0); skip.append("보유 중"); continue
        t=q.to_dict(); t["id"]=f"{q.sym}-{q.bar}"
        try:
            stt,r=sim_trade(src,q.sym,q.bar,cache,now)
            if stt is None: raise ValueError("진입봉 없음")
            (fill_closed if stt=="closed" else fill_open)(t,r)
            trades.append(t); taken.append(1); skip.append("")
            log(f"  ▲ {q.sym} {q.entry_kst} KST · 손절 {q.stop_pct:.1f}% · nsig {q.nsig} · btc48 {q.btc48:+.1f} {'★T' if q['T'] else ''}")
        except Exception as e:
            taken.append(0); skip.append(f"오류 {str(e)[:40]}"); errors.append(f"new {q.sym}: {str(e)[:80]}")
    if len(S): S["taken"]=taken; S["skip"]=skip

    # 6) 저장
    os.makedirs(os.path.dirname(F_SIG),exist_ok=True)
    if len(S):
        old=read_csv(F_SIG); pd.concat([old,S],ignore_index=True).to_csv(F_SIG,index=False,encoding="utf-8-sig")
    if trades:
        TD=pd.DataFrame(trades); first_cols=["id","status","sym","entry_kst","ep","stop0","stop_pct","exit_kst","exit_px","reason","ret","R","hold_h",
            "T","quiet","alone","crowd","nsig","btc48","rs48","reb60","backfill"]
        TD=TD[[c for c in first_cols if c in TD]+[c for c in TD if c not in first_cols]].sort_values("bar")
        TD.to_csv(F_TRD,index=False,encoding="utf-8-sig")
    if len(keys): st["last_key"]=str(keys[-1])
    st["runs"]=st.get("runs",0)+1; save_state(st)
    append_run(now,len(keys),len(S),fails,errors,t0,"ok")
    write_report(st,now)
    log(f"완료 · 신호 {len(S)} · 진입 {sum(taken) if len(S) else 0} · 오류 {len(errors)} · {time.time()-t0:.0f}초")

def append_run(now,nk,ns,fails,errors,t0,status):
    os.makedirs(os.path.dirname(F_RUN),exist_ok=True)
    row=pd.DataFrame([dict(run_utc=str(now),run_kst=str(now+KST),keys=nk,signals=ns,fails=fails,errors=" | ".join(errors)[:500],sec=round(time.time()-t0),status=status)])
    old=read_csv(F_RUN); pd.concat([old,row],ignore_index=True).to_csv(F_RUN,index=False,encoding="utf-8-sig")

# ───────────── 요약 ─────────────
def pf(x):
    x=np.asarray(x,dtype=float); g=x[x>0].sum(); l=-x[x<=0].sum()
    return round(g/l,2) if l>0 else (np.inf if g>0 else np.nan)

def slot2(o):
    keep=[]; open_=[]
    for ix,q in o.sort_values("entry_dt").iterrows():
        open_=[x for x in open_ if x[0]>q["entry_dt"]]
        if any(x[1]==q["sym"] for x in open_): continue
        if len(open_)<2: keep.append(ix); open_.append((q["exit_dt"],q["sym"]))
    return o.loc[keep]

def write_report(st,now):
    TD=read_csv(F_TRD); L=[f"# coin-forward · L1 전향 기록\n",f"갱신 {now+KST:%Y-%m-%d %H:%M} KST · 실행 {st.get('runs',0)}회 · 실시간 기록 시작 {pd.Timestamp(st['live_from'])+KST:%Y-%m-%d %H:%M} KST (그 전은 소급·집계 제외)\n"]
    if len(TD)==0:
        L.append("\n아직 거래 없음.\n")
    else:
        C=TD[(TD.status=="closed")&(TD.backfill==0)].copy()
        def row(name,x):
            if len(x)==0: return f"| {name} | 0 | | | | | |"
            r=x["ret"].values; return f"| {name} | {len(x)} | {(r>0).mean()*100:.0f}% | {pf(r)} | {r.mean():+.2f} | {r.sum():+.1f} | {x['hold_h'].mean():.1f} |"
        L+=["\n## 청산된 거래 (실시간 · 비용 0.1% 차감 · %)\n","| 구분 | n | 승률 | PF | 건당 | 합 | 보유h |","|---|---|---|---|---|---|---|",
            row("전체",C), row("★ T 조용한 장 & 혼자",C[C["T"]==1]), row("나머지",C[C["T"]==0]),
            row("조용한 장",C[C.quiet==1]), row("시끄러운 장",C[C.quiet==0]), row("혼자 nsig=1",C[C.alone==1]),
            row("2~3개 동시",C[(C.nsig>=2)&(C.nsig<=3)]), row("묶음 nsig≥4",C[C.crowd==1])]
        if len(C):
            k=slot2(C); L.append(f"\n2슬롯 시뮬: {len(k)}건 · PF {pf(k['ret'])} · 합 {k['ret'].sum():+.1f}%\n")
        L.append("\n과거 630일 기준(같은 규칙): L1 PF 1.20 · 건당 +0.22 — T/나머지 비교는 coin_quiet_alone.py 결과와 나란히 볼 것.\n")
        O=TD[TD.status=="open"]
        L+=["\n## 보유 중\n","| 종목 | 진입(KST) | 진입가 | 초기손절 | 현재 손절선 | 평가% | 최고R | T | nsig |","|---|---|---|---|---|---|---|---|---|"]
        for _,t in O.iterrows():
            L.append(f"| {t.sym} | {t.entry_kst[5:16]} | {t.ep:.6g} | {t.stop0:.6g} (-{t.stop_pct:.1f}%) | {t.stop_now:.6g} | {t.unreal:+.2f} | {t.hiR:.2f} | {'★' if t['T'] else ''} | {t.nsig} |")
        R=TD.sort_values("bar").tail(15).iloc[::-1]
        L+=["\n## 최근 거래 15\n","| 종목 | 진입(KST) | 상태 | 사유 | 결과% | R | btc48 | nsig | T | 소급 |","|---|---|---|---|---|---|---|---|---|---|"]
        for _,t in R.iterrows():
            res=f"{t.ret:+.2f}" if t.status=="closed" else f"({t.unreal:+.2f})"
            L.append(f"| {t.sym} | {t.entry_kst[5:16]} | {'청산' if t.status=='closed' else '보유'} | {t.reason if t.status=='closed' else ''} | {res} | {t.R if t.status=='closed' else ''} | {t.btc48:+.1f} | {t.nsig} | {'★' if t['T'] else ''} | {'소급' if t.backfill else ''} |")
    wk=sorted(st["univ"])[-1] if st["univ"] else None
    if wk: L.append(f"\n## 이번 주 우주 ({wk} · {len(st['univ'][wk])}종목)\n\n{' · '.join(st['univ'][wk])}\n")
    with open(F_REP,"w",encoding="utf-8") as f: f.write("\n".join(L)+"\n")

if __name__=="__main__":
    try: main()
    except SystemExit: raise
    except Exception:
        traceback.print_exc(); sys.exit(1)
