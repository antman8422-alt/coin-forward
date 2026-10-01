# ═════════════════════════════════════════════════════════════════════
#  L1(1H 구조판) 신호·관리 규칙 — coin_quiet_alone.py(과거 630일 분석)와 한 글자도 다르지 않게 복사
#  이 파일을 고치면 과거 분석과 전향 기록의 잣대가 달라진다 → 고치지 말 것 (고치면 새 저장소로)
# ═════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd

ATR_K    = 3.0
COST     = 0.10
SWING_MG = 6
STEP     = 30
BAR      = pd.Timedelta(hours=1)
TFMIN    = {"15m":15, "30m":30, "1H":60, "4H":240}
PDR      = {"15m":"15min", "30m":"30min", "1H":"1h", "4H":"4h"}
UNIV_N   = 30
EXCL_PREFIX = ("NC",)
STABLE   = {"USDC","FDUSD","TUSD","BUSD","DAI","USDP","USDE","USD1","EUR","EURI","PAXG","XAUT","BTCDOM","WBTC","WETH","STETH","BFUSD","RLUSD","XUSD"}

QUIET_BTC = 3.0   # 조용한 장 = |btc48| ≤ 3
ALONE_N   = 1     # 혼자 = nsig 1
CROWD_N   = 4     # 묶음 = nsig ≥ 4
C2_BTC    = 3.0   # 장세 C2: |BTC 48h| ≤ 3
MOV_PCT   = 10.0  #          48h 등락 절댓값 ≥ 10% 를 '큰 움직임'으로
C2_MOV    = 20.0  #          그 주 우주 중 큰 움직임 종목 비율 ≤ 20%  (coin_quiet_gate.py C2 · 2026-10-01 채택 후보)

B3 = dict(eTf="1H", look=30,  hi=10, surge=3,  brk=48, swing=24, vlen=80, vmult=1.5, box=1e9,  spike=1e9,  reb=(0.0,1e9),   sl=(4.0,10.0), hold_h=24, mgmt="30m")
L1 = dict(B3, reb=(5.0,1e9))

def week_of(dt_values):
    t=pd.to_datetime(dt_values)+BAR
    return (t-pd.to_timedelta(t.weekday,unit="D")).normalize().values

def mg_arrays(d):
    h,l,cl=d["high"],d["low"],d["close"]
    tr=pd.concat([h-l,(h-cl.shift()).abs(),(l-cl.shift()).abs()],axis=1).max(axis=1)
    return dict(dt=d["dt"].values.astype("datetime64[ns]"), atr=tr.ewm(alpha=1/14,adjust=False).mean().values,
                sw=l.rolling(SWING_MG).min().shift(1).values)

def align(pdt, step_min, mg, mg_min):
    step=np.timedelta64(step_min,"m"); ln=np.timedelta64(mg_min,"m")
    mdt=mg["dt"]; n=len(mdt)
    k=np.searchsorted(mdt, pdt+step-ln, side="right")-1
    atr_at=np.where(k>=0, mg["atr"][np.clip(k,0,n-1)], np.nan)
    ln_ns=np.timedelta64(mg_min,"m").astype("timedelta64[ns]").astype("int64")
    T0=(pdt.astype("int64")//ln_ns*ln_ns).astype("datetime64[ns]")
    i0=np.searchsorted(mdt, T0, side="left"); i0c=np.clip(i0,0,n-1)
    ok=(i0<n)&(mdt[i0c]==T0)
    sw_e=np.where(ok, mg["sw"][i0c], np.nan)
    return atr_at, sw_e

def sim_path(P, i, hold_bars, atr_at, sw_e):
    o,h,l,c=P["o"],P["h"],P["l"],P["c"]; n=len(c); ep=c[i]; sl0=sw_e[i]
    if np.isnan(sl0) or ep<=sl0: return None
    R=ep-sl0; sl=sl0; peak=ep; hiR=0.0; mae=0.0; mfe=0.0; j=i+1
    while j<n and j-i<=hold_bars*20:
        if l[j]<=sl:
            px=o[j] if o[j]<sl else sl
            return dict(ret=(px/ep-1)*100,bars=j-i,reason="트레일" if sl>ep else ("본절" if sl>=ep else "손절"),xi=j,mae=min(mae,l[j]/ep-1)*100,mfe=mfe*100)
        peak=max(peak,h[j]); hiR=max(hiR,(h[j]-ep)/R); mae=min(mae,l[j]/ep-1); mfe=max(mfe,h[j]/ep-1)
        a=atr_at[j]
        if not np.isnan(a): sl=max(sl,peak-ATR_K*a)
        if j-i>=hold_bars and hiR<1.0:
            return dict(ret=(c[j]/ep-1)*100,bars=j-i,reason="시간",xi=j,mae=mae*100,mfe=mfe*100)
        j+=1
    return None

def sim_open(P, i, hold_bars, atr_at, sw_e):
    """sim_path와 같은 규칙으로 아직 안 끝난 거래의 현재 상태 (현재 손절선 · 평가 % · 최고 R)"""
    o,h,l,c=P["o"],P["h"],P["l"],P["c"]; n=len(c); ep=c[i]; sl0=sw_e[i]
    R=ep-sl0; sl=sl0; peak=ep; hiR=0.0
    for j in range(i+1,n):
        peak=max(peak,h[j]); hiR=max(hiR,(h[j]-ep)/R)
        a=atr_at[j]
        if not np.isnan(a): sl=max(sl,peak-ATR_K*a)
    return dict(stop_now=sl, last=c[-1], unreal=(c[-1]/ep-1)*100, hiR=hiR, bars=n-1-i)

def now_js_p(ed, md, Q):
    eTf=Q["eTf"]; LK,HL,SG=Q["look"],Q["hi"],Q["surge"]; lo_,hi_=Q["reb"]; slmin,slmax=Q["sl"]
    h,l,c,v = ed["high"], ed["low"], ed["close"], ed["volume"]
    P = pd.DataFrame({"key": ed["dt"].values, "ei": np.arange(len(ed)),
        "hiB": h.rolling(Q["brk"]).max().shift(1).values, "vavg": v.rolling(Q["vlen"]).mean().shift(1).values,
        "sw": l.rolling(Q["swing"]).min().shift(1).values,
        "loP": l.rolling(LK-1, min_periods=LK//2).min().shift(1).values,
        "hiP": h.rolling(HL-1, min_periods=5).max().shift(1).values, "c3": c.shift(SG).values})
    m = pd.DataFrame({"j": np.arange(len(md)), "key": md["dt"].dt.floor(PDR[eTf]).values,
                      "high": md["high"].values, "low": md["low"].values, "close": md["close"].values, "volume": md["volume"].values})
    m = m.merge(P, on="key", how="inner").sort_values("j"); m = m[m["ei"] >= max(LK,Q["vlen"],Q["brk"])].reset_index(drop=True)
    if len(m)==0: return np.array([],dtype=int)
    g=m.groupby("key", sort=False); ph,pl,pv=g["high"].cummax(),g["low"].cummin(),g["volume"].cumsum(); cur=m["close"]
    lo=np.minimum(m["loP"],pl); hi=np.maximum(m["hiP"],ph)
    reb=(cur/lo-1)*100; box=(1-cur/hi)*100; ret3=(cur/m["c3"]-1)*100; sld=(cur/m["sw"]-1)*100; vm=pv/m["vavg"]
    ok=(reb>=lo_)&(reb<=hi_)&(box<=Q["box"])&(ret3<Q["spike"])&(ph>m["hiB"])&(vm>Q["vmult"])&(sld>=slmin)&(sld<=slmax)
    return m.loc[ok.fillna(False)].groupby("key")["j"].min().values.astype(int)

def sig_detail(ed, md, Q, j):
    """신호 30분봉 j 시점의 기록용 값: 1H 스윙 SL% · 량 배수 · 30봉 반등% (now_js_p와 같은 식)"""
    key=md["dt"].iloc[j].floor(PDR[Q["eTf"]]); e=ed.index[ed["dt"]==key]
    if len(e)==0: return dict(sl1h=np.nan, vmult=np.nan, reb30=np.nan)
    ei=int(e[0]); h,l,v=ed["high"],ed["low"],ed["volume"]
    sub=md[(md["dt"]>=key)&(md.index<=j)]
    cur=md["close"].iloc[j]
    sw=l.iloc[ei-Q["swing"]:ei].min(); vavg=v.iloc[ei-Q["vlen"]:ei].mean()
    loP=l.iloc[max(0,ei-(Q["look"]-1)):ei].min(); lo=min(loP, sub["low"].min())
    return dict(sl1h=(cur/sw-1)*100, vmult=sub["volume"].sum()/vavg, reb30=(cur/lo-1)*100)

def reb60_1h(d1):
    c=d1["close"].shift(1); l=d1["low"].shift(1)
    return ((c/l.rolling(1440,min_periods=720).min()-1)*100).values

def r48_now(d1):
    """지금 진행 중인 봉에서 아는 48h 등락 = 마지막 완성봉 종가 / 48봉 전 종가"""
    c=d1["close"].astype(float).values
    return (c[-1]/c[-49]-1)*100 if len(c)>=49 else np.nan

def ret48_series(d1):
    c=d1.set_index("dt")["close"].astype(float)
    r=(c/c.shift(48)-1)*100
    return r.shift(1)
