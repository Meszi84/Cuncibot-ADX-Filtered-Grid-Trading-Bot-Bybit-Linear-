# Cuncibot – ADX-Filtered Grid Trading Bot (Bybit / Linear)

Cuncibot egy **state-aware, ADX-szűrt grid trading bot**, amely Bybit linear (USDT-margined) perpetual piacokra készült.  
A bot célja **oldalazó piacokban profitot termelni**, miközben **trend esetén automatikusan szünetel**.

⚠️ **Ez nem “set and forget” bot.** Tudatos paraméterezést és megfigyelést igényel.

---

## Fő jellemzők

- 📊 **ADX-alapú regime filter**
  - Grid csak range piacon
  - Trend esetén automatikus pause
  - Hysteresis (`adx_on` / `adx_off`)
  - Cooldown trend után (flicker ellen)

- 🧱 **ATR-alapú dinamikus grid**
  - Lookback + ATR buffer
  - Minimum step százalék
  - Stabil rebuild (nem „elszivárgó” grid)

- 🧠 **Állapotkezelés (State-based design)**
  - `GridState` – grid validitás, reset, rebuild
  - `DayState` – napi equity, drawdown figyelés

- 🛑 **Risk management**
  - Daily drawdown stop
  - Isolated margin
  - Equity-arányos grid sizing

- 🔁 **Rebuild / Refill logika**
  - Grid újraépítés ADX után vagy range drift esetén
  - Részleges refill, ha fogynak az orderek

- 📝 **Structured JSONL logging**
  - Események: START, ADX_PAUSE, GRID_REBUILD, GRID_REFILL, DAILY_DD_STOP, ERROR
  - Elemzésre alkalmas (pandas, notebook, Grafana)

---

## Működési logika – röviden

1. Lekéri az OHLCV adatokat
2. Kiszámolja az ADX-et
3. **Ha trend van → pause**
4. **Ha range van → grid számítás**
5. ATR + lookback alapján meghatározza a grid sávot
6. Grid építés vagy utántöltés
7. Folyamatos kockázat- és állapotfigyelés

---

## Követelmények

- Python **3.10+**
- Bybit account (USDT-M perpetual)
- Könyvtárak:
  ```bash
  pip install ccxt pandas
```pip install -r requirements.txt```
