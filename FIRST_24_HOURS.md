# First 24 Hours Checklist – Cuncibot

Ez a lista arra van, hogy **ne veszíts pénzt az első napon**.

---

## Indítás előtt (kötelező)

- [ ] API kulcs env-ben (NO withdrawal)
- [ ] Testnet / kis tőke
- [ ] `leverage` ≤ 7 (ajánlott: 5)
- [ ] `daily_dd_usdt` beállítva
- [ ] Log fájl írható

---

## Első 1 óra

Figyeld a logot:

- [ ] `START_SESSION` megjelent
- [ ] Van `GRID_REBUILD`
- [ ] NINCS folyamatos `LOOP_ERROR`
- [ ] ADX pause nem villog folyamatosan

Ha:
- sok rebuild → túl szűk range
- sok pause → trendel a piac

---

## 6–12 óra után

- [ ] Equity nem zuhan
- [ ] Refill nem spammel
- [ ] Drawdown < 50% daily limit
- [ ] Grid stabilan él

Ha feszült vagy ránézni → **leverage le**

---

## 24 óra után

Nézd meg:
- GRID_REBUILD / nap
- ADX_PAUSE / nap
- Nettó PnL
- Max drawdown

Döntés:
- ✔ stabil → maradhat
- ⚠ zajos → leverage 5
- ❌ DD közel → STOP, elemzés

---

## Aranyszabály

> Nem az a jó bot, ami sokat köt.  
> Hanem ami **túléli a rossz napot**.
