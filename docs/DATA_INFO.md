# Dokumentacioni i të Dhënave - Projekti AI Tirana Weather

Ky dokument shpjegon procesin e përgatitjes së të dhënave (Data Engineering) për modelin e parashikimit të temperaturës me LSTM.

## 1. Burimi i të Dhënave
- **Stacioni:** Tirana (13601).
- **Periudha:** Janar 2020 - Janar 2026.
- **Frekuenca:** Çdo 1 orë.
- **Libraria:** Meteostat Python API.

## 2. Përshkrimi i Kolonave (Features)
| Kolona | Përshkrimi | Njësia | Statusi |
| :--- | :--- | :--- | :--- |
| `temp` | Temperatura e ajrit | °C | Target Variable |
| `rhum` | Lagështia relative | % | Input Feature |
| `prcp` | Reshjet (Shiu) | mm | Input Feature |
| `wspd` | Shpejtësia e erës | km/h | Input Feature |
| `pres` | Presioni atmosferik | hPa | Input Feature |
| `hour` | Ora e ditës (0-23) | - | Engineered Feature |
| `month` | Muaji (1-12) | - | Engineered Feature |

## 3. Procesi i Pastrimit dhe Transformimit
1. **Reindexing:** Janë krijuar rreshta për 6,183 orët që mungonin në serinë origjinale për të garantuar vazhdimësi kohore.
2. **Interpolation:** Vlerat e munguar te `temp`, `rhum` dhe `pres` janë plotësuar me metodën *Linear Interpolation*.
3. **Imputation:** Kolona `prcp` është plotësuar me `0` në vendet ku mungonte informacioni.
4. **Feature Engineering:** Janë krijuar variabla ciklike (`hour`, `month`) për të ndihmuar modelin të kuptojë sezonalitetin.

## 4. Analiza Statisikore (Insights)
- **Integriteti:** 0 vlera boshe pas pastrimit.
- **Madhësia:** 52,869 rreshta totalë.
- **Sezonaliteti:** Verifikuar përmes *Seasonal Decomposition*. Ekziston një cikël 24-orësh i fortë dhe i rregullt.
- **Korrelacioni:** U vërtetua lidhja fizike negative mes temperaturës dhe lagështisë relative.