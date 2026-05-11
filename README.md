![image](https://github.com/user-attachments/assets/575ac8bf-bdd9-4bd3-a403-a1607bd4e690)

# Solar DA Value Analysis — NL

Quantifies the **market value of Dutch solar PV (and wind) production** by combining EPEX day-ahead spot prices with NED.nl generation data on a uniform 15-minute UTC grid. Outputs interactive Plotly dashboards, profile-factor vs. installed-capacity curves, and monthly summary tables.

> **Source of truth:** `birdcurve_nl` DuckDB warehouse — not raw CSV. The CSV ingestion path is archived but kept readable for forensic/historical use.

---

## 1. System overview

```mermaid
flowchart LR
    subgraph SRC["External sources"]
        NED["NED.nl API<br/>(Solar PV, Wind on/offshore)"]
        EPEX["EPEX / ENTSO-E<br/>(NL Day-Ahead prices)"]
        CBS["CBS<br/>(installed capacity)"]
    end

    subgraph WH["birdcurve_nl warehouse (DuckDB)"]
        TS15["ts_15min<br/>NED_PV / Wind"]
        TSH["ts_hourly + 15m<br/>DA_price"]
    end

    subgraph LOADER["data_loader.py"]
        LP["load_ned_pv()"]
        LWON["load_ned_wind_onshore()"]
        LWOFF["load_ned_wind_offshore()"]
        LW["load_ned_wind() — combined"]
        LD["load_da_prices()"]
    end

    subgraph ANALYSIS["Analysis layer"]
        C3["combine_v3.py<br/>(monthly value)"]
        DPV["Dashboard_PV_…"]
        DWON["Dashboard_Wind_Onshore_…"]
        DWOFF["Dashboard_Wind_Offshore_…"]
        NEDPY["NED.py"]
    end

    subgraph OUT["Artifacts (HTML / PDF)"]
        HSOL["solar_production_plot_v3.html"]
        HPV["pv_profile_factor_vs_capacity_dashboard.html"]
        HWINDON["wind_onshore_production_plot_v3.html<br/>wind_onshore_profile_factor_vs_capacity_dashboard.html"]
        HWINDOFF["wind_offshore_production_plot_v3.html<br/>wind_offshore_profile_factor_vs_capacity_dashboard.html"]
        TBL["monthly_summary_table.html"]
    end

    NED --> TS15
    EPEX --> TSH
    CBS  -.calibration.-> ANALYSIS

    TS15 --> LP
    TS15 --> LWON & LWOFF & LW
    TSH  --> LD

    LP    --> C3 & DPV & NEDPY
    LWON  --> DWON
    LWOFF --> DWOFF
    LD    --> C3 & DPV & DWON & DWOFF & NEDPY

    C3    --> HSOL & TBL
    DPV   --> HPV
    DWON  --> HWINDON
    DWOFF --> HWINDOFF
```

---

## 2. Repository map

```mermaid
graph TD
    classDef entry fill:#0d6efd,color:#fff,stroke:#0d6efd
    classDef lib fill:#198754,color:#fff,stroke:#198754
    classDef arch fill:#6c757d,color:#fff,stroke:#6c757d
    classDef out fill:#fd7e14,color:#fff,stroke:#fd7e14

    DL["data_loader.py<br/><i>shared loaders</i>"]:::lib

    C1["combine_v1.py"]:::entry
    C2["combine_v2.py"]:::entry
    C3["combine_v3.py ★"]:::entry

    DPV1["Dashboard_PV_Profile_Factor_vs_Capacity.py ★"]:::entry
    DPV2["Dashboard_PV_Profile_Factor_vs_Capacity-standard.py"]:::entry
    DPV3["Dashboard_PV_Profile_Factor_vs_Capacity-xkcd-matplotlib.py"]:::entry
    DPVMK["Dashboard_Solar_PV_market_prices_NL.py"]:::entry

    DW1ON["Dashboard_Wind_Onshore_Profile_Factor_vs_Capacity.py"]:::entry
    DW1OFF["Dashboard_Wind_Offshore_Profile_Factor_vs_Capacity.py"]:::entry
    DWMKON["Dashboard_Wind_Onshore_market_prices_NL.py"]:::entry
    DWMKOFF["Dashboard_Wind_Offshore_market_prices_NL.py"]:::entry

    NEDPY["NED.py"]:::entry
    CMP["compar_prices_June_July_2025.py"]:::entry

    ARCH["archive/legacy_csv_ingestion/<br/>(retired CSV path)"]:::arch
    DATA["data/<br/>(legacy raw CSVs, optional)"]:::arch

    HTML["*.html / *.pdf<br/>dashboards & tables"]:::out

    DL --> C1 & C2 & C3
    DL --> DPV1 & DPV2 & DPV3 & DPVMK
    DL --> DW1 & DWMK
    DL --> NEDPY & CMP

    C3 --> HTML
    DPV1 --> HTML
    DW1 --> HTML
```

★ = recommended entry point.

---

## 3. Data flow & time-grid handling

The single hardest invariant in this repo is keeping prices and production on the **same 15-min UTC grid** across the EPEX `1h → 15min` cutover (2025-10-01).

```mermaid
flowchart TB
    subgraph IN["Inputs"]
        H["Hourly DA price<br/>(pre 2025-10-01)"]
        Q["15-min DA price<br/>(post 2025-10-01)"]
        P15["NED 15-min production"]
    end

    subgraph TZ["Timezone normalisation"]
        U["UTC tz-aware Timestamps<br/>(no naive datetimes)"]
    end

    subgraph GRID["15-min UTC grid (single source)"]
        R["pd.date_range(freq='15min', tz='UTC')"]
        FF["ffill(limit=3)<br/>1 hourly row → 4 quarters"]
    end

    subgraph CONV["Energy convention"]
        E["MWh per slot = MW × 0.25<br/>(so production × price = €)"]
    end

    subgraph DISP["Display boundary"]
        L["tz_convert('Europe/Amsterdam')<br/>only at output"]
    end

    H --> U
    Q --> U
    P15 --> U
    U --> R
    R --> FF
    FF --> CONV
    P15 --> CONV
    CONV --> DISP
```

**Why this matters.** Mixing tz-naive and tz-aware Timestamps silently drops rows in pandas 2.x. Storing local time would break at DST. The loader localises at the read boundary and only converts at display.

---

## 4. DuckDB warehouse schema

The relevant slice of `birdcurve_nl/data/birdcurve.duckdb` consumed by this project:

```mermaid
erDiagram
    TS_15MIN {
        TIMESTAMP timestamp_utc PK "tz-naive, UTC by convention"
        DOUBLE NED_PV__PV "MW, NL solar"
        DOUBLE NED_Wind_Onshore__Wind_Onshore "MW"
        DOUBLE NED_Wind_Offshore__Wind_Offshore "MW"
    }
    TS_HOURLY {
        TIMESTAMP timestamp_utc PK "tz-naive, UTC by convention"
        DOUBLE DA_price__DA_price "EUR/MWh, NL"
    }
    TS_15MIN ||--o{ TS_HOURLY : "ffilled into 4 slots/hour pre-cutover"
```

> **Override location** by setting `BIRDCURVE_DB=/path/to/your.duckdb` in the environment.

---

## 5. Profile-factor pipeline

The "profile factor" is the central economic metric: the **ratio of the production-weighted price to the time-weighted (baseload) price** in a given window. <1 means the resource is generating when it's worth less than average.

```mermaid
flowchart LR
    A["load_ned_pv()<br/>15-min MWh"] --> M["merge on time"]
    B["load_da_prices()<br/>15-min EUR/MWh"] --> M
    M --> V["value = MWh × €/MWh"]
    V --> AGG["resample('M' or 'Y')"]
    AGG --> WP["weighted price =<br/>Σ value / Σ MWh"]
    AGG --> BP["baseload price =<br/>mean(price)"]
    WP --> PF["profile_factor =<br/>weighted / baseload"]
    BP --> PF
    PF --> OUT["dashboard / table"]
```

For PV this is also plotted **against installed DC capacity** (linear-fit anchored on CBS data points in `Dashboard_PV_Profile_Factor_vs_Capacity.py`), which lets you see cannibalisation as the fleet grows.

---

## 6. Typical run sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant S as Script (e.g. combine_v3.py)
    participant L as data_loader.py
    participant D as DuckDB (read-only)
    participant P as Plotly / Matplotlib

    U->>S: python combine_v3.py
    S->>L: load_ned_pv() / load_da_prices()
    L->>D: SELECT … FROM ts_15min / ts_hourly
    D-->>L: tz-naive UTC frames
    L->>L: tz_localize('UTC') → tz_convert('Europe/Amsterdam')
    L->>L: reindex to 15-min grid + ffill DA price
    L-->>S: tidy DataFrame (time, MWh / €)
    S->>S: merge, value, resample, profile factor
    S->>P: render figure
    P-->>U: solar_production_plot_v3.html
```

---

## 7. Data coverage timeline

```mermaid
gantt
    title Roughly available time ranges by series
    dateFormat  YYYY-MM-DD
    axisFormat  %Y
    section Prices
    EPEX hourly DA          :done, h1, 2018-01-01, 2025-10-01
    EPEX 15-min DA          :active, h2, 2025-10-01, 2026-12-31
    section NED.nl
    Solar PV (15-min)       :active, p1, 2018-01-01, 2026-12-31
    Wind onshore + offshore :active, w1, 2018-01-01, 2026-12-31
    section CBS
    Installed PV capacity   :crit, c1, 2018-01-01, 2026-12-31
```

(Exact bounds are clipped at runtime to whatever has actually been ingested into `ts_15min` / `ts_hourly`.)

---

## 8. Outputs catalogue

| Artifact | Generated by | Shows |
| --- | --- | --- |
| `solar_production_plot_v3.html` | `combine_v3.py` | Yearly + monthly PV energy, market value, weighted price, profile factor |
| `monthly_summary_table.html` | `combine_v3.py` | Monthly PV production, installed capacity, market value, price metrics |
| `pv_profile_factor_vs_capacity_dashboard.html` | `Dashboard_PV_Profile_Factor_vs_Capacity.py` | Profile factor vs. installed DC capacity (yearly) |
| `pv_profile_factor_vs_capacity_dashboard_xkcd.{png,pdf}` | `…-xkcd-matplotlib.py` | Same, xkcd hand-drawn style |
| `wind_onshore_production_plot_v3.html` | `Dashboard_Wind_Onshore_market_prices_NL.py` | Onshore Wind energy + market value |
| `wind_offshore_production_plot_v3.html` | `Dashboard_Wind_Offshore_market_prices_NL.py` | Offshore Wind energy + market value |
| `wind_onshore_profile_factor_vs_capacity_dashboard.html` | `Dashboard_Wind_Onshore_Profile_Factor_vs_Capacity.py` | Onshore Wind profile factor vs. installed capacity |
| `wind_offshore_profile_factor_vs_capacity_dashboard.html` | `Dashboard_Wind_Offshore_Profile_Factor_vs_Capacity.py` | Offshore Wind profile factor vs. installed capacity |
| `compare_prices_july2025_vs_june2025*.html` | `compar_prices_June_July_2025.py` | Month-over-month price overlay |

---

## 9. Quick start

This repo ships its own `pixi.toml` — use the project env, not the global `main` env or `pip`.

```sh
# Resolve & install the project env from pixi.lock (one-time / after pulls)
pixi install

# Point at the warehouse (or rely on the default below)
export BIRDCURVE_DB=/Users/mayk/birdcurve_nl/data/birdcurve.duckdb

# Sanity-check the loaders
pixi run python data_loader.py

# Generate the headline solar-value report
pixi run python combine_v3.py

# Open results
open solar_production_plot_v3.html monthly_summary_table.html
```

If `BIRDCURVE_DB` is unset the loaders default to `/Users/mayk/birdcurve_nl/data/birdcurve.duckdb`.

> Adding a dependency? Use `pixi add <pkg>` (conda-forge) or `pixi add --pypi <pkg>` (PyPI-only). Both update `pixi.lock`. Never `pip install` into this env.

---

## 10. Customisation knobs

| Knob | Where | Effect |
| --- | --- | --- |
| `BIRDCURVE_DB` env var | shell | Point loaders at a different DuckDB file |
| `tz=` arg on loaders | `data_loader.py` | Display timezone (default `Europe/Amsterdam`) |
| `clip_future=False` | `load_da_prices()` | Keep day-ahead rows past today's midnight |
| `capacity_points` list | `Dashboard_PV_Profile_Factor_vs_Capacity.py` | Anchor years for the installed-capacity fit — update as new CBS releases land |

---

## 11. Conventions (non-obvious ones)

- **All timestamps are tz-aware UTC at rest.** Conversion to `Europe/Amsterdam` happens at display only.
- **Energy units are MWh per 15-min slot.** Multiply by `4` to recover instantaneous MW.
- **NL solar/wind data comes from NED.nl, never ENTSO-E.** ENTSO-E reports ~10% of installed NL solar — known systemic gap.
- **CSV is legacy.** Anything in `data/` and `archive/legacy_csv_ingestion/` is kept for forensic reproducibility, not for new work.

---

## 12. License

MIT License. For questions or contributions, open an issue or pull request.
