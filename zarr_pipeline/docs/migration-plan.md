# NWP 資料 Migration 計畫與紀錄 — Bronze(Parquet)+ Silver(Zarr Mode A)

> 制定 2026-06-29,完成 2026-06-30。把既有 HF `.zarr.zip` 歸檔轉到目標架構的步驟、實測發現、
> 例外與保留策略。決議見 [2026-06-25-meeting_decisions.md](2026-06-25-meeting_decisions.md)、
> 格式分析見 [2026-06-25-feedback.md](2026-06-25-feedback.md)。
>
> **狀態:jma_msm + dwd_icon Silver 皆已完成並上傳(public);go-forward 待接。**

## 1. 目標架構

| 層 | 格式 | HF dataset | 角色 |
|---|---|---|---|
| **Bronze** | **Parquet**(per-run、region-scoped、zstd) | `apac-nwp-forecast-raw`(public) | 不可變原始(go-forward 起累積) |
| **Silver** | **Zarr Mode A** `(run_init, lead, lat, lon)`、**v3 sharding** | `apac-nwp-forecast`(public) | 服務/查詢、可重切 |
| Legacy | per-run `.zarr.zip` | `apac-nwp-forecast-zip`(public) | 歷史原始,冷封存(見 §10) |

- 未來資料:**Parquet(Bronze)→ `convert_to_zarr.py --source parquet` → Mode A(Silver)**,不再產 `.zarr.zip`。
- 分塊 grid-aware:`--target-mb`(預設 8MB)依網格算 tile(jma→**158×161**、dwd→**103×100**),lead 整段、run_init=1、**原生 dtype 保留**。
- **Silver 用 run-aligned sharding**(每 run 一個 shard,內含小 chunk):兼顧少檔(可上 HF)、點查/部分讀、以及 append(新 run = 加新 shard 檔,不動既有)。

## 2. 關鍵實測:S3 保留與 jma 缺口

| 模型 | S3 `--run` 可回抓(2026-06-28 測) | 結論 |
|---|---|---|
| **dwd_icon** | 回到 ~2026-03-30(90 天)OK,98 天 fail | S3 保留 ~3 個月 ✅ |
| **jma_msm** | 只到 ~2026-05-14;更早 `modelRunUnavailable` | jma 在 S3 有缺口(~3月底→05-13),**非**保留問題 ⚠️ |

`--run` 視窗每天往前滾。同日期 dwd 可抓、jma 不可 → jma 缺口確認在 S3 端。

## 3. 各模型歷史現況(HF archive `.zarr.zip`)

| 模型 | 檔數/大小 | 範圍 | 只能靠 `.zarr.zip`(S3 已無) | 永久不存在 |
|---|---|---|---|---|
| jma_msm | 363 / ~31.6GB | 2026-05-12T12 → 06-27T15 | 05-12、05-13 | 05-12 之前(缺口) |
| dwd_icon | 404 / ~216GB | 2026-03-19 → 06-27 | 03-19~03-29(~40 檔) | — |

## 4. 決策:歷史走「`.zarr.zip` → Silver migrate」,不重抓

重抓全歷史 ≈ 數十 GB / 十幾小時,且與既有 `.zarr.zip`(同資料)重複。所以**直接 migrate `.zarr.zip`**(無損、不依賴 S3)。Bronze=Parquet 只從 go-forward 起累積;歷史原始就是 `.zarr.zip`(medallion 允許 Bronze 跨期不同格式)。

## 5. 工具(`E3/scripts/` 與 `open-meteo/`)

| 工具 | 用途 |
|---|---|
| `convert_to_zarr.py` | Parquet/`.zarr.zip` → Mode A;`--source`/`--target-mb`/`--init`/`--append`/**`--model`**(覆寫誤標)、parquet 保留原生 dtype |
| `migration_precheck.py` | migrate 前逐檔健檢(可讀性、grid/model 一致性) |
| `migrate_jma_silver.sh` | jma `.zarr.zip` → Silver(下載+建+驗證,可續跑) |
| `migrate_dwd_silver.sh` | dwd `.zarr.zip` → Silver(**分批下載→append→刪批、並行下載、seed-init**,可續跑) |
| `reshard_silver.py` | 一般 re-shard(讀整個 cube + 寫,需 2× 磁碟) |
| `reshard_silver_lowdisk.py` | **省磁碟** re-shard(逐變數、邊刪源,峰值≈源+1變數);`--shard-runs` 可調 shard 粗細 |
| `paced_upload.py` | **節流批次上傳**(分批 commit + sleep + 尊重 Retry-After),免費帳號速率限制下穩定 |
| `go_forward.sh` | go-forward per-run 範本(export→parquet→bronze→月 cube),見 §9 |
| `postprocess.py`(open-meteo) | raw export → archive parquet(加 run_init/scraped_at、int-cast、去 NaN) |

## 6. jma_msm — ✅ 完成

- 363 runs 全寫入,`(run_init=400, lead=78, 473×481)`,grid 2026-05-12→07-01、step 3h、lead-max 78。
- chunk `[1,78,158,161]`、原生 dtype。無損驗證最大差 = 0。
- Sharding:42,867 → **5,115 檔**,27GB。
- 上傳:`apac-nwp-forecast`(public),**5,115 檔 / 28.8GB**,本機↔HF 一致 ✅。本機已清。

## 7. dwd_icon — ✅ 完成(踩了三個雷,都處理掉)

- `migrate_dwd_silver.sh` 分批 migrate 404 檔 `.zarr.zip` → `(run_init=416, lead=180, 721×497)`,grid 2026-03-19→07-01、step 6、lead-max 180、chunk `[1,180,103,100]`。filled 404/416。
- Sharding(`reshard_silver_lowdisk.py`,逐變數省磁碟):6,475 檔 / 176GB。
- 上傳:`apac-nwp-forecast`(public),**6,475 檔 / 176GB**,從 HF 遠端開 cube 驗證可讀(model/變數/維度/抽點 lazy 讀)✅。本機已清。

**三個資料品質/工程問題與解法:**
1. **`model` 屬性誤標 `jma_msm`**(根因:`open-meteo/parquet_to_zarr_cube.py` 舊版硬寫 jma_msm fallback,已修)→ migrate 時用 **`--model dwd_icon`** 覆寫。
2. **snow 變數中途才加**(03-19~21 = 13 變數無 snow;03-22+ = 15 變數)→ **seed-init**:用一個 15 變數的 run 先建 cube(superset),13 變數舊 run 子集寫入、snow 留缺值哨兵(那 3 天 snow 哪裡都沒有,不可得)。
3. **HF 免費帳號限制**(見 §8)。

## 8. HF 免費帳號限制與上傳策略(重要實戰)

dwd 上傳踩到三道牆(jma 因為小,全部繞過):

| 限制 | 數字 | 解法 |
|---|---|---|
| Private 儲存 | **100GB** | **改 public**(public 寬鬆,跟 archive 一致) |
| API 請求率 | **1000 / 5 分** | 少檔(sharding)+ 節流上傳 |
| Commit 率 | **128 / 小時** | 分批 commit(一批多檔)+ sleep |

- **`hf upload-large-folder` 會雪崩**:多 worker 一遇 429 就猛重試,把額度燒光、空轉數小時。**改用 `paced_upload.py`**(分批 commit + sleep + 尊重 Retry-After)→ dwd 6475 檔穩定傳完(89 次 429 全被退讓處理)。
- **上傳一律用 huggingface_hub 1.19.0**(`open-meteo/.venv-ocean-ab/bin/python`)。miniconda 的 **0.36.2** LFS preupload 有 bug 會 HTTP/2 reset。
- **改 public**:`HfApi().update_repo_settings(repo, repo_type='dataset', private=False)`。

## 9. Go-forward — ✅ 完成(單一 cube,每日增量 append)

**設計定案:不分歷史/ongoing,就一個 cube** —— 把歷史 cube 的 run_init 軸**預先延長到 2028**,未來每天的 run 直接 region-write 進對應空槽。歷史在前、未來空槽接著,查詢就是 `open_zarr(一個 cube)`,不用 `open_mfdataset` 串。

**① 軸延長(已做,metadata-only)** — `scripts/extend_hf_cube.sh`、`scripts/extend_run_init.py`
- 只下載 cube skeleton(zarr.json + 座標 + slot_filled,幾 MB)→ resize run_init 維度 + 重寫 run_init/slot_filled 小陣列 → 只傳回 metadata。
- **資料 chunk 完全不碰**;sharding codec 保留;空槽不佔磁碟。實測:
  - jma:400 → **4792 格**(→2027-12-31),+11 metadata 檔,28.8GB 未動,filled 仍 363。
  - dwd:416 → **2612 格**(→2027-12-31),+6 metadata 檔,176GB 未動,filled 仍 404。
- 關鍵實作:`zarr.open_group(use_consolidated=False)` 才能讓 resize 持久化 + 之後 `consolidate_metadata` 重掃;run_init 用 numpy 自行做 CF 編碼(免 cftime)並 round-trip 驗證。

**② 每日 append(已做,增量)** — `scripts/go_forward.sh <model> [RUN_ISO | --backfill N]`

三種模式:
- `go_forward.sh jma_msm` — 只抓**最新**一個 run。
- `go_forward.sh jma_msm --backfill N` — **自動補洞**:讀 cube 的 slot_filled,找出最近 N 天「該有但還沒填」的 run,逐個補(防漏跑、防 export 失敗)。← **推薦的每日 cron**
- `go_forward.sh dwd_icon 2026-06-30T06` — 指定某個 run。

```
流程(每個要處理的 run):export → postprocess → parquet
  ├─ 上傳 parquet 到 bronze dataset
  └─ 下載 cube skeleton(一次,幾 MB,無 data chunk)
     → --append(region-write 進空槽,冪等:已填跳過)→ 只產生該 run 的 shard 檔
     → 最後一次 commit 上傳所有新 shard + slot_filled
```
- **本機不存完整 cube**(只暫存 skeleton,跑完即刪);**HF 既有 28.8/176GB 永不重傳**。
- backfill 對「未來/S3 沒有」的 run 自動跳過,不中斷。
- 實測:單 run(jma 20260630T09Z)filled 363→364、HF +14 檔;`--backfill 2`(jma)補 14 個缺口 run、filled 364→378、抽點 78/78 無損、一次 commit 上傳 196 檔。

**排 cron(一天一次,自動補最近 3 天的洞 — 最簡):**
```
0 6 * * *  cd <E3> && bash scripts/go_forward.sh jma_msm  --backfill 3
0 7 * * *  cd <E3> && bash scripts/go_forward.sh dwd_icon --backfill 3
```
- N 選 3~7 即可(S3 保留 ~3 個月,理論上補得更久);一天一次 = 資料最多延遲 ~1 天進 cube,對歸檔/訓練足夠。要即時可另加高頻「抓最新」cron。
- ⚠️ **2028 前要再跑一次 `extend_hf_cube.sh <cube> 2030-01-01 <step>`**(一樣 metadata-only)。

## 9b. 已驗證無損（2026-07-01）

- **格式健檢(從 HF 讀兩個 cube)**:Mode A 結構、model 屬性(jma_msm / **dwd_icon** 正確)、13/15 變數、網格/lead、run_init 軸延長到 2028 且 step 連續、原生 dtype(uint8/uint16/float32)、sharding(inner chunk 158×161 / 103×100)、抽樣有真值、空槽為哨兵 —— 全部 ✓。
- **位元級無損對比(silver vs 原始 `.zarr.zip`,`mask_and_scale=False` 比原生整數)**:
  jma 78h/39h、dwd 180h/120h(含 snow),共 4 個 run × 13/15 變數,**最大差 = 0.0、缺值格局完全一致**。

## 9c. 退役舊自動化(⚠️ 注意事項)

舊 production cron(`open-meteo/deploy_mac_mini.sh` 裝的):每天 06:00 UTC `MODELS="jma_msm dwd_icon" sweep_data_run.sh` → 找 archive 缺的 run → export → `.zarr.zip` → 傳 archive。**功能等同 `go_forward.sh --backfill`**,故可退役。但:
- **🚨 保留 himawari cron**(`himawari_daily.sh` → `apac-himawari-swr`,卫星資料,go_forward 未涵蓋)。
- **gfs/ecmwf 從未實際歸檔**(archive 只有 jma+dwd);未來要加需擴充 go_forward。
- 退役後 archive **凍結**(不再有新 `.zarr.zip`),符合 §10 冷封存。

## 10. 保留策略 + `.zarr.zip` 退役準則

- **Bronze parquet / Silver Mode A**:永久。
- **`.zarr.zip` archive**:**先不要刪** —— Silver 雖無損衍生,但 `.zarr.zip` 是歷史**唯一原始**(尤其 dwd 03-19~29、jma 早期,S3 已無)。
- **退役準則(全部滿足才考慮刪,且建議只刪「已驗證可從 Silver 完整重建」的部分)**:
  1. 下游(ML/查詢)用 Silver 跑過、確認資料正確;
  2. 確認該 run 已在 Silver(filled)且抽樣無損;
  3. 真要回收空間,優先「冷封存到更便宜的儲存」而非直接刪。
- **鐵則:刪任何 `.zarr.zip` 前,先確認該 run 已在 Silver 或 S3 仍可重抓。**

## 11. Open items

- [x] jma_msm / dwd_icon `.zarr.zip` → Silver(無損)+ sharding + 上傳(public)。
- [x] Silver 改 public;建 `apac-nwp-forecast-raw`(public)。
- [x] 修 `parquet_to_zarr_cube.py` 的 model 誤標 bug;converter 加 `--model`。
- [x] 清本機 cube + 工作區 log。
- [x] **單一 cube 設計**:run_init 軸延長到 2028(jma/dwd 皆完成,metadata-only)。
- [x] **go-forward 完成並驗證**:`go_forward.sh` 每日增量 append(jma 實測通過;dwd 同流程)。
- [ ] **排 cron**(jma 每 3h、dwd 每 6h);**2028 前再延長軸一次**。
- [ ] 來源端把舊管線重跑/修正,讓 archive 的 dwd `model` 屬性也正確(目前靠讀取時覆寫)。
- [ ] 下游驗證後,依 §10 準則決定 `.zarr.zip` archive 去留。
- [ ] (可選)若要真正高頻/即時 append serving,評估搬 S3/GCS + `append_dim`/Icechunk(業界標準;HF 適合「每日一次」節奏)。

## 12. 工具補充(go-forward 相關)

| 工具 | 用途 |
|---|---|
| `extend_run_init.py` | 延長 cube 的 run_init 軸(本機,metadata-only;resize + 重編 run_init 座標 + slot_filled) |
| `extend_hf_cube.sh` | 在 HF 上延長軸(下載 skeleton → extend → 只傳回 metadata,資料不碰) |
| `go_forward.sh` | 每日 per-run:export → bronze parquet → 增量 append 進延長後的單一 silver cube |
| `paced_upload.py` | 節流批次上傳(大量檔上傳用;go-forward 日常增量不需要) |
