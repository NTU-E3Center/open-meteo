# Open-Meteo 輻射內插方法說明（3 小時 → 1 小時）

> 整理自 open-meteo 原始碼（2026-06-12），核心實作在
> `Sources/App/Helper/Interpolation.swift:22-104`（`interpolateSolarBackwards`）。
> 本文件描述太陽輻射變數（GHI、直射、散射）如何從模型的 3 小時原生輸出
> 內插成資料庫的 1 小時步長。

## 1. 什麼時候會發生內插

`.om` 資料庫一律以 **1 小時**步長儲存（`dtSeconds = 3600`），但模型原生輸出
在預報後段會降到 3 小時一步。內插發生在**下載時**（資料寫入 `.om` 之前），
所以你從 `export` / API 拿到的尾段逐時值都是內插結果：

| 模型 | 原生逐時段 | 內插段（3h → 1h） |
|---|---|---|
| `dwd_icon` | 0–78h | **81–180h** |
| `ncep_gfs013` | 0–119h | **120–384h** |
| `jma_msm` | 全程（0–78h） | 無（不經內插）|

適用的變數型別為 `solar_backwards_averaged`（ICON：`IconVariable.swift:419-422,517-518`；
GFS：`GfsVariable.swift:256-258`），涵蓋 `shortwave_radiation`、`direct_radiation`、
`diffuse_radiation`（及其衍生的 DNI、GTI）。

## 2. 為什麼不能直接對 W/m² 內插

輻射的日週期形狀（日出 → 正午峰 → 日落 → 夜間 0）是**太陽幾何決定的剛性結構**，
週期 24 小時。3 小時取樣對這個訊號而言太疏，直接對數值做曲線內插會：

- 把正午峰削平、日出日落時刻拉歪；
- 夜間與白天交界產生負值或假亮度；
- 系統性低估每日總輻射量（對太陽能發電量估算是致命的）。

因此 Open-Meteo 改為內插「**晴空指數 kt**」——把「太陽幾何」與「雲遮蔽」兩個
成分拆開：太陽幾何用天文公式精確重建，只對變化平滑的雲遮蔽成分做內插。

## 3. 演算法逐步說明

記號：模型 3 小時值為 backwards-averaged（該時間戳代表「前一段時間的平均」）。

### Step 1 — 計算大氣層頂參考輻射（`Interpolation.swift:25-26`）

用內建太陽位置演算法（`Zensun.calculateRadiationBackwardsAveraged`，NREL SPA 簡化版）
對該格點分別算出：

- `solarLow[]`：3 小時網格上的大氣層頂 backwards-averaged 輻射
- `solar[]`：1 小時網格上的同一量

這是「該時刻物理上可能的最大輻射」，已含日地距離、赤緯、時角等因素。

### Step 2 — 換算晴空指數 kt（`Interpolation.swift:62-70`）

對內插窗口的四個相鄰 3 小時點 A、B、C、D：

```
kt = min( 模型輻射 / 大氣層頂輻射 , 0.95 × 太陽常數上限 )
```

kt ∈ [0,1]，物理意義是「大氣（主要是雲）放行了多少比例的輻射」。

邊界保護：

- **上限**：輻射不得超過大氣層頂的 95%（`radLimit`，`:33`）；
- **低仰角**：大氣層頂輻射 < 5 W/m² 時（日出日落邊緣），除法不穩定，
  該點 kt 設為 NaN，再從相鄰時段借值填補（`:71-83` 的鄰近填補鏈）。

### Step 3 — 對 kt 做 Hermite（Catmull-Rom）三次內插（`Interpolation.swift:91-95`）

```
a = -ktA/2 + 3·ktB/2 − 3·ktC/2 + ktD/2
b =  ktA − 5·ktB/2 + 2·ktC − ktD/2
c = -ktA/2 + ktC/2
d =  ktB
kt(f) = a·f³ + b·f² + c·f + d        # f ∈ [0,1) 為窗口內的小數位置
```

時間索引帶**半步偏移**（`:40`，`+ dtOld/2 − dt/2`），因為新舊值都是區間平均、
代表點在區間中心而非端點。

### Step 4 — 乘回每小時的太陽幾何（`Interpolation.swift:96`）

```
小時輻射 = kt(f) × solar[i]      # solar[i] = 該小時的大氣層頂輻射
```

夜間（`solar[i] == 0`）直接回傳 0（`:55-57`），不會出現夜間假輻射。

### Step 5 — 兩個收尾保護（`Interpolation.swift:97-103`）

- **負值回退**：kt 變化過快時三次曲線可能過衝成負值，此時退回對 kt 的
  **線性**內插（`:97-101`）；
- **量化對齊**：結果按變數的 scalefactor 四捨五入（`:103`），避免內插值
  呈現比原始資料更高的假精度。

## 4. 品質特性總結

| 面向 | 內插段的可信度 |
|---|---|
| 日週期形狀（日出/日落時刻、正午峰位置） | ✅ 精確（由天文計算重建，非內插產物） |
| 每日總輻射量 | ✅ 良好（kt 平滑假設下守恆性佳） |
| 雲造成的小時級波動 | ⚠️ **無資訊**——3 小時內的雲變化被平滑掉，這是內插無法恢復的 |
| 夜間零值 | ✅ 嚴格為 0 |
| 數值邊界 | ✅ 不會超過大氣層頂 95%、負值有回退保護 |

## 5. 對太陽能 ML 應用的建議

1. **特徵工程**：建議加一個 `is_interpolated` 旗標（ICON：lead > 78h；
   GFS：lead > 120h），讓模型知道尾段輻射的雲波動資訊量較低。
2. **不要用內插段評估小時級雲事件**（雲隙波動、瞬時遮蔽）——那段資料
   物理形狀正確但高頻內容是合成的。
3. **每日彙總層級**（日發電量、日輻射量）在內插段仍然可靠。
4. `jma_msm` 全程原生逐時，短期（≤39–78h）應用中它的輻射時間結構
   是三個模型中最真實的——但注意其直射/散射為經驗拆分
   （`JmaController.swift:253-256`，GHI 本身為原生）。

## 6. 相關程式碼索引

| 內容 | 位置 |
|---|---|
| 內插方法分派（每變數一種） | `Sources/App/Helper/Interpolation.swift:5-20` |
| 太陽輻射內插主體 | `Sources/App/Helper/Interpolation.swift:22-104` |
| 太陽位置/大氣層頂輻射 | `Sources/App/Helper/Solar/Zensun.swift` |
| ICON 變數 → 內插型別對照 | `Sources/App/Icon/IconVariable.swift:331+` |
| GFS 變數 → 內插型別對照 | `Sources/App/Gfs/GfsVariable.swift:20-21,250+` |
| 各模型原生步長（內插段範圍） | `Icon.swift:153`、`GfsDomain.swift:203`、`JmaDownloader.swift:569-570` |
| 降水的總量守恆拆分（對照用） | `Interpolation.swift:107+`（`backwardsSum`） |
