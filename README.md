# スパッタ成膜条件のベイズ最適化

RF スパッタリングによる SiO₂ 薄膜の成膜条件を、ベイズ最適化で探索するサンプルプログラムです。

## セットアップ

uv を使用して環境構築を行います。

```bash
# 仮想環境作成と依存関係のインストール
uv sync
```

## ファイル構成

```
bo/
├── README.md                     # このファイル
├── pyproject.toml                # プロジェクト設定・依存パッケージ定義 (uv)
├── uv.lock                       # 依存関係ロックファイル
├── .gitignore                    # Git除外設定
├── requirements.txt              # requirements.txt (参考用)
├── generate_dataset.py           # 合成データセット生成
├── bayesian_optimization.py      # ベイズ最適化メインスクリプト
├── data/                         # 生成されたデータ
│   ├── initial_experiments.csv   # 初期実験データ (20点)
│   └── full_dataset.csv          # 全データセット (200点)
└── results/                      # 最適化結果の出力先
    ├── optimization_history.csv  # 評価履歴
    ├── convergence.png           # 収束プロット
    ├── objective_landscape.png   # 目的関数の部分依存プロット
    └── property_evolution.png    # 膜特性の推移プロット
```

## 成膜パラメータ（探索空間）

| パラメータ | 単位 | 範囲 | 説明 |
|-----------|------|------|------|
| RF出力 | W | 50 ~ 300 | スパッタリングの投入電力 |
| Ar圧力 | Pa | 0.1 ~ 2.0 | チャンバー内ガス圧力 |
| 基板温度 | °C | 25 ~ 400 | 基板加熱温度 |
| Ar流量 | sccm | 5 ~ 50 | アルゴンガス流量 |
| T-S間距離 | mm | 40 ~ 120 | ターゲット-基板間距離 |

## 最適化対象の膜特性

| 特性 | 単位 | 目標 |
|------|------|------|
| 成膜レート | nm/min | 最大化 |
| 屈折率 (@633nm) | - | 1.46 (SiO₂バルク値) に近づける |
| 表面粗さ Ra | nm | 最小化 |
| 膜応力 | MPa | 絶対値を最小化 |

## 使い方

### 1. データセットの生成

```bash
uv run python generate_dataset.py
```

`data/` ディレクトリに初期実験データと検証用データが生成されます。

### 2. ベイズ最適化の実行

**基本的な実行（ランダム探索から開始）:**

```bash
uv run python bayesian_optimization.py
```

**初期実験データを活用して最適化:**

```bash
uv run python bayesian_optimization.py --initial-data data/initial_experiments.csv --n-calls 50
```

**オプション一覧:**

```bash
uv run python bayesian_optimization.py --help
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--n-calls` | 50 | 総評価回数 |
| `--n-initial` | 10 | 初期ランダム探索の点数 |
| `--initial-data` | None | 初期実験データCSVのパス |
| `--acq-func` | EI | 獲得関数 (EI/PI/LCB/gp_hedge) |
| `--seed` | 42 | 乱数シード |
| `--output-dir` | results | 出力ディレクトリ |

## 目的関数について

4つの膜特性を重み付きスカラー値に統合しています:

```
score = - w_rate × rate_norm
        + w_roughness × roughness_norm
        + w_ri × |RI - 1.46|_norm
        + w_stress × |stress|_norm
```

各重みは `bayesian_optimization.py` 内の `WEIGHTS` 辞書で変更可能です。

## カスタマイズのヒント

- **膜材料の変更**: `generate_dataset.py` の `simulate_deposition()` を修正
- **探索空間の変更**: `bayesian_optimization.py` の `search_space` を修正
- **目的関数の重みの変更**: `WEIGHTS` 辞書を修正
- **獲得関数の比較**: `--acq-func` オプションで EI, PI, LCB, gp_hedge を切り替え
- **実際の実験データの利用**: CSVを `--initial-data` で指定（カラム名を合わせること）

## 獲得関数の選択ガイド

| 獲得関数 | 特徴 |
|---------|------|
| **EI** (Expected Improvement) | 探索と活用のバランスが良い。一般的に推奨 |
| **PI** (Probability of Improvement) | 活用（exploitation）寄り。局所的に探索 |
| **LCB** (Lower Confidence Bound) | κパラメータで探索/活用を制御可能 |
| **gp_hedge** | 複数の獲得関数を自動切り替え |
