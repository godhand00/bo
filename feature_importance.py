"""
スパッタ成膜パラメータの寄与度分析（ランダムフォレスト）

各膜特性（成膜レート、屈折率、表面粗さ、膜応力）に対して、
成膜パラメータがどの程度影響しているかをランダムフォレストの
特徴量重要度で可視化する。

2種類の重要度を算出:
  1. 不純度ベース重要度 (MDI: Mean Decrease in Impurity)
     - ツリー内での分岐に使われた際の不純度の減少量の合計
     - 計算が速いが、連続値・高カーディナリティの特徴量を過大評価する傾向あり

  2. 順列重要度 (Permutation Importance)
     - ある特徴量の値をシャッフルしたときにスコアがどれだけ低下するかを測定
     - 計算は遅いが、より信頼性の高い重要度が得られる
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

# 日本語フォント設定
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# japanize_matplotlib が rcParams を更新しない場合のフォールバック
import matplotlib.font_manager as fm

_jp_font_path = None
for f in fm.fontManager.ttflist:
    if "IPAexGothic" in f.name or "ipaexg" in f.fname:
        _jp_font_path = f.fname
        break

if _jp_font_path is None:
    # japanize-matplotlib のバンドルフォントを直接探す
    import importlib.util

    spec = importlib.util.find_spec("japanize_matplotlib")
    if spec and spec.origin:
        from pathlib import Path as _P

        _font_dir = _P(spec.origin).parent / "fonts"
        _candidate = _font_dir / "ipaexg.ttf"
        if _candidate.exists():
            _jp_font_path = str(_candidate)
            fm.fontManager.addfont(_jp_font_path)

if _jp_font_path:
    _font_name = fm.FontProperties(fname=_jp_font_path).get_name()
    plt.rcParams["font.family"] = _font_name
    plt.rcParams["axes.unicode_minus"] = False
else:
    warnings.warn(
        "日本語フォントが見つかりません。グラフの日本語表示が文字化けする可能性があります。"
    )

# ============================================================
# 定数
# ============================================================
FEATURE_COLUMNS = [
    "rf_power_W",
    "pressure_Pa",
    "substrate_temp_C",
    "ar_flow_sccm",
    "target_substrate_distance_mm",
]

FEATURE_LABELS = [
    "RF出力 (W)",
    "Ar圧力 (Pa)",
    "基板温度 (°C)",
    "Ar流量 (sccm)",
    "T-S間距離 (mm)",
]

TARGET_COLUMNS = [
    "deposition_rate_nm_per_min",
    "refractive_index",
    "surface_roughness_Ra_nm",
    "film_stress_MPa",
]

TARGET_LABELS = [
    "成膜レート (nm/min)",
    "屈折率",
    "表面粗さ Ra (nm)",
    "膜応力 (MPa)",
]


def train_and_evaluate(
    df: pd.DataFrame,
    n_estimators: int = 200,
    seed: int = 42,
) -> dict:
    """各膜特性に対してランダムフォレストを学習し、重要度を算出する。

    Returns
    -------
    results : dict
        各ターゲットごとの {
            "model": fitted RandomForestRegressor,
            "mdi_importance": array,
            "perm_importance_mean": array,
            "perm_importance_std": array,
            "cv_r2": float (5-fold CV R² の平均),
        }
    """
    X = df[FEATURE_COLUMNS].values
    results = {}

    for col, label in zip(TARGET_COLUMNS, TARGET_LABELS):
        y = df[col].values

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X, y)

        # 5-fold CV R²
        cv_scores = cross_val_score(
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=seed,
                n_jobs=-1,
            ),
            X,
            y,
            cv=5,
            scoring="r2",
        )

        # 順列重要度
        perm_imp = permutation_importance(
            rf, X, y, n_repeats=20, random_state=seed, n_jobs=-1
        )

        results[col] = {
            "label": label,
            "model": rf,
            "mdi_importance": rf.feature_importances_,
            "perm_importance_mean": perm_imp.importances_mean,
            "perm_importance_std": perm_imp.importances_std,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
        }

        print(f"\n{'='*50}")
        print(f"【{label}】")
        print(f"  5-fold CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  {'パラメータ':<16} {'不純度重要度':>10} {'順列重要度':>10}")
        print(f"  {'-'*16} {'-'*10} {'-'*10}")
        for i, feat_label in enumerate(FEATURE_LABELS):
            print(
                f"  {feat_label:<14} {rf.feature_importances_[i]:>10.4f}"
                f" {perm_imp.importances_mean[i]:>10.4f}"
                f" +/- {perm_imp.importances_std[i]:.4f}"
            )

    return results


def plot_importance(
    results: dict,
    output_dir: str = "results",
) -> None:
    """特徴量重要度のプロットを生成する。"""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # --- 1. 不純度ベース重要度（4つの膜特性を横並び） ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, (col, info), color in zip(axes, results.items(), colors):
        sorted_idx = np.argsort(info["mdi_importance"])
        ax.barh(
            range(len(FEATURE_LABELS)),
            info["mdi_importance"][sorted_idx],
            color=color,
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_yticks(range(len(FEATURE_LABELS)))
        ax.set_yticklabels([FEATURE_LABELS[i] for i in sorted_idx])
        ax.set_xlabel("不純度ベース重要度")
        ax.set_title(f'{info["label"]}\n(R²={info["cv_r2_mean"]:.3f})', fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "ランダムフォレストによるパラメータ寄与度（不純度ベース）",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "feature_importance_mdi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n不純度ベース重要度プロット → {out / 'feature_importance_mdi.png'}")

    # --- 2. 順列重要度（4つの膜特性を横並び） ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for ax, (col, info), color in zip(axes, results.items(), colors):
        sorted_idx = np.argsort(info["perm_importance_mean"])
        ax.barh(
            range(len(FEATURE_LABELS)),
            info["perm_importance_mean"][sorted_idx],
            xerr=info["perm_importance_std"][sorted_idx],
            color=color,
            alpha=0.85,
            edgecolor="white",
            capsize=3,
        )
        ax.set_yticks(range(len(FEATURE_LABELS)))
        ax.set_yticklabels([FEATURE_LABELS[i] for i in sorted_idx])
        ax.set_xlabel("順列重要度")
        ax.set_title(f'{info["label"]}\n(R²={info["cv_r2_mean"]:.3f})', fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "ランダムフォレストによるパラメータ寄与度（順列重要度）",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "feature_importance_perm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"順列重要度プロット → {out / 'feature_importance_perm.png'}")

    # --- 3. ヒートマップ（全膜特性 × 全パラメータ） ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, imp_key, title in [
        (axes[0], "mdi_importance", "不純度ベース重要度"),
        (axes[1], "perm_importance_mean", "順列重要度"),
    ]:
        matrix = np.array(
            [results[col][imp_key] for col in TARGET_COLUMNS]
        )
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(len(FEATURE_LABELS)))
        ax.set_xticklabels(FEATURE_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(TARGET_LABELS)))
        ax.set_yticklabels(TARGET_LABELS)
        ax.set_title(title)

        # セル内に数値を表示
        for i in range(len(TARGET_LABELS)):
            for j in range(len(FEATURE_LABELS)):
                val = matrix[i, j]
                text_color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    color=text_color, fontsize=9,
                )

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "パラメータ寄与度ヒートマップ",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out / "feature_importance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ヒートマップ → {out / 'feature_importance_heatmap.png'}")


def save_importance_csv(
    results: dict,
    output_dir: str = "results",
) -> None:
    """重要度をCSVに保存する。"""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    rows = []
    for col in TARGET_COLUMNS:
        info = results[col]
        for i, feat in enumerate(FEATURE_LABELS):
            rows.append(
                {
                    "膜特性": info["label"],
                    "パラメータ": feat,
                    "不純度ベース重要度": round(info["mdi_importance"][i], 4),
                    "順列重要度（平均）": round(info["perm_importance_mean"][i], 4),
                    "順列重要度（標準偏差）": round(info["perm_importance_std"][i], 4),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = out / "feature_importance.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n重要度CSV → {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ランダムフォレストによるスパッタ成膜パラメータの寄与度分析"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/full_dataset.csv",
        help="分析対象データセットのCSVパス (default: data/full_dataset.csv)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="ランダムフォレストの決定木の数 (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="出力ディレクトリ (default: results)",
    )
    args = parser.parse_args()

    # データ読み込み
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"データファイルが見つかりません: {data_path}")
        print("先に generate_dataset.py を実行してください。")
        return

    df = pd.read_csv(data_path)
    print(f"データ読み込み: {data_path} ({len(df)} サンプル)")

    # 学習と重要度算出
    results = train_and_evaluate(
        df,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )

    # 可視化とCSV出力
    plot_importance(results, output_dir=args.output_dir)
    save_importance_csv(results, output_dir=args.output_dir)

    print("\n完了しました。")


if __name__ == "__main__":
    main()
