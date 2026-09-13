"""
スパッタ成膜条件のベイズ最適化プログラム

scikit-optimize (skopt) を用いて、スパッタ成膜の条件を最適化する。

最適化目標:
  成膜レートを最大化しつつ、表面粗さを最小化し、
  屈折率をバルクSiO2の値 (1.46) に近づける。

  → スカラー目的関数として統合:
    score = w1 * rate_normalized
            - w2 * roughness_normalized
            - w3 * |refractive_index - 1.46|_normalized
            - w4 * |stress|_normalized

    (最大化 = minimize(-score))
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from generate_dataset import simulate_deposition

# japanize-matplotlib がインストールされていれば日本語フォントを使用
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    warnings.warn(
        "japanize-matplotlib がインストールされていません。"
        "グラフの日本語表示が文字化けする可能性があります。"
    )


# ============================================================
# 探索空間の定義
# ============================================================
search_space = [
    Integer(50, 300, name="rf_power_W"),
    Real(0.1, 2.0, name="pressure_Pa"),
    Integer(25, 400, name="substrate_temp_C"),
    Real(5.0, 50.0, name="ar_flow_sccm"),
    Integer(40, 120, name="target_substrate_distance_mm"),
]

# ============================================================
# 目的関数の重み設定
# ============================================================
# 各指標を正規化してから重み付けする
# 正規化の基準値（おおよそのデータ範囲から設定）
RATE_RANGE = (0.5, 30.0)         # nm/min
ROUGHNESS_RANGE = (0.1, 3.0)     # nm
RI_TARGET = 1.46                  # SiO2 バルク値
RI_TOL = 0.03                    # 屈折率の許容偏差
STRESS_RANGE = (0, 400)          # |MPa|

# 重み（合計=1にする必要はないが、相対的な重要度を示す）
WEIGHTS = {
    "rate": 1.0,        # 成膜レート（大きいほど良い）
    "roughness": 1.0,   # 表面粗さ（小さいほど良い）
    "ri_dev": 1.5,      # 屈折率の目標値からの偏差（小さいほど良い）
    "stress": 0.5,      # 膜応力の絶対値（小さいほど良い）
}


def normalize(value: float, vmin: float, vmax: float) -> float:
    """0-1 に正規化"""
    return np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)


# シミュレーション用 RNG（再現性のため）
_sim_rng = np.random.default_rng(2024)

# 評価履歴を保持するリスト
evaluation_history: list[dict] = []


@use_named_args(search_space)
def objective(
    rf_power_W: int,
    pressure_Pa: float,
    substrate_temp_C: int,
    ar_flow_sccm: float,
    target_substrate_distance_mm: int,
) -> float:
    """ベイズ最適化の目的関数（最小化）。

    成膜シミュレーションを実行し、複合スコアを返す。
    skopt は最小化するため、良いスコアほど小さい値を返す。
    """
    result = simulate_deposition(
        power=float(rf_power_W),
        pressure=float(pressure_Pa),
        substrate_temp=float(substrate_temp_C),
        ar_flow=float(ar_flow_sccm),
        distance=float(target_substrate_distance_mm),
        rng=_sim_rng,
    )

    rate = result["deposition_rate_nm_per_min"]
    roughness = result["surface_roughness_Ra_nm"]
    ri = result["refractive_index"]
    stress = result["film_stress_MPa"]

    # 正規化
    rate_norm = normalize(rate, *RATE_RANGE)
    roughness_norm = normalize(roughness, *ROUGHNESS_RANGE)
    ri_dev_norm = min(abs(ri - RI_TARGET) / RI_TOL, 1.0)
    stress_norm = normalize(abs(stress), *STRESS_RANGE)

    # 複合スコア（最小化 = 成膜レート最大化 + 粗さ最小化 + ...）
    score = (
        -WEIGHTS["rate"] * rate_norm
        + WEIGHTS["roughness"] * roughness_norm
        + WEIGHTS["ri_dev"] * ri_dev_norm
        + WEIGHTS["stress"] * stress_norm
    )

    # 履歴に記録
    evaluation_history.append(
        {
            "iteration": len(evaluation_history) + 1,
            "rf_power_W": rf_power_W,
            "pressure_Pa": round(pressure_Pa, 3),
            "substrate_temp_C": substrate_temp_C,
            "ar_flow_sccm": round(ar_flow_sccm, 1),
            "target_substrate_distance_mm": target_substrate_distance_mm,
            "deposition_rate_nm_per_min": rate,
            "refractive_index": ri,
            "surface_roughness_Ra_nm": roughness,
            "film_stress_MPa": stress,
            "objective_score": round(score, 4),
        }
    )

    return score


def load_initial_data(csv_path: str) -> tuple[list[list], list[float]]:
    """初期実験データをCSVから読み込み、skoptのx0/y0形式に変換する。"""
    df = pd.read_csv(csv_path)
    x0 = []
    y0 = []

    for _, row in df.iterrows():
        x = [
            int(row["rf_power_W"]),
            float(row["pressure_Pa"]),
            int(row["substrate_temp_C"]),
            float(row["ar_flow_sccm"]),
            int(row["target_substrate_distance_mm"]),
        ]
        x0.append(x)

        # 目的関数を再計算（ノイズありだが、実際には実験値をそのまま使う）
        result = {
            "deposition_rate_nm_per_min": row["deposition_rate_nm_per_min"],
            "refractive_index": row["refractive_index"],
            "surface_roughness_Ra_nm": row["surface_roughness_Ra_nm"],
            "film_stress_MPa": row["film_stress_MPa"],
        }

        rate_norm = normalize(result["deposition_rate_nm_per_min"], *RATE_RANGE)
        roughness_norm = normalize(result["surface_roughness_Ra_nm"], *ROUGHNESS_RANGE)
        ri_dev_norm = min(abs(result["refractive_index"] - RI_TARGET) / RI_TOL, 1.0)
        stress_norm = normalize(abs(result["film_stress_MPa"]), *STRESS_RANGE)

        score = (
            -WEIGHTS["rate"] * rate_norm
            + WEIGHTS["roughness"] * roughness_norm
            + WEIGHTS["ri_dev"] * ri_dev_norm
            + WEIGHTS["stress"] * stress_norm
        )
        y0.append(score)

    return x0, y0


def run_optimization(
    n_calls: int = 50,
    n_initial_points: int = 10,
    initial_data_path: str | None = None,
    acq_func: str = "EI",
    seed: int = 42,
) -> object:
    """ベイズ最適化を実行する。

    Parameters
    ----------
    n_calls : int
        最適化の総評価回数（初期ランダム探索を含む）
    n_initial_points : int
        初期ランダム探索の点数（initial_data_path指定時は0に設定される）
    initial_data_path : str or None
        初期実験データのCSVパス（Noneの場合はランダム探索から開始）
    acq_func : str
        獲得関数。 'EI' (Expected Improvement), 'PI' (Probability of Improvement),
        'LCB' (Lower Confidence Bound) など
    seed : int
        乱数シード

    Returns
    -------
    result : OptimizeResult
        最適化結果
    """
    kwargs = {
        "func": objective,
        "dimensions": search_space,
        "acq_func": acq_func,
        "n_calls": n_calls,
        "random_state": seed,
        "verbose": True,
        "n_jobs": 1,
    }

    if initial_data_path and Path(initial_data_path).exists():
        print(f"初期データを読み込み中: {initial_data_path}")
        x0, y0 = load_initial_data(initial_data_path)
        kwargs["x0"] = x0
        kwargs["y0"] = y0
        kwargs["n_initial_points"] = 0  # 初期データがあるのでランダム探索不要
        print(f"  → {len(x0)} 点の初期データを使用")
    else:
        kwargs["n_initial_points"] = n_initial_points
        print(f"初期データなし。{n_initial_points} 点のランダム探索から開始")

    print(f"\n{'='*60}")
    print(f"ベイズ最適化開始")
    print(f"  獲得関数: {acq_func}")
    print(f"  総評価回数: {n_calls}")
    print(f"{'='*60}\n")

    result = gp_minimize(**kwargs)

    return result


def print_results(result) -> None:
    """最適化結果を表示する。"""
    print(f"\n{'='*60}")
    print("最適化結果")
    print(f"{'='*60}")

    param_names = [
        "RF出力 (W)",
        "Ar圧力 (Pa)",
        "基板温度 (°C)",
        "Ar流量 (sccm)",
        "T-S間距離 (mm)",
    ]

    print("\n【最適条件】")
    for name, val in zip(param_names, result.x):
        if isinstance(val, float):
            print(f"  {name}: {val:.3f}")
        else:
            print(f"  {name}: {val}")

    print(f"\n  目的関数値: {result.fun:.4f}")

    # 最適条件で膜特性を計算（ノイズなし近似）
    rng_eval = np.random.default_rng(0)
    n_eval = 10
    rates, ris, roughs, stresses = [], [], [], []
    for _ in range(n_eval):
        res = simulate_deposition(
            power=float(result.x[0]),
            pressure=float(result.x[1]),
            substrate_temp=float(result.x[2]),
            ar_flow=float(result.x[3]),
            distance=float(result.x[4]),
            rng=rng_eval,
        )
        rates.append(res["deposition_rate_nm_per_min"])
        ris.append(res["refractive_index"])
        roughs.append(res["surface_roughness_Ra_nm"])
        stresses.append(res["film_stress_MPa"])

    print("\n【最適条件での予測膜特性】（10回平均 ± 標準偏差）")
    print(f"  成膜レート:   {np.mean(rates):.2f} ± {np.std(rates):.2f} nm/min")
    print(f"  屈折率:       {np.mean(ris):.4f} ± {np.std(ris):.4f}")
    print(f"  表面粗さ Ra:  {np.mean(roughs):.2f} ± {np.std(roughs):.2f} nm")
    print(f"  膜応力:       {np.mean(stresses):.1f} ± {np.std(stresses):.1f} MPa")


def save_results(result, output_dir: str = "results") -> None:
    """最適化結果をファイルに保存する。"""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # 評価履歴をCSVに保存
    if evaluation_history:
        df_hist = pd.DataFrame(evaluation_history)
        df_hist.to_csv(out / "optimization_history.csv", index=False)
        print(f"\n評価履歴を {out / 'optimization_history.csv'} に保存しました")

    # 収束プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_convergence(result, ax=ax)
    ax.set_title("ベイズ最適化の収束曲線")
    ax.set_xlabel("評価回数")
    ax.set_ylabel("目的関数の最小値")
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=150)
    plt.close(fig)
    print(f"収束プロットを {out / 'convergence.png'} に保存しました")

    # 目的関数の依存性プロット（部分依存プロット）
    try:
        fig, axes = plt.subplots(5, 5, figsize=(18, 18))
        dim_names = [
            "RF出力 (W)",
            "Ar圧力 (Pa)",
            "基板温度 (°C)",
            "Ar流量 (sccm)",
            "T-S距離 (mm)",
        ]
        plot_objective(
            result,
            dimensions=dim_names,
            n_points=20,
        )
        plt.suptitle("目的関数の部分依存プロット", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(out / "objective_landscape.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"目的関数プロットを {out / 'objective_landscape.png'} に保存しました")
    except Exception as e:
        print(f"目的関数プロットの生成でエラー: {e}")

    # 評価履歴の推移プロット
    if evaluation_history:
        df_hist = pd.DataFrame(evaluation_history)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ("deposition_rate_nm_per_min", "成膜レート (nm/min)", "tab:blue"),
            ("refractive_index", "屈折率", "tab:orange"),
            ("surface_roughness_Ra_nm", "表面粗さ Ra (nm)", "tab:green"),
            ("film_stress_MPa", "膜応力 (MPa)", "tab:red"),
        ]

        for ax, (col, label, color) in zip(axes.flat, metrics):
            ax.scatter(
                df_hist["iteration"],
                df_hist[col],
                c=color,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                s=40,
            )
            ax.set_xlabel("評価回数")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

            # 目標値の線を引く
            if col == "refractive_index":
                ax.axhline(y=1.46, color="gray", linestyle="--", alpha=0.5, label="目標値 1.46")
                ax.legend()

        fig.suptitle("ベイズ最適化による膜特性の推移", fontsize=14)
        fig.tight_layout()
        fig.savefig(out / "property_evolution.png", dpi=150)
        plt.close(fig)
        print(f"特性推移プロットを {out / 'property_evolution.png'} に保存しました")


def main():
    parser = argparse.ArgumentParser(
        description="スパッタ成膜条件のベイズ最適化"
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=50,
        help="最適化の総評価回数 (default: 50)",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=10,
        help="初期ランダム探索の点数 (default: 10)",
    )
    parser.add_argument(
        "--initial-data",
        type=str,
        default=None,
        help="初期実験データのCSVパス",
    )
    parser.add_argument(
        "--acq-func",
        type=str,
        default="EI",
        choices=["EI", "PI", "LCB", "gp_hedge"],
        help="獲得関数 (default: EI)",
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
        help="結果の出力ディレクトリ (default: results)",
    )
    args = parser.parse_args()

    result = run_optimization(
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        initial_data_path=args.initial_data,
        acq_func=args.acq_func,
        seed=args.seed,
    )

    print_results(result)
    save_results(result, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
