"""
スパッタ成膜の合成データセット生成スクリプト

材料系: SiO2ターゲットをRFスパッタにより成膜する想定
入力パラメータ（成膜条件）:
  - RF出力 (W): 50 ~ 300
  - Ar圧力 (Pa): 0.1 ~ 2.0
  - 基板温度 (°C): 25 ~ 400
  - Ar流量 (sccm): 5 ~ 50
  - ターゲット-基板間距離 (mm): 40 ~ 120

出力（膜特性）:
  - 成膜レート (nm/min)
  - 膜の屈折率 (@ 633nm)
  - 表面粗さ Ra (nm)
  - 膜応力 (MPa, 圧縮が負)

物理的な傾向を反映した合成関数でデータを生成する。
"""

import numpy as np
import pandas as pd


def simulate_deposition(
    power: float,
    pressure: float,
    substrate_temp: float,
    ar_flow: float,
    distance: float,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """物理的傾向を模擬したスパッタ成膜シミュレーション関数。

    実際の傾向:
    - 出力↑ → 成膜レート↑, 応力の絶対値↑
    - 圧力↑ → 成膜レート↓(散乱増), 表面粗さ↑
    - 基板温度↑ → 結晶化促進 → 屈折率がバルク値に近づく, 応力緩和
    - 距離↑ → 成膜レート↓, 均一性改善
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- 正規化 ---
    p_norm = (power - 50) / 250          # 0~1
    pr_norm = (pressure - 0.1) / 1.9     # 0~1
    t_norm = (substrate_temp - 25) / 375  # 0~1
    f_norm = (ar_flow - 5) / 45          # 0~1
    d_norm = (distance - 40) / 80        # 0~1

    # === 成膜レート (nm/min) ===
    # 出力に比例、圧力・距離に反比例
    rate = (
        5.0
        + 25.0 * p_norm
        - 8.0 * pr_norm
        - 10.0 * d_norm
        + 3.0 * f_norm
        - 4.0 * p_norm * pr_norm  # 交互作用
        + rng.normal(0, 1.0)
    )
    rate = max(rate, 0.5)

    # === 屈折率 (@ 633nm, SiO2バルク値 ≈ 1.46) ===
    # 基板温度が高いほどバルク値に近づく
    # 低温・低出力では密度不足で低い値
    refractive_index = (
        1.44
        + 0.03 * t_norm
        + 0.01 * p_norm
        - 0.02 * pr_norm
        - 0.005 * d_norm
        + 0.008 * p_norm * t_norm  # 高出力+高温で緻密化
        + rng.normal(0, 0.003)
    )

    # === 表面粗さ Ra (nm) ===
    # 圧力↑, 出力↑ で粗くなる傾向、基板温度↑ で平滑化
    roughness = (
        0.3
        + 1.5 * pr_norm
        + 0.8 * p_norm
        - 0.6 * t_norm
        + 0.4 * d_norm
        + 0.5 * p_norm * pr_norm
        - 0.3 * t_norm * p_norm
        + rng.normal(0, 0.1)
    )
    roughness = max(roughness, 0.1)

    # === 膜応力 (MPa) ===
    # 圧縮応力(負)が基本、高温で緩和、高出力で増大
    stress = (
        -200
        - 150 * p_norm
        + 120 * t_norm
        + 80 * pr_norm
        + 30 * f_norm
        + 50 * p_norm * t_norm  # 高温+高出力で応力緩和
        + rng.normal(0, 20)
    )

    return {
        "deposition_rate_nm_per_min": round(rate, 2),
        "refractive_index": round(refractive_index, 4),
        "surface_roughness_Ra_nm": round(roughness, 2),
        "film_stress_MPa": round(stress, 1),
    }


def generate_dataset(
    n_samples: int = 100,
    seed: int = 42,
    strategy: str = "latin_hypercube",
) -> pd.DataFrame:
    """データセットを生成する。

    Parameters
    ----------
    n_samples : int
        サンプル数
    seed : int
        乱数シード
    strategy : str
        サンプリング戦略。'latin_hypercube' または 'random'
    """
    rng = np.random.default_rng(seed)

    # パラメータ範囲
    bounds = {
        "rf_power_W": (50, 300),
        "pressure_Pa": (0.1, 2.0),
        "substrate_temp_C": (25, 400),
        "ar_flow_sccm": (5, 50),
        "target_substrate_distance_mm": (40, 120),
    }

    if strategy == "latin_hypercube":
        # ラテン超方格サンプリングでパラメータ空間を効率的にカバー
        n_dims = len(bounds)
        samples = np.zeros((n_samples, n_dims))
        for i in range(n_dims):
            perm = rng.permutation(n_samples)
            samples[:, i] = (perm + rng.uniform(size=n_samples)) / n_samples

        # 各次元を実際の範囲にスケーリング
        params = {}
        for j, (name, (lo, hi)) in enumerate(bounds.items()):
            params[name] = lo + (hi - lo) * samples[:, j]
            # 見やすくするために丸める
            if name in ("rf_power_W", "substrate_temp_C", "target_substrate_distance_mm"):
                params[name] = np.round(params[name], 0).astype(int)
            elif name == "pressure_Pa":
                params[name] = np.round(params[name], 2)
            else:
                params[name] = np.round(params[name], 1)
    else:
        params = {
            "rf_power_W": rng.integers(50, 301, size=n_samples),
            "pressure_Pa": np.round(rng.uniform(0.1, 2.0, size=n_samples), 2),
            "substrate_temp_C": rng.integers(25, 401, size=n_samples),
            "ar_flow_sccm": np.round(rng.uniform(5, 50, size=n_samples), 1),
            "target_substrate_distance_mm": rng.integers(40, 121, size=n_samples),
        }

    # シミュレーション実行
    results = []
    for i in range(n_samples):
        result = simulate_deposition(
            power=float(params["rf_power_W"][i]),
            pressure=float(params["pressure_Pa"][i]),
            substrate_temp=float(params["substrate_temp_C"][i]),
            ar_flow=float(params["ar_flow_sccm"][i]),
            distance=float(params["target_substrate_distance_mm"][i]),
            rng=rng,
        )
        results.append(result)

    df_params = pd.DataFrame(params)
    df_results = pd.DataFrame(results)
    df = pd.concat([df_params, df_results], axis=1)

    return df


if __name__ == "__main__":
    # 初期実験データ（20点）と追加データ（80点）を生成
    df_initial = generate_dataset(n_samples=20, seed=42, strategy="latin_hypercube")
    df_initial.to_csv("data/initial_experiments.csv", index=False)
    print(f"初期実験データ: {len(df_initial)} 点を data/initial_experiments.csv に保存")
    print(df_initial.describe().round(3))

    # 全データ（検証用）
    df_full = generate_dataset(n_samples=200, seed=123, strategy="latin_hypercube")
    df_full.to_csv("data/full_dataset.csv", index=False)
    print(f"\n全データセット: {len(df_full)} 点を data/full_dataset.csv に保存")
