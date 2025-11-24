import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.svm import LinearSVC

# =========================================================
# 1. 理想分離ベクトル u と定数 gamma, D を求める関数
# =========================================================
def compute_u_gamma_D(X_bias, y):
    """
    拡張されたデータ X_bias に対して線形SVMで理想の分離超平面 u を求める。
    データ側にバイアス項が含まれているため、モデル側は原点を通る設定で良い。
    """
    # fit_intercept=False に設定（データにバイアス項があるため）
    clf = LinearSVC(C=1e10, fit_intercept=False, dual="auto", max_iter=10000)
    clf.fit(X_bias, y)

    u_ideal = clf.coef_[0]                  # 3次元の法線ベクトル
    u = u_ideal / np.linalg.norm(u_ideal)   # ||u||=1 に正規化

    gamma = np.min(y * (X_bias @ u))        # 最小マージン
    D = np.max(np.linalg.norm(X_bias, axis=1)) # 最大ノルム

    return u, gamma, D


# =========================================================
# 2. パーセプトロン学習（更新履歴も全部保存）
# =========================================================
def perceptron_train_with_history(X_bias, y, eta=0.5, max_epochs=1000):
    """
    拡張されたデータ X_bias に対してパーセプトロン学習を行う。
    重み w は3次元ベクトルとなる。
    """
    w = np.zeros(X_bias.shape[1]) # 3次元で初期化
    history = []
    k = 0

    for epoch in range(max_epochs):
        errors = 0
        for i, x_i in enumerate(X_bias):
            if y[i] * (w @ x_i) <= 0:
                # インプレース演算子 += を使用（念のため）
                w += eta * y[i] * x_i
                k += 1
                errors += 1
                history.append({"k": k, "w": w.copy(), "mis_idx": i})
        if errors == 0:
            print(f"Converged at epoch {epoch+1}")
            break
    else:
        print("Did not converge within max_epochs.")

    return w, history


# =========================================================
# 3. 証明で追う数列を履歴から計算する
# =========================================================
def compute_proof_sequences(history, u):
    """
    history から u^T w_k, ||w_k||, cosθ_k を計算する。
    u, w_k ともに3次元だが、内積計算はそのまま行える。
    """
    ks = []
    nums = []
    dens = []
    coss = []

    for h in history:
        w_k = h["w"]
        num = u @ w_k
        den = np.linalg.norm(w_k)
        # 分母が0の場合は便宜上 cos=0 とする
        cos_theta = num / den if den > 1e-9 else 0

        ks.append(h["k"])
        nums.append(num)
        dens.append(den)
        coss.append(cos_theta)

    return np.array(ks), np.array(nums), np.array(dens), np.array(coss)


# =========================================================
# 4. 動的可視化（バイアス項対応版）
# =========================================================
def animate_learning(X, y, u, gamma, D, history, eta):
    ks, nums, dens, coss = compute_proof_sequences(history, u)

    # ---- 理論境界の計算 ----
    num_bounds = ks * eta * gamma
    den_bounds = np.sqrt(ks) * eta * D
    # 0除算回避と上限1.0のクリップ
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_bounds = np.sqrt(ks) * (gamma / D)
        cos_bounds = np.nan_to_num(cos_bounds) # k=0でのNaNを0に
        cos_bounds = np.minimum(1.0, cos_bounds)

    # ---- 図のレイアウト ----
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 3)

    ax_data = fig.add_subplot(gs[:, 0])
    ax_num  = fig.add_subplot(gs[0, 1])
    ax_den  = fig.add_subplot(gs[0, 2])
    ax_cos  = fig.add_subplot(gs[1, 1:])

    # ---- 左：データを描画（元の2次元データを使用） ----
    pos = y == 1
    neg = y == -1
    ax_data.scatter(X[pos, 0], X[pos, 1], marker="o", label="Class +1", s=100, c='blue')
    ax_data.scatter(X[neg, 0], X[neg, 1], marker="x", label="Class -1", s=100, c='red')
    ax_data.set_title("Perceptron learning (Decision Boundary)")
    ax_data.legend()
    ax_data.grid(True)

    # 軸範囲の設定
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    ax_data.set_xlim(x_min, x_max)
    ax_data.set_ylim(y_min, y_max)

    xs = np.linspace(x_min, x_max, 200)
    boundary_line, = ax_data.plot([], [], lw=2, color='green')
    mis_point = ax_data.scatter([], [], s=200, facecolors="none", edgecolors="red", linewidth=2)

    # ---- 右側3つのグラフの初期設定 ----
    # 分子
    ax_num.set_title(r"Numerator: $u^T w_k$")
    ax_num.set_xlabel("Updates (k)")
    ax_num.grid(True)
    num_actual, = ax_num.plot([], [], "-", label="Actual") # 点が多いので線のみに
    num_bound,  = ax_num.plot([], [], "r--", label=r"Bound $k\eta\gamma$")
    ax_num.legend()

    # 分母
    ax_den.set_title(r"Denominator: $||w_k||$")
    ax_den.set_xlabel("Updates (k)")
    ax_den.grid(True)
    den_actual, = ax_den.plot([], [], "-", label="Actual")
    den_bound,  = ax_den.plot([], [], "r--", label=r"Bound $\sqrt{k}\eta D$")
    ax_den.legend()

    # コサイン
    ax_cos.set_title(r"Convergence: $\cos \theta_k$")
    ax_cos.set_xlabel("Updates (k)")
    ax_cos.grid(True)
    cos_actual, = ax_cos.plot([], [], "-", color='green', label="Actual")
    cos_bound,  = ax_cos.plot([], [], "r--", label=r"Lower bound $\frac{\sqrt{k}\gamma}{D}$")
    ax_cos.axhline(1.0, linestyle=":", color='black')
    ax_cos.legend(loc='lower right')

    # ---- 軸範囲の固定（重要） ----
    max_k = ks[-1] + 5 if len(ks) > 0 else 10
    for ax in [ax_num, ax_den, ax_cos]:
        ax.set_xlim(0, max_k)

    # y軸の最大値をデータに基づいて設定
    ax_num.set_ylim(0, max(nums.max(), num_bounds.max()) * 1.1 if len(nums) > 0 else 1)
    ax_den.set_ylim(0, max(dens.max(), den_bounds.max()) * 1.1 if len(dens) > 0 else 1)
    ax_cos.set_ylim(0, 1.05)


    # ---- アニメーション更新関数 ----
    def update(frame):
        h = history[frame]
        k = h["k"]
        w_k = h["w"] # 3次元ベクトル [w0, w1, w2] (w2がバイアス)
        i_mis = h["mis_idx"]

        # 決定境界の描画: w0*x + w1*y + w2*1 = 0  =>  y = -(w0*x + w2) / w1
        if abs(w_k[1]) > 1e-9:
            ys = -(w_k[0] * xs + w_k[2]) / w_k[1]
            boundary_line.set_data(xs, ys)
        elif abs(w_k[0]) > 1e-9: # 垂直線の場合 x = -w2 / w0
             boundary_line.set_data([-w_k[2]/w_k[0]]*2, [y_min, y_max])
        else:
            boundary_line.set_data([], []) # wがまだ0の時など

        # 間違えた点を赤丸で囲む（元の2次元座標を使用）
        mis_point.set_offsets([X[i_mis]])

        # 右側グラフのデータを更新
        idx = frame + 1
        num_actual.set_data(ks[:idx], nums[:idx])
        num_bound.set_data(ks[:idx], num_bounds[:idx])

        den_actual.set_data(ks[:idx], dens[:idx])
        den_bound.set_data(ks[:idx], den_bounds[:idx])

        cos_actual.set_data(ks[:idx], coss[:idx])
        cos_bound.set_data(ks[:idx], cos_bounds[:idx])

        fig.suptitle(f"Novikoff Proof Visualization (Step k={k})", fontsize=16)
        return boundary_line, mis_point, num_actual, num_bound, den_actual, den_bound, cos_actual, cos_bound

    # フレーム数が多いので間隔を短く設定
    ani = FuncAnimation(
        fig, update, frames=len(history),
        interval=100, blit=False, repeat=False
    )
    plt.tight_layout()
    plt.show()


# =========================================================
# 5. 実行部
# =========================================================
if __name__ == "__main__":
    # ---- 元の2次元データ ----
    X = np.array([
        [1.5, 4.0], [2.0, 2.5], [3.0, 4.5], # Class +1
        [1.0, 1.0], [2.5, 0.5], [0.5, 2.0]  # Class -1
    ])
    y = np.array([1, 1, 1, -1, -1, -1])

    # ---- 【重要】バイアス項に対応するため、データに 1.0 の列を追加 ----
    # これにより、データは3次元 (x1, x2, 1) になる
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

    # ---- 定理の定数を計算（拡張されたデータを使用） ----
    # u も3次元ベクトルになる
    u, gamma, D = compute_u_gamma_D(X_bias, y)
    print(f"Parameters: gamma={gamma:.4f}, D={D:.4f}, Bound (D/gamma)^2={(D/gamma)**2:.4f}")

    # ---- パーセプトロン学習（拡張されたデータを使用） ----
    eta = 0.1 # 学習率を少し小さくしてみる
    w_final, history = perceptron_train_with_history(X_bias, y, eta=eta, max_epochs=200)
    print(f"Total updates k={len(history)}")

    if len(history) > 0:
        # ---- 動的可視化 ----
        # 散布図には元のX、計算にはu(3次元)とhistory(重みも3次元)を渡す
        animate_learning(X, y, u, gamma, D, history, eta)
    else:
        print("No updates were made. The data might be already separated by initial weights.")

