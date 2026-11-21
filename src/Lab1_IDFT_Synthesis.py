# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lab 1 – Spectral analysis of deterministic signals
# ### Signal synthesis with the IDFT in matrix notation — all 15 variants
#
# | | |
# |---|---|
# | **Author** | Radosław Skrzypczyński — Informatyka II st., niestacjonarne, 1 sem., Gr. 2 |
# | **Course** | Digital Signal Processing — prof. dr hab. Vasyl Martsenyuk |
# | **Repository** | https://github.com/Slayzerus/dsp-lab1 |
#
# **Task (Sec. 7 of the lab instructions).** Synthesize a discrete-time signal by using the IDFT in matrix
# notation for different values of N. Show the matrices **W** and **K**. Plot the signal synthesized.
# The DFT coefficient vectors x_μ of the variants are given in the instructions. **All 15 variants are computed.**
#
# **Contents.** 1 Theory · 2 Helper functions · 3 Validation against the instructions · 4 Variants ·
# 5 Matrices K and W · 6 Synthesis for all variants and several N · 7 Plots of the synthesized signals ·
# 8 Effect of N · 9 Interpretation · 10 Conclusions

# %%
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from numpy.fft import ifft
from IPython.display import display

print(f"Python {platform.python_version()} | numpy {np.__version__} | matplotlib {matplotlib.__version__} | "
      f"pandas {pd.__version__}")

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
pd.options.display.precision = 3
pd.options.display.width = 200

# %% [markdown]
# ## 1. Theory
#
# * **DFT pair** (instructions, eq. 1–2), 0 ≤ k, μ ≤ N − 1:
#   X[μ] = Σₖ x[k]·e^(−j2πkμ/N)  and  x[k] = (1/N)·Σ_μ X[μ]·e^(+j2πkμ/N).
#   This is the convention of `numpy.fft.fft` / `numpy.fft.ifft`.
# * **Matrix notation** (eq. 3–6): with the column vectors x_k = (x[0], …, x[N−1])ᵀ and x_μ = (X[0], …, X[N−1])ᵀ
#
#   **DFT:** x_μ = W*·x_k  **IDFT:** x_k = (1/N)·W·x_μ
# * **Fourier matrix** (eq. 7–9): W = e^(+j2π/N ⊙ K) (element-wise), i.e. W[k, μ] = W_N^(k·μ) with the twiddle
#   factor W_N = e^(+j2π/N), and K is the outer product K = (0, 1, …, N−1)ᵀ·(0, 1, …, N−1), so K[k, μ] = k·μ.
#   Because W_N^N = 1, only the exponent (k·μ mod N) matters.
# * **Properties** (eq. 11–14): W is symmetric and W⁻¹ = W^H/N = W*/N, so W/√N is unitary. The columns
#   w_μ[k] = cos(2πkμ/N) + j·sin(2πkμ/N) are the orthogonal DFT eigensignals with the digital frequencies
#   Ω_μ = 2πμ/N. The IDFT therefore synthesizes x_k as a **linear combination of the columns of W**, weighted by
#   X[μ]/N (signal synthesis = forward problem; the DFT is the inverse problem, signal analysis).
# * **Different N.** A vector of length N₀ is used with N = N₀ and with longer IDFTs N = 2N₀ and 4N₀ by appending
#   zeros (X[μ] = 0 for N₀ ≤ μ < N). Every coefficient keeps its index μ, but its digital frequency becomes 2πμ/N.

# %%
# Plot style: the same as in the Lab 2 (windowing) notebook, so that all reports look alike.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": [f for f in ("Segoe UI", "DejaVu Sans") if f in _fonts] or ["sans-serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "mathtext.fontset": "dejavusans", "figure.titlesize": 10.5, "figure.titleweight": "semibold",
})
C_RE, C_IM = "#2a78d6", "#eb6834"          # real part / imaginary part, fixed in every figure
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f7f6f2", "#2a78d6"])


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.show()


def stem(ax, k, y, color, label=None, ms=4.0):
    """Thin stem plot: marker on a hairline stem, no baseline."""
    markers, stems, base = ax.stem(k, y, linefmt="-", markerfmt="o", basefmt=" ", label=label)
    plt.setp(stems, color=color, linewidth=0.9)
    plt.setp(markers, color=color, markersize=ms, markeredgecolor="white", markeredgewidth=0.5)
    return markers


# %% [markdown]
# ## 2. Helper functions
#
# The functions follow the template notebook of the course (`build_K_W`, `idft_matrix`, `pad_or_trim_X`).
# K is built explicitly as the outer product of a column and a row vector, as required by the task.

# %%
def build_K(N):
    """Index matrix K = (0..N-1)^T (0..N-1): K[k, mu] = k*mu (outer product, eq. 9)."""
    k = np.arange(N).reshape(-1, 1)      # column vector (N x 1)
    mu = np.arange(N).reshape(1, -1)     # row vector (1 x N)
    return k @ mu


def build_W(N):
    """Fourier matrix W = exp(+j 2pi/N (.) K) (eq. 7)."""
    return np.exp(+1j * 2*np.pi/N * build_K(N))


def pad(X, N):
    """DFT vector of length N: X[mu] = 0 for mu >= len(X) (zero-padding of the spectrum)."""
    X = np.asarray(X, dtype=complex).ravel()
    if N < X.size:
        raise ValueError("N must not be smaller than the given vector")
    out = np.zeros(N, dtype=complex)
    out[:X.size] = X
    return out


def idft_matrix(X, N=None):
    """IDFT in matrix notation: x_k = (1/N) W x_mu (eq. 6)."""
    X = pad(X, len(X) if N is None else N)
    return build_W(X.size) @ X / X.size


def dft_matrix(x):
    """DFT in matrix notation: x_mu = W* x_k (eq. 5)."""
    return np.conj(build_W(x.size)) @ x


def clean(a, d=3):
    """Round for display and remove negative zeros."""
    return np.round(a, d) + 0.0


# %% [markdown]
# ## 3. Validation against the instructions
#
# The helpers reproduce the matrices K and W for N = 4 given in eq. (10) and the initial example of Sec. 6:
# N = 8, x_μ = [8, 2, 4, 0, 0, 0, 0, 0]ᵀ (a DC value plus the first and the second eigenfrequency).

# %%
K4_expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]])
W4_expected = np.array([[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]])

X_ex = np.array([8, 2, 4, 0, 0, 0, 0, 0])
x_ex = idft_matrix(X_ex)
W8 = build_W(8)
U8 = W8/np.sqrt(8)
validation = {
    "K for N = 4 equals eq. (10)": np.array_equal(build_K(4), K4_expected),
    "W for N = 4 equals eq. (10)": np.allclose(build_W(4), W4_expected),
    "W is symmetric (W = Wᵀ)": np.allclose(W8, W8.T),
    "W⁻¹ = W*/N (eq. 11)": np.allclose(np.linalg.inv(W8), np.conj(W8)/8),
    "W/√N is unitary (eq. 14)": np.allclose(U8.conj().T @ U8, np.eye(8)),
    "eigensignals orthogonal: w₀ᴴw₁ = 0, w₁ᴴw₁ = N": np.isclose(np.vdot(W8[:, 0], W8[:, 1]), 0)
                                                    and np.isclose(np.vdot(W8[:, 1], W8[:, 1]), 8),
    "IDFT (matrix) equals numpy.fft.ifft": np.allclose(x_ex, ifft(X_ex)),
    "DFT (matrix) of x_k returns x_μ": np.allclose(dft_matrix(x_ex), X_ex),
    "x_k = (1/N)(8·w₀ + 2·w₁ + 4·w₂)": np.allclose(x_ex, (8*W8[:, 0] + 2*W8[:, 1] + 4*W8[:, 2])/8),
    "DC value: mean(x_k) = X[0]/N = 1": np.isclose(x_ex.mean(), 1),
}
display(pd.Series(validation, name="holds").to_frame())
assert all(validation.values())

print("K (N = 4):\n", build_K(4))
print("W (N = 4):\n", clean(build_W(4)))
print("x_k of the example (N = 8):\n", clean(x_ex))

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.0))
k8 = np.arange(8)
stem(ax, k8, x_ex.real, C_RE, "Re x[k]")
stem(ax, k8, x_ex.imag, C_IM, "Im x[k]")
ax.set(xlabel="sample k", ylabel="x[k]", xticks=k8)
ax.set_title("Initial example of the instructions: N = 8, x_μ = [8, 2, 4, 0, 0, 0, 0, 0]ᵀ")
ax.legend(loc="upper right")
save(fig, "fig01_example_N8")

# %% [markdown]
# ## 4. Variants
#
# The 15 vectors x_μ are taken from Sec. 7 of the instructions. Several variants are identical, so the 15 variants
# contain **9 different vectors (groups G1–G9)**. All variants are still computed individually and the identity of
# the results within a group is asserted in Section 6.

# %%
VARIANTS = {
    1: [6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    2: [10, 5, 6, 6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    3: [6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    4: [6, 2, 4, 3, 4, 5, 0, 0, 0],
    5: [6, 4, 4, 5, 3, 4, 5, 0, 0, 0, 0],
    6: [7, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    7: [6, 8, 2, 4, 3, 4, 5, 0, 0, 0],
    8: [6, 2, 4, 4, 4, 5, 0, 0, 0, 0],
    9: [6, 5, 4, 3, 4, 5, 0, 0, 0, 0],
    10: [6, 2, 4, 3, 4, 4, 0, 0, 0, 0],
    11: [6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    12: [10, 5, 6, 6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    13: [6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    14: [6, 2, 4, 3, 4, 5, 0, 0, 0],
    15: [6, 4, 4, 5, 3, 4, 5, 0, 0, 0, 0],
}
VARIANTS = {no: np.array(v, dtype=float) for no, v in VARIANTS.items()}

GROUPS = {}                                   # "G1" -> [variant numbers]
_seen = {}
for no, X in VARIANTS.items():
    key = tuple(X)
    if key not in _seen:
        _seen[key] = f"G{len(_seen) + 1}"
    GROUPS.setdefault(_seen[key], []).append(no)
GROUP_OF = {no: g for g, nos in GROUPS.items() for no in nos}

table = pd.DataFrame([{
    "No": no, "x_μ": "[" + ", ".join(f"{v:g}" for v in X) + "]ᵀ", "N₀ = length": X.size,
    "non-zero X[μ]": int(np.count_nonzero(X)), "highest non-zero μ": int(np.flatnonzero(X).max()),
    "Σ X[μ]": X.sum(), "group": GROUP_OF[no]} for no, X in VARIANTS.items()]).set_index("No")
display(table)
display(pd.DataFrame({"variants": {g: ", ".join(map(str, nos)) for g, nos in GROUPS.items()}}))
N0_SET = sorted({X.size for X in VARIANTS.values()})
print("Lengths N₀ that occur:", N0_SET)

# %% [markdown]
# * All vectors are **real and non-negative**, and only the first 6–9 coefficients are non-zero; the trailing zeros
#   only fix the IDFT length N₀ ∈ {9, 10, 11, 13}.
# * G1 (variants 1, 3, 11, 13) and G3 (variants 4, 14) have the **same non-zero coefficients** [6, 2, 4, 3, 4, 5]
#   and differ only in N₀ (10 vs 9).
# * The matrices K and W depend **only on N**, not on the coefficients, so for N = N₀ only four different pairs
#   (K, W) are needed for all 15 variants.

# %% [markdown]
# ## 5. Matrices K and W
#
# The full matrices are shown for every N₀ that occurs (N = 9, 10, 11, 13). W is complex, so its real part
# Re W[k, μ] = cos(2πkμ/N) and its imaginary part Im W[k, μ] = sin(2πkμ/N) are listed as two tables
# (3 decimals). The rows are indexed by the time index k, the columns by the frequency index μ.

# %%
def show_K_W(N):
    idx_k, idx_mu = pd.Index(range(N), name="k \\ μ"), pd.Index(range(N), name="")
    K, W = build_K(N), build_W(N)
    print(f"N = {N}:  K = outer(k, μ), shape {K.shape};  W = exp(+j2π/{N}·K), shape {W.shape}")
    print("Matrix K:")
    display(pd.DataFrame(K, index=idx_k, columns=idx_mu))
    print("Matrix W, real part Re W = cos(2π/N·K):")
    display(pd.DataFrame(clean(W.real), index=idx_k, columns=idx_mu))
    print("Matrix W, imaginary part Im W = sin(2π/N·K):")
    display(pd.DataFrame(clean(W.imag), index=idx_k, columns=idx_mu))


# %% [markdown]
# ### 5.1 N = 10 (variants 1, 3, 6, 7, 8, 9, 10, 11, 13)

# %%
show_K_W(10)

# %% [markdown]
# ### 5.2 N = 13 (variants 2, 12)

# %%
show_K_W(13)

# %% [markdown]
# ### 5.3 N = 9 (variants 4, 14)

# %%
show_K_W(9)

# %% [markdown]
# ### 5.4 N = 11 (variants 5, 15)

# %%
show_K_W(11)

# %% [markdown]
# ### 5.5 Structure of K and W for the different N used in the synthesis
#
# For variant 1 the synthesis uses N = 10, 20 and 40. The figure shows K, the effective twiddle exponent K mod N,
# Re W and Im W. Row 0 and column 0 of W are all ones (μ = 0 is the DC eigensignal, k = 0 is the first sample);
# column μ is a cosine (real part) and a sine (imaginary part) with exactly μ periods in N samples; the matrices are
# symmetric about the main diagonal.

# %%
N_PLOT = (10, 20, 40)
fig, axs = plt.subplots(len(N_PLOT), 4, figsize=(10.5, 8.3))
for r, N in enumerate(N_PLOT):
    K, W = build_K(N), build_W(N)
    panels = [(K, "K = k·μ", "Greys", None, None), (K % N, "K mod N", "Greys", 0, N - 1),
              (W.real, "Re W = cos(2πkμ/N)", DIVERGING, -1, 1), (W.imag, "Im W = sin(2πkμ/N)", DIVERGING, -1, 1)]
    for c, (M, ttl, cmap, vmin, vmax) in enumerate(panels):
        ax = axs[r, c]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.grid(False)
        ax.set_title(f"{ttl},  N = {N}", fontsize=8.5)
        ax.set_xlabel("μ", labelpad=1)
        ax.set_ylabel("k", labelpad=1)
        ticks = np.arange(0, N, max(1, N//5))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.suptitle("Matrices K and W for N = 10, 20, 40 (variant 1: N₀, 2N₀, 4N₀)")
fig.tight_layout()
save(fig, "fig02_K_W_structure")

# %% [markdown]
# ## 6. Synthesis for all variants and several N
#
# Every variant is synthesized with x_k = (1/N)·W·x_μ for N = N₀, 2N₀ and 4N₀ (zero-padded spectrum). Each result
# is checked against `numpy.fft.ifft`, against the inverse operation (DFT x_μ = W*·x_k must return the
# coefficients) and against Parseval's theorem Σ|x[k]|² = (1/N)·Σ|X[μ]|².

# %%
N_FACTORS = (1, 2, 4)
SIG = {}                                  # (variant, N) -> x_k
rows = []
for no, X0 in VARIANTS.items():
    for m in N_FACTORS:
        N = m*X0.size
        X = pad(X0, N)
        x = idft_matrix(X)
        SIG[(no, N)] = x
        kneg = (-np.arange(N)) % N
        rows.append({
            "No": no, "N": N,
            "= ifft": np.allclose(x, ifft(X)),
            "W*·x_k = x_μ": np.allclose(dft_matrix(x), X),
            "Parseval": np.isclose(np.sum(np.abs(x)**2), np.sum(np.abs(X)**2)/N),
            "x[−k] = x*[k]": np.allclose(x[kneg], np.conj(x)),
            "argmax|x| = 0": int(np.argmax(np.abs(x))) == 0,
            "x[0] = ΣX/N": np.isclose(x[0], X.sum()/N),
            "mean = X[0]/N": np.isclose(x.mean(), X[0]/N),
            "max|Im x|": np.abs(x.imag).max(),
        })
CHK = pd.DataFrame(rows)
bool_cols = ["= ifft", "W*·x_k = x_μ", "Parseval", "x[−k] = x*[k]", "argmax|x| = 0", "x[0] = ΣX/N", "mean = X[0]/N"]
summary = CHK[bool_cols].all().to_frame("holds for all 15 variants × 3 values of N")
display(summary)
assert summary.iloc[:, 0].all()
assert (CHK["max|Im x|"] > 0.1).all()          # every synthesized signal is complex

# identical vectors -> identical signals
for g, nos in GROUPS.items():
    for no in nos[1:]:
        for m in N_FACTORS:
            N = m*VARIANTS[no].size
            assert np.array_equal(SIG[(no, N)], SIG[(nos[0], N)])
print("Identical results within every group G1–G9: OK")

# %% [markdown]
# **The synthesized signals for N = N₀** (real and imaginary part of x[k], 3 decimals):

# %%
cols = pd.Index(range(max(N0_SET)), name="k")
re_tab = pd.DataFrame({no: pd.Series(clean(SIG[(no, X.size)].real)) for no, X in VARIANTS.items()}).T
im_tab = pd.DataFrame({no: pd.Series(clean(SIG[(no, X.size)].imag)) for no, X in VARIANTS.items()}).T
for tab in (re_tab, im_tab):
    tab.index.name = "No"
    tab.columns = cols
print("Re x[k], N = N₀:")
display(re_tab.fillna(""))
print("Im x[k], N = N₀:")
display(im_tab.fillna(""))

key = pd.DataFrame([{
    "No": no, "group": GROUP_OF[no], "N": N,
    "x[0] = ΣX/N": SIG[(no, N)][0].real, "DC = mean(x) = X[0]/N": SIG[(no, N)].mean().real,
    "max|x[k]|": np.abs(SIG[(no, N)]).max(), "energy Σ|x|²": np.sum(np.abs(SIG[(no, N)])**2),
} for no, X in VARIANTS.items() for N in (X.size*m for m in N_FACTORS)]).set_index(["No", "N"])
print("Key quantities for N = N₀, 2N₀, 4N₀:")
display(key)

# %% [markdown]
# ## 7. Plots of the synthesized signals
#
# One figure per group G1–G9 (the title lists the variants it covers). Each figure shows the synthesized signal for
# N = N₀, 2N₀ and 4N₀: real part (blue) and imaginary part (orange) as stems, |x[k]| as a thin grey line.
# Connecting samples by a line is used only for |x[k]| and only for visual convenience.

# %%
def plot_group(g):
    nos = GROUPS[g]
    X0 = VARIANTS[nos[0]]
    vec = ", ".join(f"{v:g}" for v in X0)
    fig, axs = plt.subplots(1, len(N_FACTORS), figsize=(10.5, 3.0), sharey=True,
                            gridspec_kw={"width_ratios": [1, 1.4, 2.2]})
    for ax, m in zip(axs, N_FACTORS):
        N = m*X0.size
        x = SIG[(nos[0], N)]
        k = np.arange(N)
        ms = 4.0 if N <= 26 else 2.6
        ax.plot(k, np.abs(x), "-", color=MUTED, lw=0.8, zorder=1, label="|x[k]|")
        stem(ax, k, x.real, C_RE, "Re x[k]", ms=ms)
        stem(ax, k, x.imag, C_IM, "Im x[k]", ms=ms)
        ax.set_title(f"N = {N}" + (" (= N₀)" if m == 1 else f" (= {m}N₀)"))
        ax.set_xlabel("sample k")
        ax.set_xlim(-0.8, N - 0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel("x[k]")
    axs[-1].legend(loc="upper right", ncols=3)
    fig.suptitle(f"{g} — variant{'s' if len(nos) > 1 else ''} {', '.join(map(str, nos))}:  x_μ = [{vec}]ᵀ",
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, f"fig{2 + int(g[1:]):02d}_{g}_variants_{'_'.join(map(str, nos))}")


# %% [markdown]
# ### 7.1 G1 — variants 1, 3, 11, 13

# %%
plot_group("G1")

# %% [markdown]
# ### 7.2 G2 — variants 2, 12

# %%
plot_group("G2")

# %% [markdown]
# ### 7.3 G3 — variants 4, 14

# %%
plot_group("G3")

# %% [markdown]
# ### 7.4 G4 — variants 5, 15

# %%
plot_group("G4")

# %% [markdown]
# ### 7.5 G5 — variant 6

# %%
plot_group("G5")

# %% [markdown]
# ### 7.6 G6 — variant 7

# %%
plot_group("G6")

# %% [markdown]
# ### 7.7 G7 — variant 8

# %%
plot_group("G7")

# %% [markdown]
# ### 7.8 G8 — variant 9

# %%
plot_group("G8")

# %% [markdown]
# ### 7.9 G9 — variant 10

# %%
plot_group("G9")

# %% [markdown]
# ### 7.10 The synthesized signal as a linear combination of eigensignals (variant 1, N = 10)
#
# x_k = (1/N)·W·x_μ = Σ_μ (X[μ]/N)·w_μ: each non-zero coefficient contributes one column of W, i.e. one complex
# exponential with μ periods in N samples, scaled by X[μ]/N. The last panel shows that the sum of the six
# contributions equals the IDFT result.

# %%
X1, N1 = VARIANTS[1], VARIANTS[1].size
W10 = build_W(N1)
k1 = np.arange(N1)
contrib = [X1[mu]/N1 * W10[:, mu] for mu in range(N1) if X1[mu] != 0]
assert np.allclose(np.sum(contrib, axis=0), SIG[(1, N1)])
fig, axs = plt.subplots(2, 4, figsize=(10.5, 4.6), sharex=True, sharey=True)
for ax, mu in zip(axs.flat, np.flatnonzero(X1)):
    c = X1[mu]/N1 * W10[:, mu]
    stem(ax, k1, c.real, C_RE, "Re", ms=3.2)
    stem(ax, k1, c.imag, C_IM, "Im", ms=3.2)
    ax.set_title(f"μ = {mu}:  (X[{mu}]/N)·w_{mu},  X[{mu}] = {X1[mu]:g}", fontsize=8.5)
ax = axs.flat[len(contrib)]
x1 = SIG[(1, N1)]
stem(ax, k1, x1.real, C_RE, "Re", ms=3.2)
stem(ax, k1, x1.imag, C_IM, "Im", ms=3.2)
ax.set_title("sum = x_k = (1/N)·W·x_μ", fontsize=8.5)
axs.flat[-1].axis("off")
axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(), loc="center")
for ax in axs[-1]:
    ax.set_xlabel("sample k")
fig.suptitle("Variant 1, N = 10: contributions of the non-zero DFT coefficients", x=0.01, ha="left")
fig.tight_layout()
save(fig, "fig12_eigensignal_decomposition_V1")

# %% [markdown]
# ## 8. Effect of N
#
# With zero-padding, x[k] = (1/N)·Σ_μ X[μ]·e^(j2πμk/N) = (1/N)·P(k/N), where P(t) = Σ_μ X[μ]·e^(j2πμt) is a
# periodic trigonometric polynomial (period 1) that does not depend on N. Increasing N therefore **samples the same
# periodic function more densely** (N samples per period) and scales the amplitude by 1/N. The figure shows P(t)
# (lines) and the scaled samples N·x[k] at t = k/N for N = N₀ (circles) and N = 4N₀ (dots) for all nine groups.

# %%
t = np.linspace(0, 1, 2001)
fig, axs = plt.subplots(3, 3, figsize=(10.5, 7.4), sharex=True)
for ax, (g, nos) in zip(axs.flat, GROUPS.items()):
    X0 = VARIANTS[nos[0]]
    mu = np.arange(X0.size)
    P = np.exp(1j*2*np.pi*np.outer(t, mu)) @ X0
    ax.plot(t, P.real, color=C_RE, lw=1.0, label="Re P(t)")
    ax.plot(t, P.imag, color=C_IM, lw=1.0, label="Im P(t)")
    for m, ms, mk in ((1, 5.0, "o"), (4, 2.6, "o")):
        N = m*X0.size
        xN = N*SIG[(nos[0], N)]
        tk = np.arange(N)/N
        assert np.allclose(xN, np.exp(1j*2*np.pi*np.outer(tk, mu)) @ X0)   # samples lie on P(t)
        face = "white" if m == 1 else None
        ax.plot(tk, xN.real, mk, ms=ms, color=C_RE, mfc=face or C_RE, mew=0.9)
        ax.plot(tk, xN.imag, mk, ms=ms, color=C_IM, mfc=face or C_IM, mew=0.9)
    ax.set_title(f"{g} (V{', V'.join(map(str, nos))}),  N₀ = {X0.size}", fontsize=8.5)
for ax in axs[-1]:
    ax.set_xlabel("t = k/N  (one period)")
for ax in axs[:, 0]:
    ax.set_ylabel("N·x[k]")
h = [plt.Line2D([], [], color=C_RE, lw=1), plt.Line2D([], [], color=C_IM, lw=1),
     plt.Line2D([], [], ls="", marker="o", color=INK2, mfc="white", ms=5),
     plt.Line2D([], [], ls="", marker="o", color=INK2, ms=2.6)]
fig.legend(h, ["Re P(t)", "Im P(t)", "N·x[k], N = N₀", "N·x[k], N = 4N₀"], loc="lower center", ncols=4,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("All N sample the same periodic function P(t) = Σ X[μ]·e^(j2πμt); x[k] = P(k/N)/N",
             x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, "fig13_effect_of_N")

# %%
# Same non-zero coefficients, different N0: G1 (N0 = 10) and G3 (N0 = 9) sample the same P(t)
g1, g3 = VARIANTS[1], VARIANTS[4]
assert np.array_equal(g1[:6], g3[:6]) and not g1[6:].any() and not g3[6:].any()
assert np.allclose(10*SIG[(1, 10)][0], 9*SIG[(4, 9)][0])
print(f"x[0]: variant 1 (N = 10) = {SIG[(1, 10)][0].real:.3f},  variant 4 (N = 9) = {SIG[(4, 9)][0].real:.3f};  "
      f"N·x[0] = ΣX = {g1.sum():g} for both")
print("x[0] of variant 1 for N = 10, 20, 40:", [round(float(SIG[(1, N)][0].real), 3) for N in (10, 20, 40)])

# %% [markdown]
# ## 9. Interpretation
#
# **Is x[k] real or complex?** Complex, for every variant and every N (max |Im x[k]| > 0.1 everywhere). A real
# signal requires a Hermitian-symmetric spectrum X[μ] = X*[(N − μ) mod N]; here all coefficients above μ = 8 are
# zero while X[1] … X[8] are not, so the symmetry is violated. The signals contain only the "positive-frequency"
# complex exponentials e^(+j2πμk/N), which rotate counter-clockwise — a sum of cosines in the real part and of sines
# in the imaginary part.
#
# **Structure of the signal.** Because all X[μ] are real, x[(−k) mod N] = x*[k]: the real part is circularly even and
# the imaginary part circularly odd (Im x[0] = 0, and Im x[N/2] = 0 for even N), which is visible in every plot.
# All coefficients are non-negative, so all exponentials are in phase at k = 0: x[0] = ΣX[μ]/N is real and is the
# maximum of |x[k]| (e.g. 24/10 = 2.4 for variant 1). The DC value is mean(x) = X[0]/N, and the energy satisfies
# Parseval's theorem. Each signal is the linear combination (1/N)·Σ X[μ]·w_μ of the eigensignals (columns of W),
# Section 7.10.
#
# **How does N change the signal?** Appending zeros to x_μ does not add new components; it only changes the
# frequency grid: index μ now corresponds to Ω_μ = 2πμ/N, i.e. μ periods in N samples. The synthesized signal is
# the same periodic function P(t) sampled with N points per period and scaled by 1/N (Section 8): for N = 2N₀ and
# 4N₀ the signal looks smoother and "slower" (more samples per period) and its amplitude halves with every doubling
# of N (x[0] of variant 1: 2.4 → 1.2 → 0.6). With N = N₀ the signal is sampled so coarsely that its shape is hard to
# recognize; the trailing zeros in the given vectors only decide the IDFT length N₀ (variants 1 and 4 have the same
# non-zero coefficients and sample the same P(t) with 10 and 9 points).
#
# **K and W.** K and W depend only on N (four pairs for N = N₀ cover all variants). W is symmetric, its columns are
# orthogonal and W⁻¹ = W*/N, so the matrix IDFT is exactly invertible by the matrix DFT and agrees with
# `numpy.fft.ifft` to machine precision.

# %% [markdown]
# ## 10. Conclusions
#
# 1. The IDFT was implemented in matrix notation x_k = (1/N)·W·x_μ with W = e^(+j2π/N ⊙ K) and K = outer(k, μ),
#    validated against eq. (10), the example of Sec. 6 and `numpy.fft.ifft`.
# 2. All 15 variants (9 different vectors) were synthesized for N = N₀, 2N₀ and 4N₀; K and W are shown for all
#    N₀ ∈ {9, 10, 11, 13}, and the synthesized signals are plotted for every variant.
# 3. The synthesized signals are complex, because the given spectra are not Hermitian-symmetric; their real part is
#    even, their imaginary part odd, and the maximum |x[0]| = ΣX/N occurs at k = 0.
# 4. Increasing N by zero-padding the spectrum samples the same periodic signal more densely and scales it by 1/N;
#    it does not change the frequency content.
