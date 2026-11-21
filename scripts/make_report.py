"""Build Report_Lab1.docx / Report_Lab1.pdf from the course template.

Usage:  py -3.14 -X utf8 scripts/make_report.py [path/to/DSP_TemplateReport.docx]
Run after the notebook has been executed (the figures are taken from figures/).
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from report_builder import Report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parents[1] / "Lab3" / "DSP_TemplateReport.docx"
REPO = "https://github.com/Slayzerus/dsp-lab1"

VARIANTS = {
    1: [6, 2, 4, 3, 4, 5, 0, 0, 0, 0], 2: [10, 5, 6, 6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    3: [6, 2, 4, 3, 4, 5, 0, 0, 0, 0], 4: [6, 2, 4, 3, 4, 5, 0, 0, 0], 5: [6, 4, 4, 5, 3, 4, 5, 0, 0, 0, 0],
    6: [7, 2, 4, 3, 4, 5, 0, 0, 0, 0], 7: [6, 8, 2, 4, 3, 4, 5, 0, 0, 0], 8: [6, 2, 4, 4, 4, 5, 0, 0, 0, 0],
    9: [6, 5, 4, 3, 4, 5, 0, 0, 0, 0], 10: [6, 2, 4, 3, 4, 4, 0, 0, 0, 0], 11: [6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    12: [10, 5, 6, 6, 2, 4, 3, 4, 5, 0, 0, 0, 0], 13: [6, 2, 4, 3, 4, 5, 0, 0, 0, 0],
    14: [6, 2, 4, 3, 4, 5, 0, 0, 0], 15: [6, 4, 4, 5, 3, 4, 5, 0, 0, 0, 0],
}
VARIANTS = {k: np.array(v, float) for k, v in VARIANTS.items()}
GROUPS, _seen = {}, {}
for no, X in VARIANTS.items():
    _seen.setdefault(tuple(X), f"G{len(_seen) + 1}")
    GROUPS.setdefault(_seen[tuple(X)], []).append(no)
GROUP_OF = {no: g for g, nos in GROUPS.items() for no in nos}


def build_W(N):
    K = np.arange(N).reshape(-1, 1) @ np.arange(N).reshape(1, -1)
    return K, np.exp(1j*2*np.pi/N*K)


def synth(X0, N):
    X = np.zeros(N, complex)
    X[:X0.size] = X0
    return build_W(N)[1] @ X / N


def f3(v):
    v = round(float(v), 3) + 0.0
    return f"{v:.3f}"


rep = Report(TEMPLATE, lab=1, date=date.today().strftime("%d.%m.%Y"),
             topic="Spectral analysis of deterministic signals: signal synthesis with the IDFT in matrix notation",
             variant="wszystkie (1–15)")

# 1 ---------------------------------------------------------------------------------------------------------------
rep.heading("1. Problem statement")
rep.para("Synthesize a discrete-time signal by using the inverse discrete Fourier transform (IDFT) in matrix "
         "notation for different values of N. Show the matrices **W** and **K**. Plot the signal synthesized. "
         "The DFT coefficient vectors x_μ of the 15 variants are given in Sec. 7 of the instructions; "
         "**all 15 variants were computed**.")
rep.para("Following the instructions (eq. 5–9), the IDFT is the matrix product "
         "**x_k = (1/N)·W·x_μ**, where x_μ = (X[0], …, X[N−1])ᵀ contains the DFT "
         "coefficients and x_k = (x[0], …, x[N−1])ᵀ the samples. The Fourier matrix is "
         "**W = e^(+j2π/N ⊙ K)** (element-wise), i.e. W[k, μ] = W_N^(kμ) with the twiddle factor "
         "W_N = e^(+j2π/N), and **K = (0, 1, …, N−1)ᵀ·(0, 1, …, N−1)** is the outer "
         "product with K[k, μ] = k·μ. The DFT is x_μ = W*·x_k, because W⁻¹ = W*/N. "
         "The columns of W are the orthogonal DFT eigensignals w_μ[k] = e^(j2πkμ/N), so the IDFT "
         "builds x_k as their linear combination weighted by X[μ]/N.")

# 2 ---------------------------------------------------------------------------------------------------------------
rep.heading("2. Input data")
rep.para("The 15 coefficient vectors contain only **9 different vectors** (groups G1–G9). The vector length "
         "N₀ ∈ {9, 10, 11, 13} defines the basic IDFT length. Every variant was synthesized for "
         "**N = N₀, 2N₀ and 4N₀**; for N > N₀ the coefficient vector is extended with zeros "
         "(X[μ] = 0 for N₀ ≤ μ < N).")
rows = [(no, "[" + ", ".join(f"{v:g}" for v in X) + "]ᵀ", X.size, f"{X.size}, {2*X.size}, {4*X.size}",
         GROUP_OF[no]) for no, X in VARIANTS.items()]
rep.table(["No", "x_μ", "N₀", "N used", "group"], rows, caption="Variants of the task", size=9,
          col_widths=[1.0, 8.0, 1.2, 2.6, 1.4])
rep.para("Environment: Python 3.14.2, NumPy 2.4.1, Matplotlib 3.10.8, pandas 3.0.6, Jupyter (ipykernel 7.4).",
         size=11)

# 3 ---------------------------------------------------------------------------------------------------------------
rep.heading("3. Commands used")
rep.subheading("a) Source code (core of the notebook Lab1_IDFT_Synthesis.ipynb)")
rep.code('''
def build_K(N):
    k = np.arange(N).reshape(-1, 1)      # column vector (N x 1)
    mu = np.arange(N).reshape(1, -1)     # row vector (1 x N)
    return k @ mu                        # outer product: K[k, mu] = k*mu

def build_W(N):
    return np.exp(+1j * 2*np.pi/N * build_K(N))    # W = exp(+j 2pi/N (.) K)

def idft_matrix(X, N=None):
    X = pad(X, len(X) if N is None else N)         # zero-padded spectrum of length N
    return build_W(X.size) @ X / X.size            # x_k = (1/N) W x_mu

def dft_matrix(x):
    return np.conj(build_W(x.size)) @ x            # x_mu = W* x_k

for no, X0 in VARIANTS.items():                    # all 15 variants
    for m in (1, 2, 4):                            # N = N0, 2N0, 4N0
        N = m * X0.size
        x = idft_matrix(X0, N)
        assert np.allclose(x, np.fft.ifft(pad(X0, N)))     # check with numpy
        assert np.allclose(dft_matrix(x), pad(X0, N))      # DFT returns x_mu
''')
rep.para("The implementation was first validated against the instructions: K and W for N = 4 equal eq. (10), "
         "W is symmetric, W⁻¹ = W*/N, W/√N is unitary, the eigensignals are orthogonal, and the "
         "initial example of Sec. 6 (N = 8, x_μ = [8, 2, 4, 0, 0, 0, 0, 0]ᵀ) gives the same signal as "
         "`numpy.fft.ifft` with the DC value mean(x) = X[0]/N = 1.")
rep.subheading("b) Screenshots: the matrices K and W")
rep.para("K and W depend only on N. The notebook lists the complete matrices for all four lengths "
         "N₀ = 9, 10, 11 and 13. The matrices for N = 10 (variants 1, 3, 6–11, 13) are reproduced "
         "below; W is given by its real part cos(2πkμ/N) and imaginary part sin(2πkμ/N).")
K10, W10 = build_W(10)
hdr = ["k \\ μ"] + [str(m) for m in range(10)]
rep.table(hdr, [[k] + list(K10[k]) for k in range(10)], caption="Matrix K for N = 10", size=8.5)
rep.table(hdr, [[k] + [f3(v) for v in W10.real[k]] for k in range(10)],
          caption="Matrix W for N = 10: real part Re W = cos(2πK/N)", size=7.5)
rep.table(hdr, [[k] + [f3(v) for v in W10.imag[k]] for k in range(10)],
          caption="Matrix W for N = 10: imaginary part Im W = sin(2πK/N)", size=7.5)
rep.figure(FIG / "fig02_K_W_structure.png",
           "Structure of K, K mod N, Re W and Im W for N = 10, 20, 40 (variant 1: N₀, 2N₀, 4N₀). "
           "Row/column 0 of W are ones (DC); column μ contains μ periods of a cosine/sine.", 16.0)
rep.link("Link to the remote repository (GitHub): ", REPO)
rep.para("The repository contains the notebook (Lab1_IDFT_Synthesis.ipynb), its PDF export "
         "(Lab1_IDFT_Synthesis.pdf), the source in plain Python (src/), all figures (figures/) and this report.",
         size=11)

# 4 ---------------------------------------------------------------------------------------------------------------
rep.heading("4. Outcomes")
rep.para("All 45 syntheses (15 variants × 3 values of N) agree with `numpy.fft.ifft` to machine precision, the "
         "matrix DFT W*·x_k returns the coefficients exactly, Parseval's theorem "
         "Σ|x[k]|² = (1/N)·Σ|X[μ]|² holds, and identical vectors (e.g. variants 1, 3, 11, 13) "
         "give identical signals. The synthesized signals are plotted below, one figure per group "
         "(blue: Re x[k], orange: Im x[k], grey line: |x[k]|).")
for g, nos in GROUPS.items():
    X0 = VARIANTS[nos[0]]
    stem = f"fig{2 + int(g[1:]):02d}_{g}_variants_{'_'.join(map(str, nos))}.png"
    vec = ", ".join(f"{v:g}" for v in X0)
    rep.figure(FIG / stem, f"{g} – variant{'s' if len(nos) > 1 else ''} {', '.join(map(str, nos))} "
                           f"(x_μ = [{vec}]ᵀ): synthesized signal for N = {X0.size}, {2*X0.size}, "
                           f"{4*X0.size}", 16.0)
rows = []
for g, nos in GROUPS.items():
    X0 = VARIANTS[nos[0]]
    xs = [synth(X0, m*X0.size) for m in (1, 2, 4)]
    rows.append([g, ", ".join(map(str, nos)), X0.size] + [f3(x[0].real) for x in xs]
                + [f3(xs[0].mean().real), f3(np.abs(xs[0].imag).max())])
rep.table(["group", "variants", "N₀", "x[0], N = N₀", "x[0], N = 2N₀", "x[0], N = 4N₀",
           "DC = X[0]/N₀", "max|Im x|, N₀"], rows,
          caption="Key values: x[0] = ΣX[μ]/N is the maximum of |x[k]|; DC = mean(x[k])", size=8.5)
rep.figure(FIG / "fig12_eigensignal_decomposition_V1.png",
           "Variant 1, N = 10: each non-zero coefficient adds one scaled column of W; their sum equals x_k", 16.0)
rep.figure(FIG / "fig13_effect_of_N.png",
           "Effect of N: the scaled samples N·x[k] for N = N₀ and 4N₀ lie on the same periodic function "
           "P(t) = Σ X[μ]·e^(j2πμt), i.e. x[k] = P(k/N)/N", 16.0)

# 5 ---------------------------------------------------------------------------------------------------------------
rep.heading("5. Conclusions")
rep.para("For the reasons given, we conclude that:")
rep.bullets([
    "The IDFT in matrix notation x_k = (1/N)·W·x_μ with W = e^(+j2π/N ⊙ K) and "
    "K = outer(k, μ) reproduces `numpy.fft.ifft` exactly; W is symmetric, its columns are orthogonal and "
    "W⁻¹ = W*/N, so the synthesis is exactly inverted by the matrix DFT. K and W depend only on N.",
    "All synthesized signals are **complex**: a real signal would require a Hermitian-symmetric spectrum "
    "X[μ] = X*[N − μ], but the given vectors contain only low indices (μ ≤ 8) followed by "
    "zeros. Because the coefficients are real, Re x[k] is circularly even and Im x[k] odd.",
    "All coefficients are non-negative, so all eigensignals are in phase at k = 0: x[0] = ΣX[μ]/N is real "
    "and is the maximum of |x[k]| (e.g. 24/10 = 2.4 for variant 1); the DC value is X[0]/N.",
    "Increasing N by appending zeros does not add frequency components: the same periodic function is sampled "
    "with N points per period and scaled by 1/N (x[0] of variant 1: 2.4 → 1.2 → 0.6 for N = 10, 20, 40). "
    "Larger N gives a smoother, more readable plot of the same signal; the trailing zeros in the given vectors "
    "only set the basic length N₀ (variants 1 and 4 share the same non-zero coefficients).",
], numbered=True)

rep.save(ROOT / "Report_Lab1.docx", ROOT / "Report_Lab1.pdf")
