# DSP Lab 1 – Signal synthesis with the IDFT in matrix notation

Digital Signal Processing (prof. dr hab. Vasyl Martsenyuk), UBB — Radosław Skrzypczyński, Informatyka II st., Gr. 2.

**Task:** synthesize a discrete-time signal by using the IDFT in matrix notation, x_k = (1/N)·W·x_μ with
W = exp(+j2π/N ⊙ K) and K = outer(k, μ), for different values of N; show the matrices W and K; plot the
synthesized signal. All 15 variants of the instructions are computed (9 different vectors, N = N₀, 2N₀, 4N₀).

| File | Content |
|---|---|
| `Lab1_IDFT_Synthesis.ipynb` | executed notebook (theory, validation, matrices K and W, synthesis, plots, interpretation) |
| `Lab1_IDFT_Synthesis.pdf` | PDF export of the notebook |
| `Report_Lab1.pdf` / `.docx` | report based on the course template |
| `src/Lab1_IDFT_Synthesis.py` | the notebook as a plain Python script (jupytext percent format) |
| `figures/` | all figures of the notebook |
| `scripts/` | PDF export of the notebook and report generator |

Reproduce (Python 3.14, numpy, matplotlib, pandas, jupytext, nbconvert):

```
py -3.14 -m jupytext --to ipynb --output Lab1_IDFT_Synthesis.ipynb src/Lab1_IDFT_Synthesis.py
py -3.14 -m jupyter nbconvert --to notebook --execute --inplace Lab1_IDFT_Synthesis.ipynb
py -3.14 -X utf8 scripts/export_notebook_pdf.py Lab1_IDFT_Synthesis.ipynb Lab1_IDFT_Synthesis.pdf
py -3.14 -X utf8 scripts/make_report.py path/to/DSP_TemplateReport.docx
```
