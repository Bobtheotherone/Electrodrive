---
{
  "id": "general_known_results_summary",
  "title": "Known Results Summary",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      7,
      8
    ],
    "ranges": [
      {
        "page": 7,
        "line_start": 69,
        "line_end": 148
      },
      {
        "page": 8,
        "line_start": 1,
        "line_end": 15
      }
    ]
  },
  "tags": [
    "summary",
    "special-cases",
    "reference-list"
  ],
  "urls": [],
  "years_mentioned": [
    1845,
    1883,
    1912,
    1934,
    1941,
    1952,
    1980,
    1988,
    1990,
    1993,
    2019
  ]
}
---

# Known Results Summary

## Raw assets

- Page 07 image: `raw/page_images/page_07.png`
- Page 07 text (PyMuPDF): `raw/text_pymupdf/page_07.txt`
- Page 07 text (pdfplumber): `raw/text_pdfplumber/page_07.txt`
- Page 08 image: `raw/page_images/page_08.png`
- Page 08 text (PyMuPDF): `raw/text_pymupdf/page_08.txt`
- Page 08 text (pdfplumber): `raw/text_pdfplumber/page_08.txt`

## Source (verbatim)

```text
Known Results Summary: To conclude this technical compendium, we list some notable special-case
solutions and references: (1) A charge above a dielectric half-space – exact potential via one image charge
(with magnitude 
)
, first derived by Sommerfeld and found in many textbooks
(e.g. Jackson, Eq. 4.19). (2) Charge outside a grounded conducting sphere – Kelvin’s image solution (1845)
. (3) Charge outside a dielectric sphere – infinite series solution (e.g. Stratton 1941, p. 147; Kirkwood
41
β = ϵ +ϵ
2
1
ϵ −ϵ
2
1
s = d/a
g/a
β
B ≈
l
β q/(4πϵ ) (a/d)
1
l+1
β → ±1
β → 1
ϵ
B =
l
B
+
l
(∞)
ΔBl
Bl
(∞)
l > 0
ΔBl
1/ϵ2
1/ϵ2
ϵ
β ≈ −1
ϵ =
2
ϵ δ
1
δ ≪ 1
g ≪ a
g/a
ln(a/g)
g/a
a/d ≪ 1
Q
Q/(4πϵ r)+
1
∼ (a/d)l
∼ O((a/d) )
3
l
σ ∼ (g/ρ)1/2
distance to contact
N
qN
O(q /q)
N
q /q ∼
N
(a/d)N
d
∼ (1 − g/a)N
q =
′
(ϵ −
1
ϵ )/(ϵ +
2
1
ϵ )q
2
1
26
7
1934, J. Chem. Phys.), no finite image; image interpreted as point + line charge
.  (4) Charge inside a
dielectric sphere – similar series (see  Neumann 1883 for early treatment
).  (5) Coated sphere with
concentric layers – series solution (see Mie theory static limit; Pyati 1993 for inversion approach
). (6)
Two conducting spheres – bispherical harmonic expansion (see Jeffery 1912, Snow 1952 for capacitance
series). (7) Dielectric sphere-plane – no closed form; see Image multipole method by Zhukauskas 1988
(approximate) or numerical BEM. (8) General layered Green’s function – spectral integrals (Sommerfeld) as
in Chew’s “Waves and Fields in Inhomogeneous Media” (1990). (9) Novel solution techniques – e.g. Majic
& Le Ru (2019) “logopoles” for fast sphere series
; Yaghjian (1980) for complex images in stratified
media. These sources provide a wealth of formulas, convergence analysis, and guidance for tackling new
configurations that might admit near-analytic solutions via clever combinations of known Green’s functions
and transforms.
```