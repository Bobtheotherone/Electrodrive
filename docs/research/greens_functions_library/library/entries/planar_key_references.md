---
{
  "id": "planar_key_references",
  "title": "Key References (Planar Media)",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      2,
      3
    ],
    "ranges": [
      {
        "page": 2,
        "line_start": 43,
        "line_end": 112
      },
      {
        "page": 3,
        "line_start": 1,
        "line_end": 2
      }
    ]
  },
  "tags": [
    "planar",
    "references",
    "sommerfeld",
    "jackson",
    "landau-lifshitz",
    "chew",
    "kong",
    "barrera"
  ],
  "urls": [],
  "years_mentioned": [
    1909,
    1965,
    1978,
    1986,
    1990,
    2015
  ]
}
---

# Key References (Planar Media)

## Raw assets

- Page 02 image: `raw/page_images/page_02.png`
- Page 02 text (PyMuPDF): `raw/text_pymupdf/page_02.txt`
- Page 02 text (pdfplumber): `raw/text_pdfplumber/page_02.txt`
- Page 03 image: `raw/page_images/page_03.png`
- Page 03 text (PyMuPDF): `raw/text_pymupdf/page_03.txt`
- Page 03 text (pdfplumber): `raw/text_pdfplumber/page_03.txt`

## Source (verbatim)

```text
Key References (Planar Media): The plane interface problem was treated by Sommerfeld (1909) for EM
fields (the static limit yields the above forms).  Landau & Lifshitz derive the image charge result for a
dielectric half-space in Electrodynamics of Continuous Media (Problem 1 to §8). Jackson (3rd ed, §4.4) also
covers a point charge above a dielectric half-space (leading to a surface-charge integral solution and the
image-charge interpretation).  Ramo, Whinnery & Van Duzer (Fields and Waves, 1965) give the closed-
form solution for the static interface using Fourier transforms
. For multiple planar layers, Barrera
et al, Am. J. Phys. 46, 1172 (1978) gave explicit Fourier-integral forms (corrected by Serra et al (2015) for
sign errors
). The general formalism for layered media Green’s functions is given in texts like Chew
(1990) and  Kong (1986), using transmission line analogies for the reflection coefficient  
. In static
cases,  
 becomes real and non-dispersive (independent of   for Laplace’s equation), which is why the
integrals often evaluate to simple image expressions – indeed for a purely dielectric discontinuity the
integrand  
 is constant, allowing analytic integration to yield the simple  
 form. This
ceases to be true for frequency-dependent or more complicated stratifications, where 
 is a function of
13
14
13
15
16
17
k = 0
18
19
20
z = 0+
q′
2q/(ϵ +
1
ϵ )
2
6
21
R
≈
eff
R
+
12
R23
O(t)
t
(ϵ −
2
ϵ )/(ϵ +
1
2
ϵ )
1
22
23
24
25
R (k)
p
Rp
k
Rp
1/
ρ + (z + d)
2
2
R (k)
p
2
the spectral parameter and no closed-form point-image exists (necessitating the so-called “Sommerfeld
integrals”).
```