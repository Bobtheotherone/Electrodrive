---
{
  "id": "general_failure_of_finite_images",
  "title": "Failure of Finite Image Solutions",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      6,
      7
    ],
    "ranges": [
      {
        "page": 6,
        "line_start": 50,
        "line_end": 89
      },
      {
        "page": 7,
        "line_start": 1,
        "line_end": 11
      }
    ]
  },
  "tags": [
    "method-of-images",
    "nonexistence",
    "uniqueness-theorem",
    "branch-cuts",
    "integral-representations"
  ],
  "urls": [],
  "years_mentioned": []
}
---

# Failure of Finite Image Solutions

## Raw assets

- Page 06 image: `raw/page_images/page_06.png`
- Page 06 text (PyMuPDF): `raw/text_pymupdf/page_06.txt`
- Page 06 text (pdfplumber): `raw/text_pdfplumber/page_06.txt`
- Page 07 image: `raw/page_images/page_07.png`
- Page 07 text (PyMuPDF): `raw/text_pymupdf/page_07.txt`
- Page 07 text (pdfplumber): `raw/text_pdfplumber/page_07.txt`

## Source (verbatim)

```text
Failure of Finite Image Solutions: In summary, apart from the canonical geometries (plane, sphere,
infinite cylinder,  maybe ellipsoid in certain orientations),  finite discrete image systems do not exist for
most conductor–dielectric setups
. The method of images “fails” in the sense that one needs either
an infinite number of images or an image distribution (line, surface, etc.). A famous example is the dielectric
η
Pν
g
C
C ∼
2πϵ a/ ln(4a/g) +
0
O(1)
l
E ∼
2πϵ A
0
gap
q
Agap
λ = g/a
λ ≪ 1
E
∼
max
λ
1
N
∼ (a − d)
/d
N+1
N+1
d ≫ a
−q
(a/d)2(N+1)
g
g
59
60
6
sphere: no finite collection of Coulomb charges can satisfy the boundary conditions
. Another example:
an ellipsoidal conductor requires an infinite series of images lying along a line (the so-called Pellat’s solution
involves an integral of line charges). Similarly, a  point charge in front of a dielectric slab cannot be
represented by a few images – one gets a continuous charge “image” smeared along the perpendicular
through the slab. These impossibility results stem from the uniqueness theorem: the Green’s function in
such cases has branch-cut singularities (the continuous spectra corresponding to continuum of image
charges). A rigorous statement is that the method of images in the form of finite discrete sources works iff
the boundary is an equipotential of a conformal inversion of a sphere or plane (so basically plane, sphere, or
special cases like coaxal cylinders in 2D). For arbitrary shapes or layered media, one resorts to infinite series
or integrals.
```