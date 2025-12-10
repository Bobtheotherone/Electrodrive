---
{
  "id": "planar_spectral_integral_representation",
  "title": "Spectral Integral Representation",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      1,
      2
    ],
    "ranges": [
      {
        "page": 1,
        "line_start": 50,
        "line_end": 222
      },
      {
        "page": 2,
        "line_start": 1,
        "line_end": 23
      }
    ]
  },
  "tags": [
    "planar",
    "sommerfeld-integral",
    "fourier",
    "reflection-coefficient",
    "spectral-representation",
    "multi-layer",
    "convergence",
    "complex-images"
  ],
  "urls": [],
  "years_mentioned": [
    1964,
    1978
  ]
}
---

# Spectral Integral Representation

## Raw assets

- Page 01 image: `raw/page_images/page_01.png`
- Page 01 text (PyMuPDF): `raw/text_pymupdf/page_01.txt`
- Page 01 text (pdfplumber): `raw/text_pdfplumber/page_01.txt`
- Page 02 image: `raw/page_images/page_02.png`
- Page 02 text (PyMuPDF): `raw/text_pymupdf/page_02.txt`
- Page 02 text (pdfplumber): `raw/text_pdfplumber/page_02.txt`

## Source (verbatim)

```text
Spectral Integral Representation: The same result can be obtained by a Fourier/Sommerfeld integral. The
full-space Green’s function in homogeneous medium is  
. For the half-space, one finds an
azimuthal Fourier-Bessel transform:
with 
 (static limit of the TM-wave reflection coefficient)
. Evaluating the
integral yields the above image expression. The term with 
 corresponds to the reflected field from
the interface. For multi-layer planar stratifications, one obtains multiple reflections: e.g. for a 3-layer (two
interfaces) system, the Green’s function can be written as a similar Sommerfeld integral but with a 
frequency-dependent reflection coefficient 
 that is a rational function encoding multiple internal
reflections. Barrera, Guzmán & Balaguer (1978) used the method of images to derive such integrals for a
q
d
ϵ1
ϵ2
q′
q
ϵ + ϵ
1
2
ϵ − ϵ
1
2
1
ϵ <
2
ϵ1
q′
q
1
z > 0
V
(ρ, z) =
above
[
+
4πϵ1
q
ρ + (z − d)
2
2
1
],
ϵ + ϵ
1
2
ϵ − ϵ
1
2
ρ + (z + d)
2
2
1
2
V
z = 0
ϵEz
3
2
z < 0
E
=
z<0
(1 − ν)Eorig
ν = ϵ +ϵ
2
1
ϵ −ϵ
2
1
Ez
4
5
z < 0
2ϵ /(ϵ +
1
1
ϵ )
2
6
ϵ =
1
1
Vz<0
2q/(ϵ +
1
ϵ )
2
ϵ →
2
∞
q =
′
−q
2
ϵ →
2
0
q =
′
+q
σ (ρ) =
b
−ν 2π(ρ +d )
2
2 3/2
q d
7
8
ν = 1
σ(ρ) = − 2π(ρ +d )
2
2 3/2
q d
9
10
F =
z
− 16πϵ ϵ d
0 1
2
q q′
ϵ >
2
ϵ1
ϵ <
2
ϵ1
11
12
G =
0
4πϵ
1
R
1
G(ρ, z, z ) =
′
[e
+
2π
1 ∫
0
∞
2ϵ k
1
1
−k∣z−z ∣′
R (k) e
]J (kρ) k dk,
p
−k(z+z )
′
0
R (k) =
p
(ϵ −
1
ϵ )/(ϵ +
2
1
ϵ )
2
6
2
e−k(z+z )
′
R (k)
p
1
three-dielectric planar configuration
. The resulting potential expressions involve piecewise-defined
integrals (or infinite image charge series) for each region that satisfy the boundary conditions at both
interfaces. In general, no finite number of discrete image charges can exactly represent the field in multi-
layer dielectric media; instead one obtains either infinite image series or continuous distributions
(integrals). For instance, a point charge between two dielectric half-spaces produces an infinite series of
images in each layer (analogous to multiple mirror images in a Fabry–Pérot cavity) which sums to the exact
Green’s function
. These series/integrals converge rapidly when the source is far from the interfaces,
but can converge slowly for near-interface sources or high-contrast interfaces. Various techniques
(asymptotic extraction, Ewald summation, numerical steepest-descent integration) are used to accelerate
convergence of Sommerfeld integrals
. For quasi-static problems, it is common to partition the
Green’s function into a “direct” term plus a rapidly decaying reflected field; e.g. one can subtract the 
singular contribution and handle it analytically, then integrate the remainder numerically with fewer
discretization points
. Approximations via discrete image charges are also used: for example, Wait &
Spies (1964) proposed complex image charges to approximate the Sommerfeld integral of a dipole field in a
conductive half-space
 (the complex image method), achieving high accuracy by fitting a few images to
the continuous spectrum. Similar rational approximations exist for static/dc fields in layered media to avoid
direct integration. However, these are approximations; the exact solution generally resides in the spectral
integral form or an infinite series of images.
```