---
{
  "id": "spherical_multiple_spheres_sphere_plane",
  "title": "Multiple Spheres and Sphere–Plane Configurations",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      5,
      6
    ],
    "ranges": [
      {
        "page": 5,
        "line_start": 72,
        "line_end": 191
      },
      {
        "page": 6,
        "line_start": 1,
        "line_end": 36
      }
    ]
  },
  "tags": [
    "sphere-plane",
    "two-spheres",
    "iterated-images",
    "bispherical-coordinates",
    "capacitance",
    "narrow-gap"
  ],
  "urls": [],
  "years_mentioned": [
    1897,
    1934,
    1977
  ]
}
---

# Multiple Spheres and Sphere–Plane Configurations

## Raw assets

- Page 05 image: `raw/page_images/page_05.png`
- Page 05 text (PyMuPDF): `raw/text_pymupdf/page_05.txt`
- Page 05 text (pdfplumber): `raw/text_pdfplumber/page_05.txt`
- Page 06 image: `raw/page_images/page_06.png`
- Page 06 text (PyMuPDF): `raw/text_pymupdf/page_06.txt`
- Page 06 text (pdfplumber): `raw/text_pdfplumber/page_06.txt`

## Source (verbatim)

```text
Multiple Spheres and Sphere–Plane Configurations: The method of images can be iterated to handle two
conducting spheres or a sphere near a conducting plane, but the result is an infinite recursive series of
images. For example, a conducting sphere above a conducting infinite plane: the plane induces an image
charge (Kelvin’s method for a plane says put 
 below the plane); but that image below the plane is like a
point charge near the sphere (below it), which induces another image in the sphere, which then induces
another in the plane, and so on ad infinitum. Physically, the two objects “mirror” charges back and forth. In
fact, using the sphere’s Kelvin formula and the plane’s mirror formula alternately yields an infinite sequence
of image charges of geometrically decreasing magnitude. In the sphere-plane case, these image charges lie
on the perpendicular axis through the sphere center, at progressively smaller distances, and the series
converges fast if the gap 
 is not too small. In the limit 
 (sphere almost touching the plane),
the  method-of-images  series  converges  slowly;  instead  one  can  use  the  method  of  bispherical
52
53
54
55
O(N)
56
a
b
ϵ2
ϵ1
r = b
r = a
r > b
V =
+
4πϵ r
1 >
q
B (a/r)
P (cos θ)
∑l
l
l+1
l
a < r < b
V =
(C r +
∑
l l
D r
)P (cos θ)
l − l−1
l
r < a V = 0
V = 0
r = a
C a +
l
l
D a
=
l
−l−1
0
V
ϵ∂ V
r
r = b
Bl
ϵ , ϵ , a, b, d
1
2
57
58
ab
b = a + δ δ ≪ a
−qa/d
ϵ2
(ϵ −
2
∞)
ϵ2
ϵ <
2
ϵ1
α = 4πϵ a
1
3
ϵ (2ϵ +ϵ )+2(ϵ −ϵ )( ) ϵ
2
1
2
1
2
b
a 3 1
ϵ (2ϵ +ϵ )−(ϵ −ϵ )( ) (2ϵ +ϵ )
2
1
2
1
2
b
a 3
1
2
2ϵ +ϵ
2
1
ϵ −ϵ
2
1
b = a
= 4πϵ a
1
3
ϵ →
2
∞
ϵ
ϵ2
ϵ1
−q
g = d − a
g ≪ a
5
coordinates. Bispherical coordinates exploit the fact that equipotentials of two-sphere systems are level
surfaces of a certain coordinate ; the Green’s function for two conducting spheres (or a sphere and a plane
as the plane is a sphere of infinite radius) can be expanded in Legendre functions 
 of non-integer degree
(the so-called toroidal harmonics). This yields an exact infinite series for the potential, from which one can
extract asymptotic behavior. For small  , the capacitance  
 of a sphere-plane system behaves as  
 (like a quasi-parallel plate with a logarithmic correction) – this comes from the
dominant large-  behavior of the series. The field in the narrow gap is extremely high (formally  
 on  small  patch  area  
),  so  discretization  of  charge  images  can  become  ill-conditioned.
Numerical solvers (BEM, etc.) use mesh refinement in the gap and on the sphere’s near-face to capture this
boundary-layer field. Often a proximity parameter
 is used: for 
, one can derive a singular
asymptotic expansion for the field: e.g. the maximum surface field on the sphere scales like  
times the field of an isolated sphere. Error estimates for truncating the image series: if we truncate after 
image pairs (sphere-plane iterations), the error in potential near the gap is on the order of the next image
charge magnitude 
. For large separation 
, only the first image (plane’s 
) is
significant and the error decays as 
 – very fast. 
For  two finite spheres (both conducting), the image method likewise gives an infinite double series of
point charges inside each sphere. This was studied by King (1934) and others; the exact series solution for
two-sphere capacitance and force was derived by Lord Rayleigh (1897) using bispherical harmonics. There
is no simple closed-form formula; one must sum the series or use approximation formulas valid in certain
limits (e.g. small  or large ). For mixed dielectric spheres influencing each other, one can again set up
dual infinite series of multipoles. For instance, two dielectric spheres in vacuum will polarize each other: one
can expand each sphere’s response in multipoles, with coefficients that depend on the external field applied
by the other sphere’s multipoles – this leads to a  matrix equation coupling the two sets of coefficients.
Solving that yields a double series (the method is analogous to multiple scattering theory). Only in limiting
cases (like one sphere much larger than the other, or extreme permittivity ratios) can one find simplified
image approximations.  Fikioris (1977) and others have looked at two-sphere polarization; generally one
must truncate for numerical evaluation.
```