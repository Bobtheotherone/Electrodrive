---
{
  "id": "spherical_dielectric_sphere",
  "title": "Dielectric Sphere in Homogeneous Medium",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      3,
      4,
      5
    ],
    "ranges": [
      {
        "page": 3,
        "line_start": 39,
        "line_end": 232
      },
      {
        "page": 4,
        "line_start": 1,
        "line_end": 178
      },
      {
        "page": 5,
        "line_start": 1,
        "line_end": 8
      }
    ]
  },
  "tags": [
    "sphere",
    "dielectric",
    "multipole-series",
    "image-distribution",
    "polarizability",
    "reaction-field",
    "convergence",
    "logopoles"
  ],
  "urls": [],
  "years_mentioned": [
    1934,
    1957,
    1975,
    1988,
    1992,
    1995,
    2012,
    2019
  ]
}
---

# Dielectric Sphere in Homogeneous Medium

## Raw assets

- Page 03 image: `raw/page_images/page_03.png`
- Page 03 text (PyMuPDF): `raw/text_pymupdf/page_03.txt`
- Page 03 text (pdfplumber): `raw/text_pdfplumber/page_03.txt`
- Page 04 image: `raw/page_images/page_04.png`
- Page 04 text (PyMuPDF): `raw/text_pymupdf/page_04.txt`
- Page 04 text (pdfplumber): `raw/text_pdfplumber/page_04.txt`
- Page 05 image: `raw/page_images/page_05.png`
- Page 05 text (PyMuPDF): `raw/text_pymupdf/page_05.txt`
- Page 05 text (pdfplumber): `raw/text_pdfplumber/page_05.txt`

## Source (verbatim)

```text
Dielectric Sphere in Homogeneous Medium: For a point charge near a dielectric (non-conducting) sphere,
no finite set of simple images exists – instead the solution can be expanded in an infinite series of
multipoles (or interpreted as an image  distribution inside the sphere). Consider a sphere of radius  ,
permittivity 
, in an infinite medium 
. Place a charge  at distance  from the center (outside the sphere).
The potential outside (
) can be expanded in spherical harmonics about the sphere:  
 
(with
 
).  Inside  the  sphere  (
),
 
. Applying the boundary conditions at 
 (continuity of 
 and 
) leads to the
coefficients (for 
):
For example, the dipole term
 gives 
, corresponding to an induced dipole
moment 
 (which matches the known polarizability of a dielectric sphere)
q
d
a < d
q
=
K
− q a/d
d
=
K
a /d
2
26
27
V
ϕ = 0
G
(r, r ) =
sphere
′
(
−
4πϵ0
1
∣r − r ∣′
1
),
∣r − (d /d )
∣
K
2
′ d^
a/d
r′
∣r ∣ >
′
a
σ(θ) = −ϵ ∂ ϕ∣
=
0
n
r=a
− 4πa (a +d −2ad cos θ)
2
2
3/2
q (a −d )
2
2
28
29
−q
30
31
q
p < a
q
=
K
′
−q a/p
a /p
2
32
33
34
a
ϵ2
ϵ1
q
d
r > d
V
(r, θ) =
ext
+
4πϵ1
q
r>
1
B
P (cos θ)
∑l=0
∞
l r l+1
al
l
r
=
>
max(r, d)
r < a
V
(r, θ) =
int
A r P (cos θ)
∑l=0
∞
l
l
l
r = a
V
ϵ∂ V
r
l ≥ 0
B
=
l
( )
,
A
=
4πϵ1
q
ϵ l + ϵ (l + 1)
2
1
ϵ − ϵ
2
1
d
a
l+1
l
−
a
d .
4πϵ1
q
ϵ l + ϵ (l + 1)
2
1
ϵ − ϵ
2
1
ϵ2
l + 1
− 2l−1
l
l = 1
B =
1
(a /d )
4πϵ1
q
2ϵ +ϵ
2
1
ϵ −ϵ
2
1
2
2
p = 4πϵ B a =
1
1
3
q
a /d
2ϵ +ϵ
2
1
ϵ −ϵ
2
1
3
35
3
. The infinite series can be summed in closed form only in limiting cases (e.g. 
 reduces to the
Kelvin image solution where only 
 term survives
). In general, the image representation is an
infinite sequence: one can show that the same solution is obtained by replacing the sphere with an infinite
set of point charges located on the line through the sphere center and the original charge (a convergent
sequence approaching the center). In fact, the classic analysis by Kirkwood (1934) interprets the series as
an infinite image charge expansion for reaction field calculations. Norris (1995)】 explicitly constructed
the image system: a point charge 
 at the Kelvin image point 
plus a continuous line
distribution of charge along the radius from the center to that image point
. This line-charge
density can be chosen so that each term of its multipole expansion reproduces the series coefficients
above. (For 
, a continuous distribution is needed – a single point image is insufficient
.)
The image line charge formulation is discussed by Norris and by Lindell (Radio Sci., 27:1–8, 1992)
, who termed it an “electrostatic image theory” for the dielectric sphere. In summary, unlike the
conducting sphere, no finite combination of discrete images can satisfy both 
 and 
 continuity for a
dielectric sphere
. The infinite-series solution (or equivalent image distribution) is the accepted
Green’s function. The series converges for field points outside the sphere if 
, and for field points
inside if 
. It converges faster when the contrast 
 is small (since higher-order multipoles
have coefficients 
) and when the charge is not too close to the sphere. If the charge
approaches the sphere (
), the series develops a slow convergence (the terms 
approach 1) and many multipoles are needed – this corresponds to the known strong field
enhancement in narrow gaps. In fact, as 
, the needed truncation order 
 to achieve a given
accuracy grows roughly like 
, where  is a factor related to 
. Practically, for 
 (10% gap), perhaps tens of terms suffice; for 
 (1% gap), hundreds of terms may
be required to capture the near-contact singular field. Techniques like the use of spherical harmonics
of the second kind (logarithmic potentials) aka “logopoles”** have been developed by Majic & Le Ru
(2019) to accelerate convergence in this regime
. These functions effectively capture the line-charge
singularity inside the sphere, yielding a more rapidly convergent expansion than standard Legendre series. 
If the point charge is  inside a dielectric sphere (radius  , permittivity  
) embedded in medium  
, one
similarly expands in harmonics. The roles of interior/exterior solutions swap: inside we have a Coulomb plus
reflected series, outside purely multipole series. The coefficients become 
 for 
, etc. When 
 (sphere in a conducting medium), only the 
 term survives inside (uniform field),
consistent with the image-charge result that a dielectric sphere in a conducting host induces a single image
outside at 
 (for a charge at radius 
)
. A curious fact noted by van Siclen (AJP 56, 1142
(1988)) is that if one places the Kelvin image charge inside a dielectric sphere (instead of the proper infinite
series), the external field it produces together with the real external charge is independent of 
. In
other words, the combination “external charge + Kelvin image” yields the correct outside potential for any
sphere permittivity (equal to the conducting-sphere solution). Of course, the interior field will be wrong
except in the conductor limit – but this trick implies that the effect of finite  
 is felt only in the  higher
multipole terms (which vanish if 
). This was generalized by Redžić et al (2012) to spheroidal cases
. Practically, it suggests using the Kelvin image as a starting approximation for large but finite 
, adding
corrections  for  the  residual  multipoles.  Indeed,  Kirkwood  (1957) and  Friedman  (1975) introduced
approximate  image-charge  methods  to  replace  the  full  series  in  molecular  solvation  models
.
Kirkwood’s  approximation  is  essentially  truncating  at  the  dipole  term  (using  one  image);  Friedman
proposed a single image located at the Kelvin image point but with magnitude chosen to match the exact
dipole moment (improving accuracy for moderate contrasts)
. These give errors on the order of a few
percent for low or moderate permittivity ratios and are widely used in “reaction field” models of solvent
36
ϵ →
2
∞
l = 0
37
38
q′
r = d
=
K
a /d
2
39
40
ϵ /ϵ
=
2
1  ∞
41
42
42
V
D
41
43
r > d
r < a
∣ϵ −
2
ϵ ∣
1
∝ (ϵ −
2
ϵ )
1
d → a+
∼ (a/d)l+1
d → a
lmax
l
∼
max
ln(γ)
ln(1/(1−a/d))
γ
ϵ +ϵ
2
1
ϵ −ϵ
2
1
d/a = 1.1
d/a = 1.01
44
45
a
ϵ2
ϵ1
A =
l
4πϵ2
q
ϵ (l+1)+ϵ l
1
2
ϵ −ϵ
1
2
( a
r′ )
l
l ≥
0
ϵ →
1
∞
l = 0
r = a /p
2
p < a
46
47
ϵ2
48
49
ϵ2
ϵ →
2
∞
50
ϵ2
51
52
51
52
4
dielectrics
. Modern extensions use multiple discrete images (e.g. 3 or 4 image charges per sphere) to
fit higher multipole moments, yielding accuracies of 10^(-4) with far fewer terms than the full series
.
Such methods, combined with FMM (fast multipole method), allow efficient  
 simulations of many-
charge systems with dielectric spheres
.
```