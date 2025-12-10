---
{
  "id": "spherical_coated_spheres",
  "title": "Coated (Layered) Spheres",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      5
    ],
    "ranges": [
      {
        "page": 5,
        "line_start": 9,
        "line_end": 71
      }
    ]
  },
  "tags": [
    "sphere",
    "coated-sphere",
    "concentric-layers",
    "dielectric-shell",
    "pyati",
    "polarizability",
    "mie-static-limit"
  ],
  "urls": [],
  "years_mentioned": [
    1941,
    1993
  ]
}
---

# Coated (Layered) Spheres

## Raw assets

- Page 05 image: `raw/page_images/page_05.png`
- Page 05 text (PyMuPDF): `raw/text_pymupdf/page_05.txt`
- Page 05 text (pdfplumber): `raw/text_pdfplumber/page_05.txt`

## Source (verbatim)

```text
Coated (Layered) Spheres: A  dielectric-coated conducting sphere is another geometry amenable to
analytic solution. Suppose a conducting core radius  is covered by a concentric dielectric shell (outer radius
, permittivity 
), embedded in medium 
. A point charge in the outside region can be solved by matching
boundary conditions at 
 (dielectric interface) and 
 (conductor). One finds an infinite series again:
outside
 
,
 
;  in  the  shell
 
,
 
; inside the conductor  
,  
. The coefficients are determined by applying (i)
 at 
 (implying 
), and (ii) continuity of 
 and 
 at 
. The algebra
leads to formulas for 
 in terms of 
. These are lengthy, but in special cases they simplify. Pyati
(Radio Sci. 28:1105, 1993) applied Kelvin inversion to this problem
: using the fact that inversion in a
sphere of radius  
 maps the coated sphere to a simpler geometry, he obtained an analytic image
solution. Essentially, the presence of the coating means the image of a charge is no longer a single charge:
one gets a doublet of images – an image charge and an image dipole (or higher multipole) located at the
Kelvin image point – to satisfy the two conditions at the interface. For example, for a very  thin coating
(
, 
), one can treat the coating as a perturbation: the image charge is approximately Kelvin’s
, plus a small dipole at the same location to account for the finite 
. The dipole moment can be
solved from a first-order perturbation of the boundary condition: it comes out proportional to 
,
which for large 
 is small. Conversely, if the shell permittivity is low, the dipole moment might reinforce the
primary image (if 
, the dipole term adds a repulsive component). In general, series solutions for
coated spheres are given in treatises like Stratton’s Electromagnetic Theory (1941, §3.25) and in many
antenna-scattering analyses (the static limit of Mie theory). The coefficients often involve the spherical
harmonic recursion for three-layer systems. The effect of the coating is to modify the effective polarization
of the sphere: e.g. the polarizability of a coated metal sphere is  
,
which reduces to the earlier 
 for 
, and to the perfect conductor value (polarizability 
) as
. This indicates that a high-  coating can nearly mimic a metal even if the core is just metal (which is
already conductor, trivial here) – more interesting is a dielectric core with a different coating, which can
produce resonant polarization effects if  
 is lower than  
. Generally, however, no  finite set of simple
images exists for coated spheres either; one again deals with infinite series or images of increasing
multipole order. Pyati’s work shows that Kelvin’s inversion can simplify the algebra but still yields an infinite
series of image multipoles (just in a perhaps more organized way).
```