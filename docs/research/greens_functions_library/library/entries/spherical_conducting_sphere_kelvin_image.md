---
{
  "id": "spherical_conducting_sphere_kelvin_image",
  "title": "Conducting Sphere (Kelvin’s Image Charge)",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      3
    ],
    "ranges": [
      {
        "page": 3,
        "line_start": 5,
        "line_end": 38
      }
    ]
  },
  "tags": [
    "sphere",
    "conductor",
    "kelvin-image",
    "grounded-boundary",
    "surface-charge-density",
    "inversion"
  ],
  "urls": [],
  "years_mentioned": [
    1845,
    1883,
    1968
  ]
}
---

# Conducting Sphere (Kelvin’s Image Charge)

## Raw assets

- Page 03 image: `raw/page_images/page_03.png`
- Page 03 text (PyMuPDF): `raw/text_pymupdf/page_03.txt`
- Page 03 text (pdfplumber): `raw/text_pdfplumber/page_03.txt`

## Source (verbatim)

```text
Conducting Sphere (Kelvin’s Image Charge): A point charge   in an infinite homogeneous medium at
distance  from the center of a grounded conducting sphere (radius 
) induces an image charge 
 located along the line from the center through the real charge, at radius  
 from the
center
. This Kelvin image produces a potential that exactly cancels 
 on the conductor surface. The
resulting Green’s function (Dirichlet, 
 on sphere) is:
for a source at 
 outside the sphere (with 
). The induced surface charge density is found by the
image method as well: 
 for a point charge on the polar axis
, and the total induced charge is 
 (the conductor shields the external field)
. Kelvin (1845)
discovered this solution
, which was a cornerstone in the development of the method of images. The 
reciprocal problem (charge inside a grounded sphere) also yields a single image: if  is at radius 
inside, then an image 
 at radius 
 outside produces zero potential on the sphere
.
These classic results are given in Maxwell’s treatise and many textbooks (e.g. Smythe, Static & Dynamic
Electricity, 1968, provides derivations). Neumann (1883) already discussed the sphere image problem and
its relation to inversion symmetry
. The method can be generalized to other conductor shapes that are 
spherical inversions of a plane or sphere: e.g. a conducting spherical shell with inner/outer radii yields
image charges for interior/exterior problems via Kelvin inversion (a charge outside a grounded spherical
shell induces one image outside and one inside the shell). Another example: a conducting circular cylinder
can be solved by images in 2D (line charges); Kelvin inversion maps it to a line and sphere configuration.
These are all special cases where a finite set of image multipoles yields an exact solution.
```