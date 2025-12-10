---
{
  "id": "planar_half_space_single_interface",
  "title": "Half‐Space  (Single  Interface)",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      1
    ],
    "ranges": [
      {
        "page": 1,
        "line_start": 2,
        "line_end": 49
      }
    ]
  },
  "tags": [
    "planar",
    "half-space",
    "single-interface",
    "image-charge",
    "dielectrics",
    "boundary-conditions",
    "surface-charge",
    "force"
  ],
  "urls": [],
  "years_mentioned": []
}
---

# Half‐Space  (Single  Interface)

## Raw assets

- Page 01 image: `raw/page_images/page_01.png`
- Page 01 text (PyMuPDF): `raw/text_pymupdf/page_01.txt`
- Page 01 text (pdfplumber): `raw/text_pdfplumber/page_01.txt`

## Source (verbatim)

```text
Half‐Space  (Single  Interface): For  a  point  charge   at  height   above  a  planar  interface  between
permittivities 
 (above, where the charge resides) and 
 (below), the potential in the upper region can be
written as the superposition of the original charge and an image charge. The image magnitude is 
= 
, located at the mirror point symmetrically below the plane
. (Notably, if 
, then 
 has
the same sign as , implying a repulsive interaction – charges are repelled from lower-permittivity regions
.) The potential for 
 is then:
in cylindrical coordinates (ρ horizontal distance)
. This satisfies the interface conditions: continuity of 
at 
 and continuity of normal 
 (no surface charge at the boundary)
. In the lower region 
, the field is uniform with no internal image charge; one finds 
 with 
 so
that 
 is continuous
. Equivalently, the potential in 
 behaves as if the free charge were
effectively scaled by a factor 
 (for 
 above, 
 is as if charge 
 were in
vacuum). In the extreme cases, (i)
 (conducting lower half-space) gives 
 and the familiar
image method for a grounded plane
; (ii)
 (infinitely “low” permittivity) gives 
 (image of
same sign, fully repulsive). The induced bound surface-charge density on the interface is 
, which for a conductor (
) recovers 
. The force on
the point charge due to the interface is 
 (attractive for 
, repulsive if 
). The
image formulation above is in fact an exact solution for the static half-space problem
.
```