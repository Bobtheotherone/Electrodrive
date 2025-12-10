---
{
  "id": "general_parameter_scaling_numerical_considerations",
  "title": "Parameter  Scaling  &  Numerical  Considerations",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      7
    ],
    "ranges": [
      {
        "page": 7,
        "line_start": 12,
        "line_end": 68
      }
    ]
  },
  "tags": [
    "scaling",
    "nondimensionalization",
    "numerical-stability",
    "series-convergence",
    "mesh-refinement",
    "error-estimation"
  ],
  "urls": [],
  "years_mentioned": []
}
---

# Parameter  Scaling  &  Numerical  Considerations

## Raw assets

- Page 07 image: `raw/page_images/page_07.png`
- Page 07 text (PyMuPDF): `raw/text_pymupdf/page_07.txt`
- Page 07 text (pdfplumber): `raw/text_pdfplumber/page_07.txt`

## Source (verbatim)

```text
Parameter  Scaling  &  Numerical  Considerations: It  is  crucial  to  nondimensionalize  lengths  by  a
characteristic scale (sphere radius, etc.) and to form contrast parameters that govern the solution. Common
dimensionless groups include the permivitty contrast
 (which appeared in many formulas above
as image-charge factors), the normalized separation
 (distance of a charge or second object in
units of sphere radius), and the gap ratio
. In series expansions, small  can be used as an expansion
parameter (for instance, for a weakly polarizable sphere, 
). In contrast, 
(huge contrast in either direction) is a singular limit – e.g. 
 corresponds to a perfect conductor image.
One often expands about those extremes: for a high-  sphere, one can write 
 where
 is 0 for 
 and 
 is proportional to 
. This yields a small parameter 
. Likewise, for a low-
 inclusion, one can treat it as a small perturbation (since 
, one might set 
 with 
 and
expand). As for geometry, small gap
 expansions often use 
 as a parameter: e.g. capacitance of
sphere-plane has an asymptotic expansion in powers of  
 and  
. For  far-field expansions, one
uses 
: e.g. the potential of a sphere of charge 
 at large distance looks like 
 dipole
term + etc., with each successive multipole 
. Truncating at dipole yields an error 
.
Many numerical schemes (method of moments, multipole accelerators) leverage such truncation for distant
interactions. 
When  implementing  image-based  solutions  or  series  numerically,  one  must  ensure  convergence and
stability. For example, the Legendre series for a dielectric sphere can suffer catastrophic cancellation for
large  if summed naively; using recurrence relations or log-scaling for the coefficients is recommended. In
boundary element collocation, placing collocation points near sharp edges or small gaps is essential: e.g.
for a sphere near a plane, a finer mesh (or denser sampling for the method of moments) is needed in the
region of closest approach, because induced surface charge density behaves like 
 near the
contact point (a square-root singularity). Mesh grading proportional to 
 is often used
to capture this. In iterative image constructions, one monitors the magnitude of the last-added image
charge: stopping when it falls below some tolerance ensures the error in potential is bounded by that
magnitude. For instance, in the two-sphere problem, if the 
th image has charge 
, the error in potential
is 
; since 
 (for large ) or 
 (for small gap), one can estimate how
many images are required.
```