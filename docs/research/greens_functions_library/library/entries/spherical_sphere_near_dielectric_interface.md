---
{
  "id": "spherical_sphere_near_dielectric_interface",
  "title": "Sphere near a Dielectric Interface",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      6
    ],
    "ranges": [
      {
        "page": 6,
        "line_start": 37,
        "line_end": 49
      }
    ]
  },
  "tags": [
    "sphere",
    "dielectric-interface",
    "hybrid-problem",
    "multipole-method",
    "numerical-methods"
  ],
  "urls": [],
  "years_mentioned": []
}
---

# Sphere near a Dielectric Interface

## Raw assets

- Page 06 image: `raw/page_images/page_06.png`
- Page 06 text (PyMuPDF): `raw/text_pymupdf/page_06.txt`
- Page 06 text (pdfplumber): `raw/text_pdfplumber/page_06.txt`

## Source (verbatim)

```text
Sphere near a Dielectric Interface: This hybrid problem (e.g. a dielectric or conducting sphere close to a
dielectric  half-space)  is  very  challenging  analytically.  No  closed-form  Green’s  function  is  known.  One
approach is the method of images with multipoles: the presence of the interface can be accounted by an
infinite set of image multipoles (located as a mirror distribution of the sphere’s charges) plus possibly
continuous distributions. For example, a conducting sphere above a dielectric plane might be approximated
by combining the sphere-plane image series (assuming the plane is conducting) with corrections for the
fact that the plane is actually dielectric. Another approach: use  bispherical coordinates if the sphere
intersects the plane (for a sphere partially embedded in a dielectric, effectively two-sphere coordinates with
one sphere of infinite radius). Some special cases have analytic results:  e.g. a conducting sphere half-
embedded in a dielectric half-space – by symmetry, this can be treated by image charges distributed along a
line (like the method of  odd extension across the interface, yielding an integral equation). In general,
however,  these  mixed  problems  are  solved  with  numerical  methods  (BEM  or  multipole  expansions
truncated at high order).
```