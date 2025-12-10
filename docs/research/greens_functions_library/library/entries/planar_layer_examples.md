---
{
  "id": "planar_layer_examples",
  "title": "Planar Layer Examples",
  "source": {
    "document": "raw/source.pdf",
    "pages": [
      2
    ],
    "ranges": [
      {
        "page": 2,
        "line_start": 24,
        "line_end": 42
      }
    ]
  },
  "tags": [
    "planar",
    "examples",
    "dielectric-slab",
    "thin-film",
    "asymptotics",
    "far-field-expansion"
  ],
  "urls": [],
  "years_mentioned": []
}
---

# Planar Layer Examples

## Raw assets

- Page 02 image: `raw/page_images/page_02.png`
- Page 02 text (PyMuPDF): `raw/text_pymupdf/page_02.txt`
- Page 02 text (pdfplumber): `raw/text_pdfplumber/page_02.txt`

## Source (verbatim)

```text
Planar Layer Examples: A classical example is a charge on the interface (say in medium 1 at 
): it
induces a surface charge of magnitude 
 on the boundary such that the field in medium 2 is as if a charge
 were placed at the interface
. Another case: a charge embedded in a dielectric slab
of finite thickness between two different dielectric half-spaces. By multiple reflections, one can sum an
infinite geometric series of image charges in each region. If the slab is symmetric (same permittivity above
and below), the series simplifies (even and odd image symmetry). If the slab is very thin (small thickness
compared to source distance), asymptotic methods can treat the two interfaces as a single effective
interface  with  an  effective  reflection  coefficient  
 etc.,  accurate  to  
 where   is
thickness. If the source is very close to one interface (small gap), the first image dominates and higher-
order images (multiple reflections) contribute corrections that can be expanded in powers of the small
parameter. Conversely, if the source is far, one can expand the field as the free-space field plus a dipole term
proportional to 
 (for a distant charge, the interface effect looks like an induced dipole
layer).
```