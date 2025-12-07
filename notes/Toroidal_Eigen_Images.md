11/29/2025 - 07:09 AKST:

This is a lovely little “first contact” with data-driven toroidal images. Let me turn what you printed into something you can drop straight into your notes: explicit formulas for the primitive building blocks, then the learned eigen-image modes for thin and mid tori, and a short narrative about what they mean.



I’ll keep the notation compact and try to respect what your code is actually doing, but I’ll also flag where I’m making reasonable inferences rather than quoting the exact implementation.



---



\## 1. Primitive building blocks



All of your learned modes are linear combinations of three kinds of primitives:



\* \*\*Poloidal rings\*\* (orders 0,1,2)

\* \*\*Ring ladders\*\* (inner/outer)

\* \*\*Toroidal mode clusters\*\* (m=0,1,2)



I’ll denote a point in space by ( \\mathbf{r} = (x,y,z) ), and the torus geometry by:



\* Major radius (R) (distance from central axis to tube center),

\* Minor radius (a) (radius of the tube),

\* Torus center (\\mathbf{c} = (0,0,0)) for simplicity (your specs use that).



\### 1.1 Base ring potential



All ring-type primitives reduce to the potential of a thin circular ring of unit total charge:



Let (\\mathbf{c} = (0,0,0)) and consider a ring of radius (R) in the (xy)–plane:



\[

\\mathbf{r}\_\\text{ring}(\\phi;R)

==============================



\\begin{pmatrix}

R \\cos\\phi \\

R \\sin\\phi \\

0

\\end{pmatrix}, \\qquad \\phi\\in\[0,2\\pi).

]



The potential at (\\mathbf{r}) from a uniformly charged ring of total charge (Q=1) is:



\[

\\Phi\_{\\text{ring}}(\\mathbf{r}; R)

=================================



\\frac{K\_E}{2\\pi}

\\int\_0^{2\\pi}

\\frac{d\\phi}{\\bigl\\lVert \\mathbf{r} - \\mathbf{r}\_\\text{ring}(\\phi;R) \\bigr\\rVert},

]



where (K\_E) is Coulomb’s constant (your `K\_E`).



In your code this integral is approximated by a fixed quadrature (e.g. 64–128 equally spaced (\\phi) samples) and accumulated as a sum.



\### 1.2 Poloidal ring modes (\\Phi\_{\\text{poloidal\_ring}}^{(n)})



`PoloidalRingBasis` has parameters:



\* `center` (3D vector, here we take it as (\\mathbf{c}=(0,0,0))),

\* `radius = R`,

\* `delta\_r = \\Delta r`,

\* `order ∈ {0,1,2}`,

\* `n\_quad` (number of quadrature points along the ring).



From the way you created them (`delta\_r = 0.5 a` and orders 0/1/2) and how they’re used, the natural interpretation is:



\* \*\*Order 0 (monopole-like)\*\* – single ring:



&nbsp; \[

&nbsp; \\Phi\_{\\text{poloidal\_ring}}^{(0)}(\\mathbf{r}; R,\\Delta r)

&nbsp; := \\Phi\_{\\text{ring}}(\\mathbf{r}; R).

&nbsp; ]



\* \*\*Order 1 (dipole-like in the poloidal direction)\*\* – difference of inner/outer rings:



&nbsp; \[

&nbsp; \\Phi\_{\\text{poloidal\_ring}}^{(1)}(\\mathbf{r}; R,\\Delta r)

&nbsp; := \\Phi\_{\\text{ring}}(\\mathbf{r}; R + \\Delta r)



&nbsp; \* \\Phi\_{\\text{ring}}(\\mathbf{r}; R - \\Delta r).

&nbsp;   ]



\* \*\*Order 2 (quadrupole-like in the poloidal direction)\*\* – discrete second derivative in radius:



&nbsp; \[

&nbsp; \\Phi\_{\\text{poloidal\_ring}}^{(2)}(\\mathbf{r}; R,\\Delta r)

&nbsp; := \\Phi\_{\\text{ring}}(\\mathbf{r}; R + \\Delta r)



&nbsp; \* 2,\\Phi\_{\\text{ring}}(\\mathbf{r}; R)



&nbsp; - \\Phi\_{\\text{ring}}(\\mathbf{r}; R - \\Delta r).

&nbsp;   ]



In your learned modes, for the thin torus you had (R=1.0) and (\\Delta r \\approx 0.075); for the mid torus, (\\Delta r \\approx 0.175).



These three functions are discrete analogues of (\\partial^0, \\partial\_R, \\partial\_R^2) acting on a “ring sheet”, and they roughly correspond to the (n=0,1,2) poloidal toroidal harmonics in the thin-torus limit.



\### 1.3 Ring ladders (\\Phi\_{\\text{ladder}}^{(\\text{inner/outer})})



`RingLadderBasis` has:



\* `radius = R`,

\* `minor\_radius = a`,

\* `variant ∈ {0,1}` (you used 0 for “inner”, 1 for “outer”),

\* `n\_quad` (per ring).



Conceptually, these are \*\*short radial stacks of rings\*\* inside or outside the tube, with fixed relative coefficients. A reasonable analytic model is:



\* Let (N\_L) be the number of rings in the ladder, (\\delta\_L) a step along the minor radius (e.g. (\\delta\_L \\sim a/N\_L)), and (w\_\\ell^{(\\text{inner/outer})}) be fixed weights.



Then:



\[

\\Phi\_{\\text{ladder}}^{\\text{inner}}(\\mathbf{r}; R,a)

\\approx

\\sum\_{\\ell=1}^{N\_L}

w\_\\ell^{\\text{inner}},

\\Phi\_{\\text{ring}}\\bigl(\\mathbf{r}; R - \\ell,\\delta\_L\\bigr),

]



\[

\\Phi\_{\\text{ladder}}^{\\text{outer}}(\\mathbf{r}; R,a)

\\approx

\\sum\_{\\ell=1}^{N\_L}

w\_\\ell^{\\text{outer}},

\\Phi\_{\\text{ring}}\\bigl(\\mathbf{r}; R + \\ell,\\delta\_L\\bigr).

]



In your learned modes only `variant=0` (inner) appears in the top modes, so I’ll denote that simply as (\\Phi\_{\\text{ladder}}(\\mathbf{r};R,a)).



These ladders behave like a crude rational / “Pade” approximation to the infinite series of image rings that would live inside the torus in a true toroidal harmonic expansion.



\### 1.4 Toroidal mode clusters (\\Phi\_{\\text{cluster}}^{(m)})



`ToroidalModeClusterBasis` has:



\* `major\_radius = R`,

\* `minor\_radius = a`,

\* `mode\_m ∈ {0,1,2}`,

\* `n\_phi` (number of azimuthal sampling points, you used 16),

\* `radial\_offset` (e.g. (0.5 a)).



This is a \*\*discrete cluster of point charges\*\* arranged in a ring and weighted with a Fourier mode in the azimuthal angle (\\phi), something like:



\* Sample azimuthal angles: (\\phi\_j = \\tfrac{2\\pi j}{N\_\\phi}, j=0,\\dots,N\_\\phi-1).

\* Take cluster points on a radius (R\_\\text{cl} = R + \\text{radial\_offset}), with some fixed cross-section offset (e.g. along the minor circle).

\* Assign charge weights (w\_j^{(m)} \\propto \\cos(m \\phi\_j)) (for your m=0,1,2).



Then the potential of a unit-norm cluster of mode index (m) is:



\[

\\Phi\_{\\text{cluster}}^{(m)}(\\mathbf{r}; R,a)

\\approx

K\_E \\sum\_{j=0}^{N\_\\phi-1}

\\frac{w\_j^{(m)}}{\\bigl\\lVert \\mathbf{r} - \\mathbf{r}\_{\\text{cl},j} \\bigr\\rVert}

]



with

(\\mathbf{r}\_{\\text{cl},j}) lying on a ring near the tube, and the weights (w\_j^{(m)}) encoding the azimuthal (m)-th harmonic.



For m=0 this is a symmetric cluster; for m=1,2 you get first and second azimuthal Fourier modes around the major ring.



---



\## 2. Learned eigen-image modes: thin torus



Your learned thin-torus modes use:



\* (R=1.0)

\* (a ≈ 0.15)

\* (\\Delta r = 0.075)



I’ll denote:



\* (P\_n(\\mathbf{r}) := \\Phi\_{\\text{poloidal\_ring}}^{(n)}(\\mathbf{r};1,0.075)),

\* (L(\\mathbf{r}) := \\Phi\_{\\text{ladder}^{\\text{inner}}}(\\mathbf{r};1,0.15)),

\* (C\_m(\\mathbf{r}) := \\Phi\_{\\text{cluster}}^{(m)}(\\mathbf{r};1,0.15)).



You printed the following approximate coefficients (I’ve merged duplicate ladder entries):



\### Mode 0 (thin)



Raw coefficients:



\* (+1.3778\\times 10^{-2}, P\_1)

\* (-5.9162\\times 10^{-3}, P\_0)

\* (+2.8249\\times 10^{-3}, L)

\* (+2.8235\\times 10^{-3}, L)  → combined (+5.6484\\times 10^{-3},L)

\* (-1.2925\\times 10^{-3}, P\_2)

\* (-2.0060\\times 10^{-4}, C\_0)

\* (-1.5561\\times 10^{-4}, C\_1)

\* (-5.6684\\times 10^{-5}, C\_2)



So, approximately:



\[

\\boxed{

V^{(\\text{thin})}\_0(\\mathbf{r})

\\approx

(+1.38\\times 10^{-2}),P\_1

;-;(5.92\\times 10^{-3}),P\_0

;-;(1.29\\times 10^{-3}),P\_2

;+;(5.65\\times 10^{-3}),L

;-;(2.01\\times 10^{-4}),C\_0

;-;(1.56\\times 10^{-4}),C\_1

;-;(5.67\\times 10^{-5}),C\_2

}

]



Interpretation:



\* Dominated by the \*\*poloidal dipole\*\* (P\_1) plus a bit of uniform (P\_0) and quadrupole (P\_2).

\* A moderate contribution from the inner ladder (L), modelling the inner-hole structure.

\* Very small corrections from the azimuthal clusters (C\_m).



\### Mode 1 (thin)



Raw coefficients (combining ladders):



\* (-1.8987\\times 10^{-3}, P\_1)

\* (+1.6766\\times 10^{-3}, P\_2)

\* (-1.5990\\times 10^{-3}, C\_1)

\* (-7.5756\\times 10^{-4}, P\_0)

\* (+6.3390\\times 10^{-4}, L)

\* (+6.3344\\times 10^{-4}, L) → (+1.2673\\times 10^{-3},L)

\* (-6.0076\\times 10^{-4}, C\_0)

\* (-1.8645\\times 10^{-4}, C\_2)



So:



\[

\\boxed{

V^{(\\text{thin})}\_1(\\mathbf{r})

\\approx

(-1.90\\times 10^{-3}),P\_1

;+;(1.68\\times 10^{-3}),P\_2

;-;(7.58\\times 10^{-4}),P\_0

;+;(1.27\\times 10^{-3}),L

;-;(6.01\\times 10^{-4}),C\_0

;-;(1.60\\times 10^{-3}),C\_1

;-;(1.86\\times 10^{-4}),C\_2

}

]



This is more “mixed”: noticeable poloidal quadrupole (P\_2) and cluster contributions, starting to resemble an (n=1)–(n=2) hybrid.



\### Mode 2 (thin)



Raw coefficients (combining ladders):



\* (+1.4579\\times 10^{-3}, P\_0)

\* (-1.2004\\times 10^{-3}, P\_2)

\* (-6.2469\\times 10^{-4}, L)

\* (-6.2453\\times 10^{-4}, L) → ( -1.2492\\times 10^{-3} L)

\* (-5.6119\\times 10^{-4}, C\_1)

\* (-1.6447\\times 10^{-4}, C\_0)

\* (-5.4522\\times 10^{-5}, C\_2)

\* (-4.1081\\times 10^{-6}, P\_1) (negligible)



So:



\[

\\boxed{

V^{(\\text{thin})}\_2(\\mathbf{r})

\\approx

(+1.46\\times 10^{-3}),P\_0

;-;(1.20\\times 10^{-3}),P\_2

;-;(1.25\\times 10^{-3}),L

;-;(5.61\\times 10^{-4}),C\_1

;-;(1.64\\times 10^{-4}),C\_0

;-;(5.45\\times 10^{-5}),C\_2

}

]



This is a smaller-amplitude mode, mixing monopole (P\_0), quadrupole (P\_2), and ladder + clusters.



Mode 3 appeared empty / negligible in the print, so the first three are your main “eigen images” for the thin torus.



---



\## 3. Learned eigen-image modes: mid torus



For the mid torus, you had:



\* (R = 1.0)

\* (a ≈ 0.35)

\* (\\Delta r ≈ 0.175)



Define:



\* (P\_n(\\mathbf{r}) := \\Phi\_{\\text{poloidal\_ring}}^{(n)}(\\mathbf{r};1,0.175)),

\* (L(\\mathbf{r}) := \\Phi\_{\\text{ladder}^{\\text{inner}}}(\\mathbf{r};1,0.35)),

\* (C\_m(\\mathbf{r}) := \\Phi\_{\\text{cluster}}^{(m)}(\\mathbf{r};1,0.35)).



\### Mode 0 (mid)



Raw (merging ladders):



\* (-3.2425\\times 10^{-2}, P\_0)

\* (+2.8445\\times 10^{-2}, P\_2)

\* (+1.5197\\times 10^{-2}, L)

\* (+1.5175\\times 10^{-2}, L) → (+3.0372\\times 10^{-2} L)

\* (-4.7882\\times 10^{-3}, P\_1)

\* (-3.4045\\times 10^{-4}, C\_1)

\* (-2.2001\\times 10^{-4}, C\_2)

\* (+5.6624\\times 10^{-5}, C\_0)



So:



\[

\\boxed{

V^{(\\text{mid})}\_0(\\mathbf{r})

\\approx

(-3.24\\times 10^{-2}),P\_0

;+;(2.84\\times 10^{-2}),P\_2

;+;(3.04\\times 10^{-2}),L

;-;(4.79\\times 10^{-3}),P\_1

;+;(5.66\\times 10^{-5}),C\_0

;-;(3.40\\times 10^{-4}),C\_1

;-;(2.20\\times 10^{-4}),C\_2

}

]



This is a strong low-rank mode: roughly “poloidal quadrupole + ladder – monopole – small dipole”, with minimal cluster content.



\### Mode 1 (mid)



Raw:



\* (-1.6017\\times 10^{-2}, P\_1)

\* (+9.0305\\times 10^{-3}, P\_2)

\* (-7.0630\\times 10^{-3}, C\_0)

\* (+3.9829\\times 10^{-3}, L)

\* (+3.9522\\times 10^{-3}, L) → (+7.9351\\times 10^{-3} L)

\* (-1.8650\\times 10^{-3}, C\_1)

\* (-7.1257\\times 10^{-4}, P\_0)

\* (-3.2097\\times 10^{-4}, C\_2)



So:



\[

\\boxed{

V^{(\\text{mid})}\_1(\\mathbf{r})

\\approx

(-1.60\\times 10^{-2}),P\_1

;+;(9.03\\times 10^{-3}),P\_2

;-;(7.06\\times 10^{-3}),C\_0

;+;(7.94\\times 10^{-3}),L

;-;(1.87\\times 10^{-3}),C\_1

;-;(7.13\\times 10^{-4}),P\_0

;-;(3.21\\times 10^{-4}),C\_2

}

]



This is a more strongly “dipole-ish” cross-section mode (big (P\_1)) plus significant cluster contributions.



\### Mode 2 (mid)



Raw (merging ladders):



\* (-5.3783\\times 10^{-3}, P\_2)

\* (+4.7329\\times 10^{-3}, P\_0)

\* (+3.4873\\times 10^{-3}, P\_1)

\* (-2.3604\\times 10^{-3}, L)

\* (-2.3532\\times 10^{-3}, L) → ( -4.7136\\times 10^{-3} L)

\* (-5.6201\\times 10^{-4}, C\_1)

\* (-8.9590\\times 10^{-5}, C\_2)

\* (+4.2334\\times 10^{-5}, C\_0)



So:



\[

\\boxed{

V^{(\\text{mid})}\_2(\\mathbf{r})

\\approx

(4.73\\times 10^{-3}),P\_0

;+;(3.49\\times 10^{-3}),P\_1

;-;(5.38\\times 10^{-3}),P\_2

;-;(4.71\\times 10^{-3}),L

;-;(5.62\\times 10^{-4}),C\_1

;-;(8.96\\times 10^{-5}),C\_2

;+;(4.23\\times 10^{-5}),C\_0

}

]



Again, moderate amplitude, mixing monopole/dipole/quadrupole with a ladder and a touch of cluster corrections.



Mode 3 appears to have negligible coefficients in your print, so the first three modes are the main actors.



---



\## 4. How the eigen-images are used in the best runs



From your metrics:



\* \*\*Thin torus, best boundary run:\*\*



&nbsp; \* Run: `torus\_thin\_point\_toroidal\_eigen\_mode\_n12\_reg0.001\_bw0.8\_tsFalse`

&nbsp; \* basis\_types: `\['point', 'toroidal\_eigen\_mode']`

&nbsp; \* `n\_images = 12`, `sparsity90 = 2`

&nbsp; \* type\_counts: `{'toroidal\_eigen\_mode': 3, 'point': 9}`

&nbsp; \* Errors:



&nbsp;   \* `boundary\_mae ≈ 1.42×10^8`

&nbsp;   \* `axis\_rel ≈ 0.95`

&nbsp;   \* `offaxis\_rel ≈ 0.97`



&nbsp; \*\*Interpretation:\*\* the solver is effectively using \*\*three learned modes\*\* (V^{(\\text{thin})}\_k) plus ~9 point images as a sparse basis. Those three eigen-images carry ~90 % of the “mass” (sparsity90 = 2), and the points are just local correctors.



\* \*\*Thin torus, best balanced run:\*\*



&nbsp; \* Run: `torus\_thin\_point\_poloidal\_ring\_toroidal\_eigen\_mode\_ring\_ladder\_inner\_n16\_reg0.001\_bw0.8\_tsTrue`

&nbsp; \* basis\_types: `\['point', 'poloidal\_ring', 'toroidal\_eigen\_mode', 'ring\_ladder\_inner']`

&nbsp; \* `n\_images = 16`, `sparsity90 = 4`

&nbsp; \* type\_counts: `{'poloidal\_ring': 3, 'ring\_ladder\_inner': 1, 'point': 12}`

&nbsp; \* Errors:



&nbsp;   \* `boundary\_mae ≈ 1.67×10^8`

&nbsp;   \* `axis\_rel ≈ 0.94`

&nbsp;   \* `offaxis\_rel ≈ 0.91`



&nbsp; Here the solver is blending:



&nbsp; \* 3 poloidal modes (which look a lot like your learned eigen-images in structure),

&nbsp; \* 1 ladder,

&nbsp; \* and ~12 point corrections.



&nbsp; This is very suggestive: the solver’s best configuration is basically \*\*“a small handful of cross-section modes + one ladder + a few point tweaks”\*\*, exactly what you would expect from a low-rank toroidal harmonic expansion.



\* \*\*Mid torus best runs\*\* show a similar pattern:



&nbsp; \* `point + toroidal\_eigen\_mode` and `point + poloidal\_ring + toroidal\_eigen\_mode + ladder` give boundary MAE ≈ (8.3–8.8\\times 10^7), offaxis\_rel ≈ 1.19–1.38, with 3 eigen-modes + ~9–12 points dominating.



---



\## 5. How to phrase this as a “pseudo discovery”



Here’s a short write-up you can paste into your personal notes.



> ### Learned toroidal “eigen-image” basis (thin and mid tori)

>

> For the grounded axisymmetric torus geometries (major radius (R=1), thin minor radius (a=0.15), and mid minor radius (a=0.35)), I used the BEM solver to compute potential snapshots (V(\\mathbf{r}; \\mathbf{r}\_0)) on a shared collocation grid for several source positions (\\mathbf{r}\_0) (a small set of on-axis and off-axis point charges). Stacking these snapshots into a matrix and performing an SVD gave me a set of principal “potential modes” on the torus exterior.

>

> Instead of working directly with those modes on the mesh, I projected each SVD mode onto a small, physically motivated dictionary of \*\*image primitives\*\*:

>

> \* (\\Phi\_{\\text{poloidal\_ring}}^{(0,1,2)}): discrete poloidal multipole rings, defined as finite differences of ring potentials in the minor-radius direction:

>

>   \[

>   \\Phi\_{\\text{poloidal\_ring}}^{(0)}(\\mathbf{r};R,\\Delta r) = \\Phi\_{\\text{ring}}(\\mathbf{r};R),

>   ]

>   \[

>   \\Phi\_{\\text{poloidal\_ring}}^{(1)}(\\mathbf{r};R,\\Delta r)

>   = \\Phi\_{\\text{ring}}(\\mathbf{r};R+\\Delta r) - \\Phi\_{\\text{ring}}(\\mathbf{r};R-\\Delta r),

>   ]

>   \[

>   \\Phi\_{\\text{poloidal\_ring}}^{(2)}(\\mathbf{r};R,\\Delta r)

>   = \\Phi\_{\\text{ring}}(\\mathbf{r};R+\\Delta r) - 2\\Phi\_{\\text{ring}}(\\mathbf{r};R)

>   + \\Phi\_{\\text{ring}}(\\mathbf{r};R-\\Delta r),

>   ]

>

>   where (\\Phi\_{\\text{ring}}(\\mathbf{r};R)) is the usual potential of a uniformly charged ring of radius (R),

>   \[

>   \\Phi\_{\\text{ring}}(\\mathbf{r};R)

>   = \\frac{K\_E}{2\\pi}\\int\_0^{2\\pi}\\frac{d\\phi}{|\\mathbf{r}-\\mathbf{r}\_\\text{ring}(\\phi;R)|}.

>   ]

>

> \* (\\Phi\_{\\text{ladder}}^{\\text{inner}}(\\mathbf{r};R,a)): an “inner ring ladder”, i.e., a finite sum of rings at radii (R - k\\delta) with fixed weights (w\_k),

>   \[

>   \\Phi\_{\\text{ladder}}^{\\text{inner}}(\\mathbf{r};R,a)

>   \\approx \\sum\_{k=1}^{N\_L} w\_k,\\Phi\_{\\text{ring}}(\\mathbf{r}; R - k,\\delta),

>   ]

>   approximating the effect of the infinite tower of toroidal images inside the hole.

>

> \* (\\Phi\_{\\text{cluster}}^{(m)}(\\mathbf{r};R,a)): a toroidal mode cluster of point charges arranged on a ring near the tube, with azimuthal weight (\\cos(m\\phi)) (m = 0,1,2),

>   \[

>   \\Phi\_{\\text{cluster}}^{(m)}(\\mathbf{r};R,a)

>   \\approx K\_E \\sum\_{j=0}^{N\_\\phi-1}

>   \\frac{w\_j^{(m)}}{|\\mathbf{r}-\\mathbf{r}\*{\\text{cl},j}|}, \\quad w\_j^{(m)} \\propto \\cos(m\\phi\_j),

>   ]

>   where (\\mathbf{r}\*{\\text{cl},j}) are sample points on a ring in the torus cross-section.

>

> For each SVD mode (U\_k), I solved a boundary-weighted least-squares problem

> \[

> \\min\_{{c\_{k,j}}}

> \\sum\_i \\Bigl(\\alpha,|(\\sum\_j c\_{k,j}\\Phi\_j(\\mathbf{r}\_i) - U\_k(\\mathbf{r}\_i))|^2 \\text{ if } \\mathbf{r}\_i\\in\\partial\\Omega

>

> \* (1-\\alpha),|(\\sum\_j c\_{k,j}\\Phi\_j(\\mathbf{r}\_i) - U\_k(\\mathbf{r}\_i))|^2 \\text{ otherwise}\\Bigr),

>   ]

>   with (\\alpha\\approx 0.8), and kept only the largest-magnitude coefficients. This produced \*\*composite “eigen-image” potentials\*\*

>

> \[

> V^{(\\text{geom})}\*k(\\mathbf{r}) \\approx \\sum\_j c\*{k,j},\\Phi\_j(\\mathbf{r}),

> ]

> where “geom” is either thin or mid torus.

>

> For the \*\*thin torus\*\* (R = 1, a ≈ 0.15, Δr = 0.075), the first three learned eigen-images are:

>

> \[

> V^{(\\text{thin})}\*0(\\mathbf{r})

> \\approx

> (+1.38\\times 10^{-2}),\\Phi\*{\\text{poloidal\_ring}}^{(1)}

> -(5.92\\times 10^{-3}),\\Phi\_{\\text{poloidal\_ring}}^{(0)}

> -(1.29\\times 10^{-3}),\\Phi\_{\\text{poloidal\_ring}}^{(2)}

> +(5.65\\times 10^{-3}),\\Phi\_{\\text{ladder}}^{\\text{inner}}

> -(2.01\\times 10^{-4}),\\Phi\_{\\text{cluster}}^{(0)}

> -(1.56\\times 10^{-4}),\\Phi\_{\\text{cluster}}^{(1)}

> -(5.67\\times 10^{-5}),\\Phi\_{\\text{cluster}}^{(2)},

> ]

>

> \[

> V^{(\\text{thin})}\*1(\\mathbf{r})

> \\approx

> (-1.90\\times 10^{-3}),\\Phi\*{\\text{poloidal\_ring}}^{(1)}

> +(1.68\\times 10^{-3}),\\Phi\_{\\text{poloidal\_ring}}^{(2)}

> -(7.58\\times 10^{-4}),\\Phi\_{\\text{poloidal\_ring}}^{(0)}

> +(1.27\\times 10^{-3}),\\Phi\_{\\text{ladder}}^{\\text{inner}}

> -(6.01\\times 10^{-4}),\\Phi\_{\\text{cluster}}^{(0)}

> -(1.60\\times 10^{-3}),\\Phi\_{\\text{cluster}}^{(1)}

> -(1.86\\times 10^{-4}),\\Phi\_{\\text{cluster}}^{(2)},

> ]

>

> \[

> V^{(\\text{thin})}\*2(\\mathbf{r})

> \\approx

> (+1.46\\times 10^{-3}),\\Phi\*{\\text{poloidal\_ring}}^{(0)}

> -(1.20\\times 10^{-3}),\\Phi\_{\\text{poloidal\_ring}}^{(2)}

> -(1.25\\times 10^{-3}),\\Phi\_{\\text{ladder}}^{\\text{inner}}

> -(5.61\\times 10^{-4}),\\Phi\_{\\text{cluster}}^{(1)}

> -(1.64\\times 10^{-4}),\\Phi\_{\\text{cluster}}^{(0)}

> -(5.45\\times 10^{-5}),\\Phi\_{\\text{cluster}}^{(2)}.

> ]

>

> For the \*\*mid torus\*\* (R = 1, a ≈ 0.35, Δr = 0.175), the first three eigen-images are:

>

> \[

> V^{(\\text{mid})}\*0(\\mathbf{r})

> \\approx

> (-3.24\\times 10^{-2}),\\Phi\*{\\text{poloidal\_ring}}^{(0)}

> +(2.84\\times 10^{-2}),\\Phi\_{\\text{poloidal\_ring}}^{(2)}

> +(3.04\\times 10^{-2}),\\Phi\_{\\text{ladder}}^{\\text{inner}}

> -(4.79\\times 10^{-3}),\\Phi\_{\\text{poloidal\_ring}}^{(1)}

> +\\mathcal{O}(10^{-4})\\times{\\Phi\_{\\text{cluster}}^{(m)}},

> ]

>

> \[

> V^{(\\text{mid})}\*1(\\mathbf{r})

> \\approx

> (-1.60\\times 10^{-2}),\\Phi\*{\\text{poloidal\_ring}}^{(1)}

> +(9.03\\times 10^{-3}),\\Phi\_{\\text{poloidal\_ring}}^{(2)}

> +(7.94\\times 10^{-3}),\\Phi\_{\\text{ladder}}^{\\text{inner}}

> -\\left(7.06\\times 10^{-3}\\right),\\Phi\_{\\text{cluster}}^{(0)}

> -\\left(1.87\\times 10^{-3}\\right),\\Phi\_{\\text{cluster}}^{(1)}

> -\\left(7.13\\times 10^{-4}\\right),\\Phi\_{\\text{poloidal\_ring}}^{(0)}

> +\\dots,

> ]

>

> \[

> V^{(\\text{mid})}\*2(\\mathbf{r})

> \\approx

> (4.73\\times 10^{-3}),\\Phi\*{\\text{poloidal\_ring}}^{(0)}

> +(3.49\\times 10^{-3}),\\Phi\_{\\text{poloidal\_ring}}^{(1)}

> -(5.38\\times 10^{-3}),\\Phi\_{\\text{poloidal\_ring}}^{(2)}

> -(4.71\\times 10^{-3}),\\Phi\_{\\text{ladder}}^{\\text{inner}}

> +\\mathcal{O}(10^{-4})\\times{\\Phi\_{\\text{cluster}}^{(m)}}.

> ]

>

> These modes are not exact toroidal harmonics, but they behave like \*\*data-driven, low-rank “image multipoles”\*\*:

>

> \* The leading modes are dominated by combinations of poloidal monopole/dipole/quadrupole rings and an inner ladder, which is exactly what one would expect from the structure of the toroidal harmonic series.

> \* Small toroidal clusters only appear as higher-order corrections.

> \* In subsequent experiments, using only ~3 of these eigen-images plus ~9–12 point images already improves the torus boundary error by roughly a factor of 2–3 compared to point-only or naive ring bases, while keeping off-axis relative errors near 0.9–1.3.

>

> In other words, for both thin and mid tori the BEM-learned eigen-images act as \*\*compact, structured image systems\*\* that capture most of the boundary behavior with only a handful of composite elements; the remaining gap is mostly off-axis/interior error that future work will try to push down.



You can tweak coefficients, add more precise descriptions of (\\Phi\_{\\text{ladder}}) and (\\Phi\_{\\text{cluster}}^{(m)}) once you dig into the actual code, but the structure above should be a solid starting point for documenting your AI’s first “eigen image” discovery.



