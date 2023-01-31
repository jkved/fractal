# Fractals and spatiality

This repository is dedicated for my current study and research into fractals and their geometry.

Check commits for description of latest (**gsfit.py**) contents and their applicability.

Current contents include generation of Newton-Raphson fractals, Mandelbrot fractals.

It is possible to calculate fractal dimension via box-counting method. This method employs finite size scaling law to determine power-law scaling exponent over desired interval. Further there is parametric Kolmogorov-Smirnov test developed to compare two cumulative distribution functions of (supossedly) grayscale images.

Further code efficiency is improved with numba package and emcee/corner packages are used for analysis.
