**About**

This project explores selected aspects of Gaussian processes (GPs) by
implementing them (a.k.a. "learning by hacking").[^fn:excuse]

**Content**

```{tableofcontents}
```


**GP intro**

Unless stated otherwise, we use the Gaussian radial basis function (a.k.a.
squared exponential) as covariance ("kernel") function

$$\kappa(\ve x_i, \ve x_j) = \exp\left(-\frac{\lVert\ve x_i - \ve x_j\rVert_2^2}{2\,\ell^2}\right)$$


Notation:

* RBF kernel length scale parameter: $\ell$ = `length_scale` (as in sklearn)
* likelihood variance $\sigma_n^2$ (a.k.a. "noise level") in GPs,
  regularization parameter $\lambda$ in KRR: $\eta$ = `noise_level` (as in `sklearn`)
* posterior predictive variance: $\sigma^2$


**Resources**

* Our own
  * [Intro to GPs and kernel methods talk][talk_gp_krr]
  * [GP talk][talk_gp]
  * [Uncertainty Quantification for Neural Networks (with GP intro)][talk_nn_uq]
* [Blog post about GP vs. KRR][gp_krr_blog]
* [This section](s:gp_pred_noise) was inspired by a [discussion over at the sklearn issue
  tracker][sklearn_issue]. Thanks!
* Books and publications listed below: {cite}`rasmussen_2006_GaussianProcessesMachine, bishop_2006_PatternRecognitionMachine, murphy_2023_ProbabilisticMachineLearningAdvanced, goertler_2019_VisualExplorationGaussiana, deisenroth_2020_PracticalGuideGaussian, kanagawa_2018_GaussianProcessesKernel`


```{bibliography}
```

**Citing**

The [source code](https://github.com/elcorto/gp_playground) and this book are
licensed under the [BSD 3-Clause License][license]. If you re-use material
from this work or just like to cite it, then either use this BibTeX
entry

```bibtex
@Online{schmerler_GPPlayground,
  author   = {Steve Schmerler},
  title    = {GP Playground},
  url      = {https://github.com/elcorto/gp_playground},
  subtitle = {Explore selected topics related to Gaussian processes},
  doi      = {10.5281/zenodo.7439202},
}
```

or [the DOI `10.5281/zenodo.7439202`][doi].


[gp_krr_blog]: https://gregorygundersen.com/blog/2020/01/06/kernel-gp-regression
[sklearn_issue]: https://github.com/scikit-learn/scikit-learn/issues/22945
[talk_gp_krr]: https://figshare.com/articles/presentation/Introduction_to_kernel_methods_and_Gaussian_processes/22032650
[talk_gp]: https://figshare.com/articles/presentation/Introduction_to_Gaussian_processes/25988176
[talk_nn_uq]: https://figshare.com/articles/presentation/Uncertainty_Quantification_for_Neural_Networks_Make_your_model_predictions_trustworthy/27891222
[license]: https://github.com/elcorto/gp_playground/blob/main/LICENSE
[doi]: https://zenodo.org/doi/10.5281/zenodo.7439202
[^fn:excuse]: And is an excuse to play with JupyterBook.
