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

* [Our own intro to GPs and kernel methods talk][talk]
* [Blog post about GP vs. KRR][gp_krr_blog]
* [This section](s:gp_pred_noise) was inspired by a [discussion over at the sklearn issue
  tracker][sklearn_issue]. Thanks!
* Books and publications listed below: {cite}`rasmussen_2006_GaussianProcessesMachine, murphy_2023_ProbabilisticMachineLearningAdvanced, goertler_2019_VisualExplorationGaussiana, deisenroth_2020_PracticalGuideGaussian, kanagawa_2018_GaussianProcessesKernel`


```{bibliography}
```

**Citing**

If you like to cite this resource, you can use this BibTeX entry:


```bibtex
@Online{schmerler_GPPlayground,
  author   = {Steve Schmerler},
  title    = {GP Playground},
  url      = {https://github.com/elcorto/gp_playground},
  subtitle = {Explore selected topics related to Gaussian processes},
  doi      = {10.5281/zenodo.7439202},
}
```


[gp_krr_blog]: https://gregorygundersen.com/blog/2020/01/06/kernel-gp-regression
[sklearn_issue]: https://github.com/scikit-learn/scikit-learn/issues/22945
[talk]: https://figshare.com/articles/presentation/Introduction_to_kernel_methods_and_Gaussian_processes/22032650
[^fn:excuse]: And is an excuse to play with JupyterBook.


