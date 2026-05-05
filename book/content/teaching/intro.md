# Teaching material

The notebooks here were used to teach GPs in several iterations of [this
course](https://github.com/jorobledo/bayesian_statistical_learning_2) at Jülich
Research Center. Note that the latter resource might be outdated. Please
consider the material here to be the up to date reference.

**Contents**

```{tableofcontents}
```

## Generate and use notebooks

If you want to run these notebooks using your local Jupyter Lab / Notebook,
then follow the steps below. We use [`jupytext`](https://jupytext.readthedocs.io) to create
notebooks from the Python scripts `book/content/teaching/notebook_*.py` in this
repo.


```sh
$ git clone <this repo>
$ cd gp_playground/book/content/teaching/

# One-time setup of venv and ipy kernel. Use https://virtualenvwrapper.readthedocs.io or
#   $ python -m venv --system-site-packages bayes-ml-course-sys
#   $ source ./bayes-ml-course-sys/bin/activate
$ mkvirtualenv --system-site-packages bayes-ml-course-sys
$ pip install -r requirements.txt

# Install custom kernel, select that in Jupyter. --sys-prefix installs into the
# current venv, while --user would install into ~/.local/share/jupyter/kernels/
$ (bayes-ml-course-sys) python -m ipykernel install --name bayes-ml-course --sys-prefix

# This script converts all *.py to *.ipynb
$ (bayes-ml-course-sys) ./py-to-notebook.py

# Start Jupyter
$ (bayes-ml-course-sys) jupyter-lab notebook_01_one_dim.ipynb
```
