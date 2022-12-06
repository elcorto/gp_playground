<p align="center">
  <img src="https://raw.githubusercontent.com/elcorto/gp_playground/main/book/logo.png" width="50%"><br>
</p>

**Check the [book content here](https://elcorto.github.io/gp_playground/root.html)**.

This project uses [Jupyter Book](https://jupyterbook.org) and
[jupytext](https://jupytext.readthedocs.io). To build locally, follow the instructions below or check the CI pipline.

```sh
# https://virtualenvwrapper.readthedocs.io/en/latest/
$ mkvirtualenv gp_play
```

else

```sh
$ python3 -m venv gp_play && . ./gp_play/bin/activate
```

Then install some variant of torch (CPU, GPU) plus the packages that we test
here. The torch install line is just an example, please check the pytorch
website for more.

We need this only for gpytorch. Comment this out in
`book/content/gp_pred_comp/notebook_comp.py` to skip. Examples are tiny, so
there is no need to install the GPU version of torch.

```sh
(gp_play) $ pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
(gp_play) $ pip install -r requirements.txt
```

Build

```sh
(gp_play) $ ./generate-book.sh

# If you only modified some notebook scripts (we use jupytext in the
# background), pass their names. Else the script will purge are rebuild all.
(gp_play) $ ./generate-book.sh book/content/gp_pred_comp/notebook_comp.py

# If you only changed markdown files
(gp_play) $ jb build book
```

View

```sh
$ some_browser book/_build/html/index.html
```
