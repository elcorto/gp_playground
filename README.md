<p align="center">
  <img src="https://raw.githubusercontent.com/elcorto/gp_playground/main/book/logo.png" width="50%"><br>
</p>

**Check the [book content here](https://elcorto.github.io/gp_playground/root.html)**


Build this repo locally: Follow the instructions below or check the CI pipline.

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

```sh
(gp_play) $ ./generate-book.sh
```
