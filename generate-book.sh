#!/bin/sh

book_dir=book

# Pass /path/to/notebook_foo.py to jupytext only that and rebuild w/o purging
# and rebuilding everything.
if [ $# -gt 0 ]; then
    src_files=$@
else
    rm -rvf $book_dir/_build
    rm -vf $(find $book_dir -name "__pycache__")
    rm -vf $(find $book_dir -name "notebook_*.md")
    rm -vf $(find $book_dir -name "notebook_*.ipynb")
    src_files=$(find $book_dir -name "notebook_*.py")
fi

# We like to use :tags: in md:myst as in
# https://jupyterbook.org/en/stable/content/metadata.html#add-tags-using-myst-markdown-notebooks
#
#     ```{code-cell}
#     :tags: [hide-input]
#
#     def foo():
#         pass
#     ```
# which we can add in py:percent like so
#
#     # %%
#     # :tags: [hide-input]
#
#     def foo():
#         pass
#
# but
#
# (1) the conversion py:percent --[jupytext --execute]--> ipynb doesn't use
# them in the created notebook
#
# (2) we need to do py:percent --[jupytext]--> md:myst and let jupyterbook
# execute the md:myst notebook since it parses the tags (AFAWK)
#
# (3) the conversion py:percent --[jupytext]--> md:myst doesn't correctly
# translate the tags, we need to do a small sed correction


# Notebook names: We have the naming convention that all files called
# notebook_*.py will be converted to a notebook and rendered as such.
# In _toc.yml, use notebook_*.md .

for src in $src_files; do
##for src in $(find $book_dir -name "notebook_test.py"); do
    echo "---> src: $src"
    tgt=$(echo $src | sed -re 's/\.py$/\.md/')
    echo "---> tgt: $tgt"
    jupytext --from py:percent --to md:myst --set-kernel python3  $src
    sed -i -re 's/^#.*(:tags:)/\1/g' $tgt
    ##jupytext --from md:myst --to ipynb --set-kernel python3 --execute $tgt
    ##jupytext --set-kernel python3 --execute $tgt
done

####jb build --all $book_dir
jb build $book_dir
