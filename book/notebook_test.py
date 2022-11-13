# %% [markdown]
# # Test!!1!1!

# This line is part of the Md cell above and will be shown.
#
# Hidden code below.

# %%
# :tags: [hide-input]

from icecream import ic
from matplotlib import pyplot as plt, is_interactive

def foo():
    pass


# %%

# Code cell here. This is a normal comment part of the code cell. Will not be
# rendered as Md.

print("is_interactive():", is_interactive())

for _ in range(3):
    print("stuff from print")

for _ in range(3):
    ic("stuff from ic")

fig, ax = plt.subplots()
ax.plot([1,2,3])


# %%
fig, ax = plt.subplots()
ax.plot([4,7,9])

# %%
# :tags: [hide-input]

# Need this if running the script outside of jupyter. Note that
# is_interactive() is also False when jupyterbook or jupytext execute this
# script / notebook.
if not is_interactive():
    plt.show()
