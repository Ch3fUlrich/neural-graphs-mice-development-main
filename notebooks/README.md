# Notebooks

The `notebooks/` folder contains examples, analyses, and narratives concerning the `neural-graphs-mice-develpment` project.

To make sure you can run the notebooks, please run

```sh
pip install -r requirements.txt
```

under the `ngmd` environment. This command will install the extra Python dependencies required by some of the notebooks.

If you have access to (a version of) the mice dataset, make sure to point to its location in `nb_config.py`. In doing so, all the notebooks will automatically know where to load the data from.
