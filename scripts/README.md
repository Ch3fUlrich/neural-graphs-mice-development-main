# Notebooks

The `scrips/` folder contains executable scripts used in the `neural-graphs-mice-develpment` project.

To make sure you can run these scripts, please run

```sh
pip install -r requirements.txt
```

under the `ngmd` environment. This command will install the extra Python dependencies required by some of the scripts.

Note that some scripts make use of environment variables `$NGMD_DATA` and `$NGMD_MODELS` to locate the dataset path and the path where to save/load model files. Please make sure these environment variables are set (e.g. in `bash_profile` or `zshrc`) before running the scripts. Alternatively, you can replace them in the scripts with the actual respective paths in your local machine.
