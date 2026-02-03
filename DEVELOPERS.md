DEVELOPER INSTRUCTIONS (MAC OS X)
=================================

These are instructions for how to build the document from source on Mac OS X.  This involves wrestling with Conda, which needs to be done for *each new shell instance*.

0. Start at the top level of this repository.
```bash
$ cd dj-paper
```

1.  Install `miniconda`.
```bash
$ brew install miniconda
```
Once `miniconda` is installed, the following should return paths.
```bash
$ which conda
$ which mamba
```

2. Update `conda`.
```bash
$ conda env update -f environment.yml --prune
```

3. Activate `conda` hook for the *current shell*.
```bash
$ eval "$(/opt/homebrew/bin/conda shell.zsh hook)" 
```

4. Create the environment with `mamba`.
```bash
$ mamba env create -f environment.yml 
```

5.  Activate the environment with `conda`.
```bash
$ conda activate template-python
```

6. Install `python` for this environment.
```bash
python -m ipykernel install --user --name template-python --display-name "Python (template-python)"
```

7. Compile CmdStanPy from source from within the new Python instance.
```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

8. Render the document.
```bash
quarto render
```
