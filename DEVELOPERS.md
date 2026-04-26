DEVELOPER INSTRUCTIONS (MAC OS X)
=================================

These are instructions for how to build the document from source on Mac OS X.  This involves wrestling with Conda, which needs to be done for *each new shell instance*.

## First-time with installation

0. Start at the top level of this repository.
```bash
cd dj-paper
```

1.  Install `miniconda`.

We recomment using Anaconda’s official command-line installer:

https://www.anaconda.com/docs/getting-started/miniconda/install

The installer page should select the appropriate macOS installer for your hardware. If choosing manually, use `MacOSX-arm64` for Apple Silicon Macs and `MacOSX-x86_64` for Intel Macs.

Alternatively, install with Homebrew:
```bash
brew install miniconda
```
Once `miniconda` is installed, the following should return paths.
```bash
which conda
which mamba
```

2. Update `conda`.
```bash
conda env update -f environment.yml --prune
```

3. Activate `conda` hook for the *current shell*.
```bash
eval "$($HOME/miniconda3/bin/conda shell.zsh hook)"
```
If installed with Homebrew:
```
eval "$(/opt/homebrew/bin/conda shell.zsh hook)" 
```

4. Create the environment with `mamba`.
```bash
mamba env create -f environment.yml 
```

5.  Activate the environment with `conda`.
```bash
conda activate template-python
```

6. Install `python` for this environment.
```bash
python -m ipykernel install --user --name template-python --display-name "Python (template-python)"
```

7. Install CmdStan using CmdStanPy.
```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```
**CmdStan troubleshooting**

If a Stan model was compiled before changing environments or toolchains, remove the generated executable so that it will rebuild on the next render.
If CmdStan itself was built before a macOS or Xcode command line tools change, rebuild CmdStan.
```
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(overwrite=True)
```

8. Render the document.
```bash
quarto render
```

## Subsequent activations

Once installed, subsequent activations require only:

```bash
eval "$(/opt/homebrew/bin/conda shell.zsh hook)"  (or eval "$(/opt/homebrew/bin/conda shell.zsh hook)")
conda activate template-python
```
