import os
import platform
import subprocess
from pathlib import Path

import cmdstanpy as csp


def _cmdstan_install_dir():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix is None:
        raise RuntimeError("Activate the template-python conda environment before rendering.")
    return Path(conda_prefix) / "cmdstan"


def _latest_cmdstan_path(install_dir):
    installations = list(install_dir.glob("cmdstan-*"))
    if not installations:
        raise RuntimeError(f"No CmdStan installation found in {install_dir}.")
    return max(installations, key=lambda path: path.stat().st_mtime)


def _fix_macos_tbb_linkage(executable):
    if platform.system() != "Darwin":
        return

    linked_libs = subprocess.run(
        ["otool", "-L", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "@rpath/libtbb.dylib" not in linked_libs:
        return

    tbb_lib = Path(csp.cmdstan_path()) / "stan/lib/stan_math/lib/tbb/libtbb.dylib"
    subprocess.run(
        ["install_name_tool", "-change", "@rpath/libtbb.dylib", str(tbb_lib), str(executable)],
        check=True,
    )


def configure_cmdstan():
    install_dir = _cmdstan_install_dir()
    try:
        cmdstan_path = _latest_cmdstan_path(install_dir)
    except RuntimeError:
        if not csp.install_cmdstan(dir=str(install_dir)):
            raise RuntimeError(f"CmdStan installation failed in {install_dir}.")
        cmdstan_path = _latest_cmdstan_path(install_dir)

    csp.set_cmdstan_path(str(cmdstan_path))
    for cmdstan_exe in ("diagnose", "print", "stansummary"):
        _fix_macos_tbb_linkage(Path(csp.cmdstan_path()) / "bin" / cmdstan_exe)


def cmdstan_model(stan_file):
    model = csp.CmdStanModel(stan_file=stan_file, force_compile=True)
    _fix_macos_tbb_linkage(model.exe_file)
    return model


if __name__ == "__main__":
    configure_cmdstan()
