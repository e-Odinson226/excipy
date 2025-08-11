from pathlib import Path
import re
from setuptools import setup

ROOT = Path(__file__).parent
PACKAGE = "excipy"                 # <- this is your only package folder

# ------------------------------------------------------------------
# extract __version__ from core/__init__.py
# ------------------------------------------------------------------
def get_version():
    text = (ROOT / PACKAGE / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1)
    raise RuntimeError("Cannot find __version__ in core/__init__.py")

setup(
    name="ExciPy",                       # PyPI / pip name
    version=get_version(),               # e.g. "0.1.0"
    description="ExciPy – exciton KMC engine (single‑folder version)",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Tolib Abdurakhmonov",
    author_email="abdurakhmonov.t.z@gmail.com",
    url="https://github.com/fizikximik",
    license="MIT",
    python_requires=">=3.8",
    packages=[PACKAGE],                  # only 'core'
    install_requires=[
        "numpy",
        "scipy",
        "ase",
        "matplotlib",
    ],
)
