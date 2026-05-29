"""Force a platform-specific, Python-agnostic wheel tag (``py3-none-<plat>``).

``dead-cst-plugin-host`` ships no compiled *Python* extension, so setuptools
would tag its wheel ``py3-none-any``. But its payload — a prebuilt runtime
dylib, the rlib dependency closure, and libstd — is platform-specific, so the
wheel must carry a platform tag (CI passes ``--plat-name``). It stays valid for
any CPython >= 3.11 (the payload is data, imported via
``dead_cst_plugin_host.__file__``, not loaded as an extension), hence the
``py3`` / ``none`` python+abi tags. Project metadata lives in ``pyproject.toml``.
"""

from setuptools import setup
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # setuptools < 70.1 (bdist_wheel still vendored by `wheel`)
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    # Mark the distribution non-pure so the wheel is built platform-specific
    # and its files land at the wheel root under platlib (not `.data/purelib`).
    def has_ext_modules(self) -> bool:
        return True


class bdist_wheel(_bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        # Keep the platform tag (honoring any `--plat-name`) but make the wheel
        # Python-agnostic: the payload is data, not a cpython extension, so it
        # works on any CPython >= 3.11.
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
