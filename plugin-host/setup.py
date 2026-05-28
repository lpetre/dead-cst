"""Force a platform-specific (but Python-version-agnostic) wheel.

``dead_cst_plugin_host`` ships prebuilt, platform-specific binaries (the runtime
dylib + its rlib / proc-macro-dylib closure + libstd) as *package data*, but has
no compiled extension of its own. Setuptools would therefore tag the wheel as a
pure ``py3-none-any`` and let pip install it on the wrong platform. Mark the
build impure so the wheel carries the right platform tag; keep the python/abi
tags at ``py3``/``none`` since the payload works for any CPython >=3.11 (the
bundled dynamic ``_native`` is abi3) and is gated at load time by the runtime
ABI fingerprint regardless.
"""

from __future__ import annotations

from setuptools import Distribution, setup

try:  # setuptools >=70.1 vendors bdist_wheel; older ones get it from `wheel`.
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - fallback for old setuptools
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    """A distribution that owns platform-specific binaries.

    There is no compiled extension here, just prebuilt dylibs shipped as
    package data — but reporting them keeps the package at the wheel root
    (platlib) and forces a platform-specific tag instead of `py3-none-any`.
    """

    def has_ext_modules(self) -> bool:
        return True


class bdist_wheel(_bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        # Keep the resolved platform tag, but advertise py3/none: the bundle
        # works for any CPython >=3.11 (abi3 _native), not a single ABI.
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
