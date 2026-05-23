Installation and compilation
============================

From a local checkout, install in editable mode and compile the pybind11 backend:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install -e .[test]
   pytest
   python -c "import cholrot; print(cholrot.backend())"

A successful compiled install prints ``cpp``.  If the extension is unavailable,
``cholrot`` falls back to the reference Python implementation and prints
``python``.

Force the pure Python backend for comparison:

.. code-block:: bash

   CHOLROT_PURE_PYTHON=1 pytest

Build a wheel and source distribution:

.. code-block:: bash

   python -m pip install build
   python -m build

Typical compiler requirements
-----------------------------

``cholrot`` uses C++17 through pybind11.

* Linux: install ``g++`` or ``clang++``.
* macOS: install Xcode Command Line Tools with ``xcode-select --install``.
* Windows: install Microsoft C++ Build Tools / Visual Studio Build Tools.

The build metadata declares ``pybind11`` as a build requirement, so ``pip`` can
install it in an isolated build environment.
