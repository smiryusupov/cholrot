# Contributing to cholrot

Thank you for considering a contribution to `cholrot`.

## License of contributions

By submitting a pull request, patch, issue comment containing code, benchmark,
test, or documentation contribution, you agree that your contribution is licensed
under the Apache License, Version 2.0, the same license as the project.

Only submit contributions that you wrote yourself or that you have the right to
submit under Apache-2.0-compatible terms.

## Development setup

```bash
python -m pip install --upgrade pip
python -m pip install -e .[test,docs]
pytest
CHOLROT_PURE_PYTHON=1 pytest
```

## Before opening a pull request

Please run:

```bash
pytest
CHOLROT_PURE_PYTHON=1 pytest
python -m build
python -m twine check dist/*
```

For numerical changes, add or update tests covering both the compiled/default
backend and the reference Python backend. Prefer tests that verify mathematical
identities against NumPy references, and add golden-value regression tests when a
change could affect sign conventions, triangular orientation, or recurrence
ordering.

## Numerical correctness

`cholrot` is a numerical linear algebra package. Changes should be conservative:

- preserve the documented upper/lower Cholesky conventions;
- raise `NonPositiveDefiniteError` for invalid downdates;
- avoid silently changing benchmark definitions;
- document any new algorithmic assumptions.
