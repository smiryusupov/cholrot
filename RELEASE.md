# Release process

`cholrot` uses GitHub Actions and `cibuildwheel` to build binary wheels for
Linux, macOS, and Windows.

## Wheel targets

The first public release builds wheels for:

- CPython 3.12, 3.13, and 3.14;
- Linux `manylinux2014` x86_64;
- macOS x86_64 and arm64;
- Windows AMD64.

Free-threaded CPython wheels such as `cp313t` and `cp314t` are not built in the
first release. Add them only after the C++ extension has been reviewed and
tested for no-GIL safety.

## Manual checks before tagging

```bash
python -m pip install -e .[test]
pytest
CHOLROT_PURE_PYTHON=1 pytest
python -m build
python -m twine check dist/*
```

## GitHub setup

```bash
git init
git add .
git commit -m "Initial cholrot package"
git branch -M main
git remote add origin git@github.com:<owner>/cholrot.git
git push -u origin main
```

Replace `<owner>` with the GitHub user or organization that owns the repository.

## Documentation

Read the Docs can be configured as soon as the GitHub repository exists. PyPI is
not required for documentation hosting.

The repository contains:

- `docs/conf.py`;
- `docs/*.rst`;
- `.readthedocs.yaml`.

Import the GitHub repository in Read the Docs and let it build from
`.readthedocs.yaml`.

## Releasing

1. Update the version in `pyproject.toml`.
2. Commit the change.
3. Tag the release:

   ```bash
   git tag v0.1.0
   git push origin main --tags
   ```

The wheels workflow builds the source distribution and binary wheels. On tag
pushes, the workflow publishes to PyPI through Trusted Publishing.

## PyPI Trusted Publishing

For a repository at `github.com/<owner>/cholrot`, configure a PyPI trusted
publisher with:

- owner: `<owner>`;
- repository: `cholrot`;
- workflow name: `wheels.yml`;
- environment name: `pypi`.

The environment name must match `.github/workflows/wheels.yml`.
