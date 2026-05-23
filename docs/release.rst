Release process
===============

``cholrot`` uses GitHub Actions and ``cibuildwheel`` to build binary wheels for
Linux, macOS, and Windows.

Supported wheel targets for the first public release are:

* CPython 3.12, 3.13, and 3.14;
* Linux ``manylinux2014`` x86_64;
* macOS x86_64 and arm64;
* Windows AMD64.

Free-threaded CPython wheels, such as ``cp313t`` and ``cp314t``, are deliberately
not built in the first release. They should be added only after the C++ extension
has been reviewed and tested for no-GIL safety.

Manual checks
-------------

Before tagging a release locally::

   python -m pip install -e .[test]
   pytest
   CHOLROT_PURE_PYTHON=1 pytest
   python -m build
   python -m twine check dist/*

GitHub repository setup
-----------------------

After creating the GitHub repository, push the package source::

   git init
   git add .
   git commit -m "Initial cholrot package"
   git branch -M main
   git remote add origin git@github.com:<owner>/cholrot.git
   git push -u origin main

Replace ``<owner>`` with the actual GitHub user or organization that owns the
repository.

Documentation setup
-------------------

Read the Docs can be configured as soon as the GitHub repository exists. PyPI is
not required for documentation hosting.

The repository contains:

* ``docs/conf.py`` for Sphinx configuration;
* ``docs/*.rst`` source pages;
* ``.readthedocs.yaml`` for the Read the Docs build configuration.

To publish the documentation, import the GitHub repository in Read the Docs and
let it build from ``.readthedocs.yaml``. The generated HTML documentation can be
linked from the GitHub repository and from the PyPI project description after the
first package release.

Release through GitHub
----------------------

1. Update the version in ``pyproject.toml``.
2. Commit the change.
3. Tag the release, for example::

      git tag v0.1.0
      git push origin main --tags

4. The ``wheels`` workflow builds the source distribution and wheels.
5. On tag pushes, the workflow publishes to PyPI through Trusted Publishing.

PyPI Trusted Publishing
-----------------------

PyPI Trusted Publishing allows the release workflow to upload distributions
without storing a PyPI API token in GitHub secrets. Configure a PyPI trusted
publisher for the GitHub Actions workflow that publishes the release.

For a repository at ``github.com/<owner>/cholrot``, use:

* owner: ``<owner>``;
* repository: ``cholrot``;
* workflow name: ``wheels.yml``;
* environment name: ``pypi``.

The environment name must match the environment configured in
``.github/workflows/wheels.yml``. If the workflow environment is renamed, update
the PyPI trusted publisher configuration to match.

For a first release, create a pending trusted publisher on PyPI before pushing
the release tag. After the tag is pushed, GitHub Actions builds the distributions
and publishes them automatically.
