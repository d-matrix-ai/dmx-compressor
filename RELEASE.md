# PyPI release

## Create public git remote if needed

- `git remote -v` to check for remote branches
- `git remote add public git@github.com:d-matrix-ai/dmx-compressor.git`
- `git push public`

## Prepare ChangeLog and Increment version in pyproject.toml

- Add an entry for every feature change into CHANGELOG.md
- Bump the version in pyproject.toml
- Commit to main

## Tag the repo

- `git tag vX.X.X`
- `git push origin --tags`
- `git push `

## Make sure poetry is installed (if needed)
`
- curl -sSL https://install.python-poetry.org | python3 -`

## Build wheel files

- Double check with `git show-ref`to make sure branch is taged correctly.
- `poetry build` will put a whl and tar.gz file into the `dist` directory

## Authentication (if needed)

In order to publish to PyPI, we need to set the secret API token.
To get this token, go to PiPY, dmx-compressor project, and view the secret API token.
To set the token for poetry:

`poetry config pypi-token.pypi [token]`

## Publish package to PyPI

Finally, to upload the package:
- `poetry publish`
