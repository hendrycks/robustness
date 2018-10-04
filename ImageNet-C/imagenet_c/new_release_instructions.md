### Cutting a new release to pypi

Bump the version in setup.py

Run the following
```bash
pip install twine
rm -rf dist && python setup.py sdist bdist_wheel && twine upload dist/*
```
