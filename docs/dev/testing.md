# Testing

The rules for writing FrontierCS 2's tests and fixtures.

- ALWAYS pass `make ci` before finishing; it runs `make test` and
  `make validate CONFIG=config/site.example.json`
- PREFER running the narrowest relevant test while iterating:
  `./python -m unittest tests.test_<module>`, or for a task's maintenance suite
  `PYTHONPATH=tests:tasks/shared/lib ./python -m pytest -q -p no:cacheprovider tests/maintenance/<task>`
- WRITE few tests, at the module's public boundary; do not test what the parser or the type of
  a value already guarantees
- IMPORT `deployment_test_guard` first in every test module: tests never write real state,
  reach the network or run cluster tools, and patch directory helpers to temporary directories
  when they need to write
- PATCH the namespace that looks a dependency up, not a re-export of it
- KEEP a task's tests with that task; `tests/` holds tests of shared code
- TREAT a failing golden fixture (`tests/fixtures/golden/`) as a behaviour change: review the
  diff, and regenerate with `python3 tests/test_golden.py --update` only deliberately, in the
  same pull request, saying why; never regenerate to silence a failure
- NEVER assume a test failure is pre-existing
- NEVER let a test depend on credentials, a registry, a cluster or a site configuration of
  your own; `config/site.example.json` is the site every test uses
