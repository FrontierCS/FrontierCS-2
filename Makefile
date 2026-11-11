PYTHON ?= ./python
# Tests dynamically import locked task/acceptance files, including in children.
export PYTHONDONTWRITEBYTECODE := 1
CONFIG ?= config/site.json
# A catalog file in place of the repository's task layers.
TASK_CATALOG ?=
TASK_PROBLEM ?= all
EXPORT ?=
PUBLISH ?=
# The image namespace is declared once, in config/image-namespace.json; the tools read it.
# IMAGE_NAMESPACE=<registry>/<namespace> names this one run's images elsewhere.
IMAGE_NAMESPACE ?=
IMAGE_NAMESPACE_ARGUMENT = $(if $(strip $(IMAGE_NAMESPACE)),--image-namespace "$(IMAGE_NAMESPACE)",)
# The checks `make ci` runs, on the example site unless CONFIG is given.
CI_CHECKS ?= lint test validate

# The repository's one entry point. ./python runs any single script on the pinned interpreter.
.PHONY: ci test validate lint fmt lock bootstrap doctor render preflight plan apply deploy build-images build-task-images smoke seed-tasks verify import-task

ci: CONFIG = config/site.example.json
ci: $(CI_CHECKS)

test:
	$(PYTHON) scripts/check_test_prerequisites.py
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

validate:
	$(PYTHON) scripts/validate_task_layout.py
	$(PYTHON) scripts/validate_task_layout.py --problems-root tasks/template
	$(PYTHON) scripts/frontier_deploy.py validate --config $(CONFIG)

# Python lint and formatting, and ShellCheck on the shell launchers.
lint:
	$(PYTHON) -m ruff check
	$(PYTHON) -m ruff format --check
	$(PYTHON) scripts/repo.py tool shellcheck python scripts/bootstrap-uv.sh

# Apply the safe lint fixes and formatting; what remains needs a manual fix.
fmt:
	$(PYTHON) -m ruff check --fix-only
	$(PYTHON) -m ruff format

# Refresh uv.lock with the pinned uv, which works even when the locked environment is stale.
lock:
	sh -c '. scripts/bootstrap-uv.sh && uv=$$(bootstrap_uv . make) && "$$uv" lock'

# Prefetch the pinned contributor tools and the Git LFS inputs the tests read.
bootstrap:
	$(PYTHON) scripts/repo.py bootstrap

# Diagnose the local environment without cloud access.
doctor:
	$(PYTHON) scripts/repo.py doctor

render:
	$(PYTHON) scripts/frontier_deploy.py render --config $(CONFIG)

preflight:
	$(PYTHON) scripts/frontier_deploy.py preflight --config $(CONFIG)

plan:
	$(PYTHON) scripts/frontier_deploy.py plan --config $(CONFIG)

apply:
	$(PYTHON) scripts/frontier_deploy.py apply --config $(CONFIG) \
		--confirm-account "$${CONFIRM_ACCOUNT}" --confirm-site "$${CONFIRM_SITE}"

deploy:
	$(PYTHON) scripts/frontier_deploy.py deploy --config $(CONFIG)

build-images:
	$(PYTHON) scripts/frontier_deploy.py build-images --config $(CONFIG)

build-task-images:
	$(PYTHON) scripts/build_task_images.py --problem $(TASK_PROBLEM) $(IMAGE_NAMESPACE_ARGUMENT)

smoke:
	$(PYTHON) scripts/frontier_deploy.py smoke --config $(CONFIG)

seed-tasks:
	$(PYTHON) scripts/frontier_deploy.py seed-tasks --config $(CONFIG) $(if $(TASK_CATALOG),--catalog $(TASK_CATALOG))

verify:
	$(PYTHON) scripts/frontier_deploy.py verify --config $(CONFIG)

import-task:
	@test -n "$(strip $(EXPORT))" || { echo "EXPORT=<dir> is required" >&2; exit 2; }
	$(PYTHON) scripts/import_task.py "$(EXPORT)" $(IMAGE_NAMESPACE_ARGUMENT) $(if $(filter 1,$(PUBLISH)),--publish,)
