PYTHON ?= ./python
# Tests dynamically import locked task/acceptance files, including in children.
export PYTHONDONTWRITEBYTECODE := 1
CONFIG ?= config/site.json
TASK_CATALOG ?= config/task-catalog.json
TASK_PROBLEM ?= $(if $(filter undefined,$(origin DEMO_PROBLEM)),all,$(DEMO_PROBLEM))
# Docker Hub namespace for task images you build or publish. The default is the
# namespace of the published, digest-pinned release images.
FRONTIER_IMAGE_NAMESPACE ?= joyemang33
TASK_IMAGE_PREFIX ?= $(if $(filter undefined,$(origin DEMO_IMAGE_PREFIX)),$(FRONTIER_IMAGE_NAMESPACE),$(DEMO_IMAGE_PREFIX))
# Deprecated aliases remain available for one release. New variables take precedence.
TASK_IMAGE_ALIAS_USED = $(filter-out undefined,$(origin DEMO_PROBLEM) $(origin DEMO_IMAGE_PREFIX))
EXPORT ?=
PUBLISH ?=
IMAGE_NAMESPACE ?= $(FRONTIER_IMAGE_NAMESPACE)

.PHONY: test validate render preflight plan apply deploy build-images build-task-images build-demo-images smoke seed-tasks verify import-task

test:
	$(PYTHON) scripts/check_test_prerequisites.py
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

validate:
	$(PYTHON) scripts/validate_task_layout.py
	$(PYTHON) scripts/frontier_deploy.py validate --config $(CONFIG)

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
	$(if $(or $(filter build-demo-images,$(MAKECMDGOALS)),$(TASK_IMAGE_ALIAS_USED)),@echo "deprecated: build-demo-images / DEMO_PROBLEM / DEMO_IMAGE_PREFIX; use build-task-images / TASK_PROBLEM / TASK_IMAGE_PREFIX (aliases retained for one release)" >&2)
	TASK_IMAGE_PREFIX="$(TASK_IMAGE_PREFIX)" $(PYTHON) scripts/build_task_images.py --problem $(TASK_PROBLEM) \
		--image-prefix $(TASK_IMAGE_PREFIX)

build-demo-images: build-task-images

smoke:
	$(PYTHON) scripts/frontier_deploy.py smoke --config $(CONFIG)

seed-tasks:
	$(PYTHON) scripts/frontier_deploy.py seed-tasks --config $(CONFIG) --catalog $(TASK_CATALOG)

verify:
	$(PYTHON) scripts/frontier_deploy.py verify --config $(CONFIG)

import-task:
	@test -n "$(strip $(EXPORT))" || { echo "EXPORT=<dir> is required" >&2; exit 2; }
	$(PYTHON) scripts/import_task.py "$(EXPORT)" --image-namespace "$(IMAGE_NAMESPACE)" $(if $(filter 1,$(PUBLISH)),--publish,)
