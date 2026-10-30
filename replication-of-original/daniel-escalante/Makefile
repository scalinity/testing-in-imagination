.PHONY: all test validate dry-run export
all:
	bash scripts/run_all.sh
test:
	.venv/bin/python -m pytest tests ../../lacdm/tests -q
validate:
	bash scripts/stage.sh validate
dry-run:
	bash scripts/run_all.sh --dry-run
export:
	bash scripts/stage.sh export
