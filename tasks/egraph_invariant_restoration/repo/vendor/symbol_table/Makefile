.PHONY: test fmt ci

test:
	cargo test --no-default-features
	cargo test --all-features

fmt:
	cargo fmt --check

ci: test fmt
