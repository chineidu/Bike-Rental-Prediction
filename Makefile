.PHONY: pre-commit-update pre-commit-install pre-commit-run

# Update pre-commit hooks to latest versions
pre-commit-update:
	prek auto-update

# Install pre-commit hooks
pre-commit-install:
	prek install

# Run pre-commit on all files
pre-commit-run:
	prek run --all-files

# Update and run pre-commit
pre-commit-refresh: pre-commit-update pre-commit-run
