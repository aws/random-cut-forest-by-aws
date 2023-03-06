- [Overview](#overview)
- [Feature Branches](#feature-branches)
- [Release Labels](#release-labels)
- [Releasing](#releasing)

## Overview

This document explains the release strategy for the Random Cut Forest project.

## Feature Branches

Do not create branches in the upstream repo, use your fork, for the exception of long lasting feature branches that require active collaboration from multiple developers. Name feature branches `feature/<thing>`. Once the work is merged to `main`, please make sure to delete the feature branch.

## Release Labels

Repositories create consistent release labels, such as `3.0.0-java`. Use release labels to target an issue or a PR for a given release.

## Releasing

The release process is run by a release manager volunteering from amongst the maintainers.

1. Create a PR to bump version to desired release candidate (e.g. 3.0.0).
2. Click run on the maven-release workflow in Github Actions which uploads the artifacts to our staging repository, creates a new tag, and a new Github release.
3. Login into the nexus staging repository, verify artifact was signed successfully and click release to officially push to maven central.