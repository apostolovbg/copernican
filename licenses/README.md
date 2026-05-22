# License Assets

## Table of Contents
- [Overview](#overview)
- [Contents](#contents)
- [Notes](#notes)

## Overview
This directory ships generated license artifacts for one
dependency surface.
Operators, auditors, and downstream recipients can use these
files to inspect the reported dependencies and bundled license
texts that shipped with this artifact.
For most users they are reference material rather than files
that need direct maintenance.

## Contents
- `THIRD_PARTY_LICENSES.md` records the dependency inputs and
  generated license inventory for this dependency surface.
- `*.txt` files store the generated upstream license texts that
  match the current direct dependency set.

## Notes
- Most users can treat these files as shipped reference material.
- DevCovenant regenerates the matching report and license texts
  together during dependency refresh work.
- Direct manual edits should be rare because refresh owns the
  generated contents.
