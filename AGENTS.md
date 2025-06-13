# Development Guidelines for Copernican Suite

This file contains instructions for any AI agent or contributor working on the Copernican Suite repository.

## Commenting and Documentation
- Start all change summaries with a `DEV NOTE` at the beginning of the modified file.
- Comment code thoroughly to explain why each change is made.
- Avoid merge conflict markers (such as those inserted by Git) in comments or documentation.
- Use plain text along with `#` for code comments and `##` for markdown headings.
- Maintain a clean and readable style consistent with the current codebase.

## Testing Requirement
Run basic syntax checks before committing:

```bash
python -m py_compile data_loaders.py plotter.py
python -m py_compile *.py
```

## Documentation Updates
- Keep `README.md` concise and reference `CHANGELOG.md` for the full history of modifications.
- When project version updates occur, update `README.md` and add an entry in `CHANGELOG.md`.
- The `doc.json` file has been deprecated; any important notes should be kept in `README.md` or here.

