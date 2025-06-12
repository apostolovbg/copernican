# Development Guidelines for Copernican Suite

This file contains instructions for any AI agent or contributor working on the Copernican Suite repository.

## Commenting and Documentation
- Start all change summaries with a `DEV NOTE` at the beginning of the modified file.
- Comment code thoroughly to explain why each change is made.
- Do **not** use sequences such as `<<<<<<<`, `=======`, or `>>>>>>>` in comments or documentation because GitHub interprets them as merge conflict markers.
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
- When project version updates occur, update `doc.json` and add an entry in `CHANGELOG.md`.

