# Data Directory

This directory should contain your PDF documents for analysis.

## Sample Usage

Place your PDF files here and reference them in the CLI:

```bash
python -m lg_magent_mvp.cli run \
  --doc "data/your_document.pdf" \
  --question "Your analysis question" \
  --out "report.json"
```

## Supported Formats

- PDF files with text, images, and tables
- Multi-page documents
- Medical reports, research papers, financial documents, etc.

## Note

PDF files are excluded from git tracking to keep the repository size manageable.
Add your own PDF files to this directory for testing and analysis.
