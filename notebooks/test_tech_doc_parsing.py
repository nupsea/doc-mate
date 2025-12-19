"""
Test technical documentation parsing from PDF.

Tests how we handle:
- Tables
- Diagrams (images)
- Code blocks
- Markdown structure
"""

import fitz  # PyMuPDF
import re
from pathlib import Path


def read_pdf_with_analysis(pdf_path, max_pages=50):
    """
    Read PDF and analyze what we get (SAMPLED for large PDFs).

    Shows:
    - Text extraction quality
    - Table detection
    - Image/diagram detection
    - Code block patterns

    Args:
        pdf_path: Path to PDF
        max_pages: Maximum pages to analyze (samples evenly if PDF is larger)
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    analysis = {
        'total_pages': total_pages,
        'pages': [],
        'has_tables': False,
        'has_images': False,
        'has_code': False
    }

    print(f"Analyzing PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")

    # Sample pages if PDF is large
    if total_pages > max_pages:
        sample_interval = total_pages // max_pages
        page_indices = list(range(0, total_pages, sample_interval))[:max_pages]
        print(f"Sampling {len(page_indices)} pages (every {sample_interval} pages)")
    else:
        page_indices = list(range(total_pages))
        print(f"Analyzing all {total_pages} pages")

    print("="*70)

    for idx in page_indices:
        page = doc[idx]
        page_num = idx + 1

        # Extract text
        text = page.get_text("text")

        # Check for images/diagrams
        images = page.get_images()

        # Detect tables (heuristic: lots of aligned content, numbers in columns)
        has_table = bool(re.search(r'(\d+\s+){3,}', text))

        # Detect code blocks (JSON, SQL, code keywords)
        has_code = bool(re.search(r'(def |class |function |import |SELECT |CREATE |INSERT |{.*:.*})', text))

        page_analysis = {
            'page_num': page_num,
            'text_length': len(text),
            'num_images': len(images),
            'likely_has_table': has_table,
            'likely_has_code': has_code,
            'text_sample': text[:500] if text else ""
        }

        analysis['pages'].append(page_analysis)

        if images:
            analysis['has_images'] = True
        if has_table:
            analysis['has_tables'] = True
        if has_code:
            analysis['has_code'] = True

        # Print sample from first few pages
        if page_num <= 3 or (images and len(analysis['pages']) <= 5):
            print(f"\nPage {page_num}:")
            print(f"  Text length: {len(text)} chars")
            print(f"  Images: {len(images)}")
            print(f"  Likely table: {has_table}")
            print(f"  Likely code: {has_code}")
            if page_num <= 3 or images:
                print(f"  Sample text:")
                print("  " + "-"*60)
                print("  " + text[:300].replace('\n', '\n  '))
                print("  " + "-"*60)

    doc.close()

    print(f"\n" + "="*70)
    print("Overall Analysis:")
    print(f"  Pages analyzed: {len(page_indices)} of {total_pages}")
    print(f"  Has images/diagrams: {analysis['has_images']}")
    print(f"  Has tables: {analysis['has_tables']}")
    print(f"  Has code: {analysis['has_code']}")

    return analysis


def parse_markdown_from_pdf(pdf_path, max_pages=100):
    """
    Extract text from PDF and try to detect markdown-like structure.

    For tech docs, we look for:
    - Headings (usually larger font or bold)
    - Code blocks (indented or monospace)
    - Lists
    - Tables (aligned columns)

    Args:
        pdf_path: Path to PDF
        max_pages: Maximum pages to process (for large PDFs)
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    full_text = []

    # Limit pages for large PDFs
    pages_to_process = min(max_pages, total_pages)

    for page_idx in range(pages_to_process):
        page = doc[page_idx]
        text = page.get_text("text")  # Basic text
        full_text.append(text)

    doc.close()
    combined_text = "\n".join(full_text)

    if total_pages > max_pages:
        print(f"Note: Processed first {max_pages} of {total_pages} pages")

    # Try to detect structure
    print("\nDetecting structure in extracted text:")
    print("="*70)

    # Look for headings (Chapter, Section patterns)
    headings = re.findall(r'^(Chapter \d+|Section \d+|\d+\.\d+\s+.+)$',
                          combined_text, re.MULTILINE)
    print(f"Potential headings found: {len(headings)}")
    if headings[:5]:
        print("Sample headings:")
        for h in headings[:5]:
            print(f"  - {h}")

    # Look for code blocks (indented lines)
    code_lines = [line for line in combined_text.split('\n')
                  if line.startswith('    ') or line.startswith('\t')]
    print(f"\nPotential code lines: {len(code_lines)}")
    if code_lines[:3]:
        print("Sample code lines:")
        for c in code_lines[:3]:
            print(f"  {c[:60]}...")

    # Look for lists
    list_items = re.findall(r'^\s*[•\-\*]\s+.+$', combined_text, re.MULTILINE)
    print(f"\nList items found: {len(list_items)}")
    if list_items[:3]:
        print("Sample list items:")
        for item in list_items[:3]:
            print(f"  {item}")

    return combined_text


def extract_tables_naive(text):
    """
    Naive table extraction from text.

    Problem: PDFs don't preserve table structure well in plain text.
    Tables often become mangled or lose alignment.
    """
    print("\nAttempting naive table extraction:")
    print("="*70)

    # Look for lines with multiple tabs or aligned spaces
    lines = text.split('\n')
    potential_table_lines = []

    for line in lines:
        # Count tabs or multiple spaces
        if '\t' in line or '  ' in line:
            # Check if it has data-like content
            if re.search(r'\d', line):  # Has numbers
                potential_table_lines.append(line)

    if potential_table_lines:
        print(f"Found {len(potential_table_lines)} potential table lines")
        print("\nSample table lines:")
        for line in potential_table_lines[:10]:
            print(f"  {line[:80]}")
    else:
        print("No clear table structure found in plain text extraction")

    print("\n⚠️  Note: Plain text extraction often loses table formatting!")
    print("   Tables in PDFs are visual layouts, not structured data.")
    print("   For better results, consider:")
    print("   - PyMuPDF's table extraction (experimental)")
    print("   - Camelot or Tabula libraries")
    print("   - OCR with table detection")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Default test path
        pdf_path = "DATA/tech-docs/DDIA.pdf"

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"Error: PDF not found at {pdf_path}")
        print("\nUsage: python test_tech_doc_parsing.py <path-to-pdf>")
        print("\nExample:")
        print("  python test_tech_doc_parsing.py DATA/tech-docs/DDIA.pdf")
        sys.exit(1)

    # Step 1: Analyze what's in the PDF
    analysis = read_pdf_with_analysis(pdf_path)

    # Step 2: Extract and parse text
    print("\n" + "="*70)
    print("EXTRACTING TEXT")
    print("="*70)
    text = parse_markdown_from_pdf(pdf_path)

    # Step 3: Try to extract tables
    print("\n" + "="*70)
    print("TABLE EXTRACTION")
    print("="*70)
    extract_tables_naive(text)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total text extracted: {len(text):,} characters")
    print(f"Pages with images: {sum(1 for p in analysis['pages'] if p['num_images'] > 0)}")
    print(f"Pages with likely tables: {sum(1 for p in analysis['pages'] if p['likely_has_table'])}")
    print(f"Pages with likely code: {sum(1 for p in analysis['pages'] if p['likely_has_code'])}")

    print("\n⚠️  KEY INSIGHT:")
    print("PDFs are visual documents, not structured data.")
    print("Text extraction gives you words but loses:")
    print("  - Table structure (rows/columns)")
    print("  - Diagram content (images → no text)")
    print("  - Code formatting (indentation may be preserved)")
    print("\nFor RAG: Extracted text is still valuable!")
    print("  - LLM can understand context even without perfect formatting")
    print("  - Code snippets (as text) are usually readable")
    print("  - Table data (as text) provides information even if not aligned")
