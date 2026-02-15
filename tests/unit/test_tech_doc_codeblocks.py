"""
Unit tests for Tech Doc code block handling and reconstruction.
Validates that code blocks are preserved or correctly reconstructed during chunking.
"""

import unittest
from pathlib import Path
from src.content.parsers.markdown_parser import MarkdownParser

class TestTechDocCodeBlocks(unittest.TestCase):

    def setUp(self):
        self.temp_file = Path("tests/unit/temp_tech_doc.md")
        content = """# Tech Doc
## Section 1
Here is some text.

```python
def hello_world():
    # This is a long code block
    # intended to be split across chunks
    print("Hello")
    return True
```
"""
        self.temp_file.write_text(content)
        self.parser = MarkdownParser(str(self.temp_file), "test_tech")

    def tearDown(self):
        if self.temp_file.exists():
            self.temp_file.unlink()

    def test_chunking_reconstruction(self):
        """Test if broken code blocks are reconstructed in chunks."""
        # Force split
        chunks = self.parser.chunk(max_tokens=20, overlap=0)
        
        has_code_chunks = [c for c in chunks if c["metadata"].get("has_code")]
        self.assertTrue(len(has_code_chunks) > 0)
        
        for c in has_code_chunks:
            text = c["text"]
            # All code chunks should now have balanced fences due to reconstruction
            self.assertEqual(text.count("```") % 2, 0, f"Chunk {c['id']} has unbalanced code fences")

    def test_code_block_extraction(self):
        """Test extraction of assets."""
        assets = self.parser.extract_assets()
        code_assets = [a for a in assets if a["asset_type"] == "code"]
        self.assertEqual(len(code_assets), 1)
        self.assertEqual(code_assets[0]["metadata"]["language"], "python")

if __name__ == "__main__":
    unittest.main()
