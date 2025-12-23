"""Debug script to inspect chunk content."""

from src.flows.book_query import search_book_content

# Search for Sandeep in dth
result = search_book_content("Sandeep", "dth", limit=3)

print(f"Found {result['num_results']} results\n")
print("=" * 80)

for i, chunk in enumerate(result['chunks'], 1):
    print(f"\nCHUNK {i}")
    print(f"ID: {chunk['id']}")
    print(f"Metadata: {chunk.get('metadata', {})}")
    print(f"\nText content (first 1000 chars):")
    print(chunk['text'][:1000])
    print("\n" + "=" * 80)
