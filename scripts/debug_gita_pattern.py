import re

GITA_CHAPTERS = [
    'THE DISTRESS OF ARJUNA',
    'THE BOOK OF DOCTRINES',
    'VIRTUE IN WORK',
]

def create_gita_pattern():
    titles_escaped = [re.escape(title) for title in GITA_CHAPTERS]
    pattern = r'^(?:' + '|'.join(titles_escaped) + r')$'
    return pattern

pattern = create_gita_pattern()
print(f'Pattern: {pattern}')
print()

# Read Gita
with open('/Users/sethurama/DEV/LM/doc-mate/DATA/the_gita.txt', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Strip Gutenberg
start_match = re.search(r'\*\*\* START OF.*\*\*\*', content)
end_match = re.search(r'\*\*\* END OF.*\*\*\*', content)
if start_match and end_match:
    content = content[start_match.end():end_match.start()].strip()

# Try to find matches
matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
print(f'Found {len(matches)} matches using MULTILINE flag')

# Try splitting
parts = re.split(pattern, content, flags=re.MULTILINE)
print(f'Split into {len(parts)} parts')

# Show some context around where we expect to find 'THE DISTRESS OF ARJUNA'
search_str = 'THE DISTRESS OF ARJUNA'
idx = content.find(search_str)
if idx != -1:
    print(f'\nFound "{search_str}" at position {idx}')
    print('Context:')
    print(repr(content[idx-50:idx+len(search_str)+50]))
else:
    print(f'\nCould not find "{search_str}"')

# Check if it's the TOC vs actual chapter
all_occurrences = []
pos = 0
while True:
    pos = content.find(search_str, pos)
    if pos == -1:
        break
    all_occurrences.append(pos)
    pos += 1

print(f'\nTotal occurrences of "{search_str}": {len(all_occurrences)}')
for i, pos in enumerate(all_occurrences):
    print(f'  Occurrence {i+1} at position {pos}')
    print(f'    Context: {repr(content[pos-20:pos+len(search_str)+20])}')
