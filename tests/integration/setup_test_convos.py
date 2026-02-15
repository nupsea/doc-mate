import os
import sys
import asyncio

# Add src to path
sys.path.append(os.getcwd())

from src.flows.document_ingest import ingest_document

async def ingest_test_data():
    print("Ingesting test conversation logs...")
    
    # 1. Office Chat
    print("\n--- Ingesting Office Chat ---")
    await ingest_document(
        file_path="DATA/convos/office_chat.txt",
        title="Office Incident & Planning",
        author="Engineering Team",
        slug="office_chat",
        doc_type="conversation",
        force_update=True,
        split_pattern=r"^\[(.*?)\] (.*?): (.*)$"
    )
    
    # 2. Personal Gossip
    print("\n--- Ingesting Personal Gossip ---")
    await ingest_document(
        file_path="DATA/convos/personal_gossip.txt",
        title="Friends Group Chat",
        author="The Group",
        slug="gossip_chat",
        doc_type="conversation",
        force_update=True,
        split_pattern=r"^\[(.*?)\] (.*?): (.*)$"
    )
    
    print("\nDone! Documents ingested.")

if __name__ == "__main__":
    asyncio.run(ingest_test_data())