"""
Ingestion Pipeline - Load docs, parse metadata, chunk, embed, store in MongoDB.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

from embeddings import NomicEmbeddings

load_dotenv()

# Configuration
MONGO_DB_URL = os.environ["MONGO_DB_URL"]
OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]

DB_NAME = "product_docs_rag"
COLLECTION_NAME = "hybrid_search"
INDEX_NAME = "vector_index"


def parse_metadata_header(content: str) -> dict:
    """Parse metadata fields from document header."""
    metadata = {
        "module": None,
        "roles": [],
        "linked_apis": [],
        "feature_flags": [],
        "route": None,
        "status": None,
        "auth_requirement": None,
    }

    lines = content.split("\n")

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("Module:"):
            metadata["module"] = line.replace("Module:", "").strip()
        elif line.startswith("Roles:"):
            roles_str = line.replace("Roles:", "").strip()
            metadata["roles"] = [r.strip() for r in roles_str.split(",") if r.strip()]
        elif line.startswith("Route / URL:") or line.startswith("Route/URL:"):
            metadata["route"] = line.split(":", 1)[1].strip()
        elif line.startswith("Status:"):
            metadata["status"] = line.replace("Status:", "").strip()
        elif line.startswith("Auth Requirement:"):
            metadata["auth_requirement"] = line.replace("Auth Requirement:", "").strip()
        elif line.startswith("Feature Flags:"):
            flags_str = line.replace("Feature Flags:", "").strip()
            if flags_str:
                metadata["feature_flags"] = [f.strip() for f in flags_str.split(",") if f.strip()]
        elif line.startswith("Linked APIs:"):
            apis_str = line.replace("Linked APIs:", "").strip()
            if apis_str.startswith("-"):
                api = apis_str.lstrip("- ").strip()
                if api:
                    metadata["linked_apis"].append(api)
            # Check subsequent lines for more APIs
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                if next_line.startswith("-"):
                    api = next_line.lstrip("- ").strip()
                    if api and not any(next_line.startswith(k) for k in ["Module:", "Roles:", "Status:"]):
                        metadata["linked_apis"].append(api)
                elif next_line and ":" in next_line and not next_line.startswith("-"):
                    break

    return metadata


def extract_content_sections(content: str) -> str:
    """Extract main content, skipping the metadata header."""
    lines = content.split("\n")
    content_lines = []
    in_content = False

    for line in lines:
        if line.strip().startswith("###"):
            in_content = True
        if in_content:
            content_lines.append(line)

    return "\n".join(content_lines).strip()


def load_document(md_path: Path) -> Document | None:
    """Load a single markdown document with parsed metadata."""
    try:
        content = md_path.read_text(encoding="utf-8")

        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else md_path.stem

        metadata = parse_metadata_header(content)
        main_content = extract_content_sections(content) or content

        metadata["source_file"] = md_path.name
        metadata["title"] = title

        # Prepend title/routes so they're searchable (not just in metadata)
        searchable_text = f"{title}\n\n"
        if metadata["route"]:
            searchable_text += f"Route: {metadata['route']}\n"
        if metadata["linked_apis"]:
            searchable_text += f"APIs: {', '.join(metadata['linked_apis'])}\n"
        searchable_text += f"\n{main_content}"

        return Document(page_content=searchable_text, metadata=metadata)

    except Exception as e:
        print(f"  Error loading {md_path.name}: {e}")
        return None


def load_all_documents(docs_dir: Path) -> list[Document]:
    """Load all markdown documents from the directory."""
    md_files = sorted(docs_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    documents = []
    for md_path in md_files:
        doc = load_document(md_path)
        if doc:
            documents.append(doc)

    print(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 2000, chunk_overlap: int = 200) -> list[Document]:
    """
    Split large documents into smaller chunks.
    Chunk size 2000 with 200 overlap balances context preservation vs retrieval precision.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_docs = []
    for doc in documents:
        if len(doc.page_content) > chunk_size:
            splits = text_splitter.split_documents([doc])
            for i, split in enumerate(splits):
                split.metadata["chunk_index"] = i
                split.metadata["total_chunks"] = len(splits)
            chunked_docs.extend(splits)
        else:
            doc.metadata["chunk_index"] = 0
            doc.metadata["total_chunks"] = 1
            chunked_docs.append(doc)

    print(f"Total chunks: {len(chunked_docs)}")
    return chunked_docs


def get_mongo_client():
    return MongoClient(MONGO_DB_URL)


def setup_mongodb_collection():
    """Set up MongoDB collection, clearing existing data."""
    client = get_mongo_client()
    db = client[DB_NAME]

    if COLLECTION_NAME in db.list_collection_names():
        print("Clearing existing collection...")
        db[COLLECTION_NAME].delete_many({})
    else:
        print("Creating new collection...")
        db.create_collection(COLLECTION_NAME)

    return client, db[COLLECTION_NAME]


def create_vector_store(collection, documents: list[Document]):
    """Generate embeddings with Ollama and store in MongoDB."""
    embeddings = NomicEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    print(f"Creating embeddings for {len(documents)} chunks...")
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )
    print(f"Stored {len(documents)} documents")
    return vector_store


def create_vector_search_index(collection):
    """Create vector search index (768 dims for nomic-embed-text, cosine similarity)."""
    try:
        collection.create_search_index({
            "name": INDEX_NAME,
            "type": "vectorSearch",
            "definition": {
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 768,
                    "similarity": "cosine"
                }]
            }
        })
        print(f"Created vector search index: {INDEX_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("Vector index already exists")
        else:
            print(f"Index note: {e}")


def main():
    """Run the ingestion pipeline."""
    print("=" * 50)
    print("INGESTION PIPELINE")
    print("=" * 50)

    docs_dir = Path(__file__).parent.parent / "product-documentation"
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs not found: {docs_dir}")

    # 1. Load
    print("\n1. Loading documents...")
    documents = load_all_documents(docs_dir)

    # 2. Chunk
    print("\n2. Chunking...")
    chunked_documents = chunk_documents(documents)

    # 3. Embed & Store
    print("\n3. Embedding and storing...")
    client, collection = setup_mongodb_collection()

    try:
        create_vector_store(collection, chunked_documents)
        create_vector_search_index(collection)
        print(f"\nDone! {len(documents)} docs -> {len(chunked_documents)} chunks in {DB_NAME}.{COLLECTION_NAME}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
