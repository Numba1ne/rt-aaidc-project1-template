import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # Use LangChain's RecursiveCharacterTextSplitter for better chunking
        # It preserves sentence boundaries and handles text structure better
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,  # Overlap between chunks to preserve context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents with 'content' and 'metadata' keys
        """
        print(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        # Process each document
        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            if not content:
                print(f"Warning: Document {doc_idx} has no content, skipping...")
                continue
            
            # Chunk the document
            chunks = self.chunk_text(content)
            
            # Prepare data for ChromaDB
            if chunks:
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                    
                    # Create metadata for this chunk (include original doc metadata)
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": chunk_idx,
                        "document_index": doc_idx
                    }
                    
                    all_chunks.append(chunk)
                    all_metadatas.append(chunk_metadata)
                    all_ids.append(chunk_id)
                
                print(f"  Document {doc_idx + 1}/{len(documents)}: Created {len(chunks)} chunks")
        
        # Create embeddings for all chunks at once (more efficient)
        if all_chunks:
            embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=False)
            
            # Convert embeddings to list format for ChromaDB
            embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
            # Add all chunks to ChromaDB in one batch
            self.collection.add(
                ids=all_ids,
                embeddings=embeddings_list,
                documents=all_chunks,
                metadatas=all_metadatas
            )
            print(f"Successfully added {len(all_chunks)} chunks to vector database")
        else:
            print("No chunks to add to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # Create embedding for the query
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        
        # Convert to list format for ChromaDB
        query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=query_embedding_list,
            n_results=n_results
        )
        
        # ChromaDB returns results in a specific format
        # Handle the case where results might be empty
        if not results or not results.get("documents") or len(results["documents"]) == 0:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }
        
        # ChromaDB returns nested lists, so we need to extract the first element
        # since we only queried with one embedding
        return {
            "documents": results["documents"][0] if results.get("documents") else [],
            "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
            "distances": results["distances"][0] if results.get("distances") else [],
            "ids": results["ids"][0] if results.get("ids") else [],
        }
