import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    source: str
    page: Optional[int]
    chunk_index: int
    content: str
    metadata: Dict[str, Any]

class DocumentParser:
    """Parses various document types into raw pages/text with metadata."""
    
    @staticmethod
    def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
        pages_data = []
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_data.append({
                        "text": text.strip(),
                        "page": page_idx + 1
                    })
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF '{file_path}': {e}")
        return pages_data

    @staticmethod
    def parse_docx(file_path: str) -> List[Dict[str, Any]]:
        try:
            import docx
            doc = docx.Document(file_path)
            full_text = []
            for p in doc.paragraphs:
                if p.text.strip():
                    full_text.append(p.text.strip())
            return [{"text": "\n\n".join(full_text), "page": 1}]
        except Exception as e:
            raise RuntimeError(f"Error parsing DOCX '{file_path}': {e}")

    @staticmethod
    def parse_text(file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [{"text": content.strip(), "page": 1}]
        except Exception as e:
            raise RuntimeError(f"Error reading text file '{file_path}': {e}")

    @classmethod
    def load_document(cls, file_path: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return cls.parse_docx(file_path)
        elif ext in [".txt", ".md", ".csv", ".json", ".py", ".html"]:
            return cls.parse_text(file_path)
        else:
            # Fallback to plain text read
            return cls.parse_text(file_path)


class TextChunker:
    """Recursively splits text by paragraphs, sentences, and words with overlap."""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        # Normalize whitespace
        text = re.sub(r'\r\n|\r', '\n', text)
        
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        # Split by double newlines (paragraphs), then single newlines, then sentences, then spaces
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_len = len(para)
            
            if current_length + para_len + 2 <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_len + 2
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                
                # If a single paragraph is larger than chunk_size, split by sentences
                if para_len > self.chunk_size:
                    sub_sentences = re.split(r'(?<=[.?!])\s+', para)
                    sub_chunk = []
                    sub_len = 0
                    for sent in sub_sentences:
                        if sub_len + len(sent) + 1 <= self.chunk_size:
                            sub_chunk.append(sent)
                            sub_len += len(sent) + 1
                        else:
                            if sub_chunk:
                                chunks.append(" ".join(sub_chunk))
                            sub_chunk = [sent]
                            sub_len = len(sent)
                    if sub_chunk:
                        current_chunk = [ " ".join(sub_chunk) ]
                        current_length = sub_len
                    else:
                        current_chunk = []
                        current_length = 0
                else:
                    current_chunk = [para]
                    current_length = para_len
                    
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        # Apply overlap between chunks if appropriate
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    prev_chunk = chunks[i - 1]
                    overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) >= self.chunk_overlap else prev_chunk
                    overlapped_chunks.append(f"... {overlap_text}\n{chunk}")
            return overlapped_chunks
            
        return chunks

    def process_document(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        filename = os.path.basename(file_path)
        pages_data = DocumentParser.load_document(file_path)
        
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0
        
        for page_data in pages_data:
            page_text = page_data["text"]
            page_num = page_data.get("page", 1)
            
            raw_chunks = self.split_text(page_text)
            for chunk_content in raw_chunks:
                if not chunk_content.strip():
                    continue
                chunk_id = f"{doc_id}_p{page_num}_c{global_chunk_idx}"
                meta = {
                    "doc_id": doc_id,
                    "source": filename,
                    "page": page_num,
                    "chunk_index": global_chunk_idx,
                    "length": len(chunk_content)
                }
                all_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source=filename,
                    page=page_num,
                    chunk_index=global_chunk_idx,
                    content=chunk_content,
                    metadata=meta
                ))
                global_chunk_idx += 1
                
        return all_chunks
