"""
Sequence/Text embedding backend with support for:
- SBERT (sentence-transformers)
- HF mean-pool (transformers)
- ProtBERT/ProtTrans preprocessing helper
- DNABERT usage via HF
- optional ESM (fair-esm) wrapper (best-effort)
"""
from typing import List, Optional, Dict
import logging, os, threading
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

try:
    import esm
    HAS_ESM = True
except Exception:
    HAS_ESM = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL_MAP = {
    "sbert": "sentence-transformers/all-mpnet-base-v2",
    "protbert": "Rostlab/prot_bert",
    "dnabert": "zhihan1996/DNAbert-6",
}

class SequenceEmbeddingBackend:
    def __init__(self, model_map: Optional[Dict[str,str]] = None, device: Optional[str] = None):
        self.model_map = model_map or DEFAULT_MODEL_MAP
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._hf_lock = threading.Lock()
        self._hf_cache = {}
        self._sbert_cache = {}
        logger.info(f"SequenceEmbeddingBackend device={self.device}")

    def _get_sbert(self, model_id: str):
        if not HAS_SBERT:
            raise RuntimeError("sentence-transformers not installed")
        if model_id in self._sbert_cache:
            return self._sbert_cache[model_id]
        model = SentenceTransformer(model_id)
        if torch.cuda.is_available():
            try:
                model = model.to("cuda")
            except Exception:
                logger.warning("SBERT GPU move failed")
        self._sbert_cache[model_id] = model
        return model

    def _get_hf(self, model_id: str):
        with self._hf_lock:
            if model_id in self._hf_cache:
                return self._hf_cache[model_id]
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model = AutoModel.from_pretrained(model_id)
            model.eval()
            model.to(self.device)
            self._hf_cache[model_id] = (tok, model)
            return tok, model

    def _prep_prot_seq(self, seq: str) -> str:
        cleaned = "".join([c for c in seq.strip().upper() if c.isalpha()])
        return " ".join(list(cleaned))

    def embed_texts_sbert(self, texts: List[str], model_key: str = "sbert", batch_size: int = 64) -> np.ndarray:
        model_id = self.model_map.get(model_key, model_key)
        sbert = self._get_sbert(model_id)
        emb = sbert.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=batch_size)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def embed_texts_hf(self, texts: List[str], model_key: str, batch_size: int = 16) -> np.ndarray:
        model_id = self.model_map.get(model_key, model_key)
        tok, model = self._get_hf(model_id)
        all_emb = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                toks = tok(batch, padding=True, truncation=True, return_tensors="pt")
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = model(**toks)
                last = out.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1).expand(last.size()).float()
                summed = torch.sum(last * mask, dim=1).cpu().numpy()
                counts = mask.sum(1).cpu().numpy()
                counts[counts == 0] = 1.0
                mean = summed / counts
                all_emb.append(mean)
        emb = np.vstack(all_emb)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def embed_proteins(self, sequences: List[str], model_key: str = "protbert", prefer_sbert: bool = False, batch_size: int = 32) -> np.ndarray:
        model_id = self.model_map.get(model_key, model_key)
        texts = [self._prep_prot_seq(s) for s in sequences]
        use_sbert = prefer_sbert or model_id.startswith("sentence-transformers/")
        if use_sbert and HAS_SBERT:
            return self.embed_texts_sbert(texts, model_key, batch_size=batch_size)
        else:
            return self.embed_texts_hf(texts, model_key, batch_size=batch_size)

    def embed_dna(self, sequences: List[str], model_key: str = "dnabert", batch_size: int = 32) -> np.ndarray:
        texts = [s.strip().upper().replace(" ", "") for s in sequences]
        return self.embed_texts_hf(texts, model_key, batch_size=batch_size)

    def embed_with_esm(self, sequences: List[str], esm_model_name: Optional[str] = None, batch_size: int = 16) -> np.ndarray:
        if not HAS_ESM:
            raise RuntimeError("ESM package not available")
        model_data = esm_model_name or "esm2_t6_8M_UR50D"
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_data)
        model = model.eval().to(self.device)
        batch_converter = alphabet.get_batch_converter()
        all_emb = []
        for i in range(0, len(sequences), batch_size):
            batch = [(f"s{i+j}", sequences[i+j]) for j in range(min(batch_size, len(sequences)-i))]
            labels, seqs, toks = batch_converter(batch)
            toks = toks.to(self.device)
            with torch.no_grad():
                results = model(toks, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            seq_lens = (toks != alphabet.padding_idx).sum(1).cpu().numpy()
            for j in range(token_representations.size(0)):
                length = seq_lens[j] - 2
                rep = token_representations[j, 1:1+length].mean(0).cpu().numpy()
                all_emb.append(rep)
        emb = np.vstack(all_emb)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms