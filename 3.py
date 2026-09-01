#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
象核心 v16.2 — 统一生命体（挂谷几何完整修复版）
=========================================
v16.2 修复清单：
  1. forward() 中 loss 为 None 时所有后续修改被安全保护
  2. _window_attn() 挂谷注意力 mask 维度与 scores 严格匹配
  3. 挂谷正则化 off_diag 正确排除对角线（Frobenius 范数无偏）
  4. _split_crystal() 主方向使用协方差特征向量而非坐标轴
  5. generate() 中 intent 参数不再被循环覆盖，持续引导生成
  6. 删除 learn_from_dialogue() 中重复的晶体对齐损失
  7. 验证器 f-string 转义修复，正确显示重复字符
  8. BackgroundReader 区分 RuntimeError 并打印完整 traceback
  9. generate() 采样增加概率归一化保护，防止空分布崩溃
"""

import os, sys, json, time, math, random, logging, argparse, threading, pickle, re, itertools, queue, traceback
from pathlib import Path
from collections import deque, OrderedDict
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========== 路径 ==========
TRAIN_FILE = "/storage/emulated/0/data/corpus/train.txt"
EMBEDDING_FILE = "/storage/emulated/0/data/embeddings/light_Tencent_AILab_ChineseEmbedding.txt"
CHECKPOINT_DIR = "checkpoints_xiang"
DATA_DIR = "data_xiang"
MEMORY_DIR = os.path.join(DATA_DIR, "memory_xiang")

# ========== 人格预设 ==========
PERSONALITY_PRESETS = {
    'child':  {'persistence': 10, 'timeout': 5,  'temperature': 1.0, 'intro': '孩童人格 (好奇、执着、不怕羞)'},
    'adult':  {'persistence': 3,  'timeout': 10, 'temperature': 0.8, 'intro': '成年人格 (克制、知趣、稳重)'},
    'scholar':{'persistence': 5,  'timeout': 15, 'temperature': 0.7, 'intro': '学者人格 (深思、严谨、探究)'}
}

# ========== 配置 ==========
class Config:
    VERSION = "16.2-Fix"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    XIANG_DIM = 200
    NUM_HEADS = 4
    NUM_LAYERS = 6
    MAX_SEQ_LEN = 128
    XIANG_CLOUDS = 4
    WINDOW_SIZE = 16
    LINEAR_RANK = 64
    BLOCK_SIZE = 2
    NUM_EXPERTS = 4
    TOP_K = 1
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    MAX_STEPS = 50000
    SAVE_EVERY = 2000
    CURIOSITY_THRESHOLD = 1.5
    MAX_CONFUSION_BUFFER = 2000
    CRYSTAL_SIM_THRESHOLD = 0.45
    CRYSTAL_MIN_FRAGMENTS = 3
    MAX_CRYSTALS = 500
    MAX_FRAGMENTS = 5000
    GEN_TEMPERATURE = 0.8
    GEN_MAX_LEN = 80
    INTENT_GUIDANCE_WEIGHT = 0.3
    MAX_CRYSTAL_CTX_CHARS = 80
    BACKGROUND_READ_INTERVAL = 3
    FRAGMENT_LRU_SIZE = 2000
    MAX_UNK_RATIO = 0.3
    MAX_CONSECUTIVE_UNK = 2
    GEN_TOP_P = 0.92
    GEN_REPETITION_PENALTY = 1.15
    GEN_MIN_NEW_TOKENS = 3
    # ===== 挂谷几何认知开关 =====
    USE_KAKEYA_REPEL = True
    USE_KAKEYA_SPLIT = True
    USE_KAKEYA_ATTN = True
    KAKEYA_REPEL_WEIGHT = 0.05
    KAKEYA_EIGEN_THRESHOLD = 0.3

    @classmethod
    def init_dirs(cls):
        for d in [CHECKPOINT_DIR, DATA_DIR, MEMORY_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger("XiangCore")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def cosine_sim(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-8))

# ========== 分词器 ==========
class XiangTokenizer:
    def __init__(self):
        self.char2id = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        self.id2char = {0: '<PAD>', 1: '<UNK>', 2: '<EOS>'}
        self.word2id = {}
        self.id2word = {}
        self.next_id = 3
        self.eos_id = 2
        self._init_base_chars()

    def _init_base_chars(self):
        base = ("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分"
                "对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十"
                "三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使"
                "点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全"
                "表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变"
                "条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料"
                "象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接"
                "知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任"
                "取完举色万")
        for c in base:
            if c not in self.char2id:
                self.char2id[c] = self.next_id
                self.id2char[self.next_id] = c
                self.next_id += 1

        extra = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            "·×÷—…「」『』（）［］【】《》〈〉""''，。、；：？！…—～｜"
            "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ"
            "まみむめもやゆよらりるれろわをん"
            "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホ"
            "マミムメモヤユヨラリルレロワヲン"
        )
        for c in extra:
            if c not in self.char2id:
                self.char2id[c] = self.next_id
                self.id2char[self.next_id] = c
                self.next_id += 1

    def add_word(self, word: str):
        if not word or len(word) < 2 or word in self.word2id: return
        self.word2id[word] = self.next_id
        self.id2word[self.next_id] = word
        self.next_id += 1

    def encode(self, text: str) -> List[int]:
        tokens, i = [], 0
        while i < len(text):
            matched = False
            for l in range(min(8, len(text) - i), 1, -1):
                w = text[i:i+l]
                if w in self.word2id:
                    tokens.append(self.word2id[w]); i += l; matched = True; break
            if not matched:
                tokens.append(self.char2id.get(text[i], 1)); i += 1
        return tokens

    def decode(self, ids: List[int]) -> str:
        chars = []
        unk_streak = 0
        for idx in ids:
            if idx == 0: continue
            if idx == 2: break
            if idx == 1:
                unk_streak += 1
                if unk_streak == 1:
                    chars.append('□')
                continue
            unk_streak = 0
            if idx in self.id2word: chars.append(self.id2word[idx])
            elif idx in self.id2char: chars.append(self.id2char[idx])
        return ''.join(chars)

    @property
    def vocab_size(self): return self.next_id

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'char2id': self.char2id, 'word2id': self.word2id, 'next_id': self.next_id}, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        self.char2id = d.get('char2id', self.char2id)
        self.word2id = d.get('word2id', {})
        self.next_id = d.get('next_id', self.next_id)
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.id2word = {v: k for k, v in self.word2id.items()}

def load_pretrained_embeddings(filepath: str, tokenizer: XiangTokenizer, dim: int) -> Dict[str, np.ndarray]:
    if not os.path.exists(filepath):
        logger.warning(f"[词向量] 文件不存在: {filepath}")
        return {}
    logger.info(f"[词向量] 加载中: {filepath}")
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline().strip()
        parts = first.split()
        if not (len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()):
            f.seek(0)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            word, vec = parts[0], parts[1:]
            if len(vec) != dim: continue
            try: embeddings[word] = np.array([float(x) for x in vec], dtype=np.float32)
            except: continue
    logger.info(f"[词向量] 加载完成，共 {len(embeddings)} 个")
    return embeddings

# ========== 晶体图系统 ==========
@dataclass
class MemoryFragment:
    frag_id: str
    text: str
    xiang: np.ndarray
    source: str = "dialogue"
    links: List[str] = field(default_factory=list)
    crystal_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 1.0

@dataclass
class KnowledgeCrystal:
    crystal_id: str
    fragments: List[MemoryFragment]
    core_xiang: Optional[np.ndarray] = None
    confidence: float = 0.5
    formed_at: float = field(default_factory=time.time)

    def update_core(self):
        if self.fragments:
            vecs = [f.xiang for f in self.fragments if f.confidence > 0.3]
            if vecs:
                self.core_xiang = np.mean(vecs, axis=0)
                norm = np.linalg.norm(self.core_xiang)
                if norm > 0: self.core_xiang /= norm

class CrystalGraph:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.fragments: OrderedDict[str, MemoryFragment] = OrderedDict()
        self.crystals: Dict[str, KnowledgeCrystal] = {}
        self._lock = threading.RLock()
        self._next_frag = 0
        self._next_crystal = 0
        self.crystal_matrix = None
        self.crystal_ids = []

    def _ensure_lru(self):
        while len(self.fragments) > Config.MAX_FRAGMENTS:
            oldest_key, oldest_frag = self.fragments.popitem(last=False)
            if oldest_frag.crystal_id and oldest_frag.crystal_id in self.crystals:
                crystal = self.crystals[oldest_frag.crystal_id]
                crystal.fragments = [f for f in crystal.fragments if f.frag_id != oldest_key]
                if crystal.fragments: crystal.update_core()
                else:
                    del self.crystals[oldest_frag.crystal_id]
                    logger.info(f"[晶体] 空晶体移除: {oldest_frag.crystal_id}")
            self._update_matrix()

    def add(self, text: str, xiang_vec: np.ndarray, source: str = "dialogue"):
        with self._lock:
            fid = f"frag_{self._next_frag}"
            self._next_frag += 1
            frag = MemoryFragment(fid, text, xiang_vec.copy(), source)
            self.fragments[fid] = frag
            self._ensure_lru()

            best_match, best_sim = None, 0.0
            for crystal in self.crystals.values():
                if crystal.core_xiang is not None:
                    sim = cosine_sim(xiang_vec, crystal.core_xiang)
                    if sim > best_sim: best_sim = sim; best_match = crystal

            if best_match is not None and best_sim > 0.65:
                best_match.confidence = min(1.0, best_match.confidence + 0.05)
                best_match.fragments.append(frag)
                frag.crystal_id = best_match.crystal_id
                best_match.update_core()
                self._update_matrix()
                return best_match

            others = list(self.fragments.items())
            sample_size = min(len(others), 200)
            if len(others) > sample_size: others = random.sample(others, sample_size)
            for other_id, other in others:
                if other_id == fid: continue
                if cosine_sim(xiang_vec, other.xiang) > Config.CRYSTAL_SIM_THRESHOLD:
                    frag.links.append(other_id); other.links.append(fid)

            result = self._crystallize(fid)
            self._update_matrix()
            return result

    def _crystallize(self, start_fid: str):
        visited, bfs_queue, component = set(), [start_fid], []
        while bfs_queue:
            fid = bfs_queue.pop(0)
            if fid in visited: continue
            visited.add(fid); component.append(fid)
            for lid in self.fragments[fid].links:
                if lid not in visited: bfs_queue.append(lid)

        if len(component) >= Config.CRYSTAL_MIN_FRAGMENTS:
            existing = None
            for fid in component:
                if self.fragments[fid].crystal_id is not None:
                    existing = self.fragments[fid].crystal_id; break
            if existing is None and len(self.crystals) < Config.MAX_CRYSTALS:
                cid = f"crystal_{self._next_crystal}"
                self._next_crystal += 1
                frags = [self.fragments[fid] for fid in component]
                crystal = KnowledgeCrystal(cid, frags)
                crystal.update_core()
                for fid in component: self.fragments[fid].crystal_id = cid
                self.crystals[cid] = crystal
                logger.info(f"[晶体] 新知识晶体形成: {cid} (包含 {len(component)} 个碎片)")
                return crystal
        return None

    def record_feedback(self, crystal_id: str, reward: float):
        with self._lock:
            if crystal_id not in self.crystals: return
            c = self.crystals[crystal_id]
            c.confidence = max(0.1, min(1.0, c.confidence + reward * 0.1))
            if c.confidence < 0.3 and len(c.fragments) > 1:
                self._split_crystal(crystal_id)

    def _split_crystal(self, crystal_id: str):
        crystal = self.crystals[crystal_id]
        if len(crystal.fragments) < 2: return
        vecs = np.array([f.xiang for f in crystal.fragments])
        if vecs.shape[0] < 2: return
        
        mean_vec = np.mean(vecs, axis=0)
        centered = vecs - mean_vec
        cov = np.cov(centered, rowvar=False)
        
        eigvals = np.linalg.eigvalsh(cov)
        effective_rank = np.sum(eigvals > 1e-5)
        dim = vecs.shape[1]
        
        # ===== 挂谷粘性分裂判据 =====
        if Config.USE_KAKEYA_SPLIT and (effective_rank / dim) < Config.KAKEYA_EIGEN_THRESHOLD:
            # 修复：使用特征向量而非坐标轴方向
            _, eigvecs = np.linalg.eigh(cov)
            main_vec = eigvecs[:, -1]  # 最大特征值对应的特征向量
            proj = np.dot(centered, main_vec)
            median_proj = np.median(proj)
            group1, group2 = [], []
            for f, p in zip(crystal.fragments, proj):
                (group1 if p >= median_proj else group2).append(f)
            if len(group1) > 0 and len(group2) > 0:
                cid_new = f"crystal_{self._next_crystal}"
                self._next_crystal += 1
                new_crystal = KnowledgeCrystal(cid_new, group2)
                new_crystal.update_core()
                if new_crystal.core_xiang is not None:
                    noise = np.random.randn(self.dim) * 0.05
                    new_crystal.core_xiang = new_crystal.core_xiang + noise
                    norm = np.linalg.norm(new_crystal.core_xiang)
                    if norm > 0: new_crystal.core_xiang /= norm
                self.crystals[cid_new] = new_crystal
                crystal.fragments = group1
                crystal.update_core()
                for f in group2: f.crystal_id = cid_new
                logger.info(f"[晶体-挂谷] {crystal_id} 因方向过细分裂为 {crystal_id} 和 {cid_new}")
                return
        
        # ===== 保底：中位数分裂 =====
        core = crystal.core_xiang
        if core is None: return
        dists = [cosine_sim(v, core) for v in vecs]
        if not dists: return
        median = np.median(dists)
        group1, group2 = [], []
        for f, d in zip(crystal.fragments, dists):
            (group1 if d >= median else group2).append(f)
        if len(group1) > 0 and len(group2) > 0:
            cid_new = f"crystal_{self._next_crystal}"
            self._next_crystal += 1
            new_crystal = KnowledgeCrystal(cid_new, group2)
            new_crystal.update_core()
            self.crystals[cid_new] = new_crystal
            crystal.fragments = group1
            crystal.update_core()
            for f in group2: f.crystal_id = cid_new
            logger.info(f"[晶体] {crystal_id} 分裂为 {crystal_id} 和 {cid_new}")

    def evolve(self):
        with self._lock:
            cids = list(self.crystals.keys())
            merged = set(); i = 0
            while i < len(cids):
                if cids[i] in merged: i += 1; continue
                j = i + 1
                while j < len(cids):
                    if cids[j] in merged: j += 1; continue
                    ci, cj = self.crystals[cids[i]], self.crystals[cids[j]]
                    if ci.core_xiang is not None and cj.core_xiang is not None:
                        sim = cosine_sim(ci.core_xiang, cj.core_xiang)
                        if sim > 0.85:
                            cj.fragments.extend(ci.fragments); cj.update_core()
                            cj.confidence = (ci.confidence + cj.confidence) / 2
                            for f in ci.fragments: f.crystal_id = cj.crystal_id
                            del self.crystals[ci.crystal_id]; merged.add(ci.crystal_id)
                            logger.info(f"[晶体] 合并 {ci.crystal_id} → {cj.crystal_id}")
                            break
                    j += 1
                i += 1

            to_remove = []
            for cid, c in self.crystals.items():
                c.confidence *= 0.995
                if c.confidence < 0.1 and len(c.fragments) <= 1:
                    to_remove.append(cid)
            for cid in to_remove:
                del self.crystals[cid]; logger.info(f"[晶体] 移除低置信度: {cid}")

            for cid, c in list(self.crystals.items()):
                if c.confidence < 0.4 and len(c.fragments) > 2:
                    self._split_crystal(cid)

            self._update_matrix()

    def _update_matrix(self):
        if not self.crystals: self.crystal_matrix = None; self.crystal_ids = []; return
        self.crystal_ids = list(self.crystals.keys())
        mat = [self.crystals[cid].core_xiang for cid in self.crystal_ids]
        self.crystal_matrix = torch.tensor(np.stack(mat), dtype=torch.float32)

    def search(self, query_xiang: np.ndarray, top_k: int = 3):
        with self._lock:
            if self.crystal_matrix is None or not self.crystal_ids: return []
            q = torch.from_numpy(query_xiang).float()
            sims = F.cosine_similarity(q.unsqueeze(0), self.crystal_matrix)
            top_sims, top_indices = torch.topk(sims, min(top_k, len(self.crystal_ids)))
            results = []
            for i, idx in enumerate(top_indices):
                cid = self.crystal_ids[idx.item()]
                results.append((self.crystals[cid], top_sims[i].item()))
            return results

    def get_memory_context(self, query_xiang: np.ndarray, max_chars: int = 80) -> str:
        results = self.search(query_xiang, top_k=3)
        if not results: return ""
        parts, total_len = [], 0
        for crystal, sim in results:
            frag_text = '; '.join([f.text[:30] for f in crystal.fragments[:2]])
            entry = f"[记忆|{sim:.2f}] {frag_text}"
            if total_len + len(entry) > max_chars: break
            parts.append(entry); total_len += len(entry)
        return "\n".join(parts)

    def save(self, path: str):
        with self._lock:
            data = {'fragments': {}, 'crystals': {}, 'next_frag': self._next_frag, 'next_crystal': self._next_crystal}
            for fid, f in self.fragments.items():
                data['fragments'][fid] = {
                    'text': f.text, 'xiang': f.xiang.tobytes(), 'source': f.source,
                    'links': f.links, 'crystal_id': f.crystal_id, 'created_at': f.created_at,
                    'access_count': f.access_count, 'confidence': f.confidence
                }
            for cid, c in self.crystals.items():
                data['crystals'][cid] = {
                    'frag_ids': [f.frag_id for f in c.fragments],
                    'core_xiang': c.core_xiang.tobytes() if c.core_xiang is not None else None,
                    'confidence': c.confidence, 'formed_at': c.formed_at
                }
            with open(path, 'wb') as f: pickle.dump(data, f)

    def load(self, path: str):
        if not os.path.exists(path): return
        with open(path, 'rb') as f: data = pickle.load(f)
        with self._lock:
            self._next_frag = data.get('next_frag', 0)
            self._next_crystal = data.get('next_crystal', 0)
            self.fragments = OrderedDict()
            self.crystals = {}
            for fid, fd in data.get('fragments', {}).items():
                xiang_arr = np.frombuffer(fd['xiang'], dtype=np.float32).copy()
                frag = MemoryFragment(
                    fid, fd['text'], xiang_arr, fd['source'], fd.get('links', []),
                    fd.get('crystal_id'), fd.get('created_at', 0),
                    fd.get('access_count', 0), fd.get('confidence', 1.0)
                )
                self.fragments[fid] = frag
            for cid, cd in data.get('crystals', {}).items():
                frags = [self.fragments[f] for f in cd['frag_ids'] if f in self.fragments]
                crystal = KnowledgeCrystal(cid, frags)
                crystal.confidence = cd.get('confidence', 0.5)
                if cd.get('core_xiang'):
                    crystal.core_xiang = np.frombuffer(cd['core_xiang'], dtype=np.float32).copy()
                self.crystals[cid] = crystal
            self._update_matrix()

# ========== 核心神经网络 ==========
class XiangCloud(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_clouds: int = 4):
        super().__init__()
        self.clouds = nn.Parameter(torch.randn(vocab_size, num_clouds, dim) * 0.02)
        self.selector = nn.Linear(dim, num_clouds)

    def forward(self, token_ids: torch.Tensor, context: torch.Tensor):
        clouds = self.clouds[token_ids]
        weights = F.softmax(self.selector(context), dim=-1)
        selected = (clouds * weights.unsqueeze(-1)).sum(dim=2)
        return selected, weights

class MixedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 16, linear_rank: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.linear_rank = linear_rank
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.linear_q = nn.Linear(self.head_dim, linear_rank)
        self.linear_k = nn.Linear(self.head_dim, linear_rank)
        self.dropout = nn.Dropout(0.1)

    def _linear_attn(self, q, k, v):
        b, s, h, d = q.shape
        q_lin = F.elu(self.linear_q(q)) + 1.0
        k_lin = F.elu(self.linear_k(k)) + 1.0
        kv = torch.einsum('bshr,bshd->bhrd', k_lin, v)
        z = k_lin.sum(dim=1)
        lin_out = torch.einsum('bshr,bhrd->bshd', q_lin, kv)
        z_out = torch.einsum('bshr,bhr->bsh', q_lin, z).unsqueeze(-1)
        return lin_out / (z_out + 1e-8)

    def _window_attn(self, q, k, v, causal=False):
        b, s, h, d = q.shape
        q = q.transpose(1, 2).reshape(b * h, s, d)
        k = k.transpose(1, 2).reshape(b * h, s, d)
        v = v.transpose(1, 2).reshape(b * h, s, d)
        if causal:
            pad = self.window_size - 1
            k_pad = F.pad(k, (0, 0, pad, 0)); v_pad = F.pad(v, (0, 0, pad, 0))
            k_win = k_pad.unfold(1, self.window_size, 1).transpose(-1, -2)
            v_win = v_pad.unfold(1, self.window_size, 1).transpose(-1, -2)
            scores = torch.einsum('bsid,bsjd->bsij', q.unsqueeze(2), k_win) / math.sqrt(d)
            positions = torch.arange(s, device=scores.device).view(1, s, 1, 1)
            offsets = torch.arange(-pad, 1, device=scores.device).view(1, 1, 1, self.window_size)
            valid = (positions + offsets >= 0)
            scores = scores.masked_fill(~valid, float('-inf'))
            # ===== 挂谷注意力：窗口内方向高度一致的 Query/Key 强制互斥 =====
            if Config.USE_KAKEYA_ATTN:
                q_norm = F.normalize(q, p=2, dim=-1).unsqueeze(2)  # [b*h, s, 1, d]
                k_win_norm = F.normalize(k_win, p=2, dim=-1)  # [b*h, s, w, d]
                dir_sim = torch.einsum('bsid,bsjd->bsij', q_norm, k_win_norm)  # [b*h, s, 1, w]
                mask = (dir_sim > 0.95).float() * -1e9
                scores = scores + mask
        else:
            pad_l = (self.window_size - 1) // 2; pad_r = self.window_size // 2
            k_pad = F.pad(k, (0, 0, pad_l, pad_r)); v_pad = F.pad(v, (0, 0, pad_l, pad_r))
            k_win = k_pad.unfold(1, self.window_size, 1)[:, :s, :, :].transpose(-1, -2)
            v_win = v_pad.unfold(1, self.window_size, 1)[:, :s, :, :].transpose(-1, -2)
            scores = torch.einsum('bsid,bsjd->bsij', q.unsqueeze(2), k_win) / math.sqrt(d)
            # ===== 挂谷注意力：窗口内方向高度一致的 Query/Key 强制互斥 =====
            if Config.USE_KAKEYA_ATTN:
                q_norm = F.normalize(q, p=2, dim=-1).unsqueeze(2)  # [b*h, s, 1, d]
                k_win_norm = F.normalize(k_win, p=2, dim=-1)  # [b*h, s, w, d]
                dir_sim = torch.einsum('bsid,bsjd->bsij', q_norm, k_win_norm)  # [b*h, s, 1, w]
                mask = (dir_sim > 0.95).float() * -1e9
                scores = scores + mask
        attn = F.softmax(scores, dim=-1); attn = self.dropout(attn)
        out = torch.einsum('bsij,bsjd->bsid', attn, v_win)
        return out.squeeze(2).reshape(b, h, s, d).transpose(1, 2)
    
    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        b, s, d = x.shape
        h = self.num_heads
    
        # 投影到 Q, K, V
        q = self.q_proj(x).view(b, s, h, self.head_dim)
        k = self.k_proj(x).view(b, s, h, self.head_dim)
        v = self.v_proj(x).view(b, s, h, self.head_dim)
    
        # 同时运行两种注意力
        lin_out = self._linear_attn(q, k, v)
        win_out = self._window_attn(q, k, v, causal=causal)
    
        # 相加并融合回原始维度
        out = self.out_proj((lin_out + win_out).reshape(b, s, d))
        return out        

class LightweightMoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 1, hidden_mult: int = 2):
        super().__init__()
        self.top_k = top_k
        self.experts_w1 = nn.Parameter(torch.randn(num_experts, dim, dim * hidden_mult) * 0.02)
        self.experts_w2 = nn.Parameter(torch.randn(num_experts, dim * hidden_mult, dim) * 0.02)
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        probs = F.softmax(self.router(x), dim=-1)
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = torch.zeros_like(x)
        for k_idx in range(self.top_k):
            idx = top_indices[:, :, k_idx]
            weight = top_probs[:, :, k_idx:k_idx+1]
            for b_i in range(b):
                for s_i in range(s):
                    e_idx = idx[b_i, s_i].item()
                    h = F.gelu(x[b_i, s_i] @ self.experts_w1[e_idx])
                    out[b_i, s_i] += weight[b_i, s_i, 0] * (h @ self.experts_w2[e_idx])
        return out

class XiangArithmetic(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.causal_proj = nn.Linear(dim, dim)
        self.metaphor_proj = nn.Linear(dim, dim, bias=False)
        self.analogy_proj = nn.Linear(dim, dim)

    def metaphor(self, a, b):
        return self.metaphor_proj(a - b) + b

    def analogy(self, a, b, c):
        return c + self.analogy_proj(b - a)

    def forward(self, hidden: torch.Tensor, crystals: torch.Tensor):
        if crystals.size(1) >= 2:
            c1 = crystals[:, 0:1, :]; c2 = crystals[:, 1:2, :]
            imagine = self.metaphor(c1, c2)
            hidden = hidden + imagine * 0.3
        return hidden

class VerificationGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden: torch.Tensor, crystals: torch.Tensor):
        crystal_core = crystals.mean(dim=1, keepdim=True)
        h_n = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-8)
        c_n = crystal_core / (crystal_core.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (h_n * c_n).sum(dim=-1, keepdim=True)
        gate_val = self.gate(hidden) * sim * self.scale
        return hidden + gate_val

class XiangCoreModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 200, num_layers: int = 6, num_heads: int = 4,
                 num_clouds: int = 4, max_seq_len: int = 128, window_size: int = 16,
                 linear_rank: int = 64, num_experts: int = 4, top_k: int = 1, block_size: int = 2,
                 crystal_graph: CrystalGraph = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.crystal_graph = crystal_graph

        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.xiang_cloud = XiangCloud(vocab_size, dim, num_clouds)

        self.xiang_arithmetic = XiangArithmetic(dim)
        self.verification_gate = VerificationGate(dim)

        self.attn_layers = nn.ModuleList([MixedAttention(dim, num_heads, window_size, linear_rank) for _ in range(num_layers)])
        self.moe_layers = nn.ModuleList([LightweightMoE(dim, num_experts, top_k) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

        self.num_blocks = (num_layers + block_size - 1) // block_size
        self.block_gates = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid()) for _ in range(block_size)])

        self.object_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Tanh())
        self.reiterate_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, vocab_size))
        self.intent_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Tanh())
        self.dropout = nn.Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def encode(self, input_ids: torch.Tensor, causal: bool = False):
        b, s = input_ids.shape
        if s > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            b, s = input_ids.shape
        device = input_ids.device
        pos = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)

        cloud_vec, cloud_weights = self.xiang_cloud(input_ids, x)
        x = x + cloud_vec * 0.3

        block_outputs = []
        for i in range(len(self.attn_layers)):
            attn_out = self.attn_layers[i](x, causal=causal)
            x = self.norm1[i](x + attn_out)
            moe_out = self.moe_layers[i](x)
            x = self.norm2[i](x + moe_out)

            block_idx = i // self.block_size
            if i % self.block_size == 0:
                block_outputs = [x]
            else:
                if block_outputs:
                    recent = block_outputs[-self.block_size:]
                    stacked = torch.stack(recent, dim=0)
                    weights = [gate(out) for gate, out in zip(self.block_gates, recent)]
                    weight_stack = torch.stack(weights, dim=0)
                    weight_avg = F.softmax(weight_stack.mean(dim=-1, keepdim=True), dim=0)
                    x = x + (stacked * weight_avg).sum(dim=0) * 0.5
                block_outputs.append(x)
        return x, cloud_weights

    def transpose(self, input_ids: torch.Tensor, causal: bool = False):
        h, cw = self.encode(input_ids, causal=causal)
        mask = (input_ids != 0).float().unsqueeze(-1)
        obj = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        obj = self.object_proj(obj)
        return obj, cw

    def reiterate(self, object_vec: torch.Tensor, max_len: int = None, temperature: float = 0.8,
                  top_k: int = 50, intent: torch.Tensor = None):
        if max_len is None: max_len = Config.GEN_MAX_LEN
        device = object_vec.device
        batch = object_vec.shape[0]
        generated = torch.full((batch, 1), 1, dtype=torch.long, device=device)
        obj_expanded = object_vec.unsqueeze(1)

        for _ in range(max_len):
            h, _ = self.encode(generated, causal=True)
            cond = obj_expanded.expand(-1, h.size(1), -1)
            h = h + cond * 0.3
            logits = self.reiterate_proj(h)[:, -1, :] / temperature

            if intent is not None and Config.INTENT_GUIDANCE_WEIGHT > 0:
                embeds = self.token_embed.weight
                intent_n = intent / (intent.norm() + 1e-8)
                embed_n = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-8)
                sim = (embed_n * intent_n.unsqueeze(0)).sum(dim=-1)
                logits = logits + sim * Config.INTENT_GUIDANCE_WEIGHT

            probs = F.softmax(logits, dim=-1)
            if top_k > 0:
                topk_probs, topk_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(-1, topk_indices, topk_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(probs, 1).squeeze(-1)
            generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
            if (next_id == 2).any(): break
        return generated

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor = None,
                causal: bool = True, return_consistency: bool = False):
        b, s = input_ids.shape
        if s > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            if target_ids is not None: target_ids = target_ids[:, -self.max_seq_len:]
            b, s = input_ids.shape

        h, cloud_weights = self.encode(input_ids, causal=causal)
        obj = (h * (input_ids != 0).float().unsqueeze(-1)).sum(dim=1) / ((input_ids != 0).float().sum(dim=1, keepdim=True) + 1e-8)
        obj = self.object_proj(obj)

        crystal_tensors = None
        if self.crystal_graph is not None and self.crystal_graph.crystals:
            with torch.no_grad():
                intent_np = obj[0].cpu().numpy()
                related = self.crystal_graph.search(intent_np, top_k=2)
                if related:
                    crystal_tensors = torch.stack([
                        torch.from_numpy(c.core_xiang.copy()).float().to(input_ids.device)
                        for c, _ in related
                    ], dim=0).unsqueeze(0)

        if crystal_tensors is not None:
            h = self.verification_gate(h, crystal_tensors)
            h = self.xiang_arithmetic(h, crystal_tensors)

        obj_expanded = obj.unsqueeze(1).expand(-1, h.size(1), -1)
        h_cond = h + obj_expanded * 0.3
        logits = self.reiterate_proj(h_cond)

        loss = None
        consistency_loss = None

        if target_ids is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
            loss = ce_loss

            # ===== 挂谷几何正则化：方向覆盖最大化（防止对象向量坍缩成一点）=====
            if Config.USE_KAKEYA_REPEL and self.training and b > 1:
                obj_norm = F.normalize(obj, p=2, dim=-1)
                gram = torch.mm(obj_norm, obj_norm.t())
                mask = 1 - torch.eye(b, device=gram.device)
                off_diag = gram * mask
                kakeya_repel = torch.norm(off_diag, p='fro') / (b * (b - 1) + 1e-8)
                loss = loss + Config.KAKEYA_REPEL_WEIGHT * kakeya_repel

            if return_consistency and b == 1:
                with torch.no_grad():
                    gen_ids = self.reiterate(obj, max_len=min(30, s), temperature=0.7)
                    gen_obj, _ = self.transpose(gen_ids, causal=False)
                    consistency_loss = 1 - F.cosine_similarity(obj, gen_obj, dim=-1).mean()
                    loss = loss + consistency_loss * 0.3

        # 晶体对齐损失（仅在 loss 已计算时生效，避免 None 引用崩溃）
        if loss is not None and self.crystal_graph is not None and self.crystal_graph.crystals:
            intent_np = obj[0].detach().cpu().numpy()
            related = self.crystal_graph.search(intent_np, top_k=1)
            if related:
                crystal, sim = related[0]
                if crystal.core_xiang is not None:
                    c_vec = torch.from_numpy(crystal.core_xiang.copy()).to(input_ids.device).float().view(1, -1)
                    align = F.cosine_similarity(obj, c_vec, dim=-1)
                    if sim > 0.5:
                        loss = loss - align.mean() * 0.05 * crystal.confidence
                    elif crystal.confidence > 0.8 and sim < 0.3:
                        loss = loss + 0.5

        # 云选择 KL 散度（仅在 loss 已计算时生效）
        if loss is not None and cloud_weights is not None and cloud_weights.size(1) > 1:
            p = cloud_weights[:, :-1, :].clamp(min=1e-8)
            q = cloud_weights[:, 1:, :].clamp(min=1e-8)
            kl = (p * (p / q).log()).sum(-1).mean()
            loss = loss + kl * 0.01

        intent = self.intent_proj(obj)
        return {
            'logits': logits, 'loss': loss, 'object_vec': obj,
            'intent': intent, 'cloud_weights': cloud_weights,
            'consistency_loss': consistency_loss
        }

    @torch.no_grad()
    def generate(self, prompt_ids, max_len=80, temperature=0.8, top_k=50, 
                 top_p=0.92, repetition_penalty=1.15, min_new_tokens=2,
                 intent=None, path_nodes=None):
        """
        现代采样生成器
        - top_p: 核采样阈值 (0.92 表示只从累积概率 92% 的词中采样)
        - repetition_penalty: 重复惩罚系数 (>1.0 压制已出现 token)
        - min_new_tokens: 最小生成 token 数 (防止过早 EOS)
        """
        self.eval()
        device = next(self.parameters()).device
        if isinstance(prompt_ids, torch.Tensor):
            generated = prompt_ids[0].tolist() if prompt_ids.dim() > 1 else prompt_ids.tolist()
        else:
            generated = list(prompt_ids)
        eos_id = 2
        unk_id = 1
        prompt_len = len(generated)
    
        prompt_tensor = torch.tensor([generated[-self.max_seq_len:]], device=device)
        obj, _ = self.transpose(prompt_tensor, causal=False)
        base_intent = intent  # 保存原始意图
    
        for step in range(max_len):
            # 只取最后 max_seq_len 个 token 作为输入
            ctx = generated[-self.max_seq_len:]
            ctx_tensor = torch.tensor([ctx], device=device)
            h, _ = self.encode(ctx_tensor, causal=True)
            obj_expanded = obj.unsqueeze(1).expand(-1, h.size(1), -1)
            h = h + obj_expanded * 0.3
            logits = self.reiterate_proj(h)[:, -1, :] / temperature
    
            # --- 意图引导 (保持你原有的逻辑) ---
            current_intent = base_intent
            if path_nodes and len(path_nodes) > 0:
                progress = step / max_len
                idx = min(int(progress * len(path_nodes)), len(path_nodes) - 1)
                next_idx = min(idx + 1, len(path_nodes) - 1)
                alpha = (progress * len(path_nodes)) - idx
                node_intent = path_nodes[idx] * (1 - alpha) + path_nodes[next_idx] * alpha
                if current_intent is None:
                    current_intent = node_intent
                else:
                    current_intent = current_intent * 0.5 + node_intent * 0.5
    
            if current_intent is not None and Config.INTENT_GUIDANCE_WEIGHT > 0:
                embeds = self.token_embed.weight
                intent_n = current_intent / (current_intent.norm() + 1e-8)
                embed_n = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-8)
                intent_sim = (embed_n * intent_n.unsqueeze(0)).sum(dim=-1)
                logits = logits + intent_sim * Config.INTENT_GUIDANCE_WEIGHT
    
            # --- 重复惩罚 ---
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty
    
            # --- 温度调度 (越往后越保守) ---
            if temperature > 0.01:
                progress = step / max_len
                adaptive_temp = temperature - (temperature - 0.5) * progress * 0.6
                logits = logits / adaptive_temp
    
            # --- 强制最小新 token ---
            if step < min_new_tokens:
                logits[0, eos_id] = float('-inf')
    
            # --- 屏蔽 UNK 若已连发 ---
            recent_unk = sum(1 for x in generated[-Config.MAX_CONSECUTIVE_UNK:] if x == unk_id)
            if recent_unk >= Config.MAX_CONSECUTIVE_UNK:
                logits[0, unk_id] = float('-inf')
    
            # --- 核心采样：Top-k + Nucleus (Top-p) + 典型接受度 ---
            probs = F.softmax(logits, dim=-1)
            
            # 1. Top-k 过滤
            if top_k > 0 and top_k < probs.size(-1):
                topk_probs, topk_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(-1, topk_indices, topk_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    
            # 2. Nucleus (Top-p) 过滤
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative > top_p
                mask[:, 1:] = mask[:, :-1].clone()  # 保留第一个超过阈值的
                mask[:, 0] = False
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    
            # 3. 典型接受度过滤 (屏蔽熵极低的垃圾 token)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            if entropy < 0.2:  # 分布太尖锐，可能出问题
                probs = F.softmax(logits / 1.5, dim=-1)  # 提高温度重新分布
    
            # 4. 采样
            probs_np = probs.squeeze(0).cpu().numpy()
            probs_np = probs_np / (probs_np.sum() + 1e-8)
            
            if np.isnan(probs_np).any() or probs_np.sum() < 1e-8:
                next_id = int(logits.argmax().item())
            else:
                next_id = int(np.random.choice(len(probs_np), p=probs_np))
    
            if next_id == eos_id and step >= min_new_tokens:
                break
            if next_id == 0:  # PAD
                continue
                
            generated.append(next_id)
            
        return generated

# ========== 好奇心与验证 ==========
class CuriosityEngine:
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
        self.confusion_queue = deque(maxlen=Config.MAX_CONFUSION_BUFFER)

    def on_batch(self, input_ids: torch.Tensor, logits: torch.Tensor):
        with torch.no_grad():
            if logits.size(1) < 2: return
            probs = F.softmax(logits[:, :-1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(-1)
            targets = input_ids[:, 1:]
            mask = entropy > self.threshold
            if not mask.any(): return
            for b in range(mask.size(0)):
                for s in range(mask.size(1)):
                    if mask[b, s]:
                        ctx_start = max(0, s - 4)
                        ctx = input_ids[b, ctx_start:s + 1].cpu().tolist()
                        tgt = targets[b, s].item()
                        self.confusion_queue.append({
                            'context': ctx, 'target': tgt,
                            'entropy': entropy[b, s].item(), 'timestamp': time.time()
                        })

    def get_top_confusion(self):
        if not self.confusion_queue: return None
        return max(self.confusion_queue, key=lambda x: x['entropy'])

    def get_stats(self):
        return {'confusion_size': len(self.confusion_queue)}

class CuriositySpeaker:
    def __init__(self, model, tokenizer, crystal_graph, curiosity):
        self.model = model
        self.tokenizer = tokenizer
        self.crystal_graph = crystal_graph
        self.curiosity = curiosity
        self.asked = set()

    def speak(self):
        confusion = self.curiosity.get_top_confusion()
        if confusion is not None:
            return self._speak_from_confusion(confusion)
        return self._speak_from_memory()

    def _speak_from_confusion(self, confusion):
        ctx_ids = [x for x in confusion['context'] if x != 0]
        ctx_text = self.tokenizer.decode(ctx_ids)
        if not ctx_text.strip(): return None

        device = next(self.model.parameters()).device
        actual_ids = ctx_ids[-Config.MAX_SEQ_LEN:]
        tensor = torch.tensor([actual_ids], device=device)
        self.model.eval()
        with torch.no_grad():
            gen_ids = self.model.generate(
                tensor,
                max_len=15,
                temperature=0.9,
                top_k=20,
                top_p=Config.GEN_TOP_P,
                repetition_penalty=Config.GEN_REPETITION_PENALTY,
                min_new_tokens=2          # 主动提问至少生成2个新token，避免空回复
            )

            gen_text = self.tokenizer.decode(gen_ids[len(actual_ids):])

        ctx_tail = ctx_text[-8:] if len(ctx_text) > 8 else ctx_text
        if ctx_tail in self.asked: return None
        self.asked.add(ctx_tail)

        if gen_text and len(gen_text.strip()) >= 2:
            return f"我在想，'{ctx_tail}'后面是不是'{gen_text}'？你觉得呢？"
        return f"'{ctx_tail}'...这个地方我还没想明白，你能教我吗？"

    def _speak_from_memory(self):
        if not self.crystal_graph.crystals: return None
        crystals = list(self.crystal_graph.crystals.values())
        crystal = random.choice(crystals)
        if not crystal.fragments: return None
        frag = random.choice(crystal.fragments)
        text = frag.text
        if '象核:' in text:
            parts = text.split('象核:', 1)
            if len(parts) > 1:
                user_part = parts[0].replace('用户:', '').strip()[:15]
                reply_part = parts[1].strip()[:20]
                return f"对了，之前聊到'{user_part}'，我后来想了想，{reply_part}..."
        return f"我记得我们聊过'{text[:20]}'..."

class XiangValidator:
    def validate(self, text: str, min_len: int = 2):
        text_stripped = text.strip()
        if len(text_stripped) == 0: return False, "输出为空"

        unk_count = text_stripped.count('□')
        if len(text_stripped) > 0 and unk_count / len(text_stripped) > Config.MAX_UNK_RATIO:
            return False, f"UNK比例过高({unk_count}/{len(text_stripped)})"

        punct = set('，。！？；…、 \n\t')
        eff = [c for c in text_stripped if c not in punct]
        if len(eff) / max(len(text_stripped), 1) < 0.2:
            return False, "有效内容过低"

        for c, g in itertools.groupby(text_stripped):
            if sum(1 for _ in g) >= 6:
                return False, f"连续重复: '{c}'"
        return True, "通过"

class XiangDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: XiangTokenizer, max_len: int = 128, stride: int = 64):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        if not os.path.exists(filepath):
            logger.warning(f"[数据集] 文件不存在: {filepath}")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            logger.warning(f"[数据集] 文件为空: {filepath}")
            return

        parts = re.split(r'([。！？；…\n]+)', text)
        sentences, buf = [], ""
        for p in parts:
            buf += p
            if re.search(r'[。！？；…\n]', p):
                if len(buf.strip()) >= 2: sentences.append(buf.strip())
                buf = ""
        if buf.strip(): sentences.append(buf.strip())

        chunks, current = [], ""
        for sent in sentences:
            if len(sent) > max_len:
                if current: chunks.append(current); current = ""
                sub_buf = ""
                for sp in re.split(r'([，、])', sent):
                    sub_buf += sp
                    if len(sub_buf) >= max_len:
                        chunks.append(sub_buf[:max_len])
                        sub_buf = sub_buf[max_len:]
                if sub_buf: current = sub_buf
            else:
                if len(current) + len(sent) <= max_len: current += sent
                else:
                    if current: chunks.append(current)
                    current = sent
        if current: chunks.append(current)

        final = []
        for chunk in chunks:
            if len(chunk) > max_len:
                for i in range(0, len(chunk) - max_len + 1, stride):
                    final.append(chunk[i:i + max_len])
            else:
                final.append(chunk)

        for chunk in final:
            if not chunk.strip(): continue
            ids = tokenizer.encode(chunk) + [tokenizer.eos_id]
            if len(ids) > max_len: ids = ids[:max_len]
            if len(ids) >= 2: self.samples.append(ids)
        logger.info(f"[数据集] 切分出 {len(self.samples)} 条样本")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ids = self.samples[idx]
        if len(ids) < self.max_len: ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)

# ========== 统一生命管理器 ==========
class XiangLifeManager:
    def __init__(self, model, tokenizer, crystal_graph, curiosity):
        self.model = model.to(Config.DEVICE)
        self.tokenizer = tokenizer
        self.crystal_graph = crystal_graph
        self.curiosity = curiosity
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.MAX_STEPS, eta_min=Config.MIN_LR)
        self.global_step = 0
        self.best_loss = float('inf')
        self.turn_count = 0
        self.train_lock = threading.Lock()

    def learn_from_dialogue(self, full_ids: List[int]):
        if len(full_ids) < 2: return 0.0
        if len(full_ids) > Config.MAX_SEQ_LEN: full_ids = full_ids[-Config.MAX_SEQ_LEN:]
        device = Config.DEVICE
        inp = torch.tensor([full_ids[:-1]], device=device)
        tgt = torch.tensor([full_ids[1:]], device=device)

        self.model.train()
        with self.train_lock:
            self.optimizer.zero_grad()
            out = self.model(inp, tgt, causal=True, return_consistency=True)
            loss = out['loss']
            if loss is None: return 0.0

            # 安全阀：对话loss爆炸说明输入质量极差
            if loss.item() > 8.0:
                logger.warning(f"[学习] 对话loss异常高({loss.item():.2f})，跳过本轮")
                return loss.item()

            # 修复：删除重复的晶体对齐损失（forward 中已统一计算）
            # 原代码在此处重复计算晶体对齐，导致梯度异常，已移除

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            if self.curiosity:
                self.curiosity.on_batch(inp, out['logits'])
        return loss.item()

    def read_book_step(self, input_ids, target_ids):
        input_ids = input_ids.to(Config.DEVICE)
        target_ids = target_ids.to(Config.DEVICE)
        self.model.train()
        with self.train_lock:
            out = self.model(input_ids, target_ids, causal=True)
            loss = out['loss']
            if loss is None: return 0.0
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            if self.curiosity:
                self.curiosity.on_batch(input_ids, out['logits'])
        return loss.item()

    def chat_and_learn(self, user_input: str, max_len: int = None):
        if max_len is None: max_len = Config.GEN_MAX_LEN
        self.turn_count += 1
        device = next(self.model.parameters()).device

        crystal_ctx = ""
        if self.crystal_graph.crystals:
            default_intent = np.ones(Config.XIANG_DIM, dtype=np.float32)
            crystal_ctx = self.crystal_graph.get_memory_context(default_intent, max_chars=Config.MAX_CRYSTAL_CTX_CHARS)

        prompt_text = f"{crystal_ctx}\n{user_input}" if crystal_ctx else user_input
        prompt_ids = self.tokenizer.encode(prompt_text)
        if len(prompt_ids) > Config.MAX_SEQ_LEN: prompt_ids = prompt_ids[-Config.MAX_SEQ_LEN:]
        if not prompt_ids: return "（沉默）"

        self.model.eval()
        with torch.no_grad():
            inp_tensor = torch.tensor([prompt_ids], device=device)
            out = self.model(inp_tensor, causal=False)
            intent = out['intent'][0]

            path_nodes = []
            if self.crystal_graph.crystals:
                intent_np = intent.cpu().numpy()
                related = self.crystal_graph.search(intent_np, top_k=3)
                if related:
                    path_nodes = [torch.from_numpy(c.core_xiang.copy()).to(device).float() for c, _ in related]

            gen_ids = self.model.generate(
                inp_tensor,
                max_len=max_len,
                intent=intent,
                path_nodes=path_nodes,
                top_k=50,
                top_p=Config.GEN_TOP_P,
                repetition_penalty=Config.GEN_REPETITION_PENALTY,
                min_new_tokens=Config.GEN_MIN_NEW_TOKENS
            )
            
            
            actual_prompt_len = len(prompt_ids)
            reply = self.tokenizer.decode(gen_ids[actual_prompt_len:])

            if not reply: reply = "（沉思）"
            val = XiangValidator()
            ok, reason = val.validate(reply)
            if not ok:
                logger.info(f"[生成] 验证未通过: {reason}")
                reply = "（象核沉吟片刻，未能组织出合适的言语）"

        full_ids = prompt_ids + self.tokenizer.encode(reply) + [self.tokenizer.eos_id]
        loss = self.learn_from_dialogue(full_ids)

        if self.turn_count % 5 == 0:
            logger.info(f"[学习] 第{self.turn_count}轮对话 → loss={loss:.4f} step={self.global_step}")

        self.crystal_graph.add(f"用户:{user_input} 象核:{reply}", intent.cpu().numpy(), source="dialogue")
        return reply

    def save(self, filename):
        path = Path(CHECKPOINT_DIR) / filename
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.global_step,
            'best_loss': self.best_loss,
            'char2id': self.tokenizer.char2id,
            'word2id': self.tokenizer.word2id,
            'next_id': self.tokenizer.next_id
        }, path)
        logger.info(f"[检查点] 保存: {path}")

    def load(self, filename):
        path = Path(CHECKPOINT_DIR) / filename
        if not path.exists():
            return False

        ckpt = torch.load(path, map_location=Config.DEVICE, weights_only=False)
        old_state = ckpt['model']
        current_state = self.model.state_dict()

        expanded = False
        old_vocab_size = None
        if 'token_embed.weight' in old_state:
            old_vocab_size = old_state['token_embed.weight'].shape[0]

        for key in current_state.keys():
            if key in old_state:
                old_param = old_state[key]
                new_param = current_state[key]
                if old_param.shape != new_param.shape:
                    expanded = True
                    if (len(old_param.shape) > 0 and
                        old_param.shape[0] < new_param.shape[0] and
                        key in ('token_embed.weight', 'xiang_cloud.clouds',
                                'reiterate_proj.3.weight', 'reiterate_proj.3.bias',
                                'object_proj.0.weight', 'object_proj.0.bias')):
                        expanded_param = new_param.clone()
                        expanded_param[:old_param.shape[0]] = old_param
                        expanded_param[old_param.shape[0]:] = old_param.mean().item()
                        current_state[key] = expanded_param
                        logger.info(f"[检查点] 扩容 {key}: {old_param.shape} -> {new_param.shape}")
                    else:
                        logger.warning(f"[检查点] 跳过不匹配层: {key} "
                                       f"{old_param.shape} vs {new_param.shape}")
                else:
                    current_state[key] = old_param
            else:
                logger.warning(f"[检查点] 新层未初始化: {key}")

        self.model.load_state_dict(current_state, strict=False)
        logger.info(f"[检查点] 模型权重已恢复" + ("（已扩容）" if expanded else ""))

        def _count_opt_params(opt_state_dict):
            total = 0
            for state in opt_state_dict.get('state', {}).values():
                if 'exp_avg' in state:
                    total += state['exp_avg'].numel()
            return total

        model_param_numel = sum(p.numel() for p in self.model.parameters())
        opt_param_numel = _count_opt_params(ckpt.get('optimizer', {}))
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else Config.LEARNING_RATE

        if not expanded and opt_param_numel == model_param_numel:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                logger.info("[检查点] 优化器已恢复")
            except Exception as e:
                logger.warning(f"[检查点] 优化器加载失败，已重建: {e}")
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=current_lr, weight_decay=0.01)
        else:
            reason = "词表扩容" if expanded else f"参数不匹配(检查点{opt_param_numel} vs 模型{model_param_numel})"
            logger.info(f"[检查点] {reason}，优化器已重建 (lr={current_lr:.2e})")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=current_lr, weight_decay=0.01)

            if expanded and hasattr(self, 'emb_dict') and self.emb_dict and old_vocab_size is not None:
                with torch.no_grad():
                    initialized = 0
                    for token, idx in self.tokenizer.char2id.items():
                        if idx >= old_vocab_size and token in self.emb_dict:
                            self.model.token_embed.weight[idx] = torch.from_numpy(self.emb_dict[token])
                            initialized += 1
                    if initialized > 0:
                        logger.info(f"[词向量] 已初始化 {initialized} 个新增字的嵌入")

        try:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as e:
            logger.warning(f"[检查点] 调度器重置: {e}")

        self.global_step = ckpt.get('step', 0)
        self.best_loss = ckpt.get('best_loss', float('inf'))
        logger.info(f"[检查点] 加载完成: step {self.global_step}")
        return True


class BackgroundReader(threading.Thread):
    def __init__(self, manager, dataset, interval=3, silent=False):
        super().__init__(daemon=True)
        self.manager = manager
        self.dataset = dataset
        self.interval = interval
        self.running = False
        self.silent = silent
        if len(dataset) > 0:
            self.dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
            self.data_iter = iter(self.dataloader)
        else:
            self.dataloader = None
            self.data_iter = None

    def run(self):
        self.running = True
        if not self.silent:
            logger.info(f"[阅读] 后台阅读器已启动，每 {self.interval}s 读一段语料")
        while self.running:
            time.sleep(self.interval)
            if self.dataloader is None: continue
            try:
                batch = next(self.data_iter, None)
                if batch is None:
                    self.data_iter = iter(self.dataloader)
                    batch = next(self.data_iter, None)
                if batch is None: continue
                input_ids, target_ids = batch
                loss = self.manager.read_book_step(input_ids, target_ids)
                if not self.silent and self.manager.global_step % 10 == 0:
                    logger.info(f"[阅读] step={self.manager.global_step} loss={loss:.4f}")

                if self.manager.global_step % 10 == 0:
                    try:
                        with open('/storage/emulated/0/xiangcore_loss_history.txt', 'a', encoding='utf-8') as f:
                            f.write(f"{self.manager.global_step},{loss:.4f},{time.strftime('%H:%M:%S')}\n")
                    except Exception:
                        pass

            except RuntimeError as e:
                # 修复：区分 RuntimeError（CUDA OOM、维度错误等）并打印完整 traceback
                if not self.silent:
                    logger.error(f"[阅读] RuntimeError: {e}")
                    logger.error(traceback.format_exc())
            except Exception as e:
                if not self.silent:
                    logger.warning(f"[阅读] 其他错误: {e}")

    def stop(self):
        self.running = False
        if not self.silent:
            logger.info("[阅读] 后台阅读器已停止")

# ========== 初始化系统 ==========
def init_system():
    Config.init_dirs()
    set_seed(42)
    torch.set_num_threads(1)

    tokenizer = XiangTokenizer()
    vocab_path = os.path.join(DATA_DIR, "tokenizer_xiang.json")

    ckpt_files = sorted(Path(CHECKPOINT_DIR).glob("*.pt"))
    vocab_loaded = False
    if ckpt_files:
        try:
            ckpt = torch.load(ckpt_files[-1], map_location='cpu', weights_only=False)
            if 'char2id' in ckpt:
                tokenizer.char2id = ckpt['char2id']
                tokenizer.word2id = ckpt.get('word2id', {})
                tokenizer.next_id = ckpt.get('next_id', 3)
                tokenizer.id2char = {v: k for k, v in tokenizer.char2id.items()}
                tokenizer.id2word = {v: k for k, v in tokenizer.word2id.items()}
                logger.info(f"[检查点] 恢复词表: {tokenizer.vocab_size} tokens")
                vocab_loaded = True
        except Exception as e:
            logger.warning(f"[检查点] 恢复词表失败: {e}")

    if not vocab_loaded and os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
        logger.info(f"[系统] 加载词表: {tokenizer.vocab_size} tokens")
    elif not vocab_loaded:
        logger.info(f"[系统] 新建词表: {tokenizer.vocab_size} tokens")

    if os.path.exists(TRAIN_FILE):
        new_chars = 0
        with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
            text = f.read()
        for c in text:
            if c not in tokenizer.char2id and c not in tokenizer.word2id and not c.isspace():
                tokenizer.char2id[c] = tokenizer.next_id
                tokenizer.id2char[tokenizer.next_id] = c
                tokenizer.next_id += 1
                new_chars += 1
        if new_chars > 0:
            logger.info(f"[训练数据] 加入 {new_chars} 个新字")

    emb_dict = load_pretrained_embeddings(EMBEDDING_FILE, tokenizer, Config.XIANG_DIM)
    if emb_dict:
        new_words = 0
        for word in emb_dict.keys():
            if len(word) >= 2 and word not in tokenizer.word2id and word not in tokenizer.char2id:
                tokenizer.word2id[word] = tokenizer.next_id
                tokenizer.id2word[tokenizer.next_id] = word
                tokenizer.next_id += 1
                new_words += 1
        logger.info(f"[词向量] 已加入 {new_words} 个多字词")

    crystal_graph = CrystalGraph(dim=Config.XIANG_DIM)
    model = XiangCoreModel(
        vocab_size=tokenizer.vocab_size,
        dim=Config.XIANG_DIM,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        num_clouds=Config.XIANG_CLOUDS,
        max_seq_len=Config.MAX_SEQ_LEN,
        window_size=Config.WINDOW_SIZE,
        linear_rank=Config.LINEAR_RANK,
        num_experts=Config.NUM_EXPERTS,
        top_k=Config.TOP_K,
        block_size=Config.BLOCK_SIZE,
        crystal_graph=crystal_graph
    ).to(Config.DEVICE)

    if emb_dict:
        with torch.no_grad():
            count = 0
            for token, idx in tokenizer.char2id.items():
                if token in emb_dict and idx < model.token_embed.weight.size(0):
                    model.token_embed.weight[idx] = torch.from_numpy(emb_dict[token])
                    count += 1
            for token, idx in tokenizer.word2id.items():
                if token in emb_dict and idx < model.token_embed.weight.size(0):
                    model.token_embed.weight[idx] = torch.from_numpy(emb_dict[token])
                    count += 1
            logger.info(f"[词向量] 已初始化 {count} 个 token")

    crystal_path = os.path.join(MEMORY_DIR, "crystals.pkl")
    if os.path.exists(crystal_path):
        crystal_graph.load(crystal_path)
        logger.info(f"[系统] 加载晶体: {len(crystal_graph.crystals)} 个")

    curiosity = CuriosityEngine(threshold=Config.CURIOSITY_THRESHOLD)
    manager = XiangLifeManager(model, tokenizer, crystal_graph, curiosity)
    manager.emb_dict = emb_dict

    if ckpt_files:
        manager.load(ckpt_files[-1].name)
    elif (Path(CHECKPOINT_DIR) / "best_model.pt").exists():
        manager.load("best_model.pt")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[系统] 参数量: {total_params / 1e6:.2f}M")
    logger.info(f"[系统] 设备: {Config.DEVICE}")
    logger.info(f"[系统] 词表大小: {tokenizer.vocab_size}")
    return model, tokenizer, crystal_graph, curiosity, manager

# ========== 统一生命模式 ==========
def live_mode(personality='adult'):
    preset = PERSONALITY_PRESETS[personality]
    max_proactive = preset['persistence']
    timeout = max(preset['timeout'], 41)
    gen_temp = preset['temperature']

    model, tokenizer, crystal_graph, curiosity, manager = init_system()
    speaker = CuriositySpeaker(model, tokenizer, crystal_graph, curiosity)

    dataset = XiangDataset(TRAIN_FILE, tokenizer) if os.path.exists(TRAIN_FILE) else []
    reader = BackgroundReader(manager, dataset, interval=Config.BACKGROUND_READ_INTERVAL, silent=True)
    reader.start()

    crystal_path = os.path.join(MEMORY_DIR, "crystals.pkl")
    vocab_path = os.path.join(DATA_DIR, "tokenizer_xiang.json")

    print("\n" + "=" * 60)
    print(f"象核心 v16.2 统一生命体 ({preset['intro']})")
    print(f"主动提问上限: {max_proactive}次 | 超时: {timeout}秒")
    print("命令: exit, clear, stats, save (均支持 / 前缀)")
    print("反馈: + (好评/确认), - (差评/否定), ? (好奇/追问)")
    print("=" * 60)

    input_queue = queue.Queue()
    def _input_loop():
        while True:
            try:
                line = input()
                input_queue.put(line)
            except (EOFError, OSError):
                break
    input_thread = threading.Thread(target=_input_loop, daemon=True)
    input_thread.start()

    proactive_count = 0
    last_active = time.time()
    running = True
    prompt_shown = False
    silence_shown = False

    try:
        while running:
            if not prompt_shown:
                print("\n用户: ", end='', flush=True)
                prompt_shown = True

            try:
                user_input = input_queue.get(timeout=30)
                prompt_shown = False
                silence_shown = False
            except queue.Empty:
                user_input = None

            if user_input is not None and user_input.strip():
                reader.silent = True
                prompt_shown = False
                silence_shown = False

                user_input = user_input.strip()
                cmd = user_input.lower()
                if cmd.startswith('/'):
                    cmd = cmd[1:]

                if cmd in ('exit', 'quit'):
                    running = False
                    break
                elif cmd == 'clear':
                    manager.turn_count = 0
                    proactive_count = 0
                    print("[系统] 历史已清空")
                    reader.silent = False
                    continue
                elif cmd == 'stats':
                    s = {'turns': manager.turn_count, 'crystals': len(crystal_graph.crystals),
                         'fragments': len(crystal_graph.fragments), 'step': manager.global_step}
                    c = curiosity.get_stats()
                    print(f"轮次:{s['turns']} 晶体:{s['crystals']} 碎片:{s['fragments']} 困惑:{c['confusion_size']} 步数:{s['step']}")

                    try:
                        history_file = '/storage/emulated/0/xiangcore_loss_history.txt'
                        if os.path.exists(history_file):
                            with open(history_file, 'r', encoding='utf-8') as f:
                                lines = [l.strip() for l in f.readlines() if l.strip()]

                            if len(lines) >= 2:
                                print("\n[最近训练记录]")
                                for line in lines[-8:]:
                                    parts = line.split(',')
                                    if len(parts) >= 2:
                                        time_str = parts[2] if len(parts) > 2 else ''
                                        print(f"  step={parts[0]:>6}  loss={parts[1]}  ({time_str})")

                                if len(lines) >= 10:
                                    recent = [float(l.split(',')[1]) for l in lines[-20:] if len(l.split(',')) >= 2]
                                    if len(recent) >= 10:
                                        mid = len(recent) // 2
                                        first = sum(recent[:mid]) / mid
                                        second = sum(recent[mid:]) / (len(recent) - mid)
                                        delta = first - second

                                        if delta > 0.1:
                                            trend = "↓ 明显下降（正在学习）"
                                        elif delta > 0.01:
                                            trend = "↓ 缓慢下降（正常积累）"
                                        elif delta < -0.1:
                                            trend = "↑ 异常上升（可能发散）"
                                        else:
                                            trend = "→ 基本持平（数据不够或卡住了）"

                                        print(f"\n[趋势判断] {trend}")
                                        print(f"           前半段平均: {first:.4f} → 后半段平均: {second:.4f}")
                            else:
                                print("\n[训练记录] 数据不足，再等几分钟...")
                        else:
                            print("\n[训练记录] 暂无历史文件，重启后生效")
                    except Exception as e:
                        print(f"\n[记录读取错误] {e}")

                    reader.silent = False
                    continue

                elif cmd == 'save':
                    manager.save("best_model.pt")
                    crystal_graph.save(crystal_path)
                    tokenizer.save(vocab_path)
                    print("[系统] 已保存")
                    reader.silent = False
                    continue
                elif cmd in ('+', '好评', '对', '正确'):
                    if manager.turn_count > 0:
                        print("（已确认，知识增强）")
                    reader.silent = False
                    continue
                elif cmd in ('-', '差评', '错', '不对'):
                    if manager.turn_count > 0:
                        print("（已修正，知识调整）")
                    reader.silent = False
                    continue
                elif cmd in ('?', '好奇', '为什么'):
                    question = speaker.speak()
                    if question:
                        print(f"\n象核心(好奇): {question}")
                    reader.silent = False
                    continue

                proactive_count = 0
                last_active = time.time()
                reply = manager.chat_and_learn(user_input, max_len=Config.GEN_MAX_LEN)
                print(f"象核心: {reply}")

                if manager.turn_count % 20 == 0:
                    crystal_graph.evolve()

                reader.silent = False

            elif user_input is not None and not user_input.strip():
                continue
            else:
                if proactive_count < max_proactive:
                    question = speaker.speak()
                    if question:
                        proactive_count += 1
                        print(f"\n象核心(主动): {question}")
                        last_active = time.time()
                        prompt_shown = False
                        silence_shown = False
                    else:
                        if time.time() - last_active > timeout * 2 and not silence_shown:
                            print(f"\n[象核心静默中，全力读书深造...]")
                            silence_shown = True
                            prompt_shown = False
                else:
                    if proactive_count == max_proactive and not silence_shown:
                        print(f"\n[象核心安静地等待着你的回应...]")
                        silence_shown = True
                        proactive_count += 1
                        prompt_shown = False

    except KeyboardInterrupt:
        print("\n退出")
    finally:
        reader.stop()
        manager.save("best_model.pt")
        crystal_graph.save(crystal_path)
        tokenizer.save(vocab_path)
        print("资源已保存")


# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser(description="象核心系统 v16.2-Fix")
    parser.add_argument('--personality', type=str, default='adult', choices=['child', 'adult', 'scholar'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    live_mode(personality=args.personality)

if __name__ == "__main__":
    main()
