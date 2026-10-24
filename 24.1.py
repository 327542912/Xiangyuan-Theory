# -*- coding: utf-8 -*-
# =========================================================
# XiangCore v25 —— 古文 + 长句一次性解决版
#
# 相比 v24.1 的核心升级：
#   [长句] 6 层 dilated depthwise conv 堆叠（感受野 ±63），替换 kernel=9 单层
#   [古文] 20 个文言虚词独立 GRAM 标签（之/乎/者/也/矣/焉/哉/而/以/于/为/则/乃/其/所/故/虽/若/何/夫）
#   [古文] 文言自动检测 + 文言逐字分词模式
#   [通用] lang 通道（现代/文言双标记）
#   [通用] seq_max 128 → 256
#   [保留] 无 Attention / 完整象不压缩 / 因果残差 / 晶体检索 / 不确定性加权
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import json
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "dim": 256,
    "seq_max": 256,           # [长句] 128 → 256
    "hier_depth": 3,
    "codebook_size": 128,
    "batch_size": 4,
    "lr": 1e-4,
    "block_align": 16,
    "crystal_cap": 5000,
    "max_steps": 2000,
    "vq_beta_start": 0.05,
    "vq_beta_warmup": 500,
    "vq_beta_end": 0.25,
    "ms_n_layers": 6,         # [长句] 多尺度 conv 层数（dilation=1,2,4,8,16,32）
    "save_path": "/storage/emulated/0/大模型/v25_ckpt.pt",
    "train_json": "/storage/emulated/0/大模型/v25_train.json",
    "holdout_json": "/storage/emulated/0/大模型/v25_holdout.json",
}

# =========================================================
# 标签体系
# =========================================================
POS_DEF = {
    "PAD": 0, "n": 1, "v": 2, "a": 3, "r": 4, "u": 5, "p": 6,
    "d": 7, "m": 8, "q": 9, "c": 10, "x": 11,
}
NUM_POS = len(POS_DEF)

# [古文] 20 个文言虚词独立标签（不再全部归为 NONE）
GRAM_DEF = {
    "NONE": 0,
    # 现代汉语（15 类）
    "ASP_PERF": 1, "ASP_EXP": 2, "ASP_DUR": 3, "ASP_PROG": 4,
    "VOICE_PASS": 5, "VOICE_BA": 6, "VOICE_CMP": 7,
    "STRUCT_DE": 8, "MOOD_Q": 9, "MOOD_E": 10,
    "NEG": 11, "MOD_CAN": 12, "MOD_WILL": 13, "MOD_MUST": 14, "COPULA": 15,
    # [古文] 文言虚词（20 类）
    "CL_ZHI": 16,   # 之：定语 / 代词
    "CL_HU": 17,    # 乎：疑问 / 介词
    "CL_ZHE": 18,   # 者：代词 / 判断
    "CL_YE": 19,    # 也：判断语气
    "CL_YI": 20,    # 矣：完成语气
    "CL_YAN": 21,   # 焉：疑问 / 兼词
    "CL_ZAI": 22,   # 哉：感叹
    "CL_ER": 23,    # 而：连词
    "CL_YI2": 24,   # 以：介词 / 连词
    "CL_YU": 25,    # 于：介词
    "CL_WEI": 26,   # 为：介词 / 动词
    "CL_ZE": 27,    # 则：连词
    "CL_NAI": 28,   # 乃：副词
    "CL_QI": 29,    # 其：代词
    "CL_SUO": 30,   # 所：助词
    "CL_GU": 31,    # 故：连词（因此）
    "CL_SUI": 32,   # 虽：连词（虽然）
    "CL_RUO": 33,   # 若：连词（如果）
    "CL_HE": 34,    # 何：疑问代词
    "CL_FU": 35,    # 夫：发语词
}
NUM_GRAM = len(GRAM_DEF)

ROLE_DEF = {
    "NONE": 0, "SUBJ": 1, "PRED": 2, "OBJ": 3,
    "ATTR": 4, "ADV": 5, "COMP": 6,
    "TOPIC": 7, "ASPECT": 8, "FUNC": 9,
}
NUM_ROLES = len(ROLE_DEF)

# [通用] lang 通道
NUM_LANG = 2  # 0=现代, 1=文言


# =========================================================
# 现代汉语语法识别
# =========================================================
GRAM_RULES = {
    "了": "ASP_PERF", "过": "ASP_EXP", "着": "ASP_DUR",
    "被": "VOICE_PASS", "把": "VOICE_BA", "比": "VOICE_CMP",
    "的": "STRUCT_DE", "地": "STRUCT_DE", "得": "STRUCT_DE",
    "吗": "MOOD_Q", "呢": "MOOD_Q", "吧": "MOOD_E", "啊": "MOOD_E",
    "不": "NEG", "没": "NEG", "别": "NEG",
    "是": "COPULA",
    "能": "MOD_CAN", "会": "MOD_CAN",
    "要": "MOD_WILL", "将": "MOD_WILL",
}
GRAM_RULES_MULTI = {
    "正在": "ASP_PROG", "可以": "MOD_CAN",
    "必须": "MOD_MUST", "应该": "MOD_MUST",
    "已经": "ASP_PERF", "曾经": "ASP_EXP",
}

# [古文] 文言虚词表
CLASSICAL_GRAM_RULES = {
    "之": "CL_ZHI", "乎": "CL_HU", "者": "CL_ZHE", "也": "CL_YE",
    "矣": "CL_YI", "焉": "CL_YAN", "哉": "CL_ZAI", "耳": "CL_YE",
    "兮": "CL_YE", "而": "CL_ER", "以": "CL_YI2", "于": "CL_YU",
    "於": "CL_YU", "为": "CL_WEI", "則": "CL_ZE", "则": "CL_ZE",
    "乃": "CL_NAI", "其": "CL_QI", "所": "CL_SUO", "故": "CL_GU",
    "雖": "CL_SUI", "虽": "CL_SUI", "若": "CL_RUO", "何": "CL_HE",
    "夫": "CL_FU", "蓋": "CL_FU", "盖": "CL_FU", "遂": "CL_GU",
    "耶": "CL_HU", "欤": "CL_HU", "歟": "CL_HU",
}

# [古文] 文言虚词 → 词性
CLASSICAL_FUNC_POS = {
    "之": "u", "乎": "u", "者": "u", "也": "u", "矣": "u", "焉": "u",
    "哉": "u", "耳": "u", "兮": "u", "而": "c", "以": "p", "于": "p",
    "於": "p", "为": "v", "则": "c", "則": "c", "乃": "d", "其": "r",
    "所": "u", "故": "c", "虽": "c", "雖": "c", "若": "c", "何": "r",
    "夫": "u", "蓋": "u", "盖": "u", "遂": "d", "耶": "u", "欤": "u", "歟": "u",
}

# [通用] 文言检测集合
_CLASSICAL_SET = set(CLASSICAL_GRAM_RULES.keys())
_MODERN_SET = set(GRAM_RULES.keys())


def is_classical(text):
    """[古文] 检测是否为文言（有文言虚词且现代虚词少）"""
    if len(text) < 4:
        return False
    n_cl = sum(1 for c in text if c in _CLASSICAL_SET)
    n_mod = sum(1 for c in text if c in _MODERN_SET)
    if n_cl == 0:
        return False
    return n_cl > n_mod and n_cl / len(text) > 0.1


def detect_grammar_tag(text, char_pos, prev_char, next_char):
    """[通用] 从字符上下文识别语法功能标签（现代 + 文言混合）"""
    # 多字优先
    for pat, tag in GRAM_RULES_MULTI.items():
        if text[char_pos:char_pos + len(pat)] == pat:
            return GRAM_DEF[tag]
        for i in range(1, len(pat)):
            if char_pos - i >= 0 and text[char_pos - i:char_pos - i + len(pat)] == pat:
                return GRAM_DEF[tag]
    c = text[char_pos]
    # 文言优先（如果字符在文言表里）
    if c in CLASSICAL_GRAM_RULES:
        return GRAM_DEF[CLASSICAL_GRAM_RULES[c]]
    # 现代
    if c in GRAM_RULES:
        return GRAM_DEF[GRAM_RULES[c]]
    return GRAM_DEF["NONE"]


def pos_to_cat(flag):
    if flag.startswith("n"): return POS_DEF["n"]
    if flag.startswith("v"): return POS_DEF["v"]
    if flag.startswith("a"): return POS_DEF["a"]
    if flag.startswith("r"): return POS_DEF["r"]
    if flag.startswith("u"): return POS_DEF["u"]
    if flag.startswith("p"): return POS_DEF["p"]
    if flag.startswith("d"): return POS_DEF["d"]
    if flag.startswith("m"): return POS_DEF["m"]
    if flag.startswith("q"): return POS_DEF["q"]
    if flag.startswith("c"): return POS_DEF["c"]
    return POS_DEF["x"]


def build_roles_from_grammar(pos_ids, gram_ids):
    n = len(pos_ids)
    roles = [ROLE_DEF["NONE"]] * n
    if n == 0:
        return roles

    gram_to_role = {
        "ASP_PERF": "ASPECT", "ASP_EXP": "ASPECT",
        "ASP_DUR": "ASPECT", "ASP_PROG": "ASPECT",
        "VOICE_PASS": "FUNC", "VOICE_BA": "FUNC", "VOICE_CMP": "FUNC",
        "STRUCT_DE": "FUNC",
        "MOOD_Q": "FUNC", "MOOD_E": "FUNC",
        "NEG": "ADV",
        "MOD_CAN": "ADV", "MOD_WILL": "ADV", "MOD_MUST": "ADV",
        "COPULA": "PRED",
        # [古文]
        "CL_ZHI": "FUNC", "CL_HU": "FUNC", "CL_ZHE": "FUNC", "CL_YE": "FUNC",
        "CL_YI": "ASPECT", "CL_YAN": "FUNC", "CL_ZAI": "FUNC", "CL_ER": "FUNC",
        "CL_YI2": "FUNC", "CL_YU": "FUNC", "CL_ZE": "FUNC", "CL_NAI": "ADV",
        "CL_SUO": "FUNC", "CL_GU": "FUNC", "CL_SUI": "FUNC", "CL_RUO": "FUNC",
        "CL_FU": "FUNC",
    }
    gram_rev = {v: k for k, v in GRAM_DEF.items()}
    for i, g in enumerate(gram_ids):
        gname = gram_rev.get(g, "NONE")
        if gname in gram_to_role:
            roles[i] = ROLE_DEF[gram_to_role[gname]]

    pred_idx = -1
    for i, (p, g) in enumerate(zip(pos_ids, gram_ids)):
        if roles[i] != ROLE_DEF["NONE"]:
            continue
        if p == POS_DEF["v"] or gram_ids[i] == GRAM_DEF["COPULA"]:
            roles[i] = ROLE_DEF["PRED"]
            if pred_idx == -1:
                pred_idx = i

    if pred_idx >= 0:
        first_noun = -1
        for i in range(pred_idx):
            if pos_ids[i] == POS_DEF["n"] and roles[i] == ROLE_DEF["NONE"]:
                if first_noun == -1:
                    first_noun = i
                    roles[i] = ROLE_DEF["SUBJ"]
                else:
                    roles[i] = ROLE_DEF["TOPIC"]
        for i in range(pred_idx + 1, n):
            if pos_ids[i] == POS_DEF["n"] and roles[i] == ROLE_DEF["NONE"]:
                roles[i] = ROLE_DEF["OBJ"]
                break

    for i in range(n):
        if roles[i] != ROLE_DEF["NONE"]:
            continue
        if pos_ids[i] == POS_DEF["a"]:
            roles[i] = ROLE_DEF["ATTR"]
        elif pos_ids[i] == POS_DEF["d"]:
            roles[i] = ROLE_DEF["ADV"]
    return roles


# =========================================================
# 词表 + 数据转换（含文言逐字模式）
# =========================================================
class CharVocab:
    def __init__(self, all_texts):
        chars = sorted(set("".join(all_texts)))
        self.pad, self.unk, self.bos, self.eos = 0, 1, 2, 3
        self.char2id = {c: i + 4 for i, c in enumerate(chars)}
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = len(self.char2id) + 4

    def encode(self, chars):
        return [self.char2id.get(c, self.unk) for c in chars]

    def decode(self, ids):
        return "".join(
            self.id2char.get(i, "") for i in ids
            if i not in (self.pad, self.bos, self.eos)
        )


def _modern_tags(text):
    """[现代] jieba 分词 + 现代语法识别"""
    import jieba.posseg
    pos_list = []
    for word, flag in jieba.posseg.cut(text):
        pos_list.extend([pos_to_cat(flag)] * len(word))
    if len(pos_list) != len(text):
        pos_list = [POS_DEF["x"]] * len(text)

    gram_list = []
    for i, c in enumerate(text):
        prev_c = text[i - 1] if i > 0 else ""
        next_c = text[i + 1] if i + 1 < len(text) else ""
        gram_list.append(detect_grammar_tag(text, i, prev_c, next_c))

    ids = list(text)
    roles = build_roles_from_grammar(pos_list, gram_list)
    if len(roles) != len(text):
        roles = [ROLE_DEF["NONE"]] * len(text)
    return ids, pos_list, gram_list, roles


def _classical_tags(text):
    """[古文] 文言逐字模式：每个字单独分词 + 位置规则推断词性"""
    ids = list(text)
    pos_list = []
    gram_list = []
    for i, c in enumerate(text):
        # gram：查文言表 + 现代表
        prev_c = text[i - 1] if i > 0 else ""
        next_c = text[i + 1] if i + 1 < len(text) else ""
        g = detect_grammar_tag(text, i, prev_c, next_c)
        gram_list.append(g)

        # pos：文言虚词查表，实词按位置推断
        if c in CLASSICAL_FUNC_POS:
            pos_list.append(POS_DEF[CLASSICAL_FUNC_POS[c]])
        elif c in GRAM_RULES:
            pos_list.append(POS_DEF["u"])
        else:
            # 实词：默认按位置推断
            # 句首/前一个虚词是"之/者/所" → 名词；否则动词
            if i == 0:
                pos_list.append(POS_DEF["n"])
            elif pos_list[i - 1] in (POS_DEF["u"], POS_DEF["p"], POS_DEF["c"]):
                pos_list.append(POS_DEF["n"])
            elif i == len(text) - 1 and text[i] in "矣也乎哉焉":
                pos_list.append(POS_DEF["u"])
            else:
                pos_list.append(POS_DEF["n"])  # 保守默认名词

    roles = build_roles_from_grammar(pos_list, gram_list)
    if len(roles) != len(text):
        roles = [ROLE_DEF["NONE"]] * len(text)
    return ids, pos_list, gram_list, roles


def text_to_three_tags(text):
    """[通用] 主入口：自动判断现代/文言并返回四元组"""
    if is_classical(text):
        return _classical_tags(text)
    else:
        try:
            return _modern_tags(text)
        except Exception:
            return _classical_tags(text)  # 兜底


# =========================================================
# 多尺度 depthwise conv（[长句] 核心改动）
# =========================================================
class MultiScaleConv(nn.Module):
    """[长句] 6 层 dilated depthwise conv 堆叠，感受野 ±63
    dilation = 1, 2, 4, 8, 16, 32 → 总感受野 63
    纯卷积，无 QKV，无位置加权，不是 Attention
    """
    def __init__(self, dim, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=dilation,
                          dilation=dilation, groups=dim),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.LayerNorm(dim),
                nn.GELU(),
            ))

    def forward(self, x):
        # x: [B, L, D]
        for layer in self.layers:
            h = layer(x.transpose(1, 2)).transpose(1, 2)
            x = x + h  # 残差
        return x


# =========================================================
# VQ
# =========================================================
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, dim):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.register_buffer("usage", torch.zeros(codebook_size))

    def forward(self, x, beta):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        cb = self.embedding.weight
        d = (x_flat.pow(2).sum(1, keepdim=True)
             - 2 * x_flat @ cb.t()
             + cb.pow(2).sum(1).unsqueeze(0))
        idx = d.argmin(-1)
        quantized = cb[idx].reshape(B, L, D)

        if self.training:
            with torch.no_grad():
                self.usage += torch.bincount(idx, minlength=self.codebook_size).float()

        commit_loss = beta * F.mse_loss(x, quantized.detach())
        q_st = x + (quantized - x).detach()
        return q_st, idx.reshape(B, L), commit_loss

    @torch.no_grad()
    def restart_dead_codes(self, x, thresh=0.5):
        dead = (self.usage < thresh).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            self.usage.zero_()
            return 0
        flat = x.reshape(-1, self.dim)
        if flat.shape[0] == 0:
            return 0
        picks = torch.randint(0, flat.shape[0], (len(dead),), device=flat.device)
        self.embedding.weight.data[dead] = flat[picks].detach()
        self.usage.zero_()
        return len(dead)


# =========================================================
# Encoder（[长句] 加 MultiScaleConv；[古文] 加 lang 通道）
# =========================================================
class StructXiangEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg["dim"]
        self.token_emb = nn.Embedding(cfg["vocab_size"], self.dim)
        self.pos_emb = nn.Embedding(cfg["seq_max"], self.dim)
        self.pos_tag_emb = nn.Embedding(NUM_POS, self.dim)
        self.gram_emb = nn.Embedding(NUM_GRAM, self.dim)
        self.role_emb = nn.Embedding(NUM_ROLES, self.dim)
        # [古文] lang 通道：0=现代，1=文言
        self.lang_emb = nn.Embedding(NUM_LANG, self.dim)
        # 6 通道融合
        self.fuse = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        # [长句] 6 层多尺度 conv
        self.multi_scale = MultiScaleConv(self.dim, n_layers=cfg["ms_n_layers"])

    def forward(self, token_ids, pos_tags, gram_tags, role_tags, lang_tags):
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)[None, :].repeat(B, 1)
        x = torch.cat([
            self.token_emb(token_ids),
            self.pos_emb(pos),
            self.pos_tag_emb(pos_tags),
            self.gram_emb(gram_tags),
            self.role_emb(role_tags),
            self.lang_emb(lang_tags),
        ], dim=-1)
        x = self.fuse(x)
        # 多尺度上下文融合
        x = self.multi_scale(x)
        return x


# =========================================================
# 层级 pooling
# =========================================================
class HierPooling(nn.Module):
    def __init__(self, cfg, depth):
        super().__init__()
        self.depth = depth
        self.dim = cfg["dim"]
        self.recon_heads = nn.ModuleList()
        for _ in range(depth - 1):
            self.recon_heads.append(nn.Linear(self.dim, self.dim * 2))

    def forward(self, X_full):
        levels = [X_full]
        recon_preds = []
        current = X_full
        for i in range(self.depth - 1):
            B, L, D = current.shape
            if L % 2 != 0:
                current = F.pad(current, (0, 0, 0, 1))
                L += 1
            even = current[:, 0::2, :]
            odd = current[:, 1::2, :]
            pooled = (even + odd) / 2
            recon = self.recon_heads[i](pooled).reshape(B, pooled.shape[1], 2, D)
            recon_preds.append(recon)
            levels.append(pooled)
            current = pooled
        return levels, recon_preds


def compute_hier_loss(levels, recon_preds, token_mask):
    if not recon_preds:
        return torch.tensor(0.0, device=levels[0].device)

    losses = []
    mask = token_mask.float()
    for i, recon in enumerate(recon_preds):
        src = levels[i]
        B, L2, D = src.shape
        L_i_recon = recon.shape[1]
        L2_use = min(L2, L_i_recon * 2)
        if L2_use % 2 != 0:
            L2_use -= 1
        L_i = L2_use // 2
        tgt = src[:, :L2_use, :].reshape(B, L_i, 2, D)
        L_min = min(L_i, L_i_recon)
        tgt = tgt[:, :L_min, :, :]
        recon = recon[:, :L_min, :, :]

        m_even = mask[:, 0:2*L_min:2]
        m_odd = mask[:, 1:2*L_min:2]
        m_pair = (m_even * m_odd).unsqueeze(-1).unsqueeze(-1)

        diff = (recon - tgt).pow(2) * m_pair
        denom = m_pair.sum() * D * 2 + 1e-6
        losses.append(diff.sum() / denom)

        mask = (m_even * m_odd).float()

    return torch.stack(losses).mean()


# =========================================================
# Transition（[长句] 加 MultiScaleConv）
# =========================================================
class StructTransition(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["dim"]
        self.multi_scale = MultiScaleConv(dim, n_layers=cfg["ms_n_layers"])
        self.pointwise = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.LayerNorm(dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
        )
        self.vq = VectorQuantizer(cfg["codebook_size"], dim)

    def forward(self, X, beta):
        mixed = self.multi_scale(X)
        delta = self.pointwise(mixed)
        Y_cont = X + torch.tanh(delta)
        Y_quant, Y_codes, vq_loss = self.vq(Y_cont, beta)
        return Y_cont, Y_quant, Y_codes, vq_loss


# =========================================================
# Decoder（[长句] 加 MultiScaleConv；[P0-2] 保留 gen_mask）
# =========================================================
class StructureDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["dim"]
        self.multi_scale = MultiScaleConv(dim, n_layers=cfg["ms_n_layers"])
        self.role_head = nn.Linear(dim, NUM_ROLES)
        self.word_head = nn.Linear(dim, cfg["vocab_size"])
        self.role_bias = nn.Embedding(NUM_ROLES, cfg["vocab_size"])
        self.pos_head = nn.Linear(dim, NUM_POS)
        self.gram_head = nn.Linear(dim, NUM_GRAM)
        self.gen_embed = nn.Parameter(torch.zeros(dim))

    def forward(self, Y, gen_mask=None):
        h = self.multi_scale(Y)

        if gen_mask is not None:
            h = h + gen_mask.unsqueeze(-1) * self.gen_embed

        role_logits = self.role_head(h)
        role_soft = F.softmax(role_logits, dim=-1)

        if self.training:
            role_bias_w = F.dropout(self.role_bias.weight, p=0.3)
        else:
            role_bias_w = self.role_bias.weight
        bias = torch.einsum("blr,rv->blv", role_soft, role_bias_w)
        bias = 0.5 * torch.tanh(bias * 2.0)

        word_logits = self.word_head(h) + bias
        pos_logits = self.pos_head(h)
        gram_logits = self.gram_head(h)

        return {
            "role_logits": role_logits,
            "word_logits": word_logits,
            "pos_logits": pos_logits,
            "gram_logits": gram_logits,
        }


# =========================================================
# 晶体库
# =========================================================
class StructCrystalBank:
    def __init__(self, max_cap, dim, device):
        self.max_cap = max_cap
        self.dim = dim
        self.device = device
        self.bank = []
        self.recent_gaps = []

    @torch.no_grad()
    def add(self, x):
        s = F.normalize(x.detach().cpu().clone().float(), p=2, dim=-1)
        self.bank.append(s)
        if len(self.bank) > self.max_cap:
            self.bank.pop(0)

    @torch.no_grad()
    def structure_sim(self, q):
        if not self.bank:
            return []
        q = F.normalize(q.float(), p=2, dim=-1)
        sims = []
        for b in self.bank:
            ml = min(q.shape[0], b.shape[0])
            sims.append(float(F.cosine_similarity(
                q[:ml].unsqueeze(0), b[:ml].unsqueeze(0), dim=-1
            ).mean()))
        return sims

    @torch.no_grad()
    def ambiguity_check(self, q):
        sims = self.structure_sim(q)
        if len(sims) < 5:
            return False, 0.0, 0.0
        top2 = sorted(sims, reverse=True)[:2]
        gap = top2[0] - top2[1]
        self.recent_gaps.append(gap)
        if len(self.recent_gaps) > 100:
            self.recent_gaps.pop(0)
        if len(self.recent_gaps) >= 20:
            thresh = float(np.percentile(self.recent_gaps, 20))
        else:
            thresh = 0.01
        return gap < thresh, top2[0], top2[1]


# =========================================================
# 不确定性加权
# =========================================================
class UncertaintyWeight(nn.Module):
    def __init__(self, num_loss):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(num_loss))

    def forward(self, losses):
        total = 0.0
        for i, ls in enumerate(losses):
            lv = self.log_var[i].clamp(-5.0, 5.0)
            w = 0.5 * torch.exp(-lv)
            w = torch.clamp(w, min=0.1)
            total = total + w * ls + 0.5 * lv
        return total


# =========================================================
# 主模型
# =========================================================
class XiangSymbolWorldV25(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = StructXiangEncoder(cfg)
        self.hier = HierPooling(cfg, cfg["hier_depth"])
        self.transition = StructTransition(cfg)
        self.decoder = StructureDecoder(cfg)
        self.uncert = UncertaintyWeight(6)
        self.register_buffer("mean_embed", torch.zeros(cfg["dim"]))

    def set_mean_embed(self, v):
        self.mean_embed.copy_(v)

    def forward(self, in_tok, in_pos, in_gram, in_role, in_lang,
                out_tok, out_pos, out_gram, out_role, beta):
        B, L = in_tok.shape

        X_raw = self.encoder(in_tok, in_pos, in_gram, in_role, in_lang)
        X = X_raw - self.mean_embed

        X_levels, X_recons = self.hier(X)

        Y_cont, Y_quant, Y_codes, vq_loss = self.transition(X, beta)
        Y = Y_quant

        input_lens = (in_tok != 0).sum(dim=1)
        pos_idx = torch.arange(L, device=Y.device).unsqueeze(0)
        gen_mask = (pos_idx >= input_lens.unsqueeze(1)).float()

        out = self.decoder(Y, gen_mask)

        token_mask = (in_tok != 0)

        loss_word = F.cross_entropy(
            out["word_logits"].reshape(-1, self.cfg["vocab_size"]),
            out_tok.reshape(-1), ignore_index=0,
        )
        loss_pos = F.cross_entropy(
            out["pos_logits"].reshape(-1, NUM_POS),
            out_pos.reshape(-1), ignore_index=0,
        )
        loss_gram = F.cross_entropy(
            out["gram_logits"].reshape(-1, NUM_GRAM),
            out_gram.reshape(-1), ignore_index=0,
        )
        loss_role = F.cross_entropy(
            out["role_logits"].reshape(-1, NUM_ROLES),
            out_role.reshape(-1), ignore_index=0,
        )
        pad_mask = (out_tok == 0)
        if pad_mask.any():
            pad_role_pred = out["role_logits"][pad_mask]
            pad_role_target = torch.zeros(pad_role_pred.shape[0], dtype=torch.long, device=pad_role_pred.device)
            loss_role = loss_role + 0.5 * F.cross_entropy(pad_role_pred, pad_role_target)
        loss_hier = compute_hier_loss(X_levels, X_recons, token_mask)

        losses = [loss_word, loss_pos, loss_gram, loss_role, loss_hier, vq_loss]
        total = self.uncert(losses)

        bias_reg = 0.01 * self.decoder.role_bias.weight.pow(2).mean()
        total = total + bias_reg

        word_probs = F.softmax(out["word_logits"], dim=-1)
        non_pad_pair = (out_tok[:, :-1] != 0) & (out_tok[:, 1:] != 0)
        adj_repeat = (word_probs[:, :-1] * word_probs[:, 1:]).sum(-1)
        adj_repeat = adj_repeat * non_pad_pair.float()
        loss_repeat = adj_repeat.sum() / (non_pad_pair.float().sum() + 1e-6) * 0.1
        total = total + loss_repeat

        return total, losses, {
            "X": X, "Y": Y, "Y_cont": Y_cont, "Y_codes": Y_codes, "out": out,
        }


# =========================================================
# 数据
# =========================================================
def load_pairs(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [tuple(x) for x in json.load(f)]


def build_sample(text, vocab):
    """[通用] 构建样本：ids, pos, gram, role, lang"""
    ids, pos, gram, role = text_to_three_tags(text)
    tok_ids = vocab.encode(ids)
    # [古文] lang：整句标记
    lang = 1 if is_classical(text) else 0
    lang_list = [lang] * len(ids)
    return tok_ids, pos, gram, role, lang_list


def pad_to_block(seqs, pad_values, block=16, max_len=256):
    L = len(seqs[0])
    pad_to = ((L + block - 1) // block) * block
    pad_to = min(pad_to, max_len)
    out = []
    for s, p in zip(seqs, pad_values):
        out.append((s + [p] * pad_to)[:pad_to])
    return out


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, vocab):
        self.samples = []
        for q, a in pairs:
            try:
                qi = build_sample(q, vocab)
                ai = build_sample(a, vocab)
                self.samples.append((qi, ai))
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        (qi, qp, qg, qr, ql), (ai, ap, ag, ar, al) = self.samples[i]
        return qi, qp, qg, qr, ql, ai, ap, ag, ar, al


def collate_pad(batch):
    qis, qps, qgs, qrs, qls, ais, aps, ags, ars, als = zip(*batch)
    max_len = max(
        max(len(x) for x in qis),
        max(len(x) for x in ais),
    )
    def pad_list(items, pad_val):
        out = []
        for s in items:
            s2 = (s + [pad_val] * max_len)[:max_len]
            out.append(torch.tensor(s2, dtype=torch.long))
        return torch.stack(out, 0)
    return (
        pad_list(qis, 0), pad_list(qps, POS_DEF["PAD"]),
        pad_list(qgs, GRAM_DEF["NONE"]), pad_list(qrs, ROLE_DEF["NONE"]),
        pad_list(qls, 0),
        pad_list(ais, 0), pad_list(aps, POS_DEF["PAD"]),
        pad_list(ags, GRAM_DEF["NONE"]), pad_list(ars, ROLE_DEF["NONE"]),
        pad_list(als, 0),
    )


# =========================================================
# 推理
# =========================================================
@torch.no_grad()
def infer(model, crystal, text, vocab, beta=0.25, add_to_crystal=True):
    model.eval()
    try:
        ids, pos, gram, role = text_to_three_tags(text)
    except Exception:
        return {"text": "(分词失败)"}
    if not ids:
        return {"text": "(空)"}

    tok_ids = vocab.encode(ids)
    lang = 1 if is_classical(text) else 0
    lang_list = [lang] * len(ids)
    tok_ids, pos, gram, role, lang_list = pad_to_block(
        [tok_ids, pos, gram, role, lang_list],
        [0, POS_DEF["PAD"], GRAM_DEF["NONE"], ROLE_DEF["NONE"], 0],
        max_len=model.cfg["seq_max"],
    )

    tin = torch.tensor([tok_ids], dtype=torch.long, device=DEVICE)
    tpos = torch.tensor([pos], dtype=torch.long, device=DEVICE)
    tgram = torch.tensor([gram], dtype=torch.long, device=DEVICE)
    trole = torch.tensor([role], dtype=torch.long, device=DEVICE)
    tlang = torch.tensor([lang_list], dtype=torch.long, device=DEVICE)

    X_raw = model.encoder(tin, tpos, tgram, trole, tlang)
    X = (X_raw - model.mean_embed)[0]

    amb, s1, s2 = crystal.ambiguity_check(X)
    if amb and add_to_crystal:
        return {"text": f"【象歧义】top1={s1:.3f} top2={s2:.3f}", "ambiguous": True}

    Y_cont, Y_quant, Y_codes, _ = model.transition(X[None], beta)
    Y = Y_quant

    input_len = len(ids)
    L = Y.shape[1]
    pos_idx = torch.arange(L, device=DEVICE).unsqueeze(0)
    gen_mask = (pos_idx >= torch.tensor([input_len], device=DEVICE).unsqueeze(1)).float()

    out = model.decoder(Y, gen_mask)
    word_ids = out["word_logits"].argmax(-1)[0].cpu().tolist()
    role_ids = out["role_logits"].argmax(-1)[0].cpu().tolist()

    max_out = int(input_len * 1.5) + 2
    truncated = []
    repeat_count = 1
    prev_wid = -1
    for wid in word_ids:
        if wid in (0, 2, 3):
            break
        if wid == prev_wid:
            repeat_count += 1
            if repeat_count >= 4:
                break
        else:
            repeat_count = 1
        truncated.append(wid)
        prev_wid = wid
        if len(truncated) >= max_out:
            break
    if len(truncated) >= 3:
        last3 = [vocab.id2char.get(w, '') for w in truncated[-3:]]
        if last3[0] == last3[1] == last3[2] and last3[0] in '了的在是也已矣也':
            truncated = truncated[:-2]

    if add_to_crystal:
        crystal.add(X)
    return {
        "text": vocab.decode(truncated),
        "word_ids": truncated,
        "role_ids": role_ids[:len(truncated)] if truncated else role_ids,
        "X_codes": Y_codes[0].cpu().tolist(),
    }


# =========================================================
# 评测
# =========================================================
def ngrams(s, n):
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))


def overlap(a, b, n):
    if not a or not b:
        return 0.0
    ga, gb = ngrams(a, n), ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return sum((ga & gb).values()) / max(len(ga), len(gb))


def token_acc(pred, exp):
    if not exp:
        return 0.0
    n = min(len(pred), len(exp))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if pred[i] == exp[i]) / len(exp)


@torch.no_grad()
def evaluate(model, crystal, vocab, holdout_pairs, verbose=True):
    model.eval()
    rows = []
    for q, a in holdout_pairs:
        r = infer(model, crystal, q, vocab, add_to_crystal=False)
        pred = r["text"] if isinstance(r, dict) else r
        def self_rep(s, n=2):
            if len(s) < n + 1:
                return 0.0
            grams = [s[i:i+n] for i in range(len(s)-n+1)]
            if len(set(grams)) == len(grams):
                return 0.0
            c = Counter(grams)
            return sum(v - 1 for v in c.values() if v > 1) / len(grams)
        rows.append({
            "in": q, "exp": a, "got": pred,
            "tok": token_acc(pred, a),
            "rep2": overlap(pred, q, 2),
            "exp2": overlap(pred, a, 2),
            "self_rep2": self_rep(pred, 2),
        })

    if verbose and rows:
        n = len(rows)
        print(f"\n=== 评测 (n={n}) ===")
        print(f"tok_acc = {sum(r['tok'] for r in rows)/n:.3f}")
        print(f"rep_2g  = {sum(r['rep2'] for r in rows)/n:.3f}")
        print(f"exp_2g  = {sum(r['exp2'] for r in rows)/n:.3f}")
        print(f"self_rep2 = {sum(r['self_rep2'] for r in rows)/n:.3f}")
        char_hit = sum(len(set(r['got']) & set(r['exp'])) / max(len(set(r['exp'])), 1) for r in rows) / n
        print(f"char_hit_rate = {char_hit:.3f}")
        print("\n-- 前 5 条 --")
        for r in rows[:5]:
            print(f"[in] {r['in']}")
            print(f"[exp] {r['exp'][:30]}")
            print(f"[got] {r['got'][:30]}")
    return rows


# =========================================================
# 主训练
# =========================================================
if __name__ == "__main__":
    train_pairs = load_pairs(CFG["train_json"])
    holdout_pairs = load_pairs(CFG["holdout_json"])

    if train_pairs is None:
        print(f"[错误] 找不到训练数据：{CFG['train_json']}")
        print("请先运行 gen_v25_data.py 生成数据")
        exit(1)

    print(f"[data] train={len(train_pairs)} holdout={len(holdout_pairs or [])}")

    all_texts = []
    for q, a in train_pairs:
        all_texts.append(q)
        all_texts.append(a)
    vocab = CharVocab(all_texts)
    CFG["vocab_size"] = vocab.vocab_size
    print(f"[vocab] size={vocab.vocab_size}")
    print(f"[tags] POS={NUM_POS}, GRAM={NUM_GRAM}, ROLE={NUM_ROLES}, LANG={NUM_LANG}")
    print(f"[v25] 多尺度 conv {CFG['ms_n_layers']} 层 / seq_max={CFG['seq_max']} / 文言虚词 {NUM_GRAM - 16} 类")

    model = XiangSymbolWorldV25(CFG).to(DEVICE)
    crystal = StructCrystalBank(CFG["crystal_cap"], CFG["dim"], DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-5)

    ds = CausalDataset(train_pairs, vocab)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=CFG["batch_size"], shuffle=True,
        drop_last=True, collate_fn=collate_pad,
    )

    print("🔧 预热 mean_embed...")
    with torch.no_grad():
        xs = []
        for q, _ in train_pairs[:min(500, len(train_pairs))]:
            try:
                ids, pos, gram, role = text_to_three_tags(q)
                if not ids:
                    continue
                tok_ids = vocab.encode(ids)
                lang = 1 if is_classical(q) else 0
                lang_list = [lang] * len(ids)
                tok_ids, pos, gram, role, lang_list = pad_to_block(
                    [tok_ids, pos, gram, role, lang_list],
                    [0, POS_DEF["PAD"], GRAM_DEF["NONE"], ROLE_DEF["NONE"], 0],
                    max_len=CFG["seq_max"],
                )
                tin = torch.tensor([tok_ids], dtype=torch.long, device=DEVICE)
                tpos = torch.tensor([pos], dtype=torch.long, device=DEVICE)
                tgram = torch.tensor([gram], dtype=torch.long, device=DEVICE)
                trole = torch.tensor([role], dtype=torch.long, device=DEVICE)
                tlang = torch.tensor([lang_list], dtype=torch.long, device=DEVICE)
                xs.append(model.encoder(tin, tpos, tgram, trole, tlang))
            except Exception:
                continue
        if xs:
            mean_vec = torch.cat(xs, 0).mean(dim=(0, 1))
            mean_vec = F.normalize(mean_vec, dim=0)
            model.set_mean_embed(mean_vec)
    print("✅ mean_embed 完成")

    start_step = 0
    if os.path.exists(CFG["save_path"]):
        ckpt = torch.load(CFG["save_path"], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        print(f"✅ resume step={start_step}")

    step = start_step
    model.train()
    while step < CFG["max_steps"]:
        for batch in loader:
            (in_tok, in_pos, in_gram, in_role, in_lang,
             out_tok, out_pos, out_gram, out_role, out_lang) = [b.to(DEVICE) for b in batch]

            if step < CFG["vq_beta_warmup"]:
                beta = CFG["vq_beta_start"] + (
                    (CFG["vq_beta_end"] - CFG["vq_beta_start"])
                    * step / CFG["vq_beta_warmup"]
                )
            else:
                beta = CFG["vq_beta_end"]

            opt.zero_grad()
            total, losses, info = model(
                in_tok, in_pos, in_gram, in_role, in_lang,
                out_tok, out_pos, out_gram, out_role, beta,
            )
            if torch.isnan(total) or torch.isinf(total):
                step += 1
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            if step % 5 == 0:
                with torch.no_grad():
                    for i in range(info["X"].shape[0]):
                        crystal.add(info["X"][i])

            if step > 0 and step % 200 == 0:
                n = model.transition.vq.restart_dead_codes(info["Y_cont"], thresh=0.5)
                if n > 0:
                    print(f"  [VQ] restarted {n} dead codes")

            step += 1
            if step % 20 == 0:
                lw, lp, lg, lr_, lh, lv = [x.item() for x in losses]
                print(
                    f"step={step:04d} tot={total.item():.3f} "
                    f"word={lw:.3f} pos={lp:.3f} gram={lg:.3f} role={lr_:.3f} "
                    f"hier={lh:.4f} vq={lv:.4f} beta={beta:.3f}"
                )

            if step % 500 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                }, CFG["save_path"])
                print(f"💾 saved step={step}")
                if holdout_pairs:
                    evaluate(model, crystal, vocab, holdout_pairs[:20], verbose=True)
                model.train()

            if step >= CFG["max_steps"]:
                break

    print("✅ 训练完成")