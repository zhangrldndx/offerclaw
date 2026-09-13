# -*- coding: utf-8 -*-
"""2026-08-10 review 修复的回归钉死(卫生规则=严格档,仅论文域默认启用):血缘契约 / 硬上限 / 引文过滤 / 英文信息量口径。

每条都对应一个实测过的真实缺陷,不是假想:
① 同为 768 维的错模型写入 e5 集合不报错 → 块永不可检索且污染距离门;
② 无空行长段无上限 → 实测 18379 字符块(embedding 只吃前 512 token,35% 正文不可见);
③ 参考文献段语义密度高、易冲 top-1,但只能让 LLM 抄引文;
⑦ 80 字符噪声门按中文调,英文 80 字符仅 ~13 词 ≈ 形同虚设。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_tools import (  # noqa: E402
    EmbeddingContractError,
    _hard_split,
    _info_chars,
    assert_collection_contract,
    embed_profile,
    is_citation_dense,
    split_markdown_document,
)


# ---------- ① 血缘 + 集合契约 ----------

def test_embed_profile_tracks_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "/models/multilingual-e5-base")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    monkeypatch.setenv("OFFERCLAW_EMBED_PREFIX", "passage: ")
    p = embed_profile()
    assert "multilingual-e5-base" in p and "768" in p and "prefix=passage:" in p


class _FakeCol:
    def __init__(self, profiles):
        self._p = profiles

    def get(self, limit=None, include=None):
        return {"metadatas": [{"embed_profile": p} for p in self._p]}


def test_contract_blocks_mismatched_model(monkeypatch):
    """核心防线:同 768 维的错模型必须被拦(否则静默污染,事后无从排查)。"""
    monkeypatch.setenv("EMBEDDING_MODEL", "/models/bge-base-zh-v1.5")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    monkeypatch.delenv("OFFERCLAW_EMBED_PREFIX", raising=False)
    col = _FakeCol(["local|multilingual-e5-base|768|prefix=passage: |maxseq=512"])
    with pytest.raises(EmbeddingContractError) as e:
        assert_collection_contract(col)
    assert "血缘不符" in str(e.value) and "RAG_COLLECTION_NAME" in str(e.value)


def test_contract_passes_when_matching(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "/models/multilingual-e5-base")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    monkeypatch.setenv("OFFERCLAW_EMBED_PREFIX", "passage: ")
    monkeypatch.setenv("OFFERCLAW_EMBED_MAX_SEQ", "512")
    assert assert_collection_contract(_FakeCol([embed_profile()])) is not None


def test_contract_tolerates_legacy_blocks():
    assert assert_collection_contract(_FakeCol([])) is None       # 空集合=首建,放行
    assert assert_collection_contract(_FakeCol([None])) is None   # 历史块无签名,只告警


# ---------- ② 硬上限 ----------

def test_hard_split_caps_length():
    long_para = "word " * 2000                       # 无空行的一大坨(论文附录/压平表格)
    parts = _hard_split(long_para, 2000)
    assert len(parts) > 1 and all(len(p) <= 2000 for p in parts)
    assert "".join(p.replace(" ", "") for p in parts) == long_para.replace(" ", "")


def test_chunker_enforces_max_chars():
    doc = "# T\n\n## S\n\n" + ("This is a long English paragraph without blank lines. " * 300)
    chunks = split_markdown_document(doc, max_chars=1500, strict_hygiene=True)
    assert chunks and max(len(c["text"]) for c in chunks) <= 1500


# ---------- ③ 引文密集段 ----------

def test_citation_dense_detected_and_dropped():
    refs = ("Yao et al., 2022. ReAct. arXiv preprint arXiv:2210.03629. "
            "Packer et al., 2023. MemGPT. arXiv preprint arXiv:2310.08560. "
            "Huang et al., 2022. Inner monologue. In Proceedings of CoRL. "
            "Wei et al., 2022. Chain of thought. In Advances in Neural Information. "
            "Shinn et al., 2023. Reflexion. arXiv preprint arXiv:2303.11366. "
            "Xiong et al., 2020. ANCE. In Proceedings of ICLR. ")
    assert is_citation_dense(refs)
    normal = ("The method interleaves reasoning and acting, following prior work "
              "(Yao et al., 2022), and we evaluate it on two interactive benchmarks "
              "with detailed ablations and analysis of failure cases in this section.")
    assert not is_citation_dense(normal)             # 正文偶尔引用不误伤
    doc = "# P\n\n## References\n\n" + refs
    # 严格档(论文域)才启用引文过滤;中文主库保持历史行为(实测:套上主库无收益)
    assert all("arxiv preprint" not in c["text"].lower()
               for c in split_markdown_document(doc, strict_hygiene=True))


# ---------- ⑦ 英文信息量口径 ----------

def test_info_chars_language_aware():
    cn = "这是一段中文正文内容用于验证信息量度量口径"
    en = "a" * len(cn)
    assert _info_chars(cn) > _info_chars(en) * 3     # 同字符数,中文信息量远高于英文


def test_english_noise_lines_filtered():
    """作者名/邮箱/脚注这类英文短行:按字符数够 80,按信息量不够 → 应被滤掉。"""
    noise = ("# Paper\n\n## Header\n\n"
             "Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran\n"
             "{shunyuy,karthikn}@princeton.edu\n"
             "Work during Google internship\n")
    assert split_markdown_document(noise, strict_hygiene=True) == []
