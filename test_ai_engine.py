# -*- coding: utf-8 -*-
"""
test_ai_engine.py - AI 语义引擎单元测试（阶段4）

覆盖：
  - 上下文感知排序（ai_context）
  - 词库自动训练（ai_trainer）

用法：
    python3 test_ai_engine.py
"""

import sys
import os
import json
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

from ai_context import AIContextEngine
from ai_trainer import AITrainer


# ------------------------------------------------------------------
# ai_context 测试
# ------------------------------------------------------------------
def test_context_ranking():
    """上文'云计算'时，'平台'应优先于'你好'"""
    engine = AIContextEngine()
    engine.train([["云计算", "平台"], ["云计算", "服务"], ["你好", "世界"]])
    cands = [
        {"phrase": "你好", "chars": [1, 2], "code": "yyy", "type": "word2"},
        {"phrase": "平台", "chars": [3, 4], "code": "xxx", "type": "word2"},
    ]
    ranked = engine.predict(["云计算"], cands)
    assert ranked[0]["phrase"] == "平台", f"期望平台在前, 实际: {ranked[0]['phrase']}"
    print("✅ test_context_ranking 通过")


def test_no_context_no_reorder():
    """无上下文时保持原序"""
    engine = AIContextEngine()
    cands = [
        {"phrase": "A", "chars": [1], "code": "a", "type": "char"},
        {"phrase": "B", "chars": [2], "code": "b", "type": "char"},
    ]
    ranked = engine.predict([], cands)
    assert [c["phrase"] for c in ranked] == ["A", "B"]
    print("✅ test_no_context_no_reorder 通过")


def test_no_model_no_reorder():
    """无训练模型时保持原序（不干预）"""
    engine = AIContextEngine()
    cands = [
        {"phrase": "A", "chars": [1], "code": "a", "type": "char"},
        {"phrase": "B", "chars": [2], "code": "b", "type": "char"},
    ]
    ranked = engine.predict(["云计算"], cands)
    assert [c["phrase"] for c in ranked] == ["A", "B"]
    print("✅ test_no_model_no_reorder 通过")


def test_model_persist():
    """模型可持久化到文件并重新加载"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        engine1 = AIContextEngine(model_path=path)
        engine1.train([["人工智能", "赋能"]])
        engine2 = AIContextEngine(model_path=path)
        cands = [
            {"phrase": "赋能", "chars": [1, 2], "code": "x", "type": "word2"},
            {"phrase": "你好", "chars": [3, 4], "code": "y", "type": "word2"},
        ]
        ranked = engine2.predict(["人工智能"], cands)
        assert ranked[0]["phrase"] == "赋能"
        print("✅ test_model_persist 通过")
    finally:
        os.unlink(path)


# ------------------------------------------------------------------
# ai_trainer 测试
# ------------------------------------------------------------------
def test_trainer_discover():
    """训练器发现高频新词并过滤已收录/低频词"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w", encoding="utf-8") as f:
        log = f.name
        now = time.time()
        for _ in range(5):
            f.write(json.dumps({"ts": now, "phrase": "算力"}) + "\n")
        for _ in range(2):
            f.write(json.dumps({"ts": now, "phrase": "你好"}) + "\n")
        for _ in range(1):
            f.write(json.dumps({"ts": now, "phrase": "量子"}) + "\n")
    try:
        trainer = AITrainer(min_freq=3)
        trainer.ingest(log)
        found = trainer.discover_new_words(existing_words={"你好"})
        words = [s["word"] for s in found]
        assert "算力" in words, f"算力应被发现: {words}"
        assert "你好" not in words, "已收录词应被过滤"
        assert "量子" not in words, "低频词应被过滤"
        print("✅ test_trainer_discover 通过")
    finally:
        os.unlink(log)


def test_trainer_export():
    """训练器导出建议与语料"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w", encoding="utf-8") as f:
        log = f.name
        for _ in range(4):
            f.write(json.dumps({"ts": time.time(), "phrase": "数据中心"}) + "\n")
    sug_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    corpus_path = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
    try:
        trainer = AITrainer(min_freq=3)
        trainer.ingest(log)
        sugg = trainer.discover_new_words(existing_words=set())
        n1 = trainer.export_suggestions(sugg, sug_path)
        n2 = trainer.export_training_corpus(corpus_path)
        assert n1 == 1, f"应导出1条建议: {n1}"
        assert n2 >= 1, f"应导出语料: {n2}"
        assert os.path.getsize(sug_path) > 0
        print("✅ test_trainer_export 通过")
    finally:
        os.unlink(log)
        os.unlink(sug_path)
        os.unlink(corpus_path)


if __name__ == "__main__":
    test_context_ranking()
    test_no_context_no_reorder()
    test_no_model_no_reorder()
    test_model_persist()
    test_trainer_discover()
    test_trainer_export()
    print("\n✅ AI 语义引擎全部单元测试通过")
