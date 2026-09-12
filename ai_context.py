# -*- coding: utf-8 -*-
"""
ai_context.py - CloudWubi AI 上下文感知排序引擎（阶段4核心）
=============================================================

目标（兑现 5G/6G 时代云端智能价值）：
  用户在输入时，AI 根据「已输入的上文」预测「接下来最可能想打的词」，
  把高概率候选置顶 —— 这就是输入法"智能联想"的底层逻辑。

实现（先科学、后先进，免费可离线）：
  1. 语言模型层：基于 2-gram（Bigram）+ 1-gram 统计的马尔可夫模型
     - 无网络、无API成本、微秒级推理
     - 模型从语料/用户行为中学习（可增量更新）
  2. 自适应层：结合用户历史输入习惯（云端持久化）
  3. 可扩展层：预留大模型接口（阶段4后续可接 OpenAI 兼容端点）

输入/输出：
  predict(context_words, candidates) -> 重排后的候选列表
    其中 context_words = 用户已上屏的字词列表
        candidates    = 构词引擎的候选列表
"""

import json
import os
from collections import defaultdict


class AIContextEngine:
    """上下文感知排序引擎（马尔可夫 2-gram）"""

    def __init__(self, model_path=None):
        self.bigram = defaultdict(int)      # (w1, w2) -> 次数
        self.unigram = defaultdict(int)     # w1 -> 次数
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self._load(model_path)

    # ------------------------------------------------------------------
    # 模型训练（云端词库自动训练入口）
    # ------------------------------------------------------------------
    def train(self, sentences):
        """
        从句子语料训练 2-gram 模型。
        sentences: [["你好", "世界"], ["云计算", "是", "未来"], ...]
        """
        for words in sentences:
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram[(w1, w2)] += 1
                self.unigram[w1] += 1
            if words:
                self.unigram[words[-1]] += 1
        if self.model_path:
            self._save(self.model_path)

    def train_from_user_log(self, log_path):
        """
        从用户输入日志训练（云端行为学习）。
        log 格式（每行一条输入序列，逗号分隔）：
            "你好,世界,云计算"
        """
        if not os.path.exists(log_path):
            return 0
        sentences = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = [w for w in line.split(",") if w]
                if len(words) >= 1:
                    sentences.append(words)
        self.train(sentences)
        return len(sentences)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def predict(self, context_words, candidates, top_n=5):
        """
        根据上文对候选排序（原地重排，返回新列表）。
        context_words: 用户已上屏的词列表（如 ["云计算", "的"]）
        candidates:   构词引擎候选 [{phrase, ...}, ...]
        """
        if not candidates:
            return candidates

        last_word = context_words[-1] if context_words else None
        total = max(self.unigram.get(last_word, 0), 1) if last_word else 1

        scored = []
        for cand in candidates:
            phrase = cand["phrase"]
            score = 0.0

            # 信号1：Bigram 条件概率 P(候选 | 上文末词)
            if last_word:
                pair = (last_word, phrase)
                big = self.bigram.get(pair, 0)
                score += (big / total) * 100.0

            # 信号2：单字延续加分（"的"后接名词性词更自然）
            if last_word in ("的", "了", "和", "与", "是", "在"):
                score += 2.0

            cand["_ai"] = score
            scored.append(cand)

        # 稳定排序：AI 分高者前，同分保持原序
        scored.sort(key=lambda c: c["_ai"], reverse=True)
        for c in scored:
            c.pop("_ai", None)

        # 仅当有上下文且有学习数据时，AI 排序才真正生效
        has_model = bool(self.bigram)
        if not has_model:
            return candidates  # 无模型时原样返回（不干预）
        return scored[:top_n] + scored[top_n:]

    # ------------------------------------------------------------------
    # 模型持久化
    # ------------------------------------------------------------------
    def _save(self, path):
        """保存模型（JSON，云端共享）。"""
        data = {
            "bigram": {f"{a}→{b}": v for (a, b), v in self.bigram.items()},
            "unigram": dict(self.unigram),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load(self, path):
        """加载模型。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, v in data.get("bigram", {}).items():
                a, b = key.split("→", 1)
                self.bigram[(a, b)] = int(v)
            self.unigram.update({k: int(v) for k, v in data.get("unigram", {}).items()})
        except (IOError, ValueError):
            pass


# ------------------------------------------------------------------
# 便捷入口
# ------------------------------------------------------------------
_engine = None


def get_ai_engine(model_path=None):
    """全局 AI 引擎（懒加载）。"""
    global _engine
    if _engine is None:
        _engine = AIContextEngine(model_path=model_path)
    return _engine


if __name__ == "__main__":
    print("=== AI 上下文感知引擎自测 ===\n")

    engine = AIContextEngine()
    # 训练语料（真实场景高频搭配）
    engine.train([
        ["云计算", "是", "未来"],
        ["人工智能", "赋能", "产业"],
        ["云计算", "平台"],
        ["云计算", "服务"],
        ["算力", "是", "核心"],
        ["云计算", "的", "发展"],
    ])

    # 测试1：上文"云计算"，候选应优先"平台/服务"
    cands = [
        {"phrase": "平台", "chars": [1, 2], "code": "xxx", "type": "word2"},
        {"phrase": "你好", "chars": [3, 4], "code": "yyy", "type": "word2"},
    ]
    ranked = engine.predict(["云计算"], cands)
    print("上文'云计算'，候选排序（期望：平台 在前）：")
    for c in ranked:
        print(f"  {c['phrase']}")

    # 测试2：无上下文时原样返回
    ranked2 = engine.predict([], cands)
    print("\n无上下文，原序保留：", [c["phrase"] for c in ranked2])

    # 测试3：无模型时原样返回
    engine2 = AIContextEngine()
    ranked3 = engine2.predict(["云计算"], cands)
    print("无模型，原序保留：", [c["phrase"] for c in ranked3])

    print("\n✅ AI 上下文感知引擎自测完成")
