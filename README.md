# CloudWubi AI — 云端 AI 引擎

> 面向 5G/6G 的云原生五笔输入法 · AI 智能组件
> 构词生成 · 语义排序 · 联邦自学习（规划中）

## 简介

CloudWubi AI 是 CloudWubi 云五笔的**云端 AI 引擎**。
承载输入法的全部智能能力：动态构词、语义重排、用户行为学习。
与端侧（<800KB 极简内核）完全解耦，**新增智能能力只需在云端发布，端侧无需更新**。

## 规划能力

| 能力 | 说明 | 状态 |
| ---- | ---- | ---- |
| 五笔文法检索 | 读取 cloudwubi-rules 公共规则库 | 📋 阶段2（已由 gateway 实现基础版） |
| 双向文法自校验 | 正向编码查字 + 反向拆字校验 | 📋 阶段3 |
| 构词生成 | 基于五笔构词公式动态生成词组 | 📋 已由 gateway 实现基础版 |
| 场景语义重排 | 根据场景标签（BP/编程/法律）调整候选权重 | 📋 阶段3 |
| 增量自训练 | 脱敏用户行为，持续优化模型 | 📋 阶段3 |
| 联邦学习 | 分布式训练，隐私保护 | 📋 阶段4 |

## 架构定位

```
cloudwubi-client（端侧 <800KB）
    ↓ 编码
cloudwubi-gateway（云端入口：查询 + 基础构词）
    ↓ 复杂语义请求
cloudwubi-ai（本仓库：AI 引擎，阶段3启用）
    ↓ 语义排序 / 自学习
返回优化后的候选
```

## 当前状态

- 本仓库为占位初始化（阶段1/2 尚未涉及 AI 引擎）
- 阶段2 的动态构词基础能力已在 `cloudwubi-gateway/phrase_engine.py` 落地
- 阶段3 启动时，本仓库将承载语义排序与自训练模块

## 技术方向

1. **不引入通用大模型**（体积/成本考量），训练**五笔专用轻量模型**
2. 用户行为数据**脱敏后才可回流**，原始输入文本永不离开端侧
3. 全部能力模块化部署，与端侧接口稳定兼容

## 许可

MIT License · 贡献规范见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 相关仓库

- [cloudwubi-client](https://github.com/zsdili/cloudwubi-client) - 端侧内核
- [cloudwubi-gateway](https://github.com/zsdili/cloudwubi-gateway) - 云端网关
- [cloudwubi-rules](https://github.com/zsdili/cloudwubi-rules) - 五笔规则库（去中心化共建）
