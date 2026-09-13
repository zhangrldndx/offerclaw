# OfferClaw V3 改进总结（Phase 1-7）

> 🗄️ **V3 发布记录 · 历史快照**。文中路由/测试/chunks 等为 **V3 时点值**，非当前口径——当前真值见 [`verification_report.md`](verification_report.md) 与 [`../README.md`](../README.md)。

> 对应指导文档：[`OfferClaw_产品级Agent化修改指导.md`](../OfferClaw_产品级Agent化修改指导.md)
> 时间窗口：V2 收口完成 → V3 七阶段全部交付
> 仓库：<https://github.com/zhangrldndx/offerclaw>

---

## 1. 一句话总览

OfferClaw 从「6 张功能卡片 + 19 路由」升级为「**8 节点 LangGraph 编排 + 24 路由 + Stepper 一页跑完整求职流程**」，全程零硬编码画像、零自动落盘、零 LLM 依赖即可跑通主回路。

---

## 2. 七个阶段的核心变更

| 阶段 | 主题 | 关键交付 | 验收 |
|---|---|---|---|
| 1 | **状态真实化** | `profile_loader.py`：13 字段 Markdown 解析 + mtime 缓存；解耦 `/api/match` 等生产链路对 `match_job.DEMO_PROFILE` 的依赖（仅保留为单测 fixture） | `/api/match` 直读真画像；编辑 `user_profile.md` 即可改变同一 JD 的结论 |
| 2 | **CareerFlow 编排化** | `career_flow.py`：8 节点 LangGraph（profile → job_input → match → gap → plan → today → resume → application_suggest）；`/api/flow/run` | 主流程**禁止落盘**，所有写意图收在 `requires_confirmation`；`skip_llm=True` 时无需 `ZHIPU_API_KEY` |
| 3 | **UI 产品化** | `static/console.html` + `/ui/console`：8 步 Stepper 一页跑完整流程 | 自动化 UI 与接口测试 |
| 4 | **JD Discovery 增强** | `build_search_queries(profile)` + `rank_candidates(jds)` + `/api/jd/queries` + `/api/jd/rank` | 候选 JD 按"当前适合 / 中长期 / 信息不足 / 暂不建议"四档归并；不爬虫、不自动投递 |
| 5 | **Resume 完整化** | `build_resume_markdown()` 6 段 MD 草稿 + `/api/resume/markdown` | 默认无 LLM；可选叠加 `/api/resume/build` 的 LLM 项目段 |
| 6 | **RAG 知识域扩展** | `rag_ingest.DEFAULT_FILES` 加入 `docs/verification_report.md`（`source_type=verification`） | 9 类 source_type：profile / log / system / doc / application / story / jd / resume / **verification** |
| 7 | **端到端验证** | `tests/test_phase3_to_7.py::test_phase7_careerflow_answers_4_core_questions` + 合成 JD 回归 | CareerFlow 能一次性回答 §10.3 四个核心问题，且不发布真实求职数据 |

---

## 3. 测试结果（按文件粒度）

| 测试文件 | 用例数 | 结果 | 涵盖阶段 |
|---|---|---|---|
| `tests/test_offerclaw_core.py` | 22 | ✅ 全部通过 | V1-V2 基线 |
| `tests/test_pipeline.py` | 7 | ✅ 全部通过 | RAG / Plan / Summary |
| `tests/test_personas.py` | 6 | ✅ 全部通过（3 e2e 默认 skipped） | 多 Persona |
| `tests/test_api.py` | 9 | ✅ 全部通过 | FastAPI |
| `tests/test_profile_loader.py` | 7 | ✅ 全部通过 | **V3 阶段 1** |
| `tests/test_career_flow.py` | 5 | ✅ 全部通过 | **V3 阶段 2** |
| `tests/test_phase3_to_7.py` | 9 | ✅ 全部通过 | **V3 阶段 3-7** |
| **合计** | **58 passed / 3 skipped** | ✅ | — |

**工程体检**

| 检查 | 结果 |
|---|---|
| `python doctor.py` | 10 OK · 0 WARN · 0 ERR |
| `python verify_pipeline.py` | 6/6 通过 |
| `/health` | `{"status":"healthy","chroma_db":"connected","collection_records":160}` |

---

## 4. 路由变化

| 类别 | V2 数量 | V3 数量 | 增量 |
|---|---|---|---|
| API 路由 | 16 | **20** | +4 |
| UI 入口 | 1 | **2** | +1（`/ui/console`）|
| 系统 | 3 | 3 | — |
| **Swagger 显示** | 19 | **24** | +5 |

**V3 新增 API**

```text
POST /api/flow/run           CareerFlow 主编排（8 节点全状态）
GET  /api/jd/queries         根据 profile 生成搜索关键词组合
POST /api/jd/rank            对一组候选 JD 调 match_job 排序
POST /api/resume/markdown    完整 Markdown 简历草稿（默认无 LLM）
GET  /ui/console             CareerFlow Stepper 控制台
```

---

## 5. 不变量与边界

V3 严格保持 V2 已建立的边界：

- ❌ 不自动登录招聘平台
- ❌ 不批量爬虫
- ❌ 不自动投递
- ❌ 不伪造经历 / 不夸大指标
- ❌ 不承诺录用概率
- ✅ CareerFlow 主流程**禁止落盘**：所有写入意图必须经 `requires_confirmation` 让用户确认
- ✅ `skip_llm=True` 时全程不读 `ZHIPU_API_KEY`，便于 CI / 离线 demo
- ✅ `ZHIPU_API_KEY` 只放 `.env.local`（已 `.gitignore`），doctor.py 自动加载并掩码显示
- ✅ 用 Markdown 文件做"运行时状态"，不引数据库 / 不引登录系统

---

## 6. 提交记录

| Commit | 阶段 | 描述 |
|---|---|---|
| historical | Phase 1 | feat(state-realization): profile_loader + drop DEMO_PROFILE |
| historical | Phase 2 | feat(career-flow): LangGraph orchestrator + /api/flow/run |
| historical | Phase 3-7 | feat(v3 phases 3-7): UI Stepper + JD ranker + Resume MD + RAG verification + acceptance tests |

---

## 7. 仍未做的（按指导明确判定）

- **真实投递场景验证（§10.2）**：需要用户线下完成实际投递 ≥ 1 次。代码侧已就绪，等用户使用数据回流。
- **生产级商业化**：明确不做。OfferClaw 定位个人 AI Agent。

---

## 8. 接下来该做的（建议优先级）

1. 用 `/ui/console` 跑一份合成目标 JD，真实数据仅在本地验证
2. 周复盘时把 `/api/flow/run` 输出的 `requires_confirmation` 中合理 patch 写回 `applications.md`
3. 等 RAG 知识库真实积累 daily_log / interview_story_bank 后，对 §9.4 六个跨文档问题做一次 50 题人工标注复评
