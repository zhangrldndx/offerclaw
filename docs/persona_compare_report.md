# Persona 泛化验证报告（阶段七）

**生成于：** 2026-04-25  
**目的：** 证明 OfferClaw 的匹配/缺口/建议链路不依赖单一硬编码 persona，可对多类候选人输出明显不同的结论。

---

## 一、测试设计

- **统一输入 JD（控制变量）：**
  ```
  岗位名称：大模型应用开发实习生
  工作地点：上海
  技术要求：Python / RAG / LangGraph / FastAPI / Prompt / Embedding
  ```
- **三类 persona（profiles/*.json）：**
  | ID | 描述 | 关键差异 |
  |---|---|---|
  | p1_demo_ai | 纯合成的计算机科学硕士 AI 应用候选人 | 只接受成都、重庆或远程，项目数 2 |
  | p2_cs_backend_intern | 纯合成的计算机本科 Python 后端候选人 | 有后端实习与工程技能 |
  | p3_phd_algo_research | 纯合成的 AI 博士算法候选人 | 方向偏预训练与算法研究 |

---

## 二、对比结果（来自 `match_job.run_match`）

| persona_file | 结论 | 缺口总数 | 缺口类别 | 建议条数 |
|---|---|---|---|---|
| p1_demo_ai | **当前暂不建议投递** | 3 | 硬门槛 / 技能 | 1 |
| p2_cs_backend_intern | **当前适合投递** | 3 | 硬门槛 / 经历 / 技能 | 2 |
| p3_phd_algo_research | **当前适合投递** | 3 | 硬门槛 / 经历 / 技能 | 2 |

> 结论分布出现 2 档（适合投递 / 暂不建议投递），证明同一 JD 在不同 persona 上**不会**输出相同结论。

---

## 三、关键观察

1. **p1 命中"当前暂不建议投递"** - 方向匹配，但合成画像的地域偏好与上海驻场岗位冲突。
2. **p2 / p3 命中"当前适合投递"** - 两个合成画像都接受上海，并分别以工程能力和研究能力覆盖岗位要求。
3. **缺口结构保持稳定**（硬门槛/经历/技能三大维度），证明数据契约（DATA_CONTRACT.md）与 match_job 的内部分类是稳定的。

---

## 四、再现命令

```powershell
$env:PYTHONIOENCODING="utf-8"
pytest tests/test_personas.py -v
```

或单测：
```python
from match_job import run_match
import json
persona = json.load(open("profiles/p1_demo_ai.json", encoding="utf-8"))
rep = run_match(persona, JD_TEXT, jd_title="LLM 实习/对比")
print(rep.conclusion, rep.gap_list, rep.suggestions)
```

---

## 五、结论

- OfferClaw 的匹配链路对 3 类合成 persona 输出不同结论与建议，覆盖多用户形态。
- `/api/profile` 通过本地、被 Git 忽略的画像文件读取用户数据；公开默认值来自 `profiles/p1_demo_ai.json`。
- `tests/test_personas.py::test_persona_matching_stable` 与 `test_persona_schema` 同时保护 schema 和合法结论。
