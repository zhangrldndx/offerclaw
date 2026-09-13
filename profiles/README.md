# OfferClaw · Persona 测试库

存放多份用户画像 JSON，用于：
1. **回归测试**：`tests/test_personas.py` 用 `pytest.mark.parametrize` 跑每个 persona × 多份 JD
2. **演示**：录 demo 时切换 persona 展示"同一套规则面对不同人输出不同结论"
3. **抽象化**：把 `match_job.DEMO_PROFILE` 写死的字典抽出来，证明匹配逻辑数据驱动

## 当前 personas

| 文件 | 描述 | 关键差异 |
|------|------|---------|
| `p1_demo_ai.json` | 合成的计算机科学硕士 AI 应用候选人 | 成都/重庆/远程；不对应真实个人 |
| `p2_cs_backend_intern.json` | 本科计算机找 Python 后端实习 | 有 1 实习 2 项目 |
| `p3_phd_algo_research.json` | AI 博士找算法岗（高门槛对照） | 全栈算法栈 |

## 字段约定

字段必须与 `match_job.DEMO_PROFILE` 完全对齐（key 不能漏，类型不能变），
否则 `run_match()` 会抛异常。校验方式：

```bash
python -m pytest tests/test_personas.py -v
```
