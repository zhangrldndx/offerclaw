# OfferClaw x 微信（OpenClaw 渠道层）

> 本文是可公开复现的部署与安全契约，只描述仓库中的通用配置。
> 真实账号、路由标识、私有端点、模型标识、主机清单、验收输出和用户数据不得写入本文或 Git。

## 1. 责任边界

OfferClaw 提供路由、业务服务、格式化和本地检索；OpenClaw 负责微信收发、身份校验、
公开问题的可选模型兜底和任务调度。微信与 Web 共用业务模块，不抓取 Web UI 文本。

```text
微信私聊
  -> openclaw-weixin
  -> offerclaw-direct-reply（模型调用之前）
  -> 固定 launcher: wechat-dispatch --stdin
  -> 本地数据桥 / 可重建索引
  -> 确定性文本返回微信
```

画像、投递、日志、计划、JD 和知识库始终来自 Git 忽略的本地 User Layer。实验环境中的
索引和 manifest 都是派生状态，删除后可由用户明确指定的本地来源重建。仓库中的
`integrations/openclaw/lab-fixtures/` 只含逐文件标注的虚构数据。

## 2. 公开基线

`setup_wechat.sh` 固定 OpenClaw、Node、Torch 和微信插件版本，以便重建环境。模型相关值不在
仓库固定，部署者必须在本机显式提供：

| 变量 | 用途 |
|---|---|
| `OFFERCLAW_MODEL_BASE_URL` | OpenAI-compatible 服务地址 |
| `OFFERCLAW_MODEL_ID` | 服务端支持的模型 ID |
| `OFFERCLAW_MODEL_PROVIDER_ID` | 本地 OpenClaw provider 名称，默认 `offerclaw-local` |
| `OFFERCLAW_MODEL_NAME` | 可选显示名 |
| `OFFERCLAW_MODEL_CONTEXT_WINDOW` | 可选上下文窗口 |
| `OFFERCLAW_MODEL_MAX_TOKENS` | 可选最大输出 token |
| `OFFERCLAW_MODEL_REF` | 可选完整 provider/model 覆盖 |

这些变量属于本地运行配置。提交前应运行 `python verify_docs.py` 和 Gitleaks；真实地址不得写入
脚本、文档、命令示例、截图或 CI 日志。

实验夹具默认设置 `RAG_RERANK=0`，避免首次问答下载大型精排模型；向量/BM25 检索仍可用于
合成知识库测试。正式运行是否开启精排由本地配置决定。

## 3. 安全配置

- `OPENAI_API_KEY` 只存 OpenClaw Secret Store，并通过 SecretRef 供 provider 与 Skill 使用。
- Key 不写入仓库、`openclaw.json`、命令参数、聊天、截图或验证报告。
- 专用 Agent 默认不开放终端、文件、内存、Secret、会话和自动化工具。
- 固定 launcher 不使用 `eval`，只接受登记过的子命令与结构化 stdin。
- 微信默认使用私聊配对，群聊关闭；账号和发送者范围必须在本机精确确认。
- 写操作先生成字段预览与短期 action ID；只有同一账号、发送者和会话的明确确认才能提交。
- 需要携带私人材料的远程生成能力默认关闭；公开兜底不得收到本地结果、路径、附件或画像。
- 自动任务只能生成建议和刷新可重建索引，不能自动修改画像、投递、计划等权威数据。
- OpenClaw 状态目录、launcher、令牌与用户文件应使用最小文件权限并纳入主机备份保护。

Secret Store 的访问控制不等同于磁盘加密。渠道路由、任务状态和最近投递摘要属于本地敏感
运行状态；需要更强静态保护时，应使用磁盘加密或受控状态存储。

## 4. 创建隔离实验副本

从仓库执行：

```bash
wsl -d Ubuntu -- bash /mnt/c/Users/<user>/Desktop/offerclaw/scripts/create_openclaw_lab.sh
```

脚本创建一次性代码快照，并排除根目录用户文件、知识库、向量库、环境文件、缓存和虚拟环境。
复制规则若发生变化，必须同步更新隐私测试。

## 5. 部署

先检查操作，不写配置：

```bash
wsl -d Ubuntu -- env \
  OFFERCLAW_DIR="$HOME/.local/share/offerclaw-lab/app" \
  OFFERCLAW_MODEL_BASE_URL='<OpenAI-compatible base URL>' \
  OFFERCLAW_MODEL_ID='<model id>' \
  bash "$HOME/.local/share/offerclaw-lab/app/setup_wechat.sh" --dry-run
```

确认输出中没有私人值后，再移除 `--dry-run`。根脚本是唯一任务定义；
`scripts/setup_openclaw_cron.sh` 只委托它的 `--cron-only` 模式。

## 6. 人工步骤

在可见终端的无回显提示中录入 Key：

```bash
openclaw --profile offerclaw-lab secrets store set OPENAI_API_KEY
```

随后运行 Secret 审计与配置校验。任何输出都不应包含明文、前后缀、长度或可关联的凭据元数据。

登录微信渠道：

```bash
openclaw --profile offerclaw-lab channels login --channel openclaw-weixin
```

扫码后只在本机确认目标账号与发送者；不要把标识复制到 issue、提交、截图或文档。群聊保持关闭。

## 7. 自动任务

仓库定义早间建议、晚间留痕提醒和周摘要三类 command 任务。它们默认 disabled，使用固定 argv
启动 launcher，结构化请求走 stdin，且不创建 Agent turn。只有用户另行明确确认后才能传
`--enable-jobs`。

本机目标通过环境变量提供：

```bash
export OFFERCLAW_WECHAT_TO='<locally confirmed private target>'
bash setup_wechat.sh --cron-only
openclaw --profile offerclaw-lab automations list --all --json
```

不要把命令输出提交到 Git，因为任务 ID、渠道目标和投递摘要都可能关联个人账号。

## 8. 合成验收场景

只使用 `integrations/openclaw/lab-fixtures/` 中的虚构画像、投递、日志、计划和附件验证：

| 场景 | 预期边界 |
|---|---|
| 查询画像、计划、日志 | 只读取实验副本中的合成文件 |
| 知识问答 | 返回答案与公开来源，不读取根目录用户文件 |
| 新建投递或留痕 | 先返回预览，不立即写入 |
| 确认或拒绝 action | 校验身份、会话、期限与 revision，保证幂等 |
| 删除投递 | 明确拒绝，不创建 action |
| 上传文档 | 只进入权限受限的临时区，不自动进入正式索引 |
| 公开问题兜底 | 不附带本地证据、路径或附件 |

## 9. 发布前验收

1. 在一次性实验副本中运行 Python 与 Node 测试。
2. 确认根目录私人文件、环境文件、知识库和索引均未进入副本。
3. 验证写操作默认只生成预览，重复确认不会重复写入。
4. 验证附件路径限制、扩展名限制、大小限制和过期清理。
5. 验证 Agent 看不到终端、文件、内存和 Secret 工具。
6. 保持自动任务 disabled，除非用户单独批准启用。
7. 对最终 Git tree 和完整可达历史运行隐私门禁与 Gitleaks。

真实环境验收结果只保存在本地受保护记录中。公开仓库只能声明上述合同与自动化测试结果，
不能声明某个真实账号、端点、模型或私人数据集已经通过验收。
