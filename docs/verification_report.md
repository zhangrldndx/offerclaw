# OfferClaw 公开发布验证报告

> 本报告只记录陌生人从公开 Git tree 可复现的检查。
> 不记录真实 API/网关配置、凭据元数据、主机身份、个人画像、投递、日志、故事库或私有评测原文。

## 1. 发布基线

| 检查 | 公开基线 |
|---|---|
| Python 测试 | **1,659 passed / 54 skipped / 0 failed** |
| FastAPI 路由 | **91** |
| 产品合同验收 | **36/36** |
| 隐私门禁 | `python verify_docs.py` 无违规 |
| 当前树 Secret 扫描 | Gitleaks 无命中 |
| CI 历史扫描 | checkout 完整历史后运行 Gitleaks |

`54 skipped` 只包括显式登记的外部 LLM/E2E、私有评测证据、本地索引/重排模型和慢速语义评测。
默认公开测试不会读取本地 `.env.local`、根目录用户文件或被忽略的评测产物。

## 2. 复现命令

在全新克隆中运行：

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q -rs
python verify_docs.py
gitleaks dir --redact .
```

发布者还应对完整可达 Git 历史运行 Gitleaks。CI 使用 `fetch-depth: 0`，避免浅克隆漏过旧提交。

## 3. 隔离要求

- 不设置 `OFFERCLAW_PRIVATE_EVAL`，私有评测证据测试必须跳过。
- 不设置真实 LLM/E2E 开关，不向外部服务发送测试内容。
- 不复制本地 `user_profile.md`、`applications.md`、`daily_log.md`、故事库、JD 或知识库。
- 不复制 `.env*`、Secret Store、DPAPI 文件、日志、截图、向量库或工作流产物。
- 只允许显式标注为合成数据的公开 fixtures。

## 4. 隐私门禁覆盖

`verify_docs.py` 会检查：

- 被误加到 Git 的私人运行路径；
- 非回环 IP 服务地址和未批准的配置端点；
- Windows、WSL、macOS 与 Linux 用户绝对路径；
- 本地 `.env.local` 中敏感值是否出现在跟踪文件；
- 本地私人画像字段是否出现在跟踪文件；
- 私有代理别名、掩码凭据长度和真实部署验收元数据；
- 关键文档指标是否与 `metrics.json` 漂移。

Gitleaks 作为第二道门禁，检测常见密钥、令牌和高熵凭据。两者都通过仍不能替代凭据轮换：
任何曾经公开过的凭据都必须在服务端撤销或更换。

## 5. 可选本地验证

本地索引、真实 LLM、微信渠道和私有评测包可在受保护环境中单独验证，但结果不得原样提交。
公开文档只保留可复现的测试合同和汇总指标；端点、模型路由、账号标识、数据规模、响应样本和
机器性能记录均留在 Git 之外。
