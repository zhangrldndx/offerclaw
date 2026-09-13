# 微信实验知识：安全状态刷新

OfferClaw 的状态刷新采用 failure-safe replace。新内容完成分块、向量化和写入后，
才删除同一来源的旧 chunk。某个来源失败时，其旧 chunk 仍可被检索，其他来源继续处理，
整个命令返回 partial。这个实验知识只用于验证微信知识库问答。
