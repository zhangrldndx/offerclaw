import assert from "node:assert/strict";
import test from "node:test";

import {
  createReplyDispatchHandler,
  parsePluginConfig,
} from "../integrations/openclaw/offerclaw-direct-reply/runtime.ts";

const config = parsePluginConfig({
  launcherPath: "/safe/offerclaw-launcher",
  accountIds: ["account-1"],
  senderIds: ["self@im.wechat"],
  timeoutMs: 45_000,
  publicLlmFallback: true,
});

function event(overrides = {}) {
  return {
    runId: "run-1",
    sessionKey: "session-1",
    originatingChannel: "openclaw-weixin",
    originatingAccountId: "account-1",
    originatingChatType: "direct",
    ctx: {
      MessageSid: "message-1",
      SessionKey: "session-1",
      SenderId: "self@im.wechat",
      AccountId: "account-1",
      ChatType: "direct",
      Provider: "openclaw-weixin",
      Timestamp: 1789238400000,
      Body: "把我当前的画像给我",
      ...overrides,
    },
  };
}

function harness({ deliveryFailure = false } = {}) {
  const replies = [];
  const blocks = [];
  const logs = [];
  return {
    replies,
    blocks,
    logs,
    logger: {
      info: (message) => logs.push(["info", message]),
      warn: (message) => logs.push(["warn", message]),
      error: (message) => logs.push(["error", message]),
    },
    context: {
      dispatcher: {
        sendBlockReply(payload) {
          if (deliveryFailure) throw new Error("delivery failed");
          blocks.push(payload.text);
          return true;
        },
        sendFinalReply(payload) {
          if (deliveryFailure) throw new Error("delivery failed");
          replies.push(payload.text);
          return true;
        },
        markComplete() {},
        async waitForIdle() {},
        getQueuedCounts: () => ({ final: replies.length }),
      },
    },
  };
}

function response(overrides = {}) {
  return {
    schema_version: "offerclaw.wechat.response.v1",
    status: "ok",
    handled: true,
    reply_text: "Windows 真实数据结果",
    route: { capability_id: "profile.read" },
    privacy: { classification: "sensitive", model_exposure: "none" },
    trace_id: "trace-random",
    ...overrides,
  };
}

test("local success is returned exactly and claims dispatch", async () => {
  const h = harness();
  let request;
  const handler = createReplyDispatchHandler(config, h.logger, async (_config, value) => {
    request = value;
    return response();
  });
  const result = await handler(event(), h.context);
  assert.equal(result.handled, true);
  assert.deepEqual(h.replies, ["Windows 真实数据结果"]);
  assert.equal(request.schema_version, "offerclaw.wechat.request.v1");
  assert.equal(request.sender_id, "self@im.wechat");
  assert.equal(request.text, "把我当前的画像给我");
  assert.deepEqual(request.reply_to, { id: "", body: "" });
  assert.equal("reply_to_message_id" in request, false);
  assert.match(request.message_id, /^weixin_[0-9a-f]{32}$/);
});

test("only a public no-attachment miss falls through", async () => {
  const h = harness();
  const handler = createReplyDispatchHandler(config, h.logger, async () =>
    response({
      handled: false,
      reply_text: "",
      privacy: { classification: "public", model_exposure: "openclaw" },
    }),
  );
  assert.equal(await handler(event({ Body: "解释一下 TCP" }), h.context), undefined);
  assert.deepEqual(h.replies, []);
});

test("redelivery produces a stable local idempotency message id", async () => {
  const h = harness();
  const ids = [];
  const handler = createReplyDispatchHandler(config, h.logger, async (_config, request) => {
    ids.push(request.message_id);
    return response();
  });
  await handler(event(), h.context);
  await handler(event(), h.context);
  const changed = event({ Timestamp: 1789238400001 });
  await handler(changed, h.context);
  assert.equal(ids.length, 2);
  assert.notEqual(ids[0], ids[1]);
  assert.deepEqual(h.replies, [
    "Windows 真实数据结果",
    "Windows 真实数据结果",
    "Windows 真实数据结果",
  ]);
});

test("long local answers are delivered in ordered blocks without rewriting", async () => {
  const h = harness();
  const handler = createReplyDispatchHandler(config, h.logger, async () =>
    response({
      reply_text: "第一段\n第二段\n第三段",
      reply_parts: ["第一段", "第二段", "第三段"],
    }),
  );
  const result = await handler(event({ Timestamp: 1789238400020 }), h.context);
  assert.equal(result.queuedBlocks, 2);
  assert.deepEqual(h.blocks, ["第一段", "第二段"]);
  assert.deepEqual(h.replies, ["第三段"]);
});

test("a sensitive miss is claimed locally", async () => {
  const h = harness();
  const handler = createReplyDispatchHandler(config, h.logger, async () =>
    response({ handled: false, reply_text: "敏感请求已停止" }),
  );
  const result = await handler(event(), h.context);
  assert.equal(result.handled, true);
  assert.deepEqual(h.replies, ["敏感请求已停止"]);
});

test("unstaged or remote-only media never reaches the dispatcher", async () => {
  const h = harness();
  let called = false;
  const handler = createReplyDispatchHandler(config, h.logger, async () => {
    called = true;
    return response();
  });
  const result = await handler(
    event({ MediaUrl: "https://example.invalid/private.pdf" }),
    h.context,
  );
  assert.equal(result.handled, true);
  assert.equal(called, false);
  assert.match(h.replies[0], /没有完成安全暂存/);
});

test("dispatcher crashes and delivery failures stay fail-closed", async () => {
  const h = harness({ deliveryFailure: true });
  const handler = createReplyDispatchHandler(config, h.logger, async () => {
    throw new Error("child crashed");
  });
  const result = await handler(event(), h.context);
  assert.equal(result.handled, true);
  assert.equal(result.queuedFinal, false);
  assert.ok(h.logs.some(([level]) => level === "error"));
});

test("wrong sender and group chats are ignored without local access", async () => {
  const h = harness();
  let calls = 0;
  const handler = createReplyDispatchHandler(config, h.logger, async () => {
    calls += 1;
    return response();
  });
  assert.equal(await handler(event({ SenderId: "other@im.wechat" }), h.context), undefined);
  const group = event({ ChatType: "group" });
  group.originatingChatType = "group";
  assert.equal(await handler(group, h.context), undefined);
  assert.equal(calls, 0);
});
