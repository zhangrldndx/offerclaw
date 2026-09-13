#!/usr/bin/env node
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";

import {
  createReplyDispatchHandler,
  invokeDispatcher,
  parsePluginConfig,
} from "../integrations/openclaw/offerclaw-direct-reply/runtime.ts";

const stateDir = process.env.OFFERCLAW_OPENCLAW_STATE_DIR
  || path.join(os.homedir(), ".openclaw-offerclaw-lab");
const root = JSON.parse(
  await fs.readFile(path.join(stateDir, "openclaw.json"), "utf8"),
);
const entry = root?.plugins?.entries?.["offerclaw-direct-reply"];
const config = parsePluginConfig(entry?.config);
if (!entry?.enabled || config.accountIds.length !== 1 || config.senderIds.length !== 1) {
  throw new Error("direct_reply_scope_not_ready");
}

let body = "";
for await (const chunk of process.stdin) body += chunk;
body = body.trim();
if (!body) throw new Error("request_text_required_on_stdin");

const replies = [];
const logs = [];
let dispatcherFailure = "";
const logger = {
  info: (message) => logs.push(["info", message]),
  warn: (message) => logs.push(["warn", message]),
  error: (message) => logs.push(["error", message]),
};
const handler = createReplyDispatchHandler(config, logger, async (pluginConfig, request) => {
  try {
    return await invokeDispatcher(pluginConfig, request);
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown";
    dispatcherFailure = [
      "launcher_not_configured", "dispatcher_timeout", "dispatcher_failed",
      "dispatcher_failed_with_stderr", "dispatcher_invalid_json",
    ].includes(message) ? message : "spawn_error";
    throw error;
  }
});
const event = {
  runId: "offerclaw-direct-reply-smoke",
  sessionKey: "offerclaw-direct-reply-smoke",
  originatingChannel: "openclaw-weixin",
  originatingAccountId: config.accountIds[0],
  originatingChatType: "direct",
  ctx: {
    Body: body,
    From: config.senderIds[0],
    AccountId: config.accountIds[0],
    OriginatingChannel: "openclaw-weixin",
    Provider: "openclaw-weixin",
    ChatType: "direct",
    SessionKey: "offerclaw-direct-reply-smoke",
    MessageSid: "offerclaw-direct-reply-smoke",
    Timestamp: Date.now(),
  },
};
const dispatcher = {
  sendFinalReply(payload) {
    replies.push(String(payload?.text ?? ""));
    return true;
  },
  markComplete() {},
  async waitForIdle() {},
  getQueuedCounts: () => ({ final: replies.length }),
};
const result = await handler(event, { dispatcher });
console.log(JSON.stringify({
  handled: result?.handled === true,
  queued_final: result?.queuedFinal === true,
  model_fallthrough: result === undefined,
  reply_chars: replies.reduce((total, text) => total + text.length, 0),
  local_log_observed: logs.some(([, message]) =>
    message.includes("offerclaw direct reply trace=")),
  errors: logs.filter(([level]) => level === "error").length,
  error_kinds: logs
    .filter(([level]) => level === "error")
    .map(([, message]) => message.match(/kind=([A-Za-z]+)/)?.[1] ?? "unknown"),
  dispatcher_failure: dispatcherFailure || null,
}));
