import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import path from "node:path";
import process from "node:process";

export const PLUGIN_VERSION = "1.1.0";
const REQUEST_SCHEMA = "offerclaw.wechat.request.v1";
const RESPONSE_SCHEMA = "offerclaw.wechat.response.v1";
const EXPECTED_CHANNEL = "openclaw-weixin";
const MAX_CHILD_BUDGET_MS = 45_000;
const MESSAGE_CACHE_TTL_MS = 15 * 60_000;
const MESSAGE_CACHE_MAX = 512;

export type PluginConfig = {
  launcherPath: string;
  accountIds: string[];
  senderIds: string[];
  timeoutMs: number;
  allowPublicModelFallback: boolean;
};

type Logger = {
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
};

type InvokeRequest = (
  config: PluginConfig,
  request: Record<string, unknown>,
) => Promise<any>;

export function parsePluginConfig(raw: unknown): PluginConfig {
  const value = (raw ?? {}) as Record<string, unknown>;
  const timeout = Number(value.timeoutMs ?? MAX_CHILD_BUDGET_MS);
  const fallback = value.publicLlmFallback ?? value.allowPublicModelFallback;
  return {
    launcherPath: String(value.launcherPath ?? "").trim(),
    accountIds: normalizeIds(value.accountIds),
    senderIds: normalizeIds(value.senderIds),
    timeoutMs: Math.min(
      Number.isFinite(timeout) && timeout > 0 ? timeout : MAX_CHILD_BUDGET_MS,
      MAX_CHILD_BUDGET_MS,
    ),
    allowPublicModelFallback: fallback !== false,
  };
}

function normalizeIds(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => String(entry ?? "").trim().toLowerCase())
    .filter(Boolean);
}

function firstText(...values: unknown[]): string {
  for (const value of values) {
    const text = String(value ?? "").trim();
    if (text) return text;
  }
  return "";
}

function requestScope(event: any) {
  const message = event?.ctx ?? {};
  const channel = firstText(
    event?.originatingChannel,
    message.OriginatingChannel,
    message.Provider,
  ).toLowerCase();
  const accountId = firstText(event?.originatingAccountId, message.AccountId);
  const senderId = firstText(message.SenderId, message.From);
  const chatType = firstText(
    event?.originatingChatType,
    message.ChatType,
  ).toLowerCase();
  return { message, channel, accountId, senderId, chatType };
}

export function isInScope(config: PluginConfig, event: any): boolean {
  const scope = requestScope(event);
  return (
    scope.channel === EXPECTED_CHANNEL &&
    scope.chatType === "direct" &&
    config.accountIds.length === 1 &&
    config.senderIds.length === 1 &&
    config.accountIds.includes(scope.accountId.toLowerCase()) &&
    config.senderIds.includes(scope.senderId.toLowerCase())
  );
}

type MediaCollection = {
  present: boolean;
  valid: boolean;
  media: Array<Record<string, unknown>>;
};

export function collectMedia(message: any): MediaCollection {
  const structured = Array.isArray(message?.media) ? message.media : [];
  const legacyPaths = Array.isArray(message?.MediaPaths)
    ? message.MediaPaths
    : message?.MediaPath
      ? [message.MediaPath]
      : [];
  const remoteOnly = Boolean(
    message?.MediaUrl ||
      (Array.isArray(message?.MediaUrls) && message.MediaUrls.length > 0),
  );
  const present = structured.length > 0 || legacyPaths.length > 0 || remoteOnly;
  const facts =
    structured.length > 0
      ? structured
      : legacyPaths.map((mediaPath: unknown) => ({
          path: mediaPath,
          contentType: message?.MediaType,
          workspaceDir: message?.MediaWorkspaceDir,
          messageId: message?.MessageSidFull ?? message?.MessageSid,
        }));

  const media: Array<Record<string, unknown>> = [];
  for (const fact of facts) {
    const localPath = firstText(fact?.path);
    if (!localPath || !path.isAbsolute(localPath)) {
      return { present, valid: false, media: [] };
    }
    const workspaceDir = firstText(
      fact?.workspaceDir,
      message?.MediaWorkspaceDir,
      path.dirname(localPath),
    );
    if (!workspaceDir || !path.isAbsolute(workspaceDir)) {
      return { present, valid: false, media: [] };
    }
    media.push({
      path: localPath,
      workspace_dir: workspaceDir,
      content_type: firstText(fact?.contentType, fact?.mimeType),
      kind: firstText(fact?.kind, fact?.type),
      message_id: firstText(
        fact?.messageId,
        message?.MessageSidFull,
        message?.MessageSid,
      ),
    });
  }

  if (remoteOnly && media.length === 0) {
    return { present, valid: false, media: [] };
  }
  return { present, valid: true, media };
}

function stableMessageId(
  event: any,
  media: Array<Record<string, unknown>>,
): string {
  const scope = requestScope(event);
  const message = scope.message;
  const timestamp = firstText(message.Timestamp);
  if (!timestamp) {
    return firstText(message.MessageSidFull, message.MessageSid, event?.runId);
  }
  const digest = createHash("sha256")
    .update(JSON.stringify([
      scope.channel,
      scope.accountId,
      scope.senderId,
      timestamp,
      firstText(message.CommandBody, message.Body, message.BodyForAgent),
      firstText(message.ReplyToId),
      media.map((item) => item.path),
    ]))
    .digest("hex");
  return `weixin_${digest.slice(0, 32)}`;
}

function buildRequest(event: any, media: Array<Record<string, unknown>>) {
  const scope = requestScope(event);
  const message = scope.message;
  return {
    schema_version: REQUEST_SCHEMA,
    message_id: stableMessageId(event, media),
    conversation_id: firstText(message.SessionKey, event?.sessionKey),
    channel: scope.channel,
    account_id: scope.accountId,
    sender_id: scope.senderId,
    is_group: false,
    text: firstText(message.CommandBody, message.Body, message.BodyForAgent),
    reply_to: {
      id: firstText(message.ReplyToId),
      body: firstText(message.ReplyToBody, message.ReplyToContent),
    },
    media,
    trigger: "user",
  };
}

function safeErrorKind(error: unknown): string {
  if (error instanceof Error && error.name) return error.name;
  return "unknown";
}

function killProcessTree(child: any): void {
  if (!child?.pid) return;
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
        windowsHide: true,
        stdio: "ignore",
      });
    } else {
      process.kill(-child.pid, "SIGKILL");
    }
  } catch {
    try {
      child.kill("SIGKILL");
    } catch {
      // The process already exited.
    }
  }
}

function scrubbedEnvironment(): NodeJS.ProcessEnv {
  const allowed = [
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONUTF8",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TZ",
    "USER",
    "USERNAME",
    "WINDIR",
    "WSL_DISTRO_NAME",
  ];
  const env: NodeJS.ProcessEnv = {};
  for (const key of allowed) {
    if (process.env[key]) env[key] = process.env[key];
  }
  return env;
}

export function invokeDispatcher(
  config: PluginConfig,
  request: Record<string, unknown>,
): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!config.launcherPath) {
      reject(new Error("launcher_not_configured"));
      return;
    }

    const child = spawn(
      config.launcherPath,
      ["wechat-dispatch", "--stdin"],
      {
        shell: false,
        detached: process.platform !== "win32",
        windowsHide: true,
        stdio: ["pipe", "pipe", "pipe"],
        env: scrubbedEnvironment(),
      },
    );
    let stdout = "";
    let stderr = "";
    let finished = false;
    const timeout = setTimeout(() => {
      if (finished) return;
      finished = true;
      killProcessTree(child);
      reject(new Error("dispatcher_timeout"));
    }, Math.min(config.timeoutMs, MAX_CHILD_BUDGET_MS));

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      if (stderr.length < 4096) stderr += chunk;
    });
    child.on("error", (error: Error) => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", (code: number | null) => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(stderr ? "dispatcher_failed_with_stderr" : "dispatcher_failed"));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch {
        reject(new Error("dispatcher_invalid_json"));
      }
    });

    child.stdin.end(`${JSON.stringify(request)}\n`, "utf8");
  });
}

async function sendLocalReply(
  dispatchContext: any,
  logger: Logger,
  text: string,
  requestedParts?: unknown,
) {
  const dispatcher = dispatchContext?.dispatcher;
  let queuedFinal = false;
  let queuedBlocks = 0;
  if (!dispatcher) {
    logger.error("offerclaw direct reply delivery status=missing_dispatcher");
    return { handled: true, queuedFinal: false, counts: {} };
  }
  try {
    const parts = Array.isArray(requestedParts)
      ? requestedParts.map((part) => String(part ?? "").trim()).filter(Boolean)
      : [];
    const completeParts = parts.length > 0 ? parts : [text];
    for (const part of completeParts.slice(0, -1)) {
      if (typeof dispatcher.sendBlockReply !== "function") {
        throw new Error("block_reply_unavailable");
      }
      if (dispatcher.sendBlockReply({ text: part })) queuedBlocks += 1;
    }
    queuedFinal = Boolean(
      dispatcher.sendFinalReply({ text: completeParts.at(-1) ?? text }),
    );
    dispatcher.markComplete();
    await dispatcher.waitForIdle();
  } catch (error) {
    logger.error(
      `offerclaw direct reply delivery status=failed kind=${safeErrorKind(error)}`,
    );
  }
  let counts = {};
  try {
    counts = dispatcher.getQueuedCounts();
  } catch {
    // Delivery is still claimed so a failed local send cannot fall through to a model.
  }
  return { handled: true, queuedFinal, queuedBlocks, counts };
}

function validResponse(result: any): boolean {
  return (
    result &&
    result.schema_version === RESPONSE_SCHEMA &&
    typeof result.handled === "boolean"
  );
}

export function createReplyDispatchHandler(
  config: PluginConfig,
  logger: Logger,
  invoke: InvokeRequest = invokeDispatcher,
) {
  const messageCache = new Map<
    string,
    { expiresAt: number; result: Promise<any> }
  >();

  const cachedInvoke = (
    request: Record<string, unknown>,
  ): Promise<any> => {
    const now = Date.now();
    for (const [key, entry] of messageCache) {
      if (entry.expiresAt <= now) messageCache.delete(key);
    }
    const key = String(request.message_id ?? "");
    const cached = messageCache.get(key);
    if (cached && cached.expiresAt > now) return cached.result;
    const result = invoke(config, request).catch((error) => {
      messageCache.delete(key);
      throw error;
    });
    messageCache.set(key, { expiresAt: now + MESSAGE_CACHE_TTL_MS, result });
    while (messageCache.size > MESSAGE_CACHE_MAX) {
      const oldest = messageCache.keys().next().value;
      if (oldest === undefined) break;
      messageCache.delete(oldest);
    }
    return result;
  };

  return async (event: any, dispatchContext: any) => {
    if (!isInScope(config, event)) return undefined;

    const media = collectMedia(event?.ctx ?? {});
    if (!media.valid) {
      logger.warn("offerclaw direct reply status=rejected reason=unstaged_media");
      return sendLocalReply(
        dispatchContext,
        logger,
        "附件没有完成安全暂存，本次未读取附件，也未调用远程模型。请重新发送附件后再试。",
      );
    }

    try {
      const result = await cachedInvoke(buildRequest(event, media.media));
      if (!validResponse(result)) throw new Error("invalid_dispatcher_response");
      const route = result.route ?? {};
      const privacy = result.privacy ?? {};
      logger.info(
        `offerclaw direct reply trace=${firstText(result.trace_id, "none")} status=${firstText(result.status, "unknown")} capability=${firstText(route.capability_id, "none")} model_exposure=${firstText(privacy.model_exposure, "none")}`,
      );

      if (result.handled) {
        return sendLocalReply(
          dispatchContext,
          logger,
          firstText(result.reply_text, "本地请求已处理，但没有可显示的结果。"),
          result.reply_parts,
        );
      }

      const publicFallbackAllowed =
        config.allowPublicModelFallback &&
        !media.present &&
        privacy.classification === "public" &&
        privacy.model_exposure === "openclaw";
      if (publicFallbackAllowed) return undefined;

      return sendLocalReply(
        dispatchContext,
        logger,
        firstText(
          result.reply_text,
          "这个请求可能涉及本地或个人数据，已停止处理，且没有发送给远程模型。",
        ),
      );
    } catch (error) {
      logger.error(
        `offerclaw direct reply status=failed kind=${safeErrorKind(error)}`,
      );
      return sendLocalReply(
        dispatchContext,
        logger,
        "OfferClaw 本地服务暂时不可用。为保护数据，本次请求没有交给远程模型。",
      );
    }
  };
}
