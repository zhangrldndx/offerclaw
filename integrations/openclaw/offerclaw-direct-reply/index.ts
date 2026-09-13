import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import {
  PLUGIN_VERSION,
  createReplyDispatchHandler,
  parsePluginConfig,
} from "./runtime.ts";

export default definePluginEntry({
  id: "offerclaw-direct-reply",
  name: "OfferClaw Direct Reply",
  description:
    "Routes scoped OpenClaw Weixin direct messages to the local OfferClaw dispatcher before any model call.",
  register(api) {
    const config = parsePluginConfig(api.pluginConfig);

    api.on(
      "reply_dispatch",
      createReplyDispatchHandler(config, api.logger),
      {
        eligibleDispatchKinds: ["agent"],
        timeoutMs: 45_000,
      },
    );

    api.logger.info(
      `offerclaw direct reply registered version=${PLUGIN_VERSION} hook=reply_dispatch`,
    );
  },
});
