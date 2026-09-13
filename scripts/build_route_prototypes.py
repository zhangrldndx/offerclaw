# -*- coding: utf-8 -*-
"""预生成顶部 RAG 路由原型向量缓存；不读取任何个人数据。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_route_registry import prebuild_route_prototype_cache


if __name__ == "__main__":
    print(json.dumps(prebuild_route_prototype_cache(), ensure_ascii=False, indent=2))
