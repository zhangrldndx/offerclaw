"""A8 工具 schema ↔ 实现一致性自动校验（把工具漂移从运行时 TypeError 左移到 CI）。

校验：①schema.required 每字段都是 fn 形参；②fn 无默认值的形参都在 required；
③schema.properties ⊆ fn 形参。两套注册体系（tools_registry.REGISTRY + tools.py 双结构）都覆盖。
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _check(name, fn, schema_params):
    sig = inspect.signature(fn)
    formal = {p for p in sig.parameters if p != "self"}
    props = set((schema_params or {}).get("properties", {}).keys())
    required = set((schema_params or {}).get("required", []))
    no_default = {p for p, v in sig.parameters.items()
                  if v.default is inspect.Parameter.empty and p != "self"
                  and v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
    assert required <= formal, f"[{name}] required 字段 {required - formal} 不在 fn 签名里"
    assert props <= formal, f"[{name}] schema properties {props - formal} 不在 fn 签名里"
    assert no_default <= required, f"[{name}] fn 必填形参 {no_default - required} 未声明进 schema.required"


def test_registry_schema_matches_fn():
    """tools_registry 注册的每个 Tool：schema 与 fn 签名一致。"""
    from tools_registry import _build_default_registry
    reg = _build_default_registry()
    assert reg.tools, "registry 不应为空"
    for name, tool in reg.tools.items():
        _check(name, tool.fn, tool.parameters)


def test_tools_py_names_aligned():
    """tools.py 的 TOOL_FUNCTIONS 与 TOOLS_SCHEMA 工具名集合完全相等（防只改一处）。"""
    import tools
    fn_names = set(tools.TOOL_FUNCTIONS.keys())
    schema_names = {s["function"]["name"] for s in tools.TOOLS_SCHEMA}
    assert fn_names == schema_names, f"两结构工具名不一致: 仅fn={fn_names - schema_names}, 仅schema={schema_names - fn_names}"


def test_tools_py_schema_matches_fn():
    """tools.py 每个 schema 的 required ↔ 对应 fn 签名一致。"""
    import tools
    for s in tools.TOOLS_SCHEMA:
        name = s["function"]["name"]
        _check(name, tools.TOOL_FUNCTIONS[name], s["function"]["parameters"])


def test_check_catches_drift():
    """自检：故意制造 schema↔fn 不一致时 _check 稳定失败（证明校验有效、非空过假阳性）。"""
    import pytest

    def fn(a, b):
        return a

    with pytest.raises(AssertionError):                       # required 含 fn 没有的字段 c
        _check("drift1", fn, {"properties": {"a": {}, "b": {}, "c": {}}, "required": ["a", "c"]})
    with pytest.raises(AssertionError):                       # fn 必填参 b 未进 required
        _check("drift2", fn, {"properties": {"a": {}}, "required": ["a"]})
