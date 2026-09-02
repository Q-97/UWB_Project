"""减负分析：计算仅保留 RA-OCCUPANCY 时，app/ 内各模块的可达/删除候选。

方法：
- 对 algorithms.py 的 AlgorithmProcessor 建类内调用图（self.method + 模块级函数 util.*/find_breathing_feature）
- 从入口 step_ra_occupancy 做闭包遍历，列出不可达方法及其行号
- 统计 util.py 中被可达代码引用的函数，列出未引用候选
- 输出 GUI mode 引用（PlotPanel/ControlPanel/App 分发）供人工核对

用法: python tools/analyze_slim.py
"""
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def class_methods(tree, class_name):
    """{method: (start, end, direct_calls:set)} 含 self.x 与 util.x / find_breathing_feature / ndimage.x 等。"""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    calls = set()
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Call):
                            f = sub.func
                            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'self':
                                calls.add(f"self.{f.attr}")
                            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                                calls.add(f"{f.value.id}.{f.attr}")
                            elif isinstance(f, ast.Name):
                                calls.add(f.id)
                    out[item.name] = (item.lineno, item.end_lineno, calls)
    return out


def reachable(methods, entry):
    """从 entry 出发，沿 self.* 与无前缀函数名做闭包。返回可达的 self 方法名集合。"""
    seen = set()
    stack = [entry]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        if m not in methods:
            continue
        for c in methods[m][2]:
            if c.startswith("self."):
                stack.append(c[len("self."):])
            elif c in methods:  # 同模块顶层函数（apply_range_bin_selection 等不在此类，仅类方法）
                stack.append(c)
    return seen


def module_functions(tree):
    """顶层函数 {name: (start, end)}"""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = (node.lineno, node.end_lineno)
    return out


def main():
    algo = parse(PROJECT_ROOT / "app" / "algorithms.py")
    methods = class_methods(algo, "AlgorithmProcessor")

    # 入口：RA-OCCUPANCY 实际链路的对外调用点
    entries = ["step_ra_occupancy"]
    reach = set()
    for e in entries:
        reach |= reachable(methods, e)
    # 构造/通用方法（保持存活）
    keep_names = {"__init__", "_reset_algo_state", "_reset_dubhe_state", "_init_music_if_needed", "init_done"}

    print("=" * 70)
    print("AlgorithmProcessor: 不可达方法（RA-OCCUPANCY 闭包之外）→ 删除候选")
    print("=" * 70)
    total = 0
    for name in sorted(methods):
        if name in reach or name in keep_names:
            continue
        s, e, _ = methods[name]
        n = e - s + 1
        total += n
        print(f"  {name:<38} L{s:>5}-{e:<5} ~{n} 行")
    print(f"  --- 不可达合计约 {total} 行 ---")
    print(f"  可达方法数 {len(reach & set(methods))} / {len(methods)}")

    print()
    print("=" * 70)
    print("可达闭包内的模块级/外部调用（需保留的依赖线索）")
    print("=" * 70)
    ext = set()
    for m in reach & set(methods):
        for c in methods[m][2]:
            if not c.startswith("self."):
                ext.add(c)
    for c in sorted(ext):
        print(f"  {c}")

    print()
    print("=" * 70)
    print("util.py 顶层函数及其被 app/ 引用计数")
    print("=" * 70)
    import re
    util_tree = parse(PROJECT_ROOT / "app" / "util.py")
    fn_map = module_functions(util_tree)
    util_refs = {}
    for p in PROJECT_ROOT.glob("app/**/*.py"):
        if p.name == "util.py":
            continue
        try:
            t = parse(p)
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
               and isinstance(node.func.value, ast.Name) and node.func.value.id == 'util':
                util_refs[node.func.attr] = util_refs.get(node.func.attr, 0) + 1
    total_util = 0
    for name, (s, e) in sorted(fn_map.items(), key=lambda kv: kv[1][0]):
        n = e - s + 1
        total_util += n
        print(f"  {name:<42} L{s:>4}-{e:<5} ~{n:>4} 行  refs={util_refs.get(name, 0)}")
    print(f"  --- util.py 合计 {total_util} 行 ---")


if __name__ == "__main__":
    main()
