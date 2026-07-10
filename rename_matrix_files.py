from pathlib import Path


TARGET_DIR_NAME = "2026_7_7_combined=4_drop=4_path=4_matrix"
REPLACEMENTS = {
    "b1": "b_1",
    "a1": "a_1",
    "a2": "a_2",
    "a3": "a_3",
}


def build_new_name(file_name: str) -> str:
    new_name = file_name
    for old, new in REPLACEMENTS.items():
        new_name = new_name.replace(old, new)
    return new_name


def rename_files(dry_run: bool = True) -> None:
    target_dir = Path(__file__).resolve().parent / TARGET_DIR_NAME
    if not target_dir.is_dir():
        raise FileNotFoundError(f"找不到目标文件夹: {target_dir}")

    rename_pairs = []
    for path in target_dir.iterdir():
        if not path.is_file():
            continue

        new_name = build_new_name(path.name)
        if new_name == path.name:
            continue

        new_path = path.with_name(new_name)
        if new_path.exists():
            raise FileExistsError(f"目标文件已存在，跳过以避免覆盖: {new_path}")

        rename_pairs.append((path, new_path))

    if not rename_pairs:
        print("没有需要重命名的文件。")
        return

    for old_path, new_path in rename_pairs:
        print(f"{old_path.name} -> {new_path.name}")
        if not dry_run:
            old_path.rename(new_path)

    if dry_run:
        print("\n当前是试运行，没有真正改名。确认无误后把 dry_run=False 再运行。")


if __name__ == "__main__":
    rename_files(dry_run=True)
