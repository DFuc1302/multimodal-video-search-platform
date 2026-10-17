"""Match folder selectors against keyframe paths, never image filenames."""

import re


def matches_folder(filename: str, selector: str) -> bool:
    needle = selector.strip().replace("\\", "/").strip("/").casefold()
    if not needle:
        return True
    folders = filename.replace("\\", "/").casefold().split("/")[:-1]
    # Dataset exports use both ``S01-V001`` and ``S01_V001``.  Treat the
    # separator as an alias while keeping matching limited to directory names.
    canonical_needle = needle.replace("-", "_")
    canonical_folders = [part.replace("-", "_") for part in folders]
    if canonical_needle in canonical_folders:
        return True
    if re.fullmatch(r"l\d+_v\d+", needle):
        return needle in folders
    if re.fullmatch(r"l\d+", needle):
        return any(re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", part)
                   for part in folders)
    return needle in "/".join(folders)
