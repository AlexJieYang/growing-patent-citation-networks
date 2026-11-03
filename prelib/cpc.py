"""Explicit CPC selectors: section/class/subclass/main-group, exact, or '*'."""
import re
import pandas as pd
from .common import normalize_code, main_group


def match_selector(codes, selector):
    original = str(selector).strip().upper().replace(" ", "")
    wildcard = original.endswith("*")
    value = normalize_code(pd.Series([original[:-1] if wildcard else original])).iloc[0]
    if not re.fullmatch(r"[A-HY](?:\d{2}(?:[A-Z](?:\d+(?:/\d*)?)?)?)?", value):
        raise ValueError(f"无效CPC选择器: {selector}")
    if wildcard or len(value) in {1, 3, 4}:
        return codes.str.startswith(value, na=False)
    if "/" not in value or value.endswith("/"):
        return main_group(codes).eq(value.rstrip("/")).fillna(False)
    return codes.eq(value).fillna(False)


def domain_mask(codes, spec):
    keep = pd.Series(False, index=codes.index)
    for selector in spec["include"]:
        keep |= match_selector(codes, selector)
    for selector in spec.get("exclude", []):
        keep &= ~match_selector(codes, selector)
    return keep
