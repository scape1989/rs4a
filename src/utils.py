import re


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

