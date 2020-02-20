import re
from src.noises import *


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def parse_noise_from_args(args, device, dim):
    """
    Given a Namespace of arguments, returns the constructed object.
    """
    kwargs = {
        "sigma": args.sigma,
        "lambd": args.lambd,
        "k": args.k,
        "j": args.j,
        "a": args.a
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return eval(args.noise)(device=device, dim=dim, **kwargs)

