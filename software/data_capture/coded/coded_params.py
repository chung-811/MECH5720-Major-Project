"""
Coded illumination parameters, shared by the Pi server and the GUI client.

A code file is a JSON object describing only the light sequence. Camera
settings (gain, colour gains, padding, ...) come from the GUI instead.

    sequence            the code. A string such as "1011001110" (spaces,
                        commas and underscores are ignored) or a list
                        [1,0,1,1,...]. 1 = white, 0 = black.      [required]
    total_duration_s    how long the whole run of bits is displayed, in
                        seconds, counting every repeat. Each bit is shown
                        for total_duration_s / (number of bits).   [required]
    repeat              how many times the sequence is played.          [1]
    invert              true = swap black and white.                [false]
    pre_roll_s          seconds of black before the sequence.         [1.0]
    post_roll_s         seconds of black after the sequence.          [0.5]

No third-party imports, so this runs on the Pi and the dev computer alike.
"""

CODE_DEFAULTS = {
    "sequence": None,
    "total_duration_s": None,
    "repeat": 1,
    "invert": False,
    "pre_roll_s": 1.0,
    "post_roll_s": 0.5,
}

WARN_FLASH_HZ = 3.0        # photosensitivity warning threshold
REFRESH_S = 1.0 / 60.0     # assumed display refresh period


def parse_sequence(seq):
    """Turn a '1011...' string or a [1, 0, 1, ...] list into a list of ints."""
    if seq is None:
        raise ValueError('"sequence" is required')

    if isinstance(seq, str):
        items = [c for c in seq if c not in " ,_\t\r\n"]
    elif isinstance(seq, (list, tuple)):
        items = list(seq)
    else:
        raise ValueError('"sequence" must be a string like "1011" or a list of 0/1')

    bits = []
    for item in items:
        if item in (0, 1, False, True, "0", "1"):
            bits.append(int(item))
        else:
            raise ValueError('"sequence" may only contain 0 and 1 (found %r)' % (item,))
    if not bits:
        raise ValueError('"sequence" is empty')
    return bits


def parse_code_params(user):
    """Apply defaults to a code dict, validate it, and add derived values.

    Returns a new dict with the code keys plus:
        bits            the bits actually shown (repeated and inverted)
        bit_duration_s  how long each bit is shown
        flashes_per_s   average black->white transitions per second
        warnings        list of human-readable warnings
    Unknown keys are ignored (and listed in the warnings).
    """
    if not isinstance(user, dict):
        raise ValueError("the code file must contain a JSON object { ... }")

    p = dict(CODE_DEFAULTS)
    p.update({k: v for k, v in user.items() if k in CODE_DEFAULTS})
    warnings = []

    unknown = sorted(set(user) - set(CODE_DEFAULTS))
    if unknown:
        warnings.append("ignored unknown key(s): %s" % ", ".join(unknown))

    base_bits = parse_sequence(p["sequence"])

    def need(name, kind, low, strict=False):
        v = p[name]
        if v is None:
            raise ValueError('"%s" is required' % name)
        if isinstance(v, bool) or not isinstance(v, kind):
            raise ValueError('"%s" must be a number' % name)
        if v < low or (strict and v == low):
            raise ValueError('"%s" must be %s %s' % (name, ">" if strict else ">=", low))

    need("total_duration_s", (int, float), 0, strict=True)
    need("repeat", int, 1)
    need("pre_roll_s", (int, float), 0)
    need("post_roll_s", (int, float), 0)
    p["invert"] = bool(p["invert"])

    bits = base_bits * p["repeat"]
    if p["invert"]:
        bits = [1 - b for b in bits]
    p["bits"] = bits
    p["bit_duration_s"] = float(p["total_duration_s"]) / len(bits)

    flashes = sum(1 for a, b in zip([0] + bits, bits) if b == 1 and a == 0)
    p["flashes_per_s"] = flashes / float(p["total_duration_s"])

    if p["bit_duration_s"] < REFRESH_S:
        warnings.append("%.1f ms per bit is shorter than one 60 Hz refresh (%.1f ms); "
                        "bits will be dropped" % (p["bit_duration_s"] * 1000, REFRESH_S * 1000))
    if p["flashes_per_s"] > WARN_FLASH_HZ:
        warnings.append("about %.1f flashes/s - rapid flashing can trigger photosensitive "
                        "seizures; make sure nobody affected can see the screen"
                        % p["flashes_per_s"])
    p["warnings"] = warnings
    return p
