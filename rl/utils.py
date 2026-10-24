
def read_links_file(path: str) -> list[dict[str, float | int]]:
    """Read the links file and return link data."""
    lines = []
    with open(path) as f:
        lines = [i.strip().split() for i in f.readlines()[2:]]  # Skip the first two header lines

    return [{
        "link_id": int(line[0]),
        "from_node": int(line[1]),
        "to_node": int(line[2]),
        "length_m": float(line[3]),
        "speed_limit_mps": float(line[4]),
        "grade": float(line[6]),
        "curvature": float(line[7]),
    } for line in lines]
