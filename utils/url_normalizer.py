import re

_NUM_RE = re.compile(r"/(\d+)(/|$)")
_UUID_RE = re.compile(
    r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

_EXT_RE = re.compile(r"\.(html?|php|asp|aspx|jsp|cgi|cfm)$", re.IGNORECASE)

_SLASH_RE = re.compile(r"/+")

def normalize_url(uri: str) -> str:
    if not uri:
        return "/"
    path = uri.split("?")[0].split("#")[0]
    path = _UUID_RE.sub("/{uuid}", path)
    prev = None
    while prev != path:
        prev = path
        path = _NUM_RE.sub(r"/{id}\2", path)

    path = _EXT_RE.sub("", path)
    path = _SLASH_RE.sub("/", path)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return path.lower() if path else "/"