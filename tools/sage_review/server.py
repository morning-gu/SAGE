import json, os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = "data/sage_eval/annotations.json"
PORT = int(os.environ.get("PORT", "8765"))
TOOL_PATH = "/tools/sage_review/index.html"


def safe_under_root(rel):
    if not rel:
        return None
    p = (ROOT / rel).resolve()
    try:
        p.relative_to(ROOT)
    except ValueError:
        return None
    return p


def working_path_for(src_path):
    return src_path.with_name(src_path.stem + "_working" + src_path.suffix)


def seed_working(src_path):
    if not src_path.exists():
        return None, "input file not found: " + str(src_path.relative_to(ROOT))
    wp = working_path_for(src_path)
    if wp.exists():
        try:
            return json.loads(wp.read_text(encoding="utf-8")), None
        except Exception:
            pass
    data = json.loads(src_path.read_text(encoding="utf-8"))
    for s in data:
        s["_review"] = {"status": "unreviewed", "notes": "", "ts": 0}
    wp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data, None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def _src(self):
        q = parse_qs(urlparse(self.path).query)
        rel = (q.get("src") or [""])[0] or DEFAULT_SRC
        return safe_under_root(rel), rel

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            self.send_response(302)
            self.send_header("Location", TOOL_PATH)
            self.end_headers()
            return
        if p == "/api/meta":
            self._json({"default_src": DEFAULT_SRC})
            return
        if p == "/api/working":
            sp, rel = self._src()
            if not sp:
                self._json({"error": "invalid src path: " + rel}, 400)
                return
            data, err = seed_working(sp)
            if err:
                self._json({"error": err}, 400)
                return
            self._json(data)
            return
        if p == "/api/original":
            sp, rel = self._src()
            if not sp or not sp.exists():
                self._json({"error": "input not found: " + rel}, 400)
                return
            self._json(json.loads(sp.read_text(encoding="utf-8")))
            return
        super().do_GET()

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/api/working":
            sp, rel = self._src()
            if not sp:
                self._json({"ok": False, "error": "invalid src path: " + rel}, 400)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"[]"
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
            wp = working_path_for(sp)
            wp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            self._json({"ok": True, "src": rel, "working": str(wp.relative_to(ROOT))})
            return
        self._json({"ok": False, "error": "not found"}, 404)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def parse_args():
    global DEFAULT_SRC, PORT
    a = sys.argv[1:]
    i = 0
    while i < len(a):
        if a[i] in ("--input", "-i") and i + 1 < len(a):
            DEFAULT_SRC = a[i + 1]
            i += 2
        elif a[i] in ("--port", "-p") and i + 1 < len(a):
            PORT = int(a[i + 1])
            i += 2
        elif a[i] in ("-h", "--help"):
            print("用法: python server.py [--input <rel-path>] [--port <n>]")
            print("  --input  输入标注文件 (相对工作区根目录), 默认 " + DEFAULT_SRC)
            print("  --port   端口, 默认 " + str(PORT))
            print("页面 URL 可用 ?src=<rel-path> 临时指定其它输入文件")
            sys.exit(0)
        else:
            i += 1


if __name__ == "__main__":
    parse_args()
    sp = safe_under_root(DEFAULT_SRC)
    if sp:
        seed_working(sp)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("SAGE 标注矫正工具已启动:", flush=True)
    print("  输入文件: " + DEFAULT_SRC, flush=True)
    print("  页面: http://localhost:" + str(PORT) + TOOL_PATH + "?src=" + DEFAULT_SRC, flush=True)
    print("  (页面 URL 的 ?src= 可指定其它输入文件)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
