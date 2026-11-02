#!/usr/bin/env python3
"""可断点续传的 Hugging Face 模型下载器（Spark Dashboard 用）。

huggingface_hub 每次下载都写到带随机后缀的新临时文件，进程一中断（断网、关机、服务重启）
已下载的部分就全部作废。这里改为：
  - 每个文件写到固定名字的 <blob>.spark-partial，中断后用 HTTP Range 从断点继续
  - 下载完校验 SHA256，再按标准 HF 缓存布局放好（blobs/ + snapshots/<commit>/ + refs/main），
    vLLM 离线模式可直接使用
  - 网络错误无限重试（指数退避），只有权限/不存在这类永久错误才退出（退出码 2）
进度写到 <job_dir>/state.json，由面板读取。

用法: hf_fetch.py <repo> <job_dir>
"""
import fnmatch, glob, hashlib, json, os, signal, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO, JOB_DIR = sys.argv[1], sys.argv[2]
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
TOKEN = os.environ.get("HF_TOKEN", "")
REPO_DIR = os.path.join(HF_HOME, "hub", "models--" + REPO.replace("/", "--"))
BLOBS = os.path.join(REPO_DIR, "blobs")
# vLLM 用不到的格式（仓库里有 safetensors 时跳过）
SKIP = ["*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot", "original/*", "onnx/*", "*.tflite"]
CHUNK = 8 * 2**20
WORKERS = 4

state = {"status": "running", "repo": REPO, "total": 0, "completed": 0, "files_total": 0, "files_done": 0,
         "attempt": 0, "last_error": None, "commit": None, "pid": os.getpid(), "updated": time.time()}
lock = threading.Lock()
done_bytes = {}  # 文件名 -> 已落盘字节数


class PermanentError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


opener = urllib.request.build_opener(NoRedirect)


def save_state(**kw):
    with lock:
        state.update(kw, completed=sum(done_bytes.values()), updated=time.time())
        tmp = os.path.join(JOB_DIR, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, os.path.join(JOB_DIR, "state.json"))


def auth_headers():
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def check_http(e):
    if e.code in (401, 403):
        raise PermanentError(f"GATED: HTTP {e.code} — add a Hugging Face token and accept the license")
    if e.code == 404:
        raise PermanentError(f"not found (HTTP 404): {e.url}")


def open_url(url, headers):
    """手动跟随跳转：跳到 CDN 时不带 Authorization。"""
    for _ in range(5):
        req = urllib.request.Request(url, headers={"User-Agent": "spark-dashboard", **headers})
        try:
            return opener.open(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                nxt = urllib.parse.urljoin(url, e.headers["Location"])
                if urllib.parse.urlparse(nxt).netloc != urllib.parse.urlparse(url).netloc:
                    headers = {k: v for k, v in headers.items() if k != "Authorization"}
                url = nxt
                continue
            check_http(e)
            raise
    raise RuntimeError("too many redirects")


def retrying(fn, what):
    delay = 5
    while True:
        try:
            return fn()
        except PermanentError:
            raise
        except Exception as e:
            with lock:
                state["attempt"] += 1
            save_state(last_error=f"{what}: {e}"[:300])
            time.sleep(delay)
            delay = min(delay * 2, 120)


def repo_info():
    url = f"https://huggingface.co/api/models/{REPO}?blobs=true"
    with open_url(url, auth_headers()) as r:
        return json.load(r)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 * 2**20):
            h.update(b)
    return h.hexdigest()


def link_snapshot(commit, rfilename, key):
    link = os.path.join(REPO_DIR, "snapshots", commit, rfilename)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    target = os.path.relpath(os.path.join(BLOBS, key), os.path.dirname(link))
    if os.path.islink(link) or os.path.exists(link):
        if os.path.exists(link) and os.path.realpath(link) == os.path.realpath(os.path.join(BLOBS, key)):
            return
        os.remove(link)
    os.symlink(target, link)


def fetch_file(commit, f):
    name, size = f["rfilename"], f.get("size") or 0
    lfs = f.get("lfs") or {}
    key = lfs.get("sha256") or f["blobId"]
    blob = os.path.join(BLOBS, key)
    if os.path.exists(blob) and os.path.getsize(blob) == size:  # 已完成（含之前 huggingface_hub 下好的）
        done_bytes[name] = size
        link_snapshot(commit, name, key)
        return
    part = blob + ".spark-partial"

    def attempt():
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        done_bytes[name] = have
        if have < size:
            url = f"https://huggingface.co/{REPO}/resolve/{commit}/{urllib.parse.quote(name)}"
            headers = {**auth_headers(), **({"Range": f"bytes={have}-"} if have else {})}
            with open_url(url, headers) as r:
                if have and r.status != 206:  # 服务器不支持续传，只能重来
                    have = 0
                with open(part, "ab" if have else "wb") as out:
                    while chunk := r.read(CHUNK):
                        out.write(chunk)
                        done_bytes[name] = done_bytes.get(name, 0) + len(chunk)
            if os.path.getsize(part) < size:
                raise IOError(f"{name}: connection closed early")
        if lfs.get("sha256") and sha256_of(part) != lfs["sha256"]:
            os.remove(part)
            raise IOError(f"{name}: checksum mismatch, re-downloading")
        os.replace(part, blob)

    retrying(attempt, name)
    done_bytes[name] = size
    save_state(last_error=None)  # 恢复正常
    link_snapshot(commit, name, key)
    with lock:
        state["files_done"] += 1


def main():
    os.makedirs(JOB_DIR, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *a: (save_state(status="paused"), os._exit(0)))
    save_state()
    info = retrying(repo_info, "repo info")
    commit = info["sha"]
    files = info.get("siblings", [])
    skip = SKIP if any(f["rfilename"].endswith(".safetensors") for f in files) else []
    files = [f for f in files if not any(fnmatch.fnmatch(f["rfilename"], p) for p in skip)]
    total = sum(f.get("size") or 0 for f in files)
    os.makedirs(BLOBS, exist_ok=True)
    # huggingface_hub 留下的随机后缀半成品无法续传，删掉腾出空间
    for orphan in glob.glob(os.path.join(BLOBS, "*.*.incomplete")):
        os.remove(orphan)
    need = total - sum(os.path.getsize(p) for p in glob.glob(os.path.join(BLOBS, "*")) if os.path.isfile(p))
    free = os.statvfs(BLOBS).f_bavail * os.statvfs(BLOBS).f_frsize
    if need > free:
        raise PermanentError(f"not enough disk space: need {need / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB")
    save_state(total=total, files_total=len(files), commit=commit)

    stop = threading.Event()
    threading.Thread(target=lambda: [save_state() for _ in iter(lambda: stop.wait(1), True)], daemon=True).start()
    with ThreadPoolExecutor(WORKERS) as pool:
        for fut in [pool.submit(fetch_file, commit, f) for f in sorted(files, key=lambda f: -(f.get("size") or 0))]:
            fut.result()
    os.makedirs(os.path.join(REPO_DIR, "refs"), exist_ok=True)
    with open(os.path.join(REPO_DIR, "refs", "main"), "w") as f:
        f.write(commit)
    stop.set()
    save_state(status="done", last_error=None)


if __name__ == "__main__":
    try:
        main()
    except PermanentError as e:
        save_state(status="error", last_error=str(e))
        sys.exit(2)
