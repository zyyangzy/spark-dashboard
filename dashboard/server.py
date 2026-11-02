#!/usr/bin/env python3
# Spark Dashboard 后端：DGX Dashboard 的轻量替代。
# 系统/网络监控 + Ollama / vLLM 模型管理 + Docker 管理，经 Caddy (端口 11000) 统一对外。
import glob, html, pwd, http.client, json, os, re, shlex, shutil, socket, subprocess, threading, time
import urllib.request, uuid, xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote, urlencode

HOST, PORT = "127.0.0.1", 11010
HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
CONF = os.path.join(HOME, ".config/spark")
TOKEN_FILE = os.path.join(CONF, "jupyter_token")
HF_TOKEN_FILE = os.path.join(CONF, "hf_token")
VSCODE_TOKEN_FILE = os.path.join(CONF, "vscode_token")
HF_CACHE = os.path.join(HOME, ".cache/huggingface")
OLLAMA = "http://127.0.0.1:11434"
VLLM_IMAGE = "nvcr.io/nvidia/vllm"
VLLM_DEFAULT_TAG = "26.10-py3"
SPARK_CTL = "/usr/local/bin/spark-ctl"
UA = {"User-Agent": "spark-dashboard/1.0"}
# 只接受本机地址，防 DNS rebinding
ALLOWED_HOST = re.compile(r"^(localhost|127\.0\.0\.1|[\w-]+\.localhost|[\w-]+\.[\w-]+\.ts\.net)(:\d+)?$")


def run(*cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def fetch(url, timeout=15, headers=None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


_cache = {}


def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def read_file(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def write_secret(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(value)


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size  # 跟随链接：新版 HF 缓存把大文件放在共享 blobs 里
            except OSError:
                pass
    return total


# ---------- 后台任务 ----------
# Docker 镜像拉取：内存任务（Docker 自己会续传）。
# 模型下载：持久化到 JOBS_DIR，面板重启、断网、关机后都能自动续传。

jobs = {}  # 内存任务 id -> {kind, name, status, completed, total, error, started}


def new_job(kind, name):
    for j in jobs.values():
        if j["kind"] == kind and j["name"] == name and j["status"] == "running":
            raise RuntimeError("already downloading")
    jid = uuid.uuid4().hex[:8]
    jobs[jid] = {"id": jid, "kind": kind, "name": name, "status": "running", "detail": "",
                 "completed": 0, "total": 0, "error": None, "started": time.time()}
    return jobs[jid]


def finish(job, error=None):
    job["status"] = "error" if error else "done"
    job["error"] = error
    job["finished"] = time.time()


def prune_jobs():
    for jid in [k for k, j in jobs.items() if j.get("finished") and time.time() - j["finished"] > 600]:
        del jobs[jid]


JOBS_DIR = os.path.join(HOME, ".local/share/spark/jobs")
HF_FETCH = os.path.join(HERE, "hf_fetch.py")
USER_ENV = {"XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus"}
ollama_threads = {}  # jid -> Thread


def _job_path(jid, name="job.json"):
    return os.path.join(JOBS_DIR, jid, name)


def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)


def _unit(jid):
    return f"spark-dl-{jid}"


def _unit_alive(jid):
    st = subprocess.run(["systemctl", "--user", "is-active", _unit(jid)], capture_output=True, text=True,
                        env={**os.environ, **USER_ENV}).stdout.strip()
    return st in ("active", "activating", "reloading")


def disk_jobs():
    out = []
    if not os.path.isdir(JOBS_DIR):
        return out
    for jid in os.listdir(JOBS_DIR):
        meta = _read_json(_job_path(jid))
        if not meta:
            continue
        st = _read_json(_job_path(jid, "state.json"), {})
        out.append({**meta, **{k: st.get(k) for k in ("status", "total", "completed", "attempt", "last_error",
                                                       "files_total", "files_done", "updated", "detail")}})
    return out


def job_view(j):
    """统一给前端的任务视图。status: running / retrying / interrupted / paused / done / error"""
    status = j.get("status") or "running"
    if j.get("paused") and status not in ("done", "error"):
        status = "paused"
    elif status in ("running", "paused") and not j.get("alive", True):
        status = "interrupted"  # 被中断，稍后自动续传
    elif status == "running" and j.get("last_error"):
        status = "retrying"  # 网络问题，下载器正在自动重试
    err = j.get("last_error") if status in ("error", "retrying", "interrupted") else None
    return {"id": j["id"], "kind": j["kind"], "name": j["name"], "status": status, "persistent": True,
            "completed": j.get("completed") or 0, "total": j.get("total") or 0, "attempt": j.get("attempt") or 0,
            "error": err, "detail": j.get("detail") or "", "started": j.get("started"),
            "files": f"{j.get('files_done') or 0}/{j['files_total']}" if j.get("files_total") else None}


def list_jobs():
    out = [{k: v for k, v in j.items() if k != "proc"} for j in jobs.values()]
    for j in disk_jobs():
        j["alive"] = _unit_alive(j["id"]) if j["kind"] == "hf" else j["id"] in ollama_threads
        out.append(job_view(j))
    return out


def start_download(kind, name):
    """新建或续传一个模型下载任务（同一模型已有未完成任务时直接续传它）。"""
    for j in disk_jobs():
        if j["kind"] == kind and j["name"] == name and j.get("status") not in ("done",):
            meta = _read_json(_job_path(j["id"]))
            meta["paused"] = False
            _write_json(_job_path(j["id"]), meta)
            if j.get("status") == "error":
                st = _read_json(_job_path(j["id"], "state.json"), {})
                _write_json(_job_path(j["id"], "state.json"), {**st, "status": "running", "last_error": None})
            _launch(meta)
            return meta
    jid = uuid.uuid4().hex[:8]
    meta = {"id": jid, "kind": kind, "name": name, "started": time.time(), "paused": False}
    _write_json(_job_path(jid), meta)
    _write_json(_job_path(jid, "state.json"), {"status": "running", "total": 0, "completed": 0, "updated": time.time()})
    _launch(meta)
    return meta


def _launch(meta):
    jid = meta["id"]
    if meta["kind"] == "hf":
        if _unit_alive(jid):
            return
        cmd = ["systemd-run", "--user", "--unit", _unit(jid), "--collect", "-q",
               "-p", "Restart=on-failure", "-p", "RestartSec=20", "-p", "RestartPreventExitStatus=2",
               "-E", f"HF_HOME={HF_CACHE}"]
        token = read_file(HF_TOKEN_FILE)
        if token:
            cmd += ["-E", f"HF_TOKEN={token}"]
        cmd += ["/usr/bin/python3", HF_FETCH, meta["name"], os.path.join(JOBS_DIR, jid)]
        subprocess.run(cmd, env={**os.environ, **USER_ENV}, capture_output=True, timeout=30)
    elif meta["kind"] in ("ollama", "image") and jid not in ollama_threads:
        worker = ollama_worker if meta["kind"] == "ollama" else image_worker
        t = threading.Thread(target=worker, args=(jid, meta["name"]), daemon=True)
        ollama_threads[jid] = t
        t.start()


def pause_job(jid):
    meta = _read_json(_job_path(jid))
    if not meta:
        return
    meta["paused"] = True
    _write_json(_job_path(jid), meta)
    if meta["kind"] == "hf":
        subprocess.run(["systemctl", "--user", "stop", _unit(jid)], env={**os.environ, **USER_ENV}, capture_output=True)


def resume_job(jid):
    meta = _read_json(_job_path(jid))
    if meta:
        start_download(meta["kind"], meta["name"])


def remove_job(jid):
    pause_job(jid)
    shutil.rmtree(os.path.join(JOBS_DIR, jid), ignore_errors=True)


def reconcile_jobs():
    """自动续传被中断的任务（面板重启、开机后），清理完成超过 10 分钟的任务。"""
    for j in disk_jobs():
        st = j.get("status")
        if st in ("done", "error"):
            if time.time() - (j.get("updated") or 0) > 600 and st == "done":
                shutil.rmtree(os.path.join(JOBS_DIR, j["id"]), ignore_errors=True)
            continue
        if not j.get("paused"):
            _launch(j)

# ---------- Ollama ----------

def ollama(path, body=None, timeout=10):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def try_ollama(path):
    try:
        return ollama(path, timeout=2)
    except Exception:
        return None


def ollama_worker(jid, name):
    """Ollama 自己会保留已下载的分块；这里负责断线重试，并把进度写盘。"""
    state_path = _job_path(jid, "state.json")
    delay = 5
    try:
        while True:
            meta = _read_json(_job_path(jid))
            if not meta or meta.get("paused"):
                return
            st = _read_json(state_path, {})
            layers, last_write = {}, 0
            try:
                req = urllib.request.Request(OLLAMA + "/api/pull", data=json.dumps({"model": name}).encode())
                with urllib.request.urlopen(req, timeout=600) as r:
                    for line in r:
                        if (_read_json(_job_path(jid)) or {}).get("paused"):
                            return  # 断开连接即暂停，Ollama 保留已下载部分
                        ev = json.loads(line)
                        if ev.get("error"):
                            raise RuntimeError(ev["error"])
                        if ev.get("digest") and ev.get("total"):
                            layers[ev["digest"]] = (ev.get("completed", 0), ev["total"])
                        if time.time() - last_write > 1:
                            st.update(status="running", detail=ev.get("status", ""), updated=time.time(),
                                      completed=sum(c for c, _ in layers.values()),
                                      total=sum(t for _, t in layers.values()), last_error=None)
                            _write_json(state_path, st)
                            last_write = time.time()
                st.update(status="done", completed=st.get("total", 0), updated=time.time(), last_error=None)
                _write_json(state_path, st)
                return
            except Exception as e:
                msg = str(e)
                permanent = any(k in msg.lower() for k in ("not found", "does not exist", "manifest unknown", "invalid"))
                st.update(status="error" if permanent else "running", last_error=msg[:300], updated=time.time(),
                          attempt=st.get("attempt", 0) + 1)
                _write_json(state_path, st)
                if permanent:
                    return
                time.sleep(delay)
                delay = min(delay * 2, 120)
    finally:
        ollama_threads.pop(jid, None)


def image_worker(jid, image):
    """Docker 镜像拉取：已完成的层 Docker 会保留，中断后重新拉取即可续上；断网自动重试。"""
    state_path = _job_path(jid, "state.json")
    delay = 5
    try:
        while True:
            meta = _read_json(_job_path(jid))
            if not meta or meta.get("paused"):
                return
            st = _read_json(state_path, {})
            img, tag = split_image(image)
            try:
                conn = DockerConn(timeout=3600)
                conn.request("POST", f"/images/create?fromImage={quote(img)}" + (f"&tag={quote(tag)}" if tag else ""))
                resp = conn.getresponse()
                if resp.status >= 400:
                    raise RuntimeError(json.loads(resp.read()).get("message"))
                layers, last_write = {}, 0
                while line := resp.readline():
                    if (_read_json(_job_path(jid)) or {}).get("paused"):
                        conn.close()
                        return
                    ev = json.loads(line)
                    if "error" in ev:
                        raise RuntimeError(ev["error"])
                    d = ev.get("progressDetail") or {}
                    if ev.get("id") and d.get("total"):
                        layers[ev["id"]] = (d.get("current", 0), d["total"])
                    elif ev.get("id") and ev.get("status") in ("Pull complete", "Already exists", "Download complete") \
                            and ev["id"] in layers:
                        layers[ev["id"]] = (layers[ev["id"]][1], layers[ev["id"]][1])
                    if time.time() - last_write > 1:
                        st.update(status="running", detail=ev.get("status", ""), updated=time.time(), last_error=None,
                                  completed=sum(c for c, _ in layers.values()), total=sum(t for _, t in layers.values()))
                        _write_json(state_path, st)
                        last_write = time.time()
                if image == TF_PREBUILT:
                    repo, _, tg = TF_IMAGE.rpartition(":")
                    docker("POST", f"/images/{quote(TF_PREBUILT, safe='')}/tag?repo={quote(repo)}&tag={quote(tg)}")
                    if not tf_image():
                        st.update(status="error", last_error="pulled image is not a TensorFold recipe image")
                        _write_json(state_path, st)
                        return
                st.update(status="done", completed=st.get("total", 0), updated=time.time(), last_error=None)
                _write_json(state_path, st)
                return
            except Exception as e:
                msg = str(e)
                permanent = any(k in msg.lower() for k in ("not found", "manifest unknown", "denied", "unauthorized"))
                st.update(status="error" if permanent else "running", last_error=msg[:300], updated=time.time(),
                          attempt=st.get("attempt", 0) + 1)
                _write_json(state_path, st)
                if permanent:
                    return
                time.sleep(delay)
                delay = min(delay * 2, 120)
    finally:
        ollama_threads.pop(jid, None)


def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


SIZE_BADGE = re.compile(r"^(e?\d+(\.\d+)?[bmk]|\d+x\d+(\.\d+)?b)$", re.I)


def ollama_search(q, page):
    page_html = fetch(f"https://ollama.com/search?{urlencode({'q': q, 'p': page})}")
    out = []
    for block in re.split(r"<li\s", page_html)[1:]:
        m = re.search(r'href="/library/([^"]+)"', block)
        if not m:
            continue
        badges = [strip_tags(b) for b in re.findall(r'<span\s+class="inline-flex items-center rounded-md[^"]*">(.*?)</span>', block)]
        dl = re.search(r'title="([\d,]+) downloads"', block)
        upd = re.search(r'title="Updated ([^"]+)">([^<]+)<', block)
        tags = re.search(r">(\d+) tags?<", block)
        out.append({"name": m.group(1),
                    "desc": strip_tags((re.search(r"<p[^>]*>(.*?)</p>", block, re.S) or [None, ""])[1]),
                    "caps": [b for b in badges if not SIZE_BADGE.match(b)],
                    "sizes": [b for b in badges if SIZE_BADGE.match(b)],
                    "downloads": int(dl.group(1).replace(",", "")) if dl else None,
                    "updated": upd.group(2).strip() if upd else "",
                    "tags": int(tags.group(1)) if tags else None})
    return {"results": out, "more": 'data-has-more="true"' in page_html}


def ollama_tags(name):
    page_html = fetch(f"https://ollama.com/library/{quote(name)}/tags")
    seen = {}
    for m in re.finditer(r'<a href="/library/([^"]+)" class="md:hidden.*?font-mono">\s*(\w+)\s*</span>\s*•\s*'
                         r'([\d.]+\s*[KMGT]?B)\s*•\s*([^<•]*?)\s*context window.*?(\w[\w ,]*?) input', page_html, re.S):
        tag, digest, size, ctx, inputs = m.groups()
        # 同一个文件的多个别名只保留最短的名字
        if digest not in seen or len(tag) < len(seen[digest]["tag"]):
            seen[digest] = {"tag": tag, "size": size.replace(" ", ""), "ctx": ctx.strip(), "inputs": inputs.strip()}
    return list(seen.values())


# ---------- Hugging Face ----------

HF_SKIP = ["*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot", "original/*", "onnx/*", "*.tflite"]


def hf_headers():
    token = read_file(HF_TOKEN_FILE)
    return {"Authorization": f"Bearer {token}"} if token else {}


HF_FIELDS = ("safetensors", "downloads", "likes", "gated", "pipeline_tag", "lastModified", "library_name", "trendingScore")
LLM_TASKS = ("text-generation", "image-text-to-text")


def hf_search(q, task):
    sort = "downloads" if q else "trendingScore"

    def query(pipeline):
        params = [("search", q), ("limit", "80"), ("sort", sort), ("direction", "-1")]
        if pipeline:
            params.append(("pipeline_tag", pipeline))
        params += [("expand[]", f) for f in HF_FIELDS]
        return json.loads(fetch("https://huggingface.co/api/models?" + urlencode(params), headers=hf_headers()))

    # 默认 "llm"：文本生成 + 图文多模态（很多新模型属于后者）
    tasks = LLM_TASKS if task == "llm" else (task,)
    seen, data = set(), []
    for m in [m for t in tasks for m in query(t)]:
        if m["id"] not in seen:
            seen.add(m["id"])
            data.append(m)
    data.sort(key=lambda m: m.get(sort) or 0, reverse=True)
    # 输入了完整仓库名就直接查，排在最前
    if re.fullmatch(r"[\w.-]+/[\w.-]+", q):
        try:
            exact = json.loads(fetch(f"https://huggingface.co/api/models/{q}?" +
                                     urlencode([("expand[]", f) for f in HF_FIELDS]), headers=hf_headers()))
            data = [exact] + [m for m in data if m["id"] != exact["id"]]
        except Exception:
            pass
    return [{"id": m["id"], "params": (m.get("safetensors") or {}).get("total"),
             "dtypes": list(((m.get("safetensors") or {}).get("parameters") or {}).keys()),
             "downloads": m.get("downloads"), "likes": m.get("likes"), "gated": bool(m.get("gated")),
             "task": m.get("pipeline_tag"), "library": m.get("library_name"), "updated": m.get("lastModified")}
            for m in data if not NOT_VLLM.search(m["id"]) and m.get("library_name") not in NOT_VLLM_LIBS][:40]


# vLLM 用不了的格式（GGUF 给 llama.cpp/Ollama，MLX 给 Apple 芯片）
NOT_VLLM = re.compile(r"(gguf|mlx|onnx)", re.I)
NOT_VLLM_LIBS = {"gguf", "mlx", "onnx", "transformers.js", "llama.cpp", "coreml"}


def hf_repo_dir(repo):
    return os.path.join(HF_CACHE, "hub", "models--" + repo.replace("/", "--"))


def hf_local():
    hub = os.path.join(HF_CACHE, "hub")
    out = []
    if not os.path.isdir(hub):
        return out
    states = {}
    for j in list_jobs():
        if j["kind"] == "hf" and j["status"] != "done":
            states[j["name"]] = j["status"]
    for d in sorted(os.listdir(hub)):
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/")
        blobs = os.path.join(hub, d, "blobs")
        names = os.listdir(blobs) if os.path.isdir(blobs) else []
        partial = any(f.endswith((".incomplete", ".spark-partial")) for f in names)
        complete = bool(names) and not partial and os.path.exists(os.path.join(hub, d, "refs", "main")) \
            and repo not in states
        out.append({"id": repo, "size": dir_size(blobs), "complete": complete, "job": states.get(repo),
                    "engine": "tensorfold" if complete and tf_eligible(repo) else "vllm",
                    "downloading": states.get(repo) in ("running", "retrying", "interrupted"),
                    "modified": os.path.getmtime(os.path.join(hub, d))})
    return out


# ---------- Docker (unix socket) ----------

class DockerConn(http.client.HTTPConnection):
    def __init__(self, timeout=30):
        super().__init__("docker", timeout=timeout)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/var/run/docker.sock")


def docker(method, path, body=None, timeout=30, raw=False):
    conn = DockerConn(timeout)
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    if resp.status >= 400:
        try:
            msg = json.loads(data).get("message", data.decode())
        except ValueError:
            msg = data.decode(errors="replace")
        raise RuntimeError(msg)
    if raw:
        return data
    return json.loads(data) if data else None


def demux_logs(data):
    """拆开 docker 非 TTY 日志的 8 字节帧头。"""
    if not data or data[0] not in (0, 1, 2):
        return data.decode(errors="replace")
    out, i = [], 0
    while i + 8 <= len(data):
        size = int.from_bytes(data[i + 4:i + 8], "big")
        out.append(data[i + 8:i + 8 + size])
        i += 8 + size
    return b"".join(out).decode(errors="replace")


def split_image(image):
    """'nvcr.io:443/nvidia/pytorch:25.01' -> ('nvcr.io:443/nvidia/pytorch', '25.01')"""
    if "@" in image:  # 按 digest 拉取：repo@sha256:...
        return image, ""
    if ":" in image.split("/")[-1]:
        img, _, tag = image.rpartition(":")
        return img, tag
    return image, "latest"


def parse_lines(text):
    return [l.strip() for l in (text or "").replace(",", "\n").splitlines() if l.strip()]


def create_and_start(body, name=""):
    path = "/containers/create" + (f"?name={quote(name)}" if name else "")
    try:
        cid = docker("POST", path, body)["Id"]
    except RuntimeError as e:
        if "No such image" not in str(e):
            raise
        img, tag = split_image(body["Image"])
        docker("POST", f"/images/create?fromImage={quote(img)}&tag={quote(tag)}", timeout=3600, raw=True)
        cid = docker("POST", path, body)["Id"]
    docker("POST", f"/containers/{cid}/start")
    return {"id": cid}


def docker_run(spec):
    ports, bindings = {}, {}
    for p in parse_lines(spec.get("ports")):
        host, _, ctr = p.rpartition(":")
        ctr = ctr if "/" in ctr else ctr + "/tcp"
        ports[ctr] = {}
        bindings.setdefault(ctr, []).append({"HostIp": "127.0.0.1", "HostPort": host or ctr.split("/")[0]})
    host_cfg = {"PortBindings": bindings,
                "Binds": [os.path.expanduser(v) for v in parse_lines(spec.get("volumes"))],
                "RestartPolicy": {"Name": spec.get("restart") or "no"}}
    if spec.get("gpu"):
        host_cfg["DeviceRequests"] = [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]
        host_cfg["IpcMode"] = "host"
    body = {"Image": spec["image"].strip(), "ExposedPorts": ports, "HostConfig": host_cfg,
            "Env": [e.strip() for e in (spec.get("env") or "").splitlines() if e.strip()]}
    if spec.get("cmd", "").strip():
        body["Cmd"] = ["/bin/sh", "-c", spec["cmd"]]
    return create_and_start(body, spec.get("name", "").strip())


def containers():
    out = []
    for c in docker("GET", "/containers/json?all=1", timeout=5):
        out.append({"id": c["Id"][:12], "name": c["Names"][0].lstrip("/"), "image": c["Image"],
                    "state": c["State"], "status": c["Status"], "created": c["Created"],
                    "labels": c.get("Labels") or {},
                    "ports": sorted({p["PublicPort"] for p in c.get("Ports", []) if p.get("PublicPort")})})
    return out


def images():
    used = {c["ImageID"] for c in docker("GET", "/containers/json?all=1", timeout=5)}
    return [{"id": i["Id"].split(":")[1][:12], "full_id": i["Id"],
             "tags": [t for t in (i.get("RepoTags") or []) if t != "<none>:<none>"],
             "size": i["Size"], "created": i["Created"], "used": i["Id"] in used}
            for i in docker("GET", "/images/json", timeout=10)]


def volumes():
    return [{"name": v["Name"], "driver": v["Driver"], "mount": v["Mountpoint"], "created": v.get("CreatedAt")}
            for v in (docker("GET", "/volumes", timeout=5).get("Volumes") or [])]


# ---------- vLLM ----------

def vllm_image():
    """本地已有的最新 NGC vLLM 镜像，没有则返回默认版本。"""
    tags = sorted(t.split(":")[1] for i in images() for t in i["tags"] if t.startswith(VLLM_IMAGE + ":"))
    return (VLLM_IMAGE + ":" + tags[-1], True) if tags else (VLLM_IMAGE + ":" + VLLM_DEFAULT_TAG, False)


def free_port(start=8000):
    used = {p for c in containers() for p in c["ports"]}
    for port in range(start, start + 100):
        if port in used:
            continue
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port")


def vllm_deploy(spec):
    repo = spec["model"]
    image, _ = vllm_image()
    port = free_port()
    args = ["vllm", "serve", repo, "--host", "0.0.0.0", "--port", "8000",
            "--gpu-memory-utilization", str(float(spec.get("gpu_mem") or 0.4)),
            "--served-model-name", repo]
    if spec.get("max_len"):
        args += ["--max-model-len", str(int(spec["max_len"]))]
    extra = spec.get("extra") or ""
    if "--tool-call-parser" not in extra:  # Open WebUI 需要工具调用；没填就按模型自动补上
        extra = f"{extra} {vllm_recommend(repo)['args']}".strip()
    args += shlex.split(extra)
    env = ["HF_HUB_OFFLINE=1"]
    token = read_file(HF_TOKEN_FILE)
    if token:
        env.append(f"HF_TOKEN={token}")
    body = {"Image": image, "Cmd": args, "Env": env,
            "Labels": {"spark.vllm": "1", "spark.model": repo},
            "ExposedPorts": {"8000/tcp": {}},
            "HostConfig": {"PortBindings": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]},
                           "Binds": [f"{HF_CACHE}:/root/.cache/huggingface"],
                           "DeviceRequests": [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}],
                           "IpcMode": "host", "RestartPolicy": {"Name": "unless-stopped"},
                           "Ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1}]}}
    name = "vllm-" + re.sub(r"[^a-z0-9]+", "-", repo.lower()).strip("-")[:50]
    return {**create_and_start(body, name), "port": port}


# ---------- vLLM 工具调用 / 推理解析器 ----------
# Open WebUI 默认发 tool_choice="auto"，vLLM 必须开 --enable-auto-tool-choice 并指定与模型匹配的解析器

def _readme_flags(text):
    tool = re.search(r"--tool-call-parser[ =]([a-z0-9_]+)", text)
    reason = re.search(r"--reasoning-parser[ =]([a-z0-9_]+)", text)
    base = re.search(r"^base_model:\s*\n?\s*-?\s*([\w.-]+/[\w.-]+)", text, re.M)
    return (tool and tool.group(1), reason and reason.group(1), base and base.group(1))


def _local_readme(repo):
    snaps = os.path.join(hf_repo_dir(repo), "snapshots")
    for snap in sorted(os.listdir(snaps)) if os.path.isdir(snaps) else []:
        text = read_file(os.path.join(snaps, snap, "README.md"))
        if text:
            return text
    return ""


def _guess_flags(repo):
    n = repo.lower()
    rules = [  # (名称匹配, 工具解析器, 推理解析器)
        (r"qwen3[.-]?coder", "qwen3_coder", None),
        (r"qwen3\.[5-9]|qwen3-next|qwen3\.\d+-flash", "qwen3_xml", "qwen3"),
        (r"qwen3", "hermes", "qwen3"),
        (r"qwen2|qwq", "hermes", None),
        (r"gpt-oss", "openai", "openai_gptoss"),
        (r"llama-?4", "llama4_pythonic", None),
        (r"llama-?3", "llama3_json", None),
        (r"mistral|ministral|devstral|magistral", "mistral", None),
        (r"deepseek-v4\.1", "deepseek_v41", "deepseek_v41"),
        (r"deepseek-v4", "deepseek_v4", "deepseek_v4"),
        (r"deepseek-v3\.[12]", "deepseek_v31", "deepseek_v3"),
        (r"glm-?4\.[56]", "glm45", "glm45"),
        (r"glm-?(4\.7|5)", "glm47", "glm47"),
        (r"gemma-?4", "gemma4", "gemma4"),
        (r"kimi-k3", "kimi_k3", "kimi_k3"),
        (r"kimi-k2", "kimi_k2", "kimi_k2"),
        (r"minimax-m3", "minimax_m3", "minimax_m3"),
        (r"minimax-m2", "minimax_m2", "minimax_m2"),
        (r"nemotron", "hermes", "nemotron_v3"),
        (r"granite", "granite", None),
    ]
    for pat, tool, reason in rules:
        if re.search(pat, n):
            return tool, reason
    return "hermes", None


def vllm_recommend(repo):
    """从模型 README（本地 → 原始模型）读取推荐的解析器，读不到再按模型名推断。"""
    tool, reason, base = _readme_flags(_local_readme(repo))
    source = "readme" if tool else None
    if not tool and base:
        try:
            btool, breason, _ = _readme_flags(fetch(f"https://huggingface.co/{base}/raw/main/README.md", timeout=8,
                                                    headers=hf_headers()))
            if btool:
                tool, reason, source = btool, reason or breason, "base"
        except Exception:
            pass
    if not tool:
        gtool, greason = _guess_flags(repo)
        tool, reason, source = gtool, reason or greason, "guess"
    args = f"--enable-auto-tool-choice --tool-call-parser {tool}"
    if reason:
        args += f" --reasoning-parser {reason}"
    return {"tool": tool, "reasoning": reason, "source": source, "base": base, "args": args}


def is_model_container(c):
    return c["labels"].get("spark.vllm") == "1" or bool(c["labels"].get("spark.engine"))


def model_port(c):
    """vLLM 发布端口；TensorFold 用主机网络，端口记在标签里。"""
    return c["ports"][0] if c["ports"] else int(c["labels"].get("spark.port") or 0) or None


def vllm_deployments():
    out = []
    for c in containers():
        if not is_model_container(c):
            continue
        engine = c["labels"].get("spark.engine") or "vllm"
        port, ready = model_port(c), False
        if c["state"] == "running" and port:
            try:
                path = "/health" if engine == "tensorfold" else "/v1/models"
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=1):
                    ready = True
            except Exception:
                pass
        out.append({**c, "ports": [port] if port else [], "engine": engine,
                    "model": c["labels"].get("spark.model"), "ready": ready})
    return out


def vllm_image_pull(job):
    img, tag = split_image(job["name"])
    try:
        conn = DockerConn(timeout=24 * 3600)
        conn.request("POST", f"/images/create?fromImage={quote(img)}" + (f"&tag={quote(tag)}" if tag else ""))
        resp = conn.getresponse()
        if resp.status >= 400:
            return finish(job, json.loads(resp.read()).get("message"))
        layers = {}
        while line := resp.readline():
            ev = json.loads(line)
            if "error" in ev:
                return finish(job, ev["error"])
            d = ev.get("progressDetail") or {}
            job["detail"] = ev.get("status", "")
            if ev.get("id") and d.get("total"):
                layers[ev["id"]] = (d.get("current", 0), d["total"])
                job["completed"] = sum(c for c, _ in layers.values())
                job["total"] = sum(t for _, t in layers.values())
        finish(job)
    except Exception as e:
        finish(job, str(e))


# ---------- TensorFold（Qwen3.8-Flash-Next 单机方案） ----------
# 配置取自 MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold（main@20e15e4 的 scripts/config.sh 与 start.sh，
# 已审查）。不执行方案自带的脚本，用相同的镜像、环境变量与参数直接创建容器；改为只监听本机。

TF_IMAGE = "tensorfold-qwen38:v1.1.4-eab8582"
TF_PREBUILT = ("ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold"
               "@sha256:5c6c42b9747318c1cd8b6cb85cb69493fff2e82cb23b61e49b9cf54cd09ac28e")
TF_CONTAINER = "qwen38-flash-next-tf"
TF_PORT = 8888
TF_SERVED_NAME = "Qwen3.8-Flash-Next"
TF_RECOMMENDED = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
TF_WEIGHTS_GIB = 81   # 加载 NVIDIA FP4 QAD 检查点占用的内存（方案实测，开启视觉）
TF_RESERVE_GIB = 9    # 引擎始终保留的空闲内存
TF_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TF_FLASHNEXT_DRAFTS": "6", "TF_FLASHNEXT_SAMPLED_HEADS": "1", "TF_FLASHNEXT_SAMPLE_CAND": "1",
    "TF_FLASHNEXT_GATHER_EARLY": "1", "TF_FLASHNEXT_HOST_DRAW_THREADS": "7", "TF_FLASHNEXT_KEEP_LATE": "1",
    "TF_FLASHNEXT_GATHER_READ": "pread", "TF_FLASHNEXT_DRAFT_AHEAD": "1", "TF_FLASHNEXT_WARM_AHEAD": "1",
    "TF_FLASHNEXT_PREFILL_TAIL": "512", "TF_FLASHNEXT_PROMPT_HC2": "1", "TF_FLASHNEXT_PROMPT_SHARED_DIRECT": "1",
    "TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS": "256", "TF_FLASHNEXT_HEAD_PIECE": "1536", "TF_FLASHNEXT_PROMPT_GEMM2": "1",
    "TF_FLASHNEXT_SHARED_SIDE": "1", "TF_FLASHNEXT_GDN_LANES": "1", "TF_FLASHNEXT_CONFIDENCE": "-0.4",
    "TF_FLASHNEXT_PRODUCT_STREAMS": "5",
    "TENSORFOLD_MAX_IMAGES": "50", "TENSORFOLD_IMAGE_TOKENS": "16384", "TENSORFOLD_VIDEO_TOKENS": "16384",
    "TENSORFOLD_VISION_WORKSPACE_MIB": "2048", "TENSORFOLD_MEMORY_RESERVE_GIB": str(TF_RESERVE_GIB),
    "TENSORFOLD_NO_UPDATE_CHECK": "1",
}


def tf_kv_gib(context):
    return -(-17888 * context // 2**30)  # fp8 KV，每 token 17888 字节，向上取整


def tf_mem_need_gib(context, vision=True):
    """与方案 start.sh 的启动前检查一致：权重 + 一个满窗口 KV + 8 + 视觉 2 + 保留 + 2。"""
    return TF_WEIGHTS_GIB + tf_kv_gib(context) + 8 + (2 if vision else 0) + TF_RESERVE_GIB + 2


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 2**20
    return 0


def hf_snapshot(repo):
    d = hf_repo_dir(repo)
    rev = read_file(os.path.join(d, "refs", "main"))
    snap = os.path.join(d, "snapshots", rev)
    return (snap, rev) if rev and os.path.isfile(os.path.join(snap, "config.json")) else (None, None)


def tf_compatible(config):
    """TensorFold 方案只跑 Qwen3.8-Flash-Next（model_type qwen4_exp）的 ModelOpt NVFP4 检查点（与方案 prepare.sh 的校验一致）。"""
    config = config or {}
    return config.get("model_type") == "qwen4_exp" and \
        str((config.get("quantization_config") or {}).get("quant_method", "")).lower() == "modelopt"


def tf_eligible(repo):
    snap, _ = hf_snapshot(repo)
    if not snap:
        return False
    try:
        return tf_compatible(json.load(open(os.path.join(snap, "config.json"))))
    except (OSError, ValueError):
        return False


def tf_search(q):
    """在 Hugging Face 上找 TensorFold 能跑的检查点；推荐（方案实测）的排最前。"""
    params = [("search", "Flash-Next"), ("limit", "300"), ("sort", "downloads"), ("direction", "-1")]
    params += [("expand[]", f) for f in HF_FIELDS + ("config",)]
    data = json.loads(fetch("https://huggingface.co/api/models?" + urlencode(params), headers=hf_headers()))
    words = q.lower().split()
    out = [{"id": m["id"], "params": (m.get("safetensors") or {}).get("total"),
            "downloads": m.get("downloads"), "likes": m.get("likes"), "gated": bool(m.get("gated")),
            "library": m.get("library_name"), "updated": m.get("lastModified"),
            "recommended": m["id"] == TF_RECOMMENDED}
           for m in data if tf_compatible(m.get("config")) and all(w in m["id"].lower() for w in words)]
    out.sort(key=lambda m: (not m["recommended"], -(m["downloads"] or 0)))
    return out


def tf_image():
    try:
        info = docker("GET", f"/images/{quote(TF_IMAGE, safe='')}/json", timeout=5)
    except Exception:
        return None
    labels = (info.get("Config") or {}).get("Labels") or {}
    return labels if labels.get("tf.kernels") == "embedded" else None


def tf_status():
    pulling = any(j["kind"] == "image" and j["name"] == TF_PREBUILT and j.get("status") not in ("done", "error")
                  and not j.get("paused") for j in disk_jobs())
    return {"present": tf_image() is not None, "pulling": pulling, "image": TF_IMAGE,
            "mem_available": mem_available_gib(), "weights_gib": TF_WEIGHTS_GIB, "reserve_gib": TF_RESERVE_GIB,
            "recommended": TF_RECOMMENDED}


def tf_deploy(spec):
    repo = spec["model"]
    context = int(spec.get("context") or 262144)
    parallel = int(spec.get("parallel") or 2)
    vision = bool(spec.get("vision", True))
    if not (1 <= context <= 262144 and 1 <= parallel <= 5):
        raise RuntimeError("context must be ≤ 262144 and parallel 1–5")
    labels = tf_image()
    if labels is None:
        raise RuntimeError("ENGINE_MISSING")
    snap, rev = hf_snapshot(repo)
    if not snap or not tf_eligible(repo):
        raise RuntimeError(f"{repo} is not a downloaded Qwen3.8-Flash-Next checkpoint")
    for c in containers():
        if c["name"] == TF_CONTAINER:
            if c["state"] == "running":
                raise RuntimeError("a TensorFold deployment is already running — remove it first")
            docker("DELETE", f"/containers/{c['id']}?force=1")
    if port_open(TF_PORT):
        raise RuntimeError(f"port {TF_PORT} is already in use")
    need, avail = tf_mem_need_gib(context, vision), mem_available_gib()
    if avail < need and not spec.get("force"):
        raise RuntimeError(f"MEM:{need}:{avail}")
    kcache = os.path.join(HOME, ".cache/tensorfold-qwen38", labels.get("tf.patches") or "default")
    os.makedirs(kcache, exist_ok=True)
    sub = os.path.relpath(snap, HF_CACHE)
    args = ["tensorfold-native", "serve", f"/root/.cache/huggingface/{sub}", "--name", TF_SERVED_NAME,
            "--host", "127.0.0.1", "--port", str(TF_PORT), "--context", str(context), "--parallel", str(parallel),
            "--max-tokens", "32768", "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20",
            "--thinking", "--kv-cache", "fp8"]
    if vision:
        args += ["--vision", "--vision-max-images", "50", "--vision-max-videos", "4", "--vision-image-tokens", "16384"]
    body = {"Image": TF_IMAGE, "Cmd": args, "Env": [f"{k}={v}" for k, v in TF_ENV.items()],
            "Labels": {"spark.engine": "tensorfold", "spark.model": repo, "spark.port": str(TF_PORT)},
            "HostConfig": {"NetworkMode": "host", "IpcMode": "host",
                           "Binds": [f"{HF_CACHE}:/root/.cache/huggingface:ro", f"{kcache}:/cache"],
                           "DeviceRequests": [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}],
                           "Ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1},
                                       {"Name": "stack", "Soft": 67108864, "Hard": 67108864}],
                           # 不自动重启：开机时其他模型可能已占内存，避免两者抢内存
                           "RestartPolicy": {"Name": "no"}}}
    return {**create_and_start(body, TF_CONTAINER), "port": TF_PORT, "need": need, "available": avail}


# ---------- Tailscale ----------
# tailscaled 以 root 运行，本用户已设为 operator（tailscale set --operator），无需 sudo 即可控制。

def ts(*args, timeout=20):
    try:
        r = subprocess.run(["tailscale", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)


def _ts_json(*args):
    code, out, err = ts(*args)
    i = out.find("{")
    if i < 0:
        raise RuntimeError(err or out or "tailscale returned no JSON")
    return json.loads(out[i:])


def ts_status():
    if not shutil.which("tailscale"):
        return {"installed": False}
    try:
        d = _ts_json("status", "--json")
    except Exception as e:
        return {"installed": True, "state": "Unavailable", "error": str(e)}
    prefs = {}
    try:
        prefs = _ts_json("debug", "prefs")
    except Exception:
        pass
    users = {str(k): v.get("LoginName") for k, v in (d.get("User") or {}).items()}

    def node(p):
        return {"id": p.get("ID"), "host": p.get("HostName"), "dns": (p.get("DNSName") or "").rstrip("."),
                "ips": p.get("TailscaleIPs") or [], "os": p.get("OS"), "online": bool(p.get("Online")),
                "last_seen": p.get("LastSeen"), "exit_node": bool(p.get("ExitNode")),
                "exit_option": bool(p.get("ExitNodeOption")), "relay": p.get("Relay"), "cur_addr": p.get("CurAddr"),
                "rx": p.get("RxBytes"), "tx": p.get("TxBytes"), "active": bool(p.get("Active")),
                "user": users.get(str(p.get("UserID")))}

    peers = sorted((node(p) for p in (d.get("Peer") or {}).values()), key=lambda n: (not n["online"], n["host"] or ""))
    routes = prefs.get("AdvertiseRoutes") or []
    tailnet = d.get("CurrentTailnet") or {}
    serve = {}
    try:
        serve = _ts_json("serve", "status", "--json")
    except Exception:
        pass
    return {"installed": True, "state": d.get("BackendState"), "auth_url": d.get("AuthURL") or None,
            "version": (d.get("Version") or "").split("-")[0], "tailnet": tailnet.get("Name"),
            "magic_dns": tailnet.get("MagicDNSEnabled"), "suffix": d.get("MagicDNSSuffix"),
            "self": node(d.get("Self") or {}), "peers": peers, "health": d.get("Health") or [],
            "serving": bool((serve or {}).get("TCP")), "serve_pending": ts_serve_state["running"],
            "serve_enable_url": ts_serve_state["enable_url"], "serve_error": ts_serve_state["error"],
            "prefs": {"accept_routes": prefs.get("RouteAll", False), "accept_dns": prefs.get("CorpDNS", True),
                      "ssh": prefs.get("RunSSH", False), "shields_up": prefs.get("ShieldsUp", False),
                      "hostname": prefs.get("Hostname") or "", "exit_node_id": prefs.get("ExitNodeID") or "",
                      "advertise_exit": "0.0.0.0/0" in routes, "operator": prefs.get("OperatorUser")},
            "user": pwd.getpwuid(os.getuid()).pw_name,
            "operator_ok": not prefs or prefs.get("OperatorUser") == pwd.getpwuid(os.getuid()).pw_name or os.getuid() == 0}


def ts_brief():
    """概览页用的简要状态。"""
    if not shutil.which("tailscale"):
        return {"installed": False}
    try:
        d = _ts_json("status", "--json")
    except Exception:
        return {"installed": True, "state": "Unavailable"}
    peers = (d.get("Peer") or {}).values()
    return {"installed": True, "state": d.get("BackendState"), "ip": ((d.get("Self") or {}).get("TailscaleIPs") or [None])[0],
            "online": sum(1 for p in peers if p.get("Online")), "peers": len(peers), "auth_url": d.get("AuthURL") or None}


def ts_action(body):
    act = body.get("action")
    if act == "up":
        return ts_up([])
    if act == "reauth":  # 切换账号 / 重新登录；不用 logout，因为 logout 会清空 operator 等全部设置
        return ts_up(["--force-reauth"])
    if act == "down":
        code, out, err = ts(act)
    elif act == "set":
        opts = {"accept_routes": "--accept-routes", "accept_dns": "--accept-dns", "ssh": "--ssh",
                "shields_up": "--shields-up", "advertise_exit": "--advertise-exit-node"}
        args = [f"{flag}={'true' if body['value'] else 'false'}" for key, flag in opts.items() if body.get("key") == key]
        if body.get("key") == "hostname":
            name = re.sub(r"[^a-zA-Z0-9-]", "", str(body.get("value") or ""))[:63]
            args = [f"--hostname={name}"]
        elif body.get("key") == "exit_node":
            ip = str(body.get("value") or "")
            if ip and not re.fullmatch(r"[0-9a-fA-F.:]+", ip):
                raise RuntimeError("bad exit node")
            args = [f"--exit-node={ip}"] + (["--exit-node-allow-lan-access=true"] if ip else [])
        if not args:
            raise RuntimeError("unknown setting")
        code, out, err = ts("set", *args)
    elif act == "serve":
        if body.get("value"):
            return ts_serve_on()
        ts_serve_state.update(running=False, enable_url=None, error=None)
        code, out, err = ts("serve", "reset")
    elif act == "ping":
        ip = str(body.get("ip") or "")
        if not re.fullmatch(r"[0-9a-fA-F.:]+", ip):
            raise RuntimeError("bad ip")
        code, out, err = ts("ping", "--c", "1", "--timeout", "5s", ip, timeout=10)
        m = re.search(r"pong from \S+ \(([^)]+)\) via (.+?) in (\S+)", out)
        if m:
            return {"ok": True, "via": m.group(2), "latency": m.group(3)}
        raise RuntimeError((err or out or "no reply").strip().splitlines()[-1])
    else:
        raise RuntimeError("unknown action")
    if code != 0:
        raise RuntimeError(err or out or f"tailscale {act} failed")
    return {"ok": True}


def ts_up(args, retried=False):
    """`tailscale up` 未登录时会一直等待浏览器登录（登录链接出现在 status 的 AuthURL 里），所以放后台；
    几秒内就退出则说明出错，把错误返回给页面。"""
    p = subprocess.Popen(["tailscale", "up", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=4)
    except subprocess.TimeoutExpired:
        threading.Thread(target=p.communicate, daemon=True).start()  # 等待登录中，后台读完输出
        return {"ok": True, "waiting": True}
    if p.returncode == 0:
        return {"ok": True}
    # 已改过设置时 tailscale 要求带上所有非默认参数，并给出完整命令：照它的建议重试一次
    m = re.search(r"^\s*tailscale up (.+)$", out, re.M)
    if m and "requires mentioning all" in out and not retried:
        return ts_up([a for a in shlex.split(m.group(1)) if a.startswith("--")] + [a for a in args if a not in m.group(1)],
                     retried=True)
    lines = [l for l in out.strip().splitlines() if l.strip()]
    if any("access denied" in l.lower() for l in lines):
        raise RuntimeError("NO_OPERATOR")
    raise RuntimeError(lines[0] if lines else f"tailscale up failed ({p.returncode})")


ts_serve_state = {"running": False, "enable_url": None, "error": None}
TS_SERVE = [["--https=443", "http://127.0.0.1:11000"],   # 面板
            ["--https=8443", "http://127.0.0.1:8080"]]   # Open WebUI


def ts_serve_on():
    """把面板分享到 tailnet。tailnet 还没开 Serve/HTTPS 时，tailscale 会打印一个启用链接并一直等待：
    把链接交给页面显示，命令留在后台，用户在浏览器里启用后它会自己完成配置。"""
    if ts_serve_state["running"]:
        return {"ok": True, "enable_url": ts_serve_state["enable_url"]}
    ts_serve_state.update(running=True, enable_url=None, error=None)
    found = threading.Event()

    def worker():
        try:
            for args in TS_SERVE:
                p = subprocess.Popen(["tailscale", "serve", "--bg", *args], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, start_new_session=True)
                out = []
                for line in p.stdout:
                    out.append(line)
                    m = re.search(r"https://login\.tailscale\.com/\S+", line)
                    if m:
                        ts_serve_state["enable_url"] = m.group(0)
                        found.set()
                if p.wait() != 0:
                    ts_serve_state["error"] = "".join(out).strip()[-300:] or "tailscale serve failed"
                    return
            ts_serve_state["enable_url"] = None
        finally:
            ts_serve_state["running"] = False
            found.set()

    threading.Thread(target=worker, daemon=True).start()
    found.wait(6)  # 已启用时几秒内完成；未启用时拿到链接就返回
    if ts_serve_state["error"]:
        raise RuntimeError(ts_serve_state["error"])
    return {"ok": True, "enable_url": ts_serve_state["enable_url"]}


def ts_netcheck():
    d = _ts_json("netcheck", "--format=json")
    lat = d.get("RegionLatency") or {}
    best = min(lat.items(), key=lambda kv: kv[1]) if lat else (None, None)
    return {"udp": d.get("UDP"), "ipv4": d.get("IPv4"), "ipv6": d.get("IPv6"), "global_v4": d.get("GlobalV4"),
            "nat_varies": d.get("MappingVariesByDestIP"), "upnp": d.get("UPnP"), "pmp": d.get("PMP"), "pcp": d.get("PCP"),
            "derp": d.get("PreferredDERP"), "derp_ms": round(best[1] / 1e6) if best[1] else None, "regions": len(lat)}


# ---------- opencode：自动把当前部署的模型配置为后端 ----------
# 生成 ~/.config/spark/opencode.json（经 OPENCODE_CONFIG 与用户自己的 ~/.config/opencode 配置合并）；
# 模型集合变化时重写并重启 spark-opencode 服务。

OPENCODE_CONFIG = os.path.join(CONF, "opencode.json")
OPENCODE_PORT = 4096
_ollama_caps = {}  # digest -> (supports tools, context length)


def _openai_models(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
            return json.load(r).get("data") or []
    except Exception:
        return []


def _ollama_model_info(m):
    key = m.get("digest") or m["name"]
    if key not in _ollama_caps:
        try:
            info = ollama("/api/show", {"model": m["name"]}, timeout=10)
            ctx = next((v for k, v in (info.get("model_info") or {}).items() if k.endswith(".context_length")), None)
            _ollama_caps[key] = ("tools" in (info.get("capabilities") or []), ctx)
        except Exception:
            return (False, None)
    return _ollama_caps[key]


def opencode_desired():
    providers, order = {}, []
    deployments = []
    try:
        deployments = vllm_deployments()
    except Exception:
        pass
    for d in deployments:
        if d["state"] != "running" or not d["ready"] or not d["ports"]:
            continue
        port, engine = d["ports"][0], d["engine"]
        models = {}
        for m in _openai_models(port):
            ctx = m.get("max_model_len") or m.get("context_length")
            models[m["id"]] = {"name": f"{m['id']} ({'TensorFold' if engine == 'tensorfold' else 'vLLM'})",
                               "tool_call": True, "reasoning": engine == "tensorfold" or None,
                               "limit": {"context": int(ctx), "output": min(32768, int(ctx) // 2)} if ctx else None}
            models[m["id"]] = {k: v for k, v in models[m["id"]].items() if v is not None}
        if models:
            pid = f"spark-{engine}-{port}"
            providers[pid] = {"npm": "@ai-sdk/openai-compatible",
                              "name": f"Spark · {'TensorFold' if engine == 'tensorfold' else 'vLLM'} :{port}",
                              "options": {"baseURL": f"http://127.0.0.1:{port}/v1"}, "models": models}
            order += [(0 if engine == "tensorfold" else 1, f"{pid}/{mid}") for mid in models]
    tags = try_ollama("/api/tags") or {}
    om = {}
    for m in tags.get("models", []):
        tools, ctx = _ollama_model_info(m)
        if tools:  # opencode 要靠工具调用读写文件、执行命令
            om[m["name"]] = {"name": f"{m['name']} (Ollama)", "tool_call": True,
                             **({"limit": {"context": int(ctx), "output": 8192}} if ctx else {})}
    if om:
        providers["spark-ollama"] = {"npm": "@ai-sdk/openai-compatible", "name": "Spark · Ollama",
                                     "options": {"baseURL": "http://127.0.0.1:11434/v1"}, "models": om}
        order += [(2, f"spark-ollama/{n}") for n in om]
    cfg = {"$schema": "https://opencode.ai/config.json", "provider": providers}
    if order:
        cfg["model"] = sorted(order)[0][1]
    return cfg


def opencode_sync(force=False):
    cfg = opencode_desired()
    old = _read_json(OPENCODE_CONFIG)
    if old == cfg and not force:
        return False
    _write_json(OPENCODE_CONFIG, cfg)
    env = {**os.environ, **USER_ENV}
    if subprocess.run(["systemctl", "--user", "is-active", "spark-opencode"], capture_output=True, text=True,
                      env=env).stdout.strip() == "active":
        subprocess.run(["systemctl", "--user", "restart", "spark-opencode"], env=env, capture_output=True, timeout=60)
    return True


def opencode_status():
    cfg = _read_json(OPENCODE_CONFIG) or {}
    return {"running": port_open(OPENCODE_PORT), "default": cfg.get("model"),
            "password": read_file(os.path.join(CONF, "opencode_token")),
            "models": [f"{pid}/{mid}" for pid, p in (cfg.get("provider") or {}).items() for mid in p.get("models", {})]}


# ---------- 系统 ----------

SCREEN_FLAG = os.path.join(CONF, "screen_off")


def screen_state():
    bl = (glob.glob("/sys/class/backlight/*") or [None])[0]
    lit = None
    if bl:
        lit = read_file(os.path.join(bl, "brightness"), "0") != "0" and read_file(os.path.join(bl, "bl_power"), "0") == "0"
    return {"off": os.path.exists(SCREEN_FLAG), "lit": lit, "service": run("systemctl", "is-active", "lid-backlight").strip() == "active"}


def desktop_state():
    return {"active": run("systemctl", "is-active", "display-manager").strip() == "active",
            "default": run("systemctl", "get-default").strip(),
            "control": os.path.exists(SPARK_CTL) and subprocess.run(
                ["sudo", "-n", "-l", SPARK_CTL], capture_output=True).returncode == 0}


def network():
    links = []
    for a in json.loads(run("ip", "-j", "addr") or "[]"):
        if a["ifname"] == "lo" or a["ifname"].startswith(("veth", "br-")):
            continue
        v4 = [f'{x["local"]}/{x["prefixlen"]}' for x in a.get("addr_info", []) if x["family"] == "inet"]
        v6 = [x["local"] for x in a.get("addr_info", []) if x["family"] == "inet6" and x.get("scope") == "global"]
        links.append({"name": a["ifname"], "up": a.get("operstate") == "UP", "mac": a.get("address"),
                      "ipv4": v4, "ipv6": v6})
    route = (json.loads(run("ip", "-j", "route", "show", "default") or "[]") or [{}])[0]
    wifi = None
    for line in run("nmcli", "-t", "-f", "ACTIVE,SSID,SIGNAL,FREQ,SECURITY", "dev", "wifi").splitlines():
        parts = re.split(r"(?<!\\):", line)
        if parts[0] == "yes":
            wifi = {"ssid": parts[1].replace("\\:", ":"), "signal": int(parts[2] or 0),
                    "freq": parts[3], "security": parts[4]}
            break
    if wifi and route.get("dev"):
        link = run("iw", "dev", route["dev"], "link")
        m = re.search(r"signal:\s*(-?\d+)", link)
        r = re.search(r"rx bitrate:\s*([\d.]+ \S+)", link)
        wifi.update(dbm=int(m.group(1)) if m else None, rate=r.group(1) if r else None)
    dns = []
    if route.get("dev"):
        dns = re.findall(r"IP4\.DNS\[\d+\]:(\S+)", run("nmcli", "-t", "-f", "IP4.DNS", "dev", "show", route["dev"]))
    return {"links": links, "gateway": route.get("gateway"), "dev": route.get("dev"), "wifi": wifi, "dns": dns}


def port_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def internet_check():
    t = time.time()
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=3):
            return {"ok": True, "latency": round((time.time() - t) * 1000)}
    except OSError:
        return {"ok": False, "latency": None}


class Sampler:
    """后台每秒采样一次，请求直接读缓存。"""

    def __init__(self):
        self.data, self.slow, self.updates, self.internet = {}, {}, None, None
        self._cpu = self._net = self._disk = None
        for fn in (self._loop, self._slow_loop, self._updates_loop, self._internet_loop):
            threading.Thread(target=fn, daemon=True).start()

    def _cpu_pct(self):
        with open("/proc/stat") as f:
            rows = [l.split() for l in f if l.startswith("cpu")]
        cur = [(int(r[4]) + int(r[5]), sum(map(int, r[1:9]))) for r in rows]
        prev, self._cpu = self._cpu, cur
        if not prev:
            return 0, []
        pct = [100 * (1 - (i - pi) / max(t - pt, 1)) for (i, t), (pi, pt) in zip(cur, prev)]
        return pct[0], pct[1:]

    def _rate(self, attr, rx, tx):
        now, prev = time.time(), getattr(self, attr)
        setattr(self, attr, (now, rx, tx))
        if not prev:
            return 0, 0
        dt = now - prev[0]
        return max(rx - prev[1], 0) / dt, max(tx - prev[2], 0) / dt

    def _net_rate(self):
        rx = tx = 0
        with open("/proc/net/dev") as f:
            for line in list(f)[2:]:
                name, vals = line.split(":", 1)
                if name.strip() == "lo" or name.strip().startswith(("docker", "veth", "br-")):
                    continue
                v = vals.split()
                rx, tx = rx + int(v[0]), tx + int(v[8])
        return self._rate("_net", rx, tx)

    def _disk_rate(self):
        rd = wr = 0
        with open("/proc/diskstats") as f:
            for line in f:
                v = line.split()
                if re.fullmatch(r"nvme\d+n\d+|sd[a-z]+|vd[a-z]+|mmcblk\d+", v[2]):
                    rd, wr = rd + int(v[5]) * 512, wr + int(v[9]) * 512
        return self._rate("_disk", rd, wr)

    def _gpu(self):
        out = run("nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw,"
                  "clocks.gr,clocks.max.gr,driver_version", "--format=csv,noheader,nounits")
        if not out.strip():
            return None
        name, util, temp, power, clk, clkmax, drv = [x.strip() for x in out.splitlines()[0].split(",")]
        apps = []
        try:
            root = ET.fromstring(run("nvidia-smi", "-q", "-x"))
            for p in root.iter("process_info"):
                mem = (p.findtext("used_memory") or "").split()
                apps.append({"pid": p.findtext("pid"), "type": p.findtext("type"),
                             "name": os.path.basename(p.findtext("process_name") or ""),
                             "mem_mib": num(mem[0]) if mem else None})
        except ET.ParseError:
            pass
        return {"name": name, "util": num(util), "temp": num(temp), "power": num(power),
                "clock": num(clk), "clock_max": num(clkmax), "driver": drv, "apps": apps}

    def _mem(self):
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k] = int(v.split()[0]) * 1024
        return {"total": info["MemTotal"], "used": info["MemTotal"] - info["MemAvailable"],
                "cached": info.get("Cached", 0), "swap_total": info["SwapTotal"],
                "swap_used": info["SwapTotal"] - info["SwapFree"]}

    def _temp(self):
        temps = []
        for z in sorted(os.listdir("/sys/class/thermal")):
            try:
                with open(f"/sys/class/thermal/{z}/temp") as f:
                    temps.append(int(f.read()) / 1000)
            except OSError:
                pass
        return max(temps) if temps else None

    def _loop(self):
        while True:
            cpu, cores = self._cpu_pct()
            rx, tx = self._net_rate()
            rd, wr = self._disk_rate()
            disk = shutil.disk_usage("/")
            self.data = {"time": time.time(), "gpu": self._gpu(), "mem": self._mem(),
                         "cpu": {"util": cpu, "cores": cores, "temp": self._temp(), "load": os.getloadavg(),
                                 "count": os.cpu_count()},
                         "disk": {"total": disk.total, "used": disk.used, "read": rd, "write": wr},
                         "net": {"rx": rx, "tx": tx, "internet": self.internet, **self.slow.get("net", {})},
                         **{k: v for k, v in self.slow.items() if k != "net"}, "updates": self.updates}
            time.sleep(1)

    @staticmethod
    def _http_ok(url):
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            return False

    def _slow_loop(self):
        while True:
            ps = try_ollama("/api/ps") or {}
            tags = try_ollama("/api/tags")
            try:
                ctrs, docker_ok = containers(), True
            except Exception:
                ctrs, docker_ok = [], False
            try:
                net = network()
            except Exception:
                net = {}
            self.slow = {
                "services": [
                    {"name": "Ollama", "ok": tags is not None, "tab": "models"},
                    {"name": "Docker", "ok": docker_ok, "tab": "docker"},
                    {"name": "Open WebUI", "ok": self._http_ok("http://127.0.0.1:8080/health"), "open": "webui"},
                    {"name": "JupyterLab", "ok": self._http_ok("http://127.0.0.1:11001/jupyter/api"), "open": "jupyter"},
                    {"name": "VS Code", "ok": port_open(11002), "open": "code"},
                    {"name": "opencode", "ok": port_open(OPENCODE_PORT), "open": "opencode"},
                ],
                "containers": [{k: v for k, v in c.items() if k != "labels"} for c in ctrs],
                "models_loaded": [{"name": m["name"], "size": m.get("size_vram") or m.get("size")}
                                  for m in ps.get("models", [])],
                "vllm": [{"name": c["name"], "model": c["labels"].get("spark.model"),
                          "engine": c["labels"].get("spark.engine") or "vllm",
                          "ports": [model_port(c)] if model_port(c) else []}
                         for c in ctrs if is_model_container(c) and c["state"] == "running"],
                "desktop": desktop_state(),
                "screen": screen_state(),
                "tailscale": ts_brief(),
                "net": net,
                "hostname": os.uname().nodename,
                "uptime": float(open("/proc/uptime").read().split()[0]),
            }
            prune_jobs()
            try:
                reconcile_jobs()
            except Exception:
                pass
            self._oc_tick = getattr(self, "_oc_tick", 0) + 1
            if self._oc_tick % 5 == 1:
                try:
                    opencode_sync()
                except Exception:
                    pass
            time.sleep(3)

    def _updates_loop(self):
        while True:
            self.updates = len(run("checkupdates", timeout=120).splitlines())
            time.sleep(1800)

    def _internet_loop(self):
        while True:
            self.internet = internet_check()
            time.sleep(15)


sampler = Sampler()


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _stream_line(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()

    def _host_ok(self):
        if ALLOWED_HOST.match(self.headers.get("Host", "")):
            return True
        self.send_error(403)
        return False

    def do_GET(self):
        if not self._host_ok():
            return
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        p = url.path
        try:
            if p == "/api/stats":
                self._json(sampler.data)
            elif p == "/api/jobs":
                self._json(list_jobs())
            elif p == "/api/ollama/models":
                tags = ollama("/api/tags").get("models", [])
                loaded = {m["name"]: m for m in ollama("/api/ps").get("models", [])}
                self._json([{"name": m["name"], "size": m["size"], "modified": m["modified_at"],
                             "params": m["details"].get("parameter_size"),
                             "quant": m["details"].get("quantization_level"),
                             "family": m["details"].get("family"),
                             "loaded": m["name"] in loaded,
                             "vram": loaded.get(m["name"], {}).get("size_vram")} for m in tags])
            elif p == "/api/ollama/search":
                query, page = q.get("q", ""), int(q.get("page", "1"))
                self._json(cached(("os", query, page), 600, lambda: ollama_search(query, page)))
            elif p == "/api/ollama/tags":
                name = q["name"]
                self._json(cached(("ot", name), 600, lambda: ollama_tags(name)))
            elif p == "/api/hf/search":
                query, task = q.get("q", ""), q.get("task", "llm")
                self._json(cached(("hf", query, task), 300, lambda: hf_search(query, task)))
            elif p == "/api/opencode":
                self._json(opencode_status())
            elif p == "/api/tailscale/status":
                self._json(ts_status())
            elif p == "/api/tailscale/netcheck":
                self._json(cached(("netcheck",), 60, ts_netcheck) if q.get("fresh") != "1" else ts_netcheck())
            elif p == "/api/tf/status":
                self._json(tf_status())
            elif p == "/api/tf/search":
                query = q.get("q", "")
                self._json(cached(("tf", query), 300, lambda: tf_search(query)))
            elif p == "/api/hf/local":
                self._json(hf_local())
            elif p == "/api/hf/token":
                self._json({"set": bool(read_file(HF_TOKEN_FILE))})
            elif p == "/api/vllm/recommend":
                self._json(vllm_recommend(q["model"]))
            elif p == "/api/vllm/status":
                image, present = vllm_image()
                self._json({"image": image, "present": present, "deployments": vllm_deployments()})
            elif p == "/api/docker/containers":
                self._json([{k: v for k, v in c.items() if k != "labels"} for c in containers()])
            elif p == "/api/docker/images":
                self._json(images())
            elif p == "/api/docker/volumes":
                self._json(volumes())
            elif p == "/api/docker/logs":
                cid, tail = q["id"], int(q.get("tail", "300"))
                data = docker("GET", f"/containers/{quote(cid)}/logs?stdout=1&stderr=1&timestamps=0&tail={tail}",
                              raw=True)
                self._json({"logs": demux_logs(data)})
            elif p == "/open/opencode":
                # 用密码向 opencode 申请一次性配对码，再让浏览器走 /auth/connect/<code>：
                # opencode 会设置 30 天的会话 cookie 并跳到首页，无需手动输入地址和密码
                import base64
                token = read_file(os.path.join(CONF, "opencode_token"))
                req = urllib.request.Request(f"http://127.0.0.1:{OPENCODE_PORT}/api/pair", data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json", "Authorization": "Basic " +
                                                      base64.b64encode(f"opencode:{token}".encode()).decode()})
                with urllib.request.urlopen(req, timeout=10) as r:
                    code = json.load(r)["code"]
                host = self.headers.get("Host", "localhost")
                port = host.rsplit(":", 1)[1] if ":" in host else "80"
                self.send_response(302)
                self.send_header("Location", f"http://opencode.localhost:{port}/auth/connect/{quote(code, safe='')}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            elif p == "/open/code":
                self.send_response(302)
                self.send_header("Location", f"/code/?tkn={read_file(VSCODE_TOKEN_FILE)}")
                self.end_headers()
            elif p == "/open/jupyter":
                self.send_response(302)
                self.send_header("Location", f"/jupyter/lab?token={read_file(TOKEN_FILE)}")
                self.end_headers()
            elif p == "/":
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        # 自定义请求头挡住跨站请求（其他网页无法不经 CORS 预检就带上它）
        if not self._host_ok():
            return
        if self.headers.get("X-Spark") != "1":
            return self.send_error(403)
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        path = urlparse(self.path).path
        try:
            if path == "/api/ollama/pull":
                return self._json(start_download("ollama", body["name"].strip()))
            if path == "/api/hf/download":
                return self._json(start_download("hf", body["repo"].strip()))
            if path == "/api/opencode/sync":
                opencode_sync(force=True)
                return self._json(opencode_status())
            if path == "/api/tailscale":
                return self._json(ts_action(body))
            if path == "/api/tf/prepare":
                return self._json(start_download("image", TF_PREBUILT))
            if path == "/api/tf/deploy":
                return self._json(tf_deploy(body))
            if path == "/api/vllm/pull-image":
                job = new_job("image", body.get("image") or vllm_image()[0])
                threading.Thread(target=vllm_image_pull, args=(job,), daemon=True).start()
                return self._json(job)
            if path == "/api/jobs/pause":
                pause_job(body["id"])
            elif path == "/api/jobs/resume":
                resume_job(body["id"])
            elif path == "/api/jobs/remove":
                remove_job(body["id"])
            elif path == "/api/vllm/deploy":
                return self._json(vllm_deploy(body))
            elif path == "/api/hf/delete":
                for j in disk_jobs():
                    if j["kind"] == "hf" and j["name"] == body["repo"]:
                        remove_job(j["id"])
                d = hf_repo_dir(body["repo"])
                if os.path.dirname(d) != os.path.join(HF_CACHE, "hub"):
                    return self._json({"error": "bad repo"}, 400)
                shutil.rmtree(d, ignore_errors=True)
            elif path == "/api/hf/token":
                write_secret(HF_TOKEN_FILE, body.get("token", "").strip())
            elif path == "/api/ollama/delete":
                req = urllib.request.Request(OLLAMA + "/api/delete", method="DELETE",
                                             data=json.dumps({"model": body["name"]}).encode())
                urllib.request.urlopen(req, timeout=30).read()
            elif path == "/api/ollama/load":
                ollama("/api/generate", {"model": body["name"], "keep_alive": -1}, timeout=600)
            elif path == "/api/ollama/unload":
                ollama("/api/generate", {"model": body["name"], "keep_alive": 0}, timeout=60)
            elif path == "/api/screen":
                # 由 root 运行的 lid-backlight 服务读取这个开关文件并控制背光
                if body.get("off"):
                    open(SCREEN_FLAG, "w").close()
                elif os.path.exists(SCREEN_FLAG):
                    os.remove(SCREEN_FLAG)
            elif path == "/api/system":
                action = body.get("action")
                if action not in ("desktop-on", "desktop-off", "reboot", "poweroff"):
                    return self._json({"error": "unknown action"}, 400)
                if not desktop_state()["control"]:
                    return self._json({"error": "NO_CONTROL"}, 403)
                subprocess.Popen(["sudo", "-n", SPARK_CTL, action], start_new_session=True)
            elif path == "/api/docker/action":
                cid, act = quote(body.get("id", "")), body.get("action")
                if not cid:
                    return self._json({"error": "missing container id"}, 400)
                if act in ("start", "stop", "restart"):
                    docker("POST", f"/containers/{cid}/{act}", timeout=60)
                elif act == "remove":
                    docker("DELETE", f"/containers/{cid}?force=1")
                else:
                    return self._json({"error": "unknown action"}, 400)
            elif path == "/api/docker/run":
                return self._json(docker_run(body))
            elif path == "/api/docker/pull":
                return self._docker_pull(body["image"].strip())
            elif path == "/api/docker/image-remove":
                docker("DELETE", f"/images/{quote(body['id'])}?force=1")
            elif path == "/api/docker/volume-remove":
                docker("DELETE", f"/volumes/{quote(body['name'])}")
            elif path == "/api/docker/prune":
                docker("POST", "/images/prune?filters=" + quote('{"dangling":["true"]}'), timeout=300)
            else:
                return self.send_error(404)
            self._json({"ok": True})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _docker_pull(self, image):
        self._stream_start()
        img, tag = split_image(image)
        conn = DockerConn(timeout=3600)
        try:
            conn.request("POST", f"/images/create?fromImage={quote(img)}&tag={quote(tag)}")
            resp = conn.getresponse()
            if resp.status >= 400:
                return self._stream_line({"error": json.loads(resp.read()).get("message")})
            layers = {}
            while line := resp.readline():
                ev = json.loads(line)
                if "error" in ev:
                    return self._stream_line({"error": ev["error"]})
                d = ev.get("progressDetail") or {}
                if ev.get("id") and d.get("total"):
                    layers[ev["id"]] = (d.get("current", 0), d["total"])
                done = sum(c for c, _ in layers.values())
                total = sum(t for _, t in layers.values())
                self._stream_line({"status": ev.get("status"), "completed": done, "total": total})
            self._stream_line({"status": "success"})
        except Exception as e:
            self._stream_line({"error": str(e)})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
