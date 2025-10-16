# app/streamlit_app.py
# Fresh UI: Transpile & Benchmark • Python ⇄ C++ / Java
# - New editor look for source & target (light, modern)
# - OpenAI / Claude only; convert + run with timing
# - No connection check; no system prompt; no offline mode
# - Swap button removed; Download buttons removed

import os
import re
import sys
import time
import shutil
import tempfile
import subprocess
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ──────────────────────────────────────────────────────────────
# Load environment variables (from project root)
# ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transpile & Benchmark • Python ⇄ C++ / Java",
    page_icon="🧪",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Fresh, light visual design for editors and buttons
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page & sidebar */
[data-testid="stAppViewContainer"] { background: #f6f8fb; }
section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e6e9f2; }

/* Cards */
.card {
  background: #ffffff;
  border: 1px solid #e6e9f2;
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 1px 3px rgba(16,24,40,.06);
}

/* Hero */
.hero-title { font-size: 1.8rem; font-weight: 750; color: #0f172a; margin: 0; }
.hero-subtitle { color: #475569; font-size: .98rem; margin-top: 6px; }

/* Editor wrapper */
.editor {
  border: 1px solid #dfe5f2;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}

/* Editor header toolbar */
.editor-header {
  background: linear-gradient(180deg, #f2f7ff 0%, #eef5ff 100%);
  border-bottom: 1px solid #dfe5f2;
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 12px;
}
.lang-chip {
  background: #eaf2ff;
  color: #0a3d91;
  border: 1px solid #bcd2ff;
  border-radius: 999px;
  padding: 4px 10px;
  font-weight: 600;
  font-size: .84rem;
}
.header-actions { display: flex; gap: 8px; }

/* Light, readable text areas */
.stTextArea textarea {
  background: #fbfcff !important;
  color: #0f172a !important;
  border: none !important;
  border-radius: 0 !important;
  min-height: 260px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
  font-size: 0.92rem !important;
}

/* Code blocks */
pre, code, .stCodeBlock, .stMarkdown pre {
  background: #fbfcff !important;
  color: #0f172a !important;
  border: 1px solid #e6e9f2 !important;
  border-radius: 12px !important;
  font-size: 0.92rem !important;
}

/* Input/select */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
  background: #fbfcff !important;
  color: #0f172a !important;
  border: 1px solid #d6dbe8 !important;
  border-radius: 10px !important;
}

/* Uniform buttons: light pastel blue, visible text, no hover change */
div.stButton > button {
  width: 100% !important;
  border-radius: 10px !important;
  border: 1px solid #bcd2ff !important;
  background: #eaf2ff !important;
  color: #0a3d91 !important;
  padding: 10px 12px !important;
  font-weight: 650 !important;
  letter-spacing: .2px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04) !important;
}

/* Utility */
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#f2f4f7; color:#374151; font-size:.78rem; margin-right:6px;
  border:1px solid #e6e9f2;
}
h2, h3, h4, label, p, span { color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Hero section
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="card" style="margin-bottom: 14px;">
  <div class="hero-title">Transpile & Benchmark</div>
  <div class="hero-subtitle">Fresh, minimal editors • Convert between <b>Python</b>, <b>C++</b>, and <b>Java</b> • Run both • See timings</div>
  <div style="margin-top:6px;">
    <span class="badge">OpenAI / Claude</span>
    <span class="badge">Local execution</span>
    <span class="badge">New editor design</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Transpile core
# ──────────────────────────────────────────────────────────────
LANGS = ["Python", "C++", "Java"]
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

SYSTEM_PROMPT_BASE = """You are a compiler-grade code transpiler.
Convert the given source code from {src} to {tgt}.
Preserve logic, naming, and structure as much as possible.
Output only valid {tgt} code (no markdown fences).
The response needs to produce an identical output in the fastest possible time."""

def extract_code(text: str) -> str:
    m = re.search(r"```[a-zA-Z0-9+]*\n(.*?)```", text, flags=re.S)
    return m.group(1).strip() if m else text.strip()

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Prefer environment; fall back to st.secrets if available; else default."""
    val = os.getenv(key)
    if val is not None:
        return val
    try:
        return st.secrets.get(key, default)
    except FileNotFoundError:
        return default

@dataclass
class LLMRequest:
    system: str
    user: str
    model: str

def call_openai(req: LLMRequest) -> str:
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY", "").strip() or get_secret("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing in .env at project root.")
    client = OpenAI(api_key=key)
    resp = client.responses.create(
        model=req.model,
        input=[{"role": "system", "content": req.system},
               {"role": "user", "content": req.user}],
        temperature=0.1,
    )
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()
    return "\n".join(
        getattr(c, "text", "")
        for item in getattr(resp, "output", [])
        for c in getattr(item, "content", [])
        if getattr(c, "type", None) in ("output_text", "text")
    ).strip()

def call_anthropic(req: LLMRequest) -> str:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "").strip() or get_secret("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing in .env at project root.")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=req.model,
        system=req.system,
        messages=[{"role": "user", "content": req.user}],
        max_tokens=4000,
    )
    return "".join(getattr(b, "text", "") for b in msg.content).strip()

# ──────────────────────────────────────────────────────────────
# Java / C++ availability helpers
# ──────────────────────────────────────────────────────────────
def _which(cmd: str, path: Optional[str] = None) -> Optional[str]:
    return shutil.which(cmd, path=path) if path else shutil.which(cmd)

def _set_path(bin_dir: str):
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH','')}" if bin_dir not in os.environ.get("PATH","") else os.environ["PATH"]

def resolve_java_paths() -> tuple[Optional[str], Optional[str]]:
    """
    Try to locate java & javac. Attempt PATH -> macOS java_home -> JAVA_HOME.
    If found via java_home/JAVA_HOME, prepend its bin to PATH for this process.
    """
    java  = _which("java")
    javac = _which("javac")

    if java and javac:
        return java, javac

    # macOS: prefer /usr/libexec/java_home
    if platform.system() == "Darwin":
        try:
            # Prefer 17; fallback to any
            for v in ("17", "21", "11", ""):
                args = ["/usr/libexec/java_home"] + (["-v", v] if v else [])
                out = subprocess.check_output(args, text=True).strip()
                if out:
                    binp = os.path.join(out, "bin")
                    _set_path(binp)
                    os.environ["JAVA_HOME"] = out
                    java  = java  or _which("java",  path=binp)
                    javac = javac or _which("javac", path=binp)
                    if java and javac:
                        return java, javac
        except Exception:
            pass

    # Any OS: try JAVA_HOME if set
    jh = os.environ.get("JAVA_HOME")
    if jh:
        binp = os.path.join(jh, "bin")
        _set_path(binp)
        java  = java  or _which("java",  path=binp)
        javac = javac or _which("javac", path=binp)

    return java, javac

def java_available() -> bool:
    j, jc = resolve_java_paths()
    return bool(j and jc)

def cpp_available() -> bool:
    return bool(_which("g++"))

# ──────────────────────────────────────────────────────────────
# Runners (local execution)
# ──────────────────────────────────────────────────────────────
def run_python(code: str, timeout_s: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "main.py"
        f.write_text(code, encoding="utf-8")
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable or "python", "-u", str(f)],
                               capture_output=True, text=True, timeout=timeout_s)
            return {"ok": p.returncode == 0, "stdout": p.stdout, "stderr": p.stderr,
                    "time_s": time.perf_counter()-t0, "compile_time_s": 0.0}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "Timeout",
                    "time_s": timeout_s, "compile_time_s": 0.0}

def run_cpp(code: str, timeout_s: int) -> dict:
    if not cpp_available():
        hint = "g++ not found on PATH."
        # Helpful hint for Streamlit Cloud
        if os.getenv("STREAMLIT_RUNTIME", "") or os.getenv("STREAMLIT_SERVER_HOST", ""):
            hint += " On Streamlit Cloud, add 'build-essential' to packages.txt."
        return {"ok": False, "stdout": "", "stderr": hint, "time_s": 0.0, "compile_time_s": 0.0}

    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / "main.cpp"
        exe = Path(td) / ("main.exe" if os.name == "nt" else "main")
        cpp.write_text(code, encoding="utf-8")
        ct0 = time.perf_counter()
        comp = subprocess.run(["g++", "-O2", "-std=c++17", str(cpp), "-o", str(exe)],
                              capture_output=True, text=True)
        ct = time.perf_counter() - ct0
        if comp.returncode != 0:
            return {"ok": False, "stdout": comp.stdout, "stderr": comp.stderr,
                    "time_s": 0.0, "compile_time_s": ct}
        t0 = time.perf_counter()
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout_s)
        return {"ok": run.returncode == 0, "stdout": run.stdout, "stderr": run.stderr,
                "time_s": time.perf_counter()-t0, "compile_time_s": ct}

def run_java(code: str, timeout_s: int) -> dict:
    java, javac = resolve_java_paths()
    if not (java and javac):
        hint = "javac/java not found on PATH."
        # Helpful hint for Streamlit Cloud
        if os.getenv("STREAMLIT_RUNTIME", "") or os.getenv("STREAMLIT_SERVER_HOST", ""):
            hint += " On Streamlit Cloud, create packages.txt with 'openjdk-17-jdk-headless'."
        else:
            hint += " Install a JDK (e.g., OpenJDK 17) and set JAVA_HOME."
        return {"ok": False, "stdout": "", "stderr": hint, "time_s": 0.0, "compile_time_s": 0.0}

    m = re.search(r"public\s+class\s+([A-Za-z_]\w*)", code)
    cname = m.group(1) if m else "Main"
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{cname}.java"
        src.write_text(code, encoding="utf-8")
        ct0 = time.perf_counter()
        comp = subprocess.run([javac, str(src)], capture_output=True, text=True, cwd=td)
        ct = time.perf_counter() - ct0
        if comp.returncode != 0:
            return {"ok": False, "stdout": comp.stdout, "stderr": comp.stderr,
                    "time_s": 0.0, "compile_time_s": ct}
        t0 = time.perf_counter()
        run = subprocess.run([java, "-cp", td, cname], capture_output=True, text=True, timeout=timeout_s)
        return {"ok": run.returncode == 0, "stdout": run.stdout, "stderr": run.stderr,
                "time_s": time.perf_counter()-t0, "compile_time_s": ct}

def run_code(lang: str, code: str, timeout_s: int) -> dict:
    if lang == "Python": return run_python(code, timeout_s)
    if lang == "C++": return run_cpp(code, timeout_s)
    if lang == "Java": return run_java(code, timeout_s)
    return {"ok": False, "stdout": "", "stderr": f"Unsupported: {lang}",
            "time_s": 0.0, "compile_time_s": 0.0}

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    engine = st.radio("Engine", ["OpenAI", "Claude 3.5 Sonnet"], index=0)
    model = st.text_input(
        "Model",
        value=DEFAULT_OPENAI_MODEL if engine == "OpenAI" else DEFAULT_ANTHROPIC_MODEL
    )
    c = st.columns([1,1])
    with c[0]:
        src_lang = st.selectbox("Source", LANGS, index=0)
    with c[1]:
        tgt_lang = st.selectbox("Target", LANGS, index=1)

    st.markdown("---")
    enable_exec = st.checkbox("Allow running code locally", value=False)
    timeout_s = st.number_input("Timeout (seconds)", 1, 250, 10)

    # Environment diagnostics (helpful on Streamlit Cloud)
    st.markdown("---")
    st.caption("Environment check")
    st.write(f"Java available: {'✅' if java_available() else '❌'}")
    st.write(f"g++ available: {'✅' if cpp_available() else '❌'}")
    if not java_available():
        st.info("If deploying on Streamlit Cloud, add `openjdk-17-jdk-headless` to packages.txt.")
    if not cpp_available():
        st.info("If deploying on Streamlit Cloud, add `build-essential` to packages.txt.")

# ──────────────────────────────────────────────────────────────
# Default examples
# ──────────────────────────────────────────────────────────────
EXAMPLES = {
    "Python": "for i in range(3):\n    print('i=', i)",
    "C++": '#include <iostream>\nusing namespace std;\nint main(){\n  for(int i=0;i<3;++i){ cout << "i=" << i << endl; }\n}',
    "Java": 'public class Main{ public static void main(String[]a){ for(int i=0;i<3;++i){ System.out.println("i=" + i); } } }'
}

# session state for source / target code
if "src_code" not in st.session_state:
    st.session_state.src_code = EXAMPLES.get("Python", "")
if "tgt_code" not in st.session_state:
    st.session_state.tgt_code = ""

# Keep example aligned with selected source
if st.session_state.src_code == "" and src_lang in EXAMPLES:
    st.session_state.src_code = EXAMPLES[src_lang]

# ──────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="editor">
      <div class="editor-header">
        <div class="lang-chip">Source • {src_lang}</div>
        <div class="header-actions"></div>
      </div>
    """, unsafe_allow_html=True)

    # Source text area
    src_text = st.text_area(
        label="",
        value=st.session_state.src_code,
        height=300,
        key="src_editor",
        placeholder=f"Paste your {src_lang} code here…",
        label_visibility="collapsed",
    )
    st.session_state.src_code = src_text

    # Toolbar row for actions (Load example, Clear)
    sbtns = st.columns([1,1,1])
    with sbtns[0]:
        if st.button("Load example", key="load_example_src"):
            st.session_state.src_code = EXAMPLES[src_lang]
            st.rerun()
    with sbtns[1]:
        if st.button("Clear", key="clear_src"):
            st.session_state.src_code = ""
            st.rerun()
    with sbtns[2]:
        st.write("")  # spacer

    st.markdown("</div>", unsafe_allow_html=True)  # editor
    st.markdown("</div>", unsafe_allow_html=True)  # card

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="editor">
      <div class="editor-header">
        <div class="lang-chip">Target • {tgt_lang}</div>
        <div class="header-actions"></div>
      </div>
    """, unsafe_allow_html=True)

    # Target display (code block)
    out_placeholder = st.empty()
    out_code = st.session_state.get("tgt_code", "")
    if out_code:
        out_placeholder.code(out_code, language=tgt_lang.lower().replace("++", "cpp"))
    else:
        out_placeholder.markdown(
            "<div style='padding:14px;color:#64748b;'>Converted code will appear here…</div>",
            unsafe_allow_html=True
        )

    tbtns = st.columns([1,1,1])
    with tbtns[0]:
        if st.button("Load example", key="load_example_tgt"):
            st.session_state.tgt_code = EXAMPLES[tgt_lang]
            st.rerun()
    with tbtns[1]:
        if st.button("Clear", key="clear_tgt"):
            st.session_state.tgt_code = ""
            st.rerun()
    with tbtns[2]:
        st.write("")  # spacer

    st.markdown("</div>", unsafe_allow_html=True)  # editor
    st.markdown("</div>", unsafe_allow_html=True)  # card

# Convert button row (centered)
c_row = st.columns([1,1,1])
with c_row[1]:
    convert = st.button("Convert ➜", key="convert_btn", use_container_width=True)

# Run & Measure row
rm = st.columns(3, gap="large")
with rm[0]:
    run_src = st.button(f"Run {src_lang}", key="run_src", use_container_width=True)
with rm[1]:
    run_tgt = st.button(f"Run {tgt_lang}", key="run_tgt", use_container_width=True)
with rm[2]:
    run_both = st.button("Run both", key="run_both", use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────
def convert_code(src_lang: str, tgt_lang: str, code: str, model: str, engine: str) -> str:
    sys_prompt = SYSTEM_PROMPT_BASE.format(src=src_lang, tgt=tgt_lang)
    user_prompt = f"Convert the following {src_lang} code into {tgt_lang}. Output only {tgt_lang} code:\n\n{code}"
    if engine == "OpenAI":
        return call_openai(LLMRequest(sys_prompt, user_prompt, model))
    else:
        return call_anthropic(LLMRequest(sys_prompt, user_prompt, model))

if convert:
    try:
        text = convert_code(src_lang, tgt_lang, st.session_state.src_code, model, engine)
        st.session_state.tgt_code = extract_code(text)
        out_placeholder.code(st.session_state.tgt_code, language=tgt_lang.lower().replace("++", "cpp"))
    except Exception as e:
        st.error(f"Conversion failed: {e}")

# ──────────────────────────────────────────────────────────────
# Run & Measure (optional local execution)
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="card" style="margin-top: 10px;">', unsafe_allow_html=True)
st.markdown("### Run & Measure")

if not enable_exec:
    st.info("Enable 'Allow running code locally' in the sidebar to execute programs.")
else:
    cols = st.columns(2, gap="large")

    def show_result(title, r, into):
        with into:
            st.markdown(f"**{title}**")
            if r["compile_time_s"]:
                st.write(f"Compile: `{r['compile_time_s']:.4f}s`")
            st.write(f"Run: `{r['time_s']:.4f}s`")
            if r["stdout"]:
                st.write("**stdout**"); st.code(r["stdout"])
            if r["stderr"]:
                st.write("**stderr**"); st.code(r["stderr"])
            if not r["ok"]:
                st.error("Execution failed.")

    if run_src or run_both:
        with st.spinner(f"Running {src_lang}…"):
            r1 = run_code(src_lang, st.session_state.src_code, timeout_s)
        show_result(f"{src_lang} (source)", r1, cols[0])

    if run_tgt or run_both:
        tcode = st.session_state.tgt_code or ""
        if not tcode:
            st.warning("No converted code yet. Click 'Convert' first.")
        else:
            with st.spinner(f"Running {tgt_lang}…"):
                r2 = run_code(tgt_lang, tcode, timeout_s)
            show_result(f"{tgt_lang} (target)", r2, cols[1])

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    '<div style="text-align:center;color:#6b7280;margin-top:10px;">Built with Streamlit • OpenAI / Claude • Fresh editor UI</div>',
    unsafe_allow_html=True
)
