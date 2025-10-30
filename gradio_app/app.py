# gradio_app/app.py
from __future__ import annotations
import os, argparse, socket, importlib
from pathlib import Path
import gradio as gr

def load_pages() -> dict[str, gr.Blocks]:
    pages = {}
    pages_dir = Path(__file__).parent / "pages"
    for py_file in pages_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        module = importlib.import_module(f"gradio_app.pages.{py_file.stem}")
        fn_name = f"create_{py_file.stem}"
        if hasattr(module, fn_name):
            pages[py_file.stem] = getattr(module, fn_name)()
    return pages

pages = load_pages()

with gr.Blocks(title="DataFlow Agent Platform") as app:
    gr.Markdown("# 🌊 DataFlow Agent 多功能平台")
    with gr.Tabs():
        for name, page in pages.items():
            with gr.Tab(name.replace("_", " ").title()):
                page.render()         

# -----------------------------------------------------------
# 启动入口
# -----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio App")
    parser.add_argument(
        "--server_port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        help="优先命令行，其次环境变量 GRADIO_SERVER_PORT，默认 7860",
    )
    return parser.parse_args()


def is_port_free(port: int) -> bool:
    """简单探测端口是否空闲"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("0.0.0.0", port)) != 0


if __name__ == "__main__":
    args = parse_args()
    port = args.server_port
    if not is_port_free(port):
        print(f"⚠️  端口 {port} 已被占用，自动切换到随机空闲端口。"
              " 如需固定端口，请换一个数字或先释放该端口。")
        port = 0  # 让 Gradio 自动选

    app.queue() 
    app.launch(server_name="0.0.0.0", server_port=port)