"""Launch SGLang with SSV4D worker runtime instrumentation enabled."""

from __future__ import annotations

import os
import sys


def main() -> None:
    os.environ.setdefault("SSV4D_RUNTIME_ENABLE", "1")
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        from sglang.launch_server import run_server

        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
