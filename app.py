import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from service import _build_ui


def main():
    parser = argparse.ArgumentParser(description="WebLogAnalyzer dev mode")
    parser.add_argument("--share",  action="store_true",
                        help="Публичная ссылка (если localhost недоступен)")
    parser.add_argument("--port",   type=int,  default=7860)
    parser.add_argument("--host",   type=str,  default=None,
                        help="Хост (по умолч. пробуем 127.0.0.1, потом 0.0.0.0)")
    args = parser.parse_args()

    demo = _build_ui()

    if args.host:
        demo.launch(server_name=args.host, server_port=args.port,
                    share=args.share, show_error=True)
        return

    for host in ("127.0.0.1", "0.0.0.0"):
        try:
            print(f"Пробуем запустить на {host}:{args.port}...")
            demo.launch(server_name=host, server_port=args.port,
                        share=args.share, show_error=True,
                        prevent_thread_lock=True)
            demo.block_thread()
            return
        except (OSError, ValueError) as e:
            print(f"  {host} недоступен: {e}")
            try:
                demo.close()
            except Exception:
                pass

    print("Localhost недоступен. Создаём публичную ссылку (share=True)...")
    demo.launch(share=True, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
