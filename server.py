"""
SmartRoute - Simple HTTP Server
Chạy: python server.py
Sau đó mở trình duyệt tại: http://localhost:8080
"""

import http.server
import socketserver
import os
import webbrowser
import threading

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Bỏ qua log không cần thiết
        pass

    def end_headers(self):
        # Thêm CORS headers để hỗ trợ ES Modules
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def open_browser():
    import time
    time.sleep(0.5)
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"=== SmartRoute HTTP Server ===")
        print(f"    http://localhost:{PORT}")
        print(f"    Press Ctrl+C to stop")
        print(f"===============================")
        threading.Thread(target=open_browser, daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Server đã dừng]")
