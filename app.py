#!/usr/bin/env python3
"""
Bilingual Telugu-English News Assistant Web Application Server
Serves the reference web application on http://127.0.0.1:8000/
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Local imports
from extractor import extract_article, chunk_text
from rag_engine import rag_pipeline, MODEL_SPECS

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class RAGRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, "File not found")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url_path = urllib.parse.urlparse(self.path).path

        if url_path == "/" or url_path == "/index.html":
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif url_path == "/static/style.css":
            self._serve_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
        elif url_path == "/static/app.js":
            self._serve_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        elif url_path == "/api/status":
            self._send_json({
                "status": "online",
                "active_model": rag_pipeline.active_model_id,
                "models": MODEL_SPECS,
                "has_article": rag_pipeline.article is not None,
                "chunk_count": len(rag_pipeline.chunks)
            })
        else:
            # Check static directory
            clean_path = url_path.lstrip("/static/")
            target = STATIC_DIR / clean_path
            if target.exists() and target.is_file():
                ctype = "text/plain"
                if target.suffix == ".html": ctype = "text/html"
                elif target.suffix == ".css": ctype = "text/css"
                elif target.suffix == ".js": ctype = "application/javascript"
                elif target.suffix == ".mp3": ctype = "audio/mpeg"
                self._serve_file(target, ctype)
            else:
                self.send_error(404, "Not Found")

    def do_POST(self):
        url_path = urllib.parse.urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            payload = {}

        if url_path == "/api/load-article":
            url = payload.get("url", "").strip()
            if not url:
                self._send_json({"error": "URL parameter is required"}, status=400)
                return

            try:
                article = extract_article(url)
                chunks = chunk_text(article["text"])
                rag_pipeline.set_article(article, chunks)

                self._send_json({
                    "status": "ok",
                    "article": article,
                    "chunk_count": len(chunks)
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif url_path == "/api/switch-model":
            model_id = payload.get("model_id", "bge_m3")
            try:
                spec = rag_pipeline.set_active_model(model_id)
                self._send_json({"status": "ok", "model": spec})
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)

        elif url_path == "/api/query":
            query = payload.get("query", "").strip()
            model_id = payload.get("model_id", rag_pipeline.active_model_id)
            is_summary = payload.get("is_summary", False)

            if not query and not is_summary:
                self._send_json({"error": "Query is required"}, status=400)
                return

            try:
                retrieval = rag_pipeline.retrieve(query, model_id=model_id, top_k=3)
                answer = rag_pipeline.generate_grounded_answer(
                    query=query,
                    retrieved_chunks=retrieval["results"],
                    is_summary=is_summary
                )

                self._send_json({
                    "status": "ok",
                    "query": query,
                    "model_id": retrieval["model_id"],
                    "model_name": retrieval["model_name"],
                    "dim": retrieval["dim"],
                    "latency_ms": retrieval["latency_ms"],
                    "answer": answer,
                    "chunks": retrieval["results"]
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif url_path == "/api/compare":
            query = payload.get("query", "").strip() or "General overview"
            try:
                e5_retrieval = rag_pipeline.retrieve(query, model_id="e5", top_k=3)
                bge_retrieval = rag_pipeline.retrieve(query, model_id="bge_m3", top_k=3)
                self._send_json({
                    "status": "ok",
                    "query": query,
                    "e5": e5_retrieval,
                    "bge_m3": bge_retrieval
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        else:
            self.send_error(404, "Endpoint Not Found")

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]} -> {args[2]}\n")


def run(host="127.0.0.1", port=8000):
    server_address = (host, port)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    print("=" * 70)
    print("🌟 Bilingual Telugu–English News Assistant Web Server")
    print("   Team Vaani - VIT-AP University")
    print(f"🚀 Running at: http://{host}:{port}/")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run(port=port)
