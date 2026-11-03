import sys
import os
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import RAGRequestHandler

class handler(RAGRequestHandler):
    """
    Vercel serverless function entrypoint.
    Inherits all GET and POST routing from RAGRequestHandler.
    """
    pass
