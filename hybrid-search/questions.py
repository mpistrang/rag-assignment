"""
Example questions for retrieval testing and evaluation.

Tries to load proprietary questions from untracked questions_proprietary.py if it exists locally.
Falls back to generic examples if not found.
"""

# Default generic examples (safe for public repos)
DEFAULT_EVAL_QUESTIONS = [
    # Semantic queries (vector strength)
    "How do I set up webhooks?",
    "user authentication not working",
    "how to reset a password",
    "what happens after a task fails",
    # Exact match queries (BM25 strength)
    "GET /api/users",
    "POST /webhooks",
    "admin role permissions",
    "API rate limits",
    # Conceptual queries (hybrid strength)
    "What permissions does an admin have?",
    "configure notification settings",
    "export user data",
]

DEFAULT_TEST_QUERIES = [
    "GET /api/events",          # Exact API route - BM25 strength
    "user can't export data",   # Semantic meaning - Vector strength
    "webhook configuration",    # Mix of both
]

# Try to load proprietary questions, fall back to defaults
try:
    from questions_proprietary import EVAL_QUESTIONS, TEST_QUERIES
except ImportError:
    EVAL_QUESTIONS = DEFAULT_EVAL_QUESTIONS
    TEST_QUERIES = DEFAULT_TEST_QUERIES
