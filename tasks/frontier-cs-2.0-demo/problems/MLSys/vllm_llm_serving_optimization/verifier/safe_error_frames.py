"""Private bounded traceback locations without exception text or frame values."""
from collections import deque
from pathlib import Path


def safe_error_frames(error):
    frames = deque(maxlen=16)
    try:
        root = Path(__file__).resolve().parents[1]
        trace = error.__traceback__
        while trace is not None:
            code = trace.tb_frame.f_code
            path = Path(code.co_filename)
            if path.is_absolute():
                try:
                    relative = path.resolve().relative_to(root)
                except (OSError, ValueError):
                    relative = None
                if (relative is not None and relative.parts[0] in {'verifier', 'judge'}
                        and code.co_name.isidentifier() and len(code.co_name) <= 80):
                    frames.append({'file': relative.as_posix(), 'line': trace.tb_lineno,
                                   'function': code.co_name})
            trace = trace.tb_next
    except Exception:
        return []
    return list(frames)
