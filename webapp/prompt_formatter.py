from typing import Callable, Dict, Iterable, List, Tuple
import re

# A strategy is (predicate, formatter)
Strategy = Tuple[Callable[[str], bool], Callable[[List[Dict], str], str]]


def _as_history(history):
    sys_msgs: List[str] = []
    conv: List[Tuple[str, str]] = []
    for t in history or []:
        if not isinstance(t, dict):
            continue
        role = (t.get('role') or '').lower()
        text = t.get('text') or t.get('content') or ''
        if role.startswith('system'):
            sys_msgs.append(text)
        elif role.startswith('user') or role in ('u', 'human'):
            conv.append(('user', text))
        else:
            conv.append(('assistant', text))
    return sys_msgs, conv


def _prepend_system(sys_msgs: List[str], s: str) -> str:
    return ('System: ' + '\n\n'.join(sys_msgs) + '\n\n' + s) if sys_msgs else s


def _instruction(history: List[Dict], prompt: str) -> str:
    sys_msgs, conv = _as_history(history)
    parts: List[str] = []
    for r, t in conv:
        parts.append(f"### {'Instruction' if r=='user' else 'Response'}:\n{t}")
    parts.append(f"### Instruction:\n{prompt}")
    parts.append("### Response:")
    return _prepend_system(sys_msgs, '\n\n'.join(parts))


def _human(history: List[Dict], prompt: str) -> str:
    sys_msgs, conv = _as_history(history)
    parts: List[str] = [("Human: " if r=='user' else "Assistant: ") + t for r, t in conv]
    parts.append("Human: " + prompt)
    parts.append("Assistant:")
    return _prepend_system(sys_msgs, '\n'.join(parts))


def _default(history: List[Dict], prompt: str) -> str:
    sys_msgs, conv = _as_history(history)
    parts: List[str] = [("User: " if r=='user' else "Assistant: ") + t for r, t in conv]
    parts.append("User: " + prompt)
    parts.append("Assistant:")
    return _prepend_system(sys_msgs, '\n'.join(parts))


STRATEGIES: Iterable[Strategy] = [
    (lambda name: bool(re.search(r"instruction|gemma", (name or '').lower())), _instruction),
    (lambda name: bool(re.search(r"gpt|chat", (name or '').lower())), _human),
    (lambda name: True, _default),
]


def format_prompt(model_name: str, history: List[Dict], prompt: str) -> str:
    formatter = next((fmt for pred, fmt in STRATEGIES if pred(model_name)), _default)
    return formatter(history, prompt)

