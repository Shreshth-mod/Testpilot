"""
Event-driven and callback dependency detection for Repository Evolution Intelligence.
Statically recognizes signal/event registrations such as:
  signal.connect(handler)
  signal.connect(receiver=handler)
  @receiver(signal)
  @receiver([signal_a, signal_b])
  event.subscribe(handler)
  event.register(handler)
  event.add_listener(handler)
Provides framework-independent, conservative event registration discovery.
"""

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EventRegistration:
    """Represents a statically identified event/signal registration."""

    event_name: str
    handler_name: str
    source_file: str
    line_number: int
    registration_type: str  # e.g. "signal_connect", "receiver_decorator", "event_subscribe"
    registration_expr: str
    is_confirmed: bool = True
    confidence: float = 0.85
    uncertainty_score: float = 0.15
    trigger_hints: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return f"{self.handler_name} registered to {self.event_name} via {self.registration_type} ({self.source_file}:{self.line_number})"


class EventRegistrationDetector:
    """
    Statically inspects Python source files for event, signal, and callback registrations.
    Avoids hardcoding framework specifics while recognizing common patterns.
    """

    # Common event registration method names
    REGISTRATION_METHODS = {"connect", "subscribe", "register", "add_listener", "listen", "on"}

    # Receiver argument keyword names
    RECEIVER_KWARGS = {"receiver", "handler", "callback", "func", "listener", "target"}

    # Database / connection keywords to reject as false positives
    NON_EVENT_RECEIVERS = {"host", "port", "user", "password", "database", "dbname", "timeout", "url", "dsn"}
    NON_EVENT_OBJECTS = {
        "db", "database", "conn", "connection", "cursor", "client", "socket",
        "session", "engine", "pool", "http", "ssh", "ftp", "tcp", "ssl",
        "s", "sock", "channel",
    }

    # Known trigger associations (framework-agnostic mapping of signal/event to operation hints)
    KNOWN_TRIGGER_MAP = {
        "post_migrate": ["migrate", "MigrationExecutor"],
        "pre_migrate": ["migrate", "MigrationExecutor"],
        "post_save": ["save", "create", "update"],
        "pre_save": ["save", "create", "update"],
        "post_delete": ["delete"],
        "pre_delete": ["delete"],
        "m2m_changed": ["add", "remove", "clear"],
        "request_started": ["request", "get", "post"],
        "request_finished": ["request", "get", "post"],
        "user_logged_in": ["login", "authenticate"],
        "user_logged_out": ["logout"],
    }

    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self._cached_py_files: Optional[list[Path]] = None
        self._custom_trigger_map: dict[str, list[str]] = {}
        self._handler_index: Optional[dict[str, list[EventRegistration]]] = None

    def register_event_mapping(self, event_name: str, trigger_hints: list[str]) -> None:
        """Extensibility API allowing external plugins or frameworks to register custom event trigger hints."""
        self._custom_trigger_map[event_name] = list(trigger_hints)

    def get_trigger_hints(self, event_name: str) -> list[str]:
        """Returns trigger operation hints for an event, falling back to generic event triggers."""
        clean_name = event_name.split(".")[-1]
        if clean_name in self._custom_trigger_map:
            return list(self._custom_trigger_map[clean_name])
        if clean_name in self.KNOWN_TRIGGER_MAP:
            return list(self.KNOWN_TRIGGER_MAP[clean_name])
        return [clean_name, f"{clean_name}.send", "send", "emit", "dispatch", "publish", "trigger", "fire"]

    def _get_repo_py_files(self) -> list[Path]:
        """Discovers and caches repo Python files, skipping vendor and virtual environments."""
        if self._cached_py_files is None:
            files: list[Path] = []
            ignored = {
                ".git", ".venv", "venv", "env", "node_modules",
                "build", "dist", ".tox", ".eggs", "__pycache__", "site-packages",
                ".chroma_db", ".pytest_cache", "htmlcov"
            }
            try:
                for dirpath, dirnames, filenames in os.walk(self.repo_root):
                    dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ignored]
                    for f in filenames:
                        if f.endswith(".py"):
                            files.append(Path(dirpath) / f)
            except Exception:
                pass
            self._cached_py_files = files
        return self._cached_py_files

    def scan_file_for_registrations(
        self,
        file_path: Path,
        target_handler_name: Optional[str] = None,
    ) -> list[EventRegistration]:
        """
        Scans a single Python file for event/signal registrations.
        If target_handler_name is provided, focuses on registrations referencing that handler.
        """
        registrations: list[EventRegistration] = []
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return registrations

        # Quick pre-filter: if searching for a specific handler and it's not in the file, skip
        if target_handler_name and target_handler_name not in content:
            return registrations

        # Check if file has any registration keywords
        has_reg_keyword = any(kw in content for kw in self.REGISTRATION_METHODS) or "receiver" in content
        if not has_reg_keyword:
            return registrations

        try:
            tree = ast.parse(content, filename=str(file_path))
        except Exception:
            return registrations

        try:
            rel_file = str(file_path.relative_to(self.repo_root)).replace("\\", "/")
        except ValueError:
            rel_file = str(file_path).replace("\\", "/")

        for node in ast.walk(tree):
            # Case 1: Method call signal.connect(handler, ...) or event.subscribe(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                if method_name in self.REGISTRATION_METHODS:
                    reg = self._extract_call_registration(node, rel_file, target_handler_name)
                    if reg:
                        registrations.append(reg)

            # Case 2: Decorator @receiver(signal) or @signal.connect on def handler(...)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if target_handler_name and node.name != target_handler_name:
                    continue
                for dec in node.decorator_list:
                    regs = self._extract_decorator_registrations(dec, node.name, rel_file, node.lineno)
                    registrations.extend(regs)

        return registrations

    def _extract_call_registration(
        self,
        call_node: ast.Call,
        rel_file: str,
        target_handler_name: Optional[str] = None,
    ) -> Optional[EventRegistration]:
        """Extracts event registration from a method call like signal.connect(handler)."""
        method_name = call_node.func.attr  # e.g. "connect"
        event_obj = call_node.func.value   # e.g. post_migrate

        # Determine event name
        event_name = self._unparse_expr(event_obj)
        if not event_name or event_name.lower() in self.NON_EVENT_OBJECTS:
            return None

        # Reject common database / socket connection calls (e.g. s.connect((ip, port)))
        for kw in call_node.keywords:
            if kw.arg in self.NON_EVENT_RECEIVERS:
                return None

        # Extract handler candidate
        handler_name: Optional[str] = None
        is_confirmed = True

        # Check positional args: connect(handler)
        if call_node.args:
            first_arg = call_node.args[0]
            if isinstance(first_arg, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)):
                # Constant or data structure literal argument -> definitely NOT a signal callback!
                return None
            elif isinstance(first_arg, ast.Name):
                handler_name = first_arg.id
            elif isinstance(first_arg, ast.Attribute):
                handler_name = first_arg.attr
            elif isinstance(first_arg, ast.Lambda):
                handler_name = "<lambda>"
                is_confirmed = False
            else:
                # Dynamic expression e.g. getattr(self, ...), get_handler()
                handler_name = self._unparse_expr(first_arg)
                is_confirmed = False

        # Check keyword args: connect(receiver=handler)
        for kw in call_node.keywords:
            if kw.arg in self.RECEIVER_KWARGS:
                if isinstance(kw.value, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)):
                    return None
                elif isinstance(kw.value, ast.Name):
                    handler_name = kw.value.id
                    is_confirmed = True
                elif isinstance(kw.value, ast.Attribute):
                    handler_name = kw.value.attr
                    is_confirmed = True
                elif isinstance(kw.value, ast.Lambda):
                    handler_name = "<lambda>"
                    is_confirmed = False
                else:
                    handler_name = self._unparse_expr(kw.value)
                    is_confirmed = False

        if not handler_name:
            return None

        # If filtering by target_handler_name, verify match
        if target_handler_name and handler_name != target_handler_name:
            if not handler_name.endswith(f".{target_handler_name}"):
                return None

        clean_event = event_name.split(".")[-1]
        hints = self.get_trigger_hints(clean_event)

        try:
            expr_str = ast.unparse(call_node)
        except Exception:
            expr_str = f"{event_name}.{method_name}({handler_name})"

        return EventRegistration(
            event_name=clean_event,
            handler_name=target_handler_name or handler_name,
            source_file=rel_file,
            line_number=call_node.lineno,
            registration_type=f"call_{method_name}",
            registration_expr=expr_str,
            is_confirmed=is_confirmed,
            confidence=0.85 if is_confirmed else 0.40,
            uncertainty_score=0.15 if is_confirmed else 0.50,
            trigger_hints=hints,
        )

    def _extract_decorator_registrations(
        self,
        dec_node: ast.AST,
        func_name: str,
        rel_file: str,
        line_number: int,
    ) -> list[EventRegistration]:
        """Extracts registration from decorators like @receiver(post_migrate) or @receiver([s1, s2])."""
        event_names: list[str] = []
        reg_type = "decorator"

        # Pattern 1: @receiver(...)
        if isinstance(dec_node, ast.Call):
            dec_func_name = self._unparse_expr(dec_node.func)
            if dec_func_name in ("receiver", "signal.receiver", "events.receiver"):
                reg_type = "receiver_decorator"
                if dec_node.args:
                    arg0 = dec_node.args[0]
                    if isinstance(arg0, (ast.List, ast.Tuple)):
                        for elt in arg0.elts:
                            ename = self._unparse_expr(elt)
                            if ename:
                                event_names.append(ename)
                    else:
                        ename = self._unparse_expr(arg0)
                        if ename:
                            event_names.append(ename)
            elif isinstance(dec_node.func, ast.Attribute) and dec_node.func.attr in self.REGISTRATION_METHODS:
                ename = self._unparse_expr(dec_node.func.value)
                reg_type = f"decorator_{dec_node.func.attr}"
                if ename:
                    event_names.append(ename)
        # Pattern 2: @signal.connect
        elif isinstance(dec_node, ast.Attribute) and dec_node.attr in self.REGISTRATION_METHODS:
            ename = self._unparse_expr(dec_node.value)
            reg_type = f"decorator_{dec_node.attr}"
            if ename:
                event_names.append(ename)

        registrations: list[EventRegistration] = []
        try:
            expr_str = ast.unparse(dec_node)
        except Exception:
            expr_str = f"@{reg_type}"

        for event_name in event_names:
            if not event_name or event_name.lower() in self.NON_EVENT_OBJECTS:
                continue
            clean_event = event_name.split(".")[-1]
            hints = self.get_trigger_hints(clean_event)
            registrations.append(
                EventRegistration(
                    event_name=clean_event,
                    handler_name=func_name,
                    source_file=rel_file,
                    line_number=line_number,
                    registration_type=reg_type,
                    registration_expr=expr_str,
                    is_confirmed=True,
                    confidence=0.90,
                    uncertainty_score=0.10,
                    trigger_hints=hints,
                )
            )

        return registrations

    def find_registrations_for_handler(self, handler_name: str) -> list[EventRegistration]:
        """
        Searches the repository for all event/signal registrations binding handler_name.
        Indexes all registrations across the repository on first lookup for O(1) query time.
        """
        if self._handler_index is None:
            index: dict[str, list[EventRegistration]] = {}
            for f in self._get_repo_py_files():
                regs = self.scan_file_for_registrations(f)
                for r in regs:
                    index.setdefault(r.handler_name, []).append(r)
            self._handler_index = index
        return list(self._handler_index.get(handler_name, []))

    @staticmethod
    def _unparse_expr(node: ast.AST) -> str:
        try:
            return ast.unparse(node).strip()
        except Exception:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return node.attr
            return ""
