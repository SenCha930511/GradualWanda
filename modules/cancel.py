import threading

_stop_event = threading.Event()


def request_stop() -> None:
    """要求目前長時間任務在下一個安全點盡快停止。"""
    _stop_event.set()


def clear_stop() -> None:
    """在開始新任務前呼叫，清除停止狀態。"""
    _stop_event.clear()


def should_stop() -> bool:
    """在長迴圈中輪詢，用來判斷是否應該提前結束。"""
    return _stop_event.is_set()


