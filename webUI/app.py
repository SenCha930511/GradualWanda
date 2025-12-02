import sys
import os
import io
import time
import re
import queue
from flask import Flask, render_template, request, jsonify, Response
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PruningConfig, LoRaConfig, EvaluateConfig, GradualConfig
from gradual_wanda import GradualWanda
from modules.cancel import request_stop, clear_stop

# 設定 log queue
log_queue = queue.Queue()
original_stdout = sys.stdout
original_stderr = sys.stderr

class StreamToQueue:
    def write(self, msg):
        if isinstance(msg, bytes):
            msg = msg.decode("utf-8")
        if msg.strip():
            clean_msg = self.clean_message(msg)
            log_queue.put(clean_msg)  # 送到前端的是清乾淨的訊息
            original_stdout.write(msg)
            original_stderr.write(msg)

    def flush(self):
        original_stdout.flush()
        original_stderr.flush()

    def clean_message(self, msg):
        # 移除 ANSI 控制碼（像 \x1b[31m、\033[A 這些）
        ansi_escape = re.compile(r'''
            \x1B   # ESC
            (?:    # Either 7-bit C1 Fe (except CSI)
                [@-Z\\-_]
            |      # Or [ for CSI, then a control sequence
                \[
                [0-?]*  # Parameter bytes
                [ -/]*  # Intermediate bytes
                [@-~]   # Final byte
            )
        ''', re.VERBOSE)
        msg = ansi_escape.sub('', msg)

        # 處理 carriage return（\r）
        msg = msg.replace('\r', '')

        return msg

# 重定向標準輸出和錯誤輸出到 queue
sys.stdout = StreamToQueue()
sys.stderr = StreamToQueue()

app = Flask(__name__)

# 事件流，用來推送 log 資料到前端
@app.route('/stream_log')
def stream_log():
    def generate():
        while True:
            msg = log_queue.get()  # 從 queue 取出訊息
            yield f"data: {msg}\n\n"  # 使用 Server-Sent Events 推送到前端
    return Response(generate(), mimetype='text/event-stream')

# GradualWanda 物件初始化（預設模型路徑）
gw = GradualWanda("/home/timmy/GradualWanda/model/models--meta-llama--Llama-2-7b-hf/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9")

@app.route('/')
def index():
    return render_template('index.html', model_path=gw.model_path)

@app.route('/set_model_path', methods=['POST'])
def set_model_path():
    data = request.get_json()
    model_path = data.get('modelPath', '')
    if model_path:
        gw.set_model_path(model_path)
        return jsonify(status="success", result="模型路徑設定完成")
    return jsonify(status="error", result="請提供模型路徑")

@app.route('/evaluate', methods=['POST'])
def evaluate():
    clear_stop()
    config = EvaluateConfig(ntrain=100, data_dir="data/", save_dir="save/", engine=["engine1"])
    body = request.get_json(silent=True) or {}
    for key, val in body.get('evaluateConfig', {}).items():
        if hasattr(config, key):
            setattr(config, key, val)
    try:
        gw.evaluate(config)
        return jsonify(status="success", result="Evaluate finished.")
    except Exception as e:
        sys.stderr.write(str(e) + "\n")  # 捕捉錯誤並寫入 stderr
        return jsonify(status="error", result="Evaluate failed.", detail=str(e))

@app.route('/prune_wanda', methods=['POST'])
def prune_wanda():
    clear_stop()
    body = request.get_json(silent=True) or {}
    cfg = body.get("pruningConfig", {})

    config = PruningConfig(
        seed=cfg.get("seed", 42),
        nsamples=cfg.get("nsamples", 100),
        sparsity_ratio=cfg.get("sparsity_ratio", 0.5),
        sparsity_type=cfg.get("sparsity_type", "unstructured"),
        use_variant=cfg.get("use_variant", True),
        save=cfg.get("save", None),
        save_model=cfg.get("save_model", None)
    )
    try:
        gw.prune_wanda(config)
        return jsonify(status="success", result="Prune Wanda 完成")
    except Exception as e:
        sys.stderr.write(str(e) + "\n")  # 捕捉錯誤並寫入 stderr
        return jsonify(status="error", result="Prune Wanda 失敗", detail=str(e))

@app.route('/evaluate_model_sparsity')
def evaluate_model_sparsity():
    try:
        sparsity = gw.evaluate_model_sparsity()
        return jsonify(status="success", result=f"模型稀疏率為: {sparsity:.4f}")
    except Exception as e:
        sys.stderr.write(str(e) + "\n")  # 捕捉錯誤並寫入 stderr
        return jsonify(status="error", result="評估稀疏度失敗", detail=str(e))

@app.route('/gradual_pruning', methods=['POST'])
def gradual_pruning():
    clear_stop()
    body = request.get_json(silent=True) or {}
    cfg = body.get("gradualConfig", {})
    config = GradualConfig(
        model_name=cfg.get("model_name", "llama2"),
        total_steps=cfg.get("total_steps", 5),
        final_sparsity=cfg.get("final_sparsity", 0.8),
        nsamples=cfg.get("nsamples", 2),
        cache_dir=cfg.get("cache_dir", "llm_weights")
    )
    try:
        gw.gradual_pruning(config)
        return jsonify(status="success", result="Gradual Pruning 完成")
    except Exception as e:
        sys.stderr.write(str(e) + "\n")  # 捕捉錯誤並寫入 stderr
        return jsonify(status="error", result="Gradual Pruning 失敗", detail=str(e))

@app.route('/lora_finetune', methods=['POST'])
def lora_finetune():
    clear_stop()
    body = request.get_json(silent=True) or {}
    cfg = body.get("loraConfig", {})
    config = LoRaConfig(
        r=cfg.get("r", 8),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("target_modules", []),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        bias=cfg.get("bias", "none"),
        task_type=cfg.get("task_type", "CAUSAL_LM"),
        epochs=cfg.get("epochs", 2),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        max_length=cfg.get("max_length", 256),
        output_dir=cfg.get("output_dir", "lora_finetuned_model")
    )
    try:
        gw.lora_finetune(config)
        return jsonify(status="success", result="LoRA Finetune 完成")
    except Exception as e:
        sys.stderr.write(str(e) + "\n")  # 捕捉錯誤並寫入 stderr
        return jsonify(status="error", result="LoRA Finetune 失敗", detail=str(e))


@app.route('/stop', methods=['POST'])
def stop():
    """
    前端「強制停止」按鈕會呼叫此端點。
    透過全域停止旗標，讓 Evaluate / 逐步減枝 等長迴圈在下一個安全點提前結束。
    """
    request_stop()
    return jsonify(status="success", result="已送出停止請求，後端將在下一個安全點中止目前任務。")


if __name__ == '__main__':
    app.run(debug=True, port=8000)
