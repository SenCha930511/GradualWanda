import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, redirect, url_for
from config import PruningConfig, LoRaConfig, EvaluateConfig, GradualConfig
from gradual_wanda import GradualWanda

app = Flask(__name__)

# 建立 Gradualgw 實例
gw = GradualWanda("/media/GradualWanda/llm_weights/models--meta-llama--Llama-2-7b-hf")

@app.route('/')
def index():
    # 把目前模型路徑一起傳給前端
    current_model_path = gw.model_path
    return render_template('index.html', model_path=current_model_path)

@app.route('/set_model_path', methods=['POST'])
def set_model_path():
    model_path = request.form.get('model_path')
    if model_path:
        gw.set_model_path(model_path)
    return redirect(url_for('index'))

@app.route('/evaluate', methods=['POST'])
def evaluate():
    config = EvaluateConfig(
        ntrain=100, 
        data_dir="data/", 
        save_dir="save/", 
        engine=["engine1", "engine2"]
    )
    gw.evaluate(config)
    return redirect(url_for('index'))

@app.route('/prune_wanda', methods=['POST'])
def prune_wanda():
    config = PruningConfig(
        seed=42, 
        nsamples=100, 
        sparsity_ratio=0.5, 
        sparsity_type="unstructured", 
        cache_dir="cache/", 
        use_variant=True,
        save=None, 
        save_model=None
    )
    gw.prune_gw(config)
    return redirect(url_for('index'))

@app.route('/evaluate_sparsity', methods=['POST'])
def evaluate_sparsity():
    sparsity = gw.evaluate_model_sparsity()
    return f"模型稀疏率為: {sparsity}"

if __name__ == '__main__':
    app.run(debug=True, port=8000)
