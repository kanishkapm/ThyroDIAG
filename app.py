#!/usr/bin/env python3
"""
================================================================================
Enhanced Web Application with LangChain AI Agent & Batch Prediction
================================================================================
Flask web UI with traditional ML, deep learning, and AI agent using LangChain.

Usage:
    python3 app.py

Author: Kanishka P M
================================================================================
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration - Use dynamic paths rather than hardcoded Downloads
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'thyroid-cancer-v3-secret')

# KNOWLEDGE BASE
# ============================================

KNOWLEDGE_BASE = """
Thyroid Cancer Information:

Cancer Types:
- PTC (Papillary): 80% of cases, good prognosis
- FTC (Follicular): 10-15%, good prognosis
- ATC (Anaplastic): 2%, aggressive
- PDTC: 3-5%, intermediate

Driver Mutations:
- BRAF V600E: MAPK pathway, aggressive
- TERT: Poor prognosis
- RAS (NRAS, HRAS, KRAS): FTC
- TP53: ATC transformation
"""

# ============================================
# MODEL LOADING
# ============================================

traditional_model = None
scaler = None
label_encoder = None
THYROID_GENES = []
THYROID_GENES_DL = []  # Genes for DL model
model_loaded = False
model_source = "N/A"
dl_model = None

try:
    traditional_model = joblib.load(MODELS_DIR / "real_data_model.pkl")
    scaler = joblib.load(MODELS_DIR / "real_data_scaler.pkl")
    label_encoder = joblib.load(MODELS_DIR / "real_data_label_encoder.pkl")

    with open(MODELS_DIR / "real_data_genes.json", 'r') as f:
        THYROID_GENES = json.load(f)

    model_loaded = True
    model_source = "Stacking Ensemble"
    print(f"✓ Traditional model loaded - {len(THYROID_GENES)} genes")
except Exception as e:
    print(f"Traditional model error: {e}")

# Load DL model
try:
    import tensorflow as tf

    # Load DL-specific components
    dl_scaler = joblib.load(MODELS_DIR / "deep_learning" / "scaler.pkl")
    dl_label_encoder = joblib.load(MODELS_DIR / "deep_learning" / "label_encoder.pkl")

    with open(MODELS_DIR / "deep_learning" / "genes.json", 'r') as f:
        THYROID_GENES_DL = json.load(f)

    dl_model = tf.keras.models.load_model(MODELS_DIR / "deep_learning" / "cnn_thyroid_model.h5")
    model_source += " + CNN (87.6%)"
    print(f"✓ Deep learning model loaded - {len(THYROID_GENES_DL)} genes")
except Exception as e:
    print(f"DL not loaded: {e}")

# ============================================
# GENE CATEGORIES
# ============================================

GENE_CATEGORIES = {
    "Thyroid Differentiation": ["TG", "TPO", "DIO1", "DIO2", "TSHR", "SLC5A5", "SLC26A4", "NKX2-1", "FOXE1", "PAX8"],
    "Cancer Drivers": ["BRAF", "RET", "NRAS", "HRAS", "KRAS", "PIK3CA", "PTEN", "TP53", "AKT1", "MAPK1", "MAP2K1", "CTNNB1"],
    "Diagnostic Markers": ["TERT", "GALECTIN3", "HBME1", "CITED1", "TFF3", "KRT19", "KRT7", "KRT8", "KRT18", "PPARG", "CDH1"],
    "EMT Markers": ["FN1", "VIM", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "CDH2"],
    "Immune Markers": ["PDCD1", "CD274", "PDCD1LG2", "CTLA4", "CD8A", "PTPRC"],
    "Proliferation": ["MKI67", "PCNA", "TOP2A", "CCNB1", "CCND1", "AURKA"],
    "Angiogenesis": ["VEGFA", "VEGFB", "KDR", "FLT1", "STAT3", "IL6"]
}

# ============================================
# FLASK ROUTES
# ============================================


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThyroDIAG Web App</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 1rem;
        }
        .container { max-width: 1400px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        header { background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 2rem; text-align: center; }
        header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        header .badge { display: inline-block; background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; margin-top: 1rem; }

        .tabs { display: flex; background: #f0f0f0; border-bottom: 2px solid #ddd; }
        .tab { flex: 1; padding: 1rem; text-align: center; cursor: pointer; background: #f0f0f0; border: none; font-size: 1rem; font-weight: 600; color: #666; }
        .tab:hover { background: #e0e0e0; }
        .tab.active { background: white; color: #1e3c72; border-bottom: 3px solid #1e3c72; }

        .tab-content { display: none; padding: 2rem; }
        .tab-content.active { display: block; }

        .gene-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .gene-input { display: flex; flex-direction: column; }
        .gene-input label { font-size: 0.8rem; font-weight: 600; color: #666; margin-bottom: 0.3rem; }
        .gene-input input { padding: 0.6rem; border: 1px solid #ddd; border-radius: 6px; font-size: 0.9rem; }
        .gene-input input:focus { outline: none; border-color: #1e3c72; }

        .category-section { margin-bottom: 1.5rem; padding: 1rem; background: #f8f9fa; border-radius: 8px; }
        .category-title { font-size: 1rem; font-weight: 600; color: #1e3c72; margin-bottom: 0.8rem; }

        .btn { padding: 0.8rem 2rem; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; margin: 0.3rem; }
        .btn-primary { background: #1e3c72; color: white; }
        .btn-primary:hover { background: #2a5298; }
        .btn-dl { background: #28a745; color: white; }
        .btn-dl:hover { background: #218838; }
        .btn-secondary { background: #6c757d; color: white; }

        .result-box { margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 12px; display: none; }
        .result-box.show { display: block; }
        .prediction { font-size: 2rem; font-weight: bold; color: #1e3c72; text-align: center; margin-bottom: 1rem; }
        .confidence { text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 1rem; }
        .prob-bar { margin: 0.5rem 0; }
        .prob-label { display: flex; justify-content: space-between; margin-bottom: 0.3rem; }
        .prob-track { height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }

        .chat-container { height: 400px; border: 1px solid #ddd; border-radius: 8px; display: flex; flex-direction: column; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 1rem; background: #f9f9f9; }
        .chat-message { margin-bottom: 1rem; padding: 0.8rem; border-radius: 8px; }
        .chat-message.user { background: #1e3c72; color: white; margin-left: 2rem; }
        .chat-message.agent { background: #e9ecef; color: #333; margin-right: 2rem; }
        .chat-input { display: flex; padding: 1rem; border-top: 1px solid #ddd; gap: 0.5rem; }
        .chat-input input { flex: 1; padding: 0.8rem; border: 1px solid #ddd; border-radius: 8px; }

        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; }
        .info-card { padding: 1.5rem; background: #f8f9fa; border-radius: 12px; border-left: 4px solid #1e3c72; }
        .info-card h3 { color: #1e3c72; margin-bottom: 1rem; }
        .info-card ul { list-style: none; }
        .info-card li { padding: 0.3rem 0; font-size: 0.9rem; }
        .info-card li::before { content: "✓ "; color: #28a745; }
        
        .upload-section { margin-bottom: 2rem; padding: 1.5rem; background: #e9ecef; border-radius: 8px; border: 2px dashed #6c757d; text-align: center; }
        .upload-section h3 { margin-bottom: 1rem; color: #1e3c72; }
        .spinner { display: none; width: 40px; height: 40px; margin: 10px auto; border: 4px solid #f3f3f3; border-top: 4px solid #1e3c72; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><i class="fas fa-atom"></i> ThyroDIAG</h1>
            <p>AI-Assisted Diagnostic Support for Thyroid Subtype Classification</p>
            <div class="badge">{{ model_source }}</div>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="showTab('predict')"><i class="fas fa-stethoscope"></i> Predict</button>
            
            <button class="tab" onclick="showTab('info')"><i class="fas fa-info-circle"></i> Info</button>
        </div>

        <div id="predict" class="tab-content active">
            <!-- CSV Upload Section Instead of Manual Typing (Manual typing forms hidden for brevity below if you prefer, or both) -->
            <div class="upload-section" id="upload-section">
                <h3><i class="fas fa-file-csv"></i> Batch Prediction via CSV</h3>
                <p style="margin-bottom: 1rem; color: #666;">Upload a CSV file containing patient gene expression data for automatic analysis.</p>
                <input type="file" id="csvFile" accept=".csv" style="display: none;" onchange="handleFileUpload(event)">
                <button type="button" class="btn btn-secondary" onclick="document.getElementById('csvFile').click()"><i class="fas fa-upload"></i> Choose File</button>
                <button type="button" class="btn btn-primary" onclick="processCSV('traditional')"><i class="fas fa-brain"></i> Predict (Ensemble)</button>
                <button type="button" class="btn btn-dl" onclick="processCSV('dl')"><i class="fas fa-network-wired"></i> Predict (DL)</button>
                <div id="uploadSpinner" class="spinner"></div>
                <div id="uploadStatus" style="margin-top: 1rem; font-weight: bold;"></div>
            </div>

            <h2 style="margin-bottom:1rem;color:#1e3c72;"><i class="fas fa-dna"></i> Or Input Single Patient Profile</h2>
            <form id="predictionForm">
                {% for category, genes in categories.items() %}
                <div class="category-section">
                    <div class="category-title"><i class="fas fa-folder"></i> {{ category }}</div>
                    <div class="gene-grid">
                        {% for gene in genes %}
                        <div class="gene-input">
                            <label>{{ gene }}</label>
                            <input type="number" step="0.01" name="{{ gene }}" value="0">
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
                <div style="text-align:center;margin-top:2rem;">
                    <button type="button" class="btn btn-primary" onclick="predict()"><i class="fas fa-brain"></i> Traditional ML</button>
                    <button type="button" class="btn btn-dl" onclick="predictDL()"><i class="fas fa-network-wired"></i> Deep Learning</button>
                    <button type="button" class="btn btn-secondary" onclick="clearForm()"><i class="fas fa-redo"></i> Clear</button>
                </div>
            </form>
            <div id="singleSpinner" class="spinner"></div>
            
            <div id="result" class="result-box">
                <h3 id="resultTitle" style="text-align: center; margin-bottom: 1rem; color: #1e3c72;">Patient Prediction Result</h3>
                <div class="prediction" id="prediction"></div>
                <div class="confidence">Confidence: <span id="confidence"></span></div>
                <div id="probabilities"></div>
            </div>
            
            <div id="batchResult" class="result-box">
                <h3 style="text-align: center; margin-bottom: 1rem; color: #1e3c72;">Batch Prediction Results</h3>
                <div id="batchTableContainer" style="overflow-x: auto;"></div>
            </div>
        </div>

        </div>

        <div id="info" class="tab-content">
            <h2 style="margin-bottom:1rem;color:#1e3c72;"><i class="fas fa-info-circle"></i> Project Info</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h3><i class="fas fa-database"></i> Data Sources</h3>
                    <ul><li>GEO: 215 samples</li><li>TCGA: 507 samples</li><li>Total: 722</li><li>Homo sapiens</li></ul>
                </div>
                <div class="info-card">
                    <h3><i class="fas fa-brain"></i> Models</h3>
                    <ul><li>Stacking Ensemble</li><li>1D CNN</li><li>LSTM</li><li>Transformer</li></ul>
                </div>
                <div class="info-card">
                    <h3><i class="fas fa-chart-line"></i> Performance</h3>
                    <ul><li>Traditional: 93%</li><li>DL Target: 95%+</li><li>CV: 94%+</li><li>6 Classes</li></ul>
                </div>
                
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.2/papaparse.min.js"></script>
    <script>
        function showTab(id) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(id).classList.add('active');
        }

        async function predict() {
            document.getElementById('singleSpinner').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            document.getElementById('batchResult').classList.remove('show');
            const form = document.getElementById('predictionForm');
            const data = Object.fromEntries(new FormData(form));
            try {
                const r = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
                const result = await r.json();
                document.getElementById('singleSpinner').style.display = 'none';
                if(result.error) alert(result.error);
                else showResult(result);
            } catch(e) { 
                document.getElementById('singleSpinner').style.display = 'none';
                alert(e); 
            }
        }

        async function predictDL() {
            document.getElementById('singleSpinner').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            document.getElementById('batchResult').classList.remove('show');
            const form = document.getElementById('predictionForm');
            const data = Object.fromEntries(new FormData(form));
            try {
                const r = await fetch('/predict_dl', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
                const result = await r.json();
                document.getElementById('singleSpinner').style.display = 'none';
                if(result.error) alert(result.error);
                else showResult(result);
            } catch(e) { 
                document.getElementById('singleSpinner').style.display = 'none';
                alert(e); 
            }
        }

        let uploadedData = null;
        
        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            document.getElementById('uploadStatus').textContent = `Loading ${file.name}...`;
            document.getElementById('uploadStatus').style.color = '#1e3c72';
            
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    uploadedData = results.data;
                    document.getElementById('uploadStatus').textContent = `Loaded ${uploadedData.length} records ready for processing.`;
                    document.getElementById('uploadStatus').style.color = '#28a745';
                },
                error: function(error) {
                    document.getElementById('uploadStatus').textContent = `Error parsing CSV: ${error.message}`;
                    document.getElementById('uploadStatus').style.color = 'red';
                }
            });
        }
        
        async function processCSV(modelType) {
            if (!uploadedData || uploadedData.length === 0) {
                alert("Please select and load a valid CSV file first.");
                return;
            }
            
            document.getElementById('uploadSpinner').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            document.getElementById('batchResult').classList.remove('show');
            
            const endpoint = modelType === 'dl' ? '/predict_dl_batch' : '/predict_batch';
            
            try {
                const r = await fetch(endpoint, {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({ patients: uploadedData })
                });
                
                const result = await r.json();
                document.getElementById('uploadSpinner').style.display = 'none';
                
                if(result.error) {
                    alert(result.error);
                } else {
                    showBatchResult(result.predictions);
                }
            } catch(e) { 
                document.getElementById('uploadSpinner').style.display = 'none';
                alert(e); 
            }
        }

        function showResult(result) {
            document.getElementById('result').classList.add('show');
            document.getElementById('prediction').textContent = result.prediction;
            document.getElementById('confidence').textContent = (result.confidence*100).toFixed(1)+'%';
            const p = document.getElementById('probabilities');
            p.innerHTML = '<h4>Class Probabilities:</h4>';
            for(const [c,prob] of Object.entries(result.probabilities)) {
                const color = prob>0.5?'#1e3c72':prob>0.2?'#2a5298':'#6c757d';
                p.innerHTML += `<div class="prob-bar"><div class="prob-label"><span>${c}</span><span>${(prob*100).toFixed(1)}%</span></div><div class="prob-track"><div class="prob-fill" style="width:${prob*100}%;background:${color};"></div></div></div>`;
            }
        }
        
        function showBatchResult(predictions) {
            const container = document.getElementById('batchTableContainer');
            document.getElementById('batchResult').classList.add('show');
            
            let html = '<table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">';
            html += '<tr style="background:#1e3c72; color:white;"><th style="padding:10px; text-align:left;">Patient ID</th><th style="padding:10px; text-align:left;">Prediction</th><th style="padding:10px; text-align:left;">Confidence</th></tr>';
            
            predictions.forEach((p, index) => {
                const id = p.id || `Patient_${index+1}`;
                const bg = index % 2 === 0 ? '#f9f9f9' : '#fff';
                const confColor = p.confidence > 0.8 ? '#28a745' : p.confidence > 0.5 ? '#fd7e14' : '#dc3545';
                html += `<tr style="background:${bg}; border-bottom:1px solid #ddd;">
                    <td style="padding:10px;">${id}</td>
                    <td style="padding:10px; font-weight:bold; color:#1e3c72;">${p.prediction}</td>
                    <td style="padding:10px; color:${confColor}; font-weight:bold;">${(p.confidence*100).toFixed(1)}%</td>
                </tr>`;
            });
            
            html += '</table>';
            container.innerHTML = html;
        }

        function clearForm() {
            document.querySelectorAll('input[type="number"]').forEach(i => i.value = 0);
            document.getElementById('result').classList.remove('show');
            document.getElementById('batchResult').classList.remove('show');
        }

        

        

        
    </script>
</body>
</html>

            <h2 style="margin-bottom:1rem;color:#1e3c72;"><i class="fas fa-info-circle"></i> Project Info</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h3><i class="fas fa-database"></i> Data Sources</h3>
                    <ul><li>GEO: 215 samples</li><li>TCGA: 507 samples</li><li>Total: 722</li><li>Homo sapiens</li></ul>
                </div>
                <div class="info-card">
                    <h3><i class="fas fa-brain"></i> Models</h3>
                    <ul><li>Stacking Ensemble</li><li>1D CNN</li><li>LSTM</li><li>Transformer</li></ul>
                </div>
                <div class="info-card">
                    <h3><i class="fas fa-chart-line"></i> Performance</h3>
                    <ul><li>Traditional: 93%</li><li>DL Target: 95%+</li><li>CV: 94%+</li><li>6 Classes</li></ul>
                </div>
                
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.2/papaparse.min.js"></script>
    <script>
        function showTab(id) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(id).classList.add('active');
        }

        async function predict() {
            document.getElementById('singleSpinner').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            document.getElementById('batchResult').classList.remove('show');
            const form = document.getElementById('predictionForm');
            const data = Object.fromEntries(new FormData(form));
            try {
                const r = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
                const result = await r.json();
                document.getElementById('singleSpinner').style.display = 'none';
                if(result.error) alert(result.error);
                else showResult(result);
            } catch(e) { 
                document.getElementById('singleSpinner').style.display = 'none';
                alert(e); 
            }
        }

        async function predictDL() {
            document.getElementById('singleSpinner').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            document.getElementById('batchResult').classList.remove('show');
            const form = document.getElementById('predictionForm');
            const data = Object.fromEntries(new FormData(form));
            try {
                const r = await fetch('/predict_dl', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
                const result = await r.json();
                document.getElementById('singleSpinner').style.display = 'none';
                if(result.error) alert(result.error);
                else showResult(result);
            } catch(e) { 
                document.getElementById('singleSpinner').style.display = 'none';
                alert(e); 
            }
        }

        let uploadedData = null;
        
        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            document.getElementById('uploadStatus').textContent = `Loading ${file.name}...`;
            document.getElementById('uploadStatus').style.color = '#1e3c72';
            
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    uploadedData = results.data;
                    document.getElementById('uploadStatus').textContent = `Loaded ${uploadedData.length} records ready for processing.`;
                    document.getElementById('uploadStatus').style.color = '#28a745';
                },
                error: function(error) {
                    document.getElementById('uploadStatus').textContent = `Error parsing CSV: ${error.message}`;
                    document.getElementById('uploadStatus').style.color = 'red';
                }
            });
        }
        
        async function processCSV(modelType) {
            if (!uploadedData || uploadedData.length === 0) {
                alert("Please select and load a valid CSV file first.");
                return;
            }
            
            document.getElementById('uploadSpinner').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            document.getElementById('batchResult').classList.remove('show');
            
            const endpoint = modelType === 'dl' ? '/predict_dl_batch' : '/predict_batch';
            
            try {
                const r = await fetch(endpoint, {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({ patients: uploadedData })
                });
                
                const result = await r.json();
                document.getElementById('uploadSpinner').style.display = 'none';
                
                if(result.error) {
                    alert(result.error);
                } else {
                    showBatchResult(result.predictions);
                }
            } catch(e) { 
                document.getElementById('uploadSpinner').style.display = 'none';
                alert(e); 
            }
        }

        function showResult(result) {
            document.getElementById('result').classList.add('show');
            document.getElementById('prediction').textContent = result.prediction;
            document.getElementById('confidence').textContent = (result.confidence*100).toFixed(1)+'%';
            const p = document.getElementById('probabilities');
            p.innerHTML = '<h4>Class Probabilities:</h4>';
            for(const [c,prob] of Object.entries(result.probabilities)) {
                const color = prob>0.5?'#1e3c72':prob>0.2?'#2a5298':'#6c757d';
                p.innerHTML += `<div class="prob-bar"><div class="prob-label"><span>${c}</span><span>${(prob*100).toFixed(1)}%</span></div><div class="prob-track"><div class="prob-fill" style="width:${prob*100}%;background:${color};"></div></div></div>`;
            }
        }
        
        function showBatchResult(predictions) {
            const container = document.getElementById('batchTableContainer');
            document.getElementById('batchResult').classList.add('show');
            
            let html = '<table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">';
            html += '<tr style="background:#1e3c72; color:white;"><th style="padding:10px; text-align:left;">Patient ID</th><th style="padding:10px; text-align:left;">Prediction</th><th style="padding:10px; text-align:left;">Confidence</th></tr>';
            
            predictions.forEach((p, index) => {
                const id = p.id || `Patient_${index+1}`;
                const bg = index % 2 === 0 ? '#f9f9f9' : '#fff';
                const confColor = p.confidence > 0.8 ? '#28a745' : p.confidence > 0.5 ? '#fd7e14' : '#dc3545';
                html += `<tr style="background:${bg}; border-bottom:1px solid #ddd;">
                    <td style="padding:10px;">${id}</td>
                    <td style="padding:10px; font-weight:bold; color:#1e3c72;">${p.prediction}</td>
                    <td style="padding:10px; color:${confColor}; font-weight:bold;">${(p.confidence*100).toFixed(1)}%</td>
                </tr>`;
            });
            
            html += '</table>';
            container.innerHTML = html;
        }

        function clearForm() {
            document.querySelectorAll('input[type="number"]').forEach(i => i.value = 0);
            document.getElementById('result').classList.remove('show');
            document.getElementById('batchResult').classList.remove('show');
        }

        

        

        
    </script>
</body>
</html>

"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,
                           genes=THYROID_GENES,
                           categories=GENE_CATEGORIES,
                           model_loaded=model_loaded,
                           model_source=model_source)


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'})

    try:
        data = request.get_json()
        gene_values = [float(data.get(gene, 0)) for gene in THYROID_GENES]

        X = pd.DataFrame([gene_values], columns=THYROID_GENES)
        X_scaled = scaler.transform(X)

        prediction_idx = traditional_model.predict(X_scaled)[0]
        prediction_proba = traditional_model.predict_proba(X_scaled)[0]
        prediction = label_encoder.inverse_transform([prediction_idx])[0]

        classes = label_encoder.classes_
        probabilities = {cls: float(proba) for cls, proba in zip(classes, prediction_proba)}

        return jsonify({
            'prediction': str(prediction),
            'probabilities': probabilities,
            'confidence': float(max(prediction_proba)),
            'model': 'Traditional ML'
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)})


@app.route('/predict_dl', methods=['POST'])
def predict_dl():
    """Make prediction using deep learning model."""
    if dl_model is None:
        return jsonify({'error': 'Deep learning model not available. Train with: python train_dl.py'})

    try:
        data = request.get_json()

        # Use DL genes only
        gene_values = [float(data.get(gene, 0)) for gene in THYROID_GENES_DL]

        # Scale using DL scaler
        X = pd.DataFrame([gene_values], columns=THYROID_GENES_DL)
        X_scaled = dl_scaler.transform(X)
        X_reshaped = X_scaled.reshape(1, len(THYROID_GENES_DL), 1)

        # Predict
        prediction_proba = dl_model.predict(X_reshaped, verbose=0)[0]
        prediction_idx = np.argmax(prediction_proba)

        # Use DL label encoder
        classes = dl_label_encoder.classes_
        prediction = classes[prediction_idx]
        probabilities = {cls: float(proba) for cls, proba in zip(classes, prediction_proba)}

        return jsonify({
            'prediction': str(prediction),
            'probabilities': probabilities,
            'confidence': float(max(prediction_proba)),
            'model': 'Deep Learning (CNN) - 87.6%'
        })
    except Exception as e:
        print(f"DL Prediction Error: {e}")
        return jsonify({'error': str(e)})


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handle batch predictions via CSV upload for Traditional Model"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'})

    try:
        data = request.get_json()
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({'error': 'No patient data provided'})
            
        results = []
        for i, patient_data in enumerate(patients):
            # Extract ID if available, otherwise generate one
            patient_id = patient_data.get('PatientID') or patient_data.get('ID') or patient_data.get('SampleID') or f"Sample_{i+1}"
            
            # Extract gene values safely
            gene_values = []
            for gene in THYROID_GENES:
                val = patient_data.get(gene, 0)
                try:
                    gene_values.append(float(val))
                except (ValueError, TypeError):
                    gene_values.append(0.0)

            X = pd.DataFrame([gene_values], columns=THYROID_GENES)
            X_scaled = scaler.transform(X)

            prediction_idx = traditional_model.predict(X_scaled)[0]
            prediction_proba = traditional_model.predict_proba(X_scaled)[0]
            prediction = label_encoder.inverse_transform([prediction_idx])[0]

            results.append({
                'id': str(patient_id),
                'prediction': str(prediction),
                'confidence': float(max(prediction_proba))
            })

        return jsonify({'predictions': results})
    except Exception as e:
        print(f"Batch Prediction Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/predict_dl_batch', methods=['POST'])
def predict_dl_batch():
    """Handle batch predictions via CSV upload for Deep Learning Model"""
    if dl_model is None:
        return jsonify({'error': 'Deep learning model not available.'})

    try:
        data = request.get_json()
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({'error': 'No patient data provided'})
            
        results = []
        for i, patient_data in enumerate(patients):
            patient_id = patient_data.get('PatientID') or patient_data.get('ID') or patient_data.get('SampleID') or f"Sample_{i+1}"
            
            gene_values = []
            for gene in THYROID_GENES_DL:
                val = patient_data.get(gene, 0)
                try:
                    gene_values.append(float(val))
                except (ValueError, TypeError):
                    gene_values.append(0.0)

            X = pd.DataFrame([gene_values], columns=THYROID_GENES_DL)
            X_scaled = dl_scaler.transform(X)
            X_reshaped = X_scaled.reshape(1, len(THYROID_GENES_DL), 1)

            prediction_proba = dl_model.predict(X_reshaped, verbose=0)[0]
            prediction_idx = np.argmax(prediction_proba)

            classes = dl_label_encoder.classes_
            prediction = classes[prediction_idx]

            results.append({
                'id': str(patient_id),
                'prediction': str(prediction),
                'confidence': float(max(prediction_proba))
            })

        return jsonify({'predictions': results})
    except Exception as e:
        print(f"DL Batch Prediction Error: {e}")
        return jsonify({'error': str(e)})

def info():
    return jsonify({
        'project': 'ThyroDIAG: AI-Assisted Thyroid Subtype Classification',
        'version': '3.0',
        'model_loaded': model_loaded,
        'model_source': model_source,
        'genes': len(THYROID_GENES),
        
        
        'data_sources': ['GEO', 'TCGA'],
        'organism': 'Homo sapiens'
    })


if __name__ == "__main__":
    print("="*60)
    print("ThyroDIAG: AI-Assisted Thyroid Classification")
    print("="*60)
    print(f"Model: {model_loaded}")
    print(f"Genes: {len(THYROID_GENES)}")
    
    print("\nOpen http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
