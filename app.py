from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Configurações
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR = os.path.join(BASE_DIR, 'dados')
MODELO_DIR = os.path.join(BASE_DIR, 'modelo')
HISTORICO_DIR = os.path.join(DADOS_DIR, 'historico')

# Criar diretórios necessários
for dir_path in [DADOS_DIR, MODELO_DIR, HISTORICO_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def carregar_modelo():
    """Carrega o modelo treinado"""
    modelo_path = os.path.join(MODELO_DIR, 'modelo_lotofacil.joblib')
    if os.path.exists(modelo_path):
        return joblib.load(modelo_path)
    return None

def treinar_modelo(df):
    """Treina o modelo com os dados históricos"""
    X = df[[f'Bola{i}' for i in range(1, 16)]].values
    y = X  # No caso da Lotofácil, queremos prever os próprios números
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    modelo_path = os.path.join(MODELO_DIR, 'modelo_lotofacil.joblib')
    joblib.dump(modelo, modelo_path)
    return modelo

def atualizar_resultados():
    """Atualiza o arquivo CSV com os últimos resultados da Lotofácil"""
    try:
        url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
        arquivo_csv = os.path.join(DADOS_DIR, 'resultados.csv')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if not data or 'dezenasSorteadasOrdemSorteio' not in data:
            raise Exception("Dados não encontrados na resposta da API")
            
        numeros = [int(n) for n in data['dezenasSorteadasOrdemSorteio']]
        if len(numeros) != 15:
            raise Exception("Número incorreto de dezenas no resultado")
            
        resultado = {
            'Concurso': [data.get('numero', 0)],
            'Data': [data.get('dataApuracao', '')],
        }
        
        for i, num in enumerate(numeros, 1):
            resultado[f'Bola{i}'] = [num]
        
        df_novo = pd.DataFrame(resultado)
        
        if os.path.exists(arquivo_csv):
            df_existente = pd.read_csv(arquivo_csv)
            if data.get('numero', 0) > df_existente['Concurso'].max():
                df = pd.concat([df_existente, df_novo], ignore_index=True)
                df.to_csv(arquivo_csv, index=False)
                treinar_modelo(df)
            else:
                df = df_existente
        else:
            df_novo.to_csv(arquivo_csv, index=False)
            treinar_modelo(df_novo)
        
        return True, data.get('numero', 0)
        
    except Exception as e:
        return False, str(e)

def gerar_numeros(metodo='aleatorio'):
    """Gera números para jogar na Lotofácil"""
    try:
        if metodo == 'aleatorio':
            numeros = list(range(1, 26))
            return np.random.choice(numeros, size=15, replace=False).tolist()
        elif metodo == 'ia':
            modelo = carregar_modelo()
            if modelo is None:
                return gerar_numeros('aleatorio')
            
            arquivo_csv = os.path.join(DADOS_DIR, 'resultados.csv')
            if not os.path.exists(arquivo_csv):
                return gerar_numeros('aleatorio')
                
            df = pd.read_csv(arquivo_csv)
            ultimo_jogo = df.iloc[-1:][['Bola' + str(i) for i in range(1, 16)]].values
            previsao = modelo.predict(ultimo_jogo)[0]
            
            numeros = np.unique(np.clip(previsao.astype(int), 1, 25))
            if len(numeros) < 15:
                complemento = np.random.choice(
                    [n for n in range(1, 26) if n not in numeros],
                    size=15-len(numeros),
                    replace=False
                )
                numeros = np.concatenate([numeros, complemento])
            elif len(numeros) > 15:
                numeros = np.random.choice(numeros, size=15, replace=False)
                
            return numeros.tolist()
    except Exception as e:
        print(f"Erro ao gerar números: {str(e)}")
        return gerar_numeros('aleatorio')

def analisar_jogo(numeros):
    """Analisa um jogo específico com base nos dados históricos"""
    try:
        arquivo_csv = os.path.join(DADOS_DIR, 'resultados.csv')
        if not os.path.exists(arquivo_csv):
            return None
            
        df = pd.read_csv(arquivo_csv)
        total_jogos = len(df)
        
        # Análise de frequência
        frequencias = {}
        for num in numeros:
            count = sum(1 for i in range(1, 16) if (df[f'Bola{i}'] == num).sum() > 0)
            frequencias[num] = (count / total_jogos) * 100
            
        # Análise de jogos similares
        jogos_similares = []
        for idx, row in df.iterrows():
            jogo_anterior = set(row[[f'Bola{i}' for i in range(1, 16)]].values)
            acertos = len(set(numeros).intersection(jogo_anterior))
            if acertos >= 11:
                jogos_similares.append({
                    'concurso': row['Concurso'],
                    'data': row['Data'],
                    'acertos': acertos
                })
                
        # Gerar gráfico de frequência
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(frequencias.keys()), y=list(frequencias.values()))
        plt.title('Frequência dos Números Escolhidos')
        plt.xlabel('Número')
        plt.ylabel('Frequência (%)')
        
        # Salvar gráfico em base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        grafico_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
                
        return {
            'frequencias': frequencias,
            'jogos_similares': jogos_similares,
            'media_frequencia': sum(frequencias.values()) / len(frequencias),
            'total_jogos_analisados': total_jogos,
            'grafico_frequencia': grafico_base64
        }
    except Exception as e:
        print(f"Erro na análise do jogo: {str(e)}")
        return None

def salvar_jogo(numeros):
    """Salva um jogo no histórico"""
    try:
        data_atual = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        arquivo_json = os.path.join(HISTORICO_DIR, f'jogo_{data_atual}.json')
        
        jogo = {
            'numeros': numeros,
            'data': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analise': analisar_jogo(numeros)
        }
        
        with open(arquivo_json, 'w', encoding='utf-8') as f:
            json.dump(jogo, f, ensure_ascii=False, indent=4)
            
        return True
    except Exception as e:
        print(f"Erro ao salvar jogo: {str(e)}")
        return False

def carregar_historico():
    """Carrega o histórico de jogos salvos"""
    try:
        jogos = []
        for arquivo in os.listdir(HISTORICO_DIR):
            if arquivo.endswith('.json'):
                with open(os.path.join(HISTORICO_DIR, arquivo), 'r', encoding='utf-8') as f:
                    jogo = json.load(f)
                    jogos.append(jogo)
        return sorted(jogos, key=lambda x: x['data'], reverse=True)
    except Exception as e:
        print(f"Erro ao carregar histórico: {str(e)}")
        return []

def gerar_estatisticas():
    """Gera estatísticas gerais dos resultados"""
    try:
        arquivo_csv = os.path.join(DADOS_DIR, 'resultados.csv')
        if not os.path.exists(arquivo_csv):
            return None
            
        df = pd.read_csv(arquivo_csv)
        
        # Frequência geral dos números
        frequencias = {}
        for num in range(1, 26):
            count = sum(1 for i in range(1, 16) if (df[f'Bola{i}'] == num).sum() > 0)
            frequencias[num] = (count / len(df)) * 100
            
        # Gerar gráfico de frequência geral
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(frequencias.keys()), y=list(frequencias.values()))
        plt.title('Frequência Geral dos Números')
        plt.xlabel('Número')
        plt.ylabel('Frequência (%)')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        grafico_frequencia = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'frequencias': frequencias,
            'total_jogos': len(df),
            'grafico_frequencia': grafico_frequencia
        }
    except Exception as e:
        print(f"Erro ao gerar estatísticas: {str(e)}")
        return None

# Rotas da API
@app.route('/api/atualizar', methods=['POST'])
def atualizar():
    """Endpoint para atualizar os resultados"""
    sucesso, resultado = atualizar_resultados()
    if sucesso:
        return jsonify({
            'success': True,
            'ultimo_concurso': resultado
        })
    return jsonify({
        'success': False,
        'error': resultado
    }), 500

@app.route('/api/gerar', methods=['POST'])
def gerar():
    """Endpoint para gerar números"""
    metodo = request.json.get('metodo', 'aleatorio')
    numeros = gerar_numeros(metodo)
    if numeros:
        analise = analisar_jogo(numeros)
        return jsonify({
            'success': True,
            'numeros': numeros,
            'analise': analise
        })
    return jsonify({
        'success': False,
        'error': 'Erro ao gerar números'
    }), 500

@app.route('/api/analisar', methods=['POST'])
def analisar():
    """Endpoint para analisar um jogo específico"""
    numeros = request.json.get('numeros')
    if not numeros or len(numeros) != 15:
        return jsonify({
            'success': False,
            'error': 'Números inválidos'
        }), 400
        
    analise = analisar_jogo(numeros)
    if analise:
        return jsonify({
            'success': True,
            'analise': analise
        })
    return jsonify({
        'success': False,
        'error': 'Erro ao analisar jogo'
    }), 500

@app.route('/api/salvar', methods=['POST'])
def salvar():
    """Endpoint para salvar um jogo"""
    numeros = request.json.get('numeros')
    if not numeros or len(numeros) != 15:
        return jsonify({
            'success': False,
            'error': 'Números inválidos'
        }), 400
        
    if salvar_jogo(numeros):
        return jsonify({
            'success': True
        })
    return jsonify({
        'success': False,
        'error': 'Erro ao salvar jogo'
    }), 500

@app.route('/api/historico', methods=['GET'])
def historico():
    """Endpoint para obter o histórico de jogos"""
    jogos = carregar_historico()
    return jsonify({
        'success': True,
        'jogos': jogos
    })

@app.route('/api/estatisticas', methods=['GET'])
def estatisticas():
    """Endpoint para obter estatísticas gerais"""
    stats = gerar_estatisticas()
    if stats:
        return jsonify({
            'success': True,
            'estatisticas': stats
        })
    return jsonify({
        'success': False,
        'error': 'Erro ao gerar estatísticas'
    }), 500

@app.route('/api/ultimo-concurso', methods=['GET'])
def ultimo_concurso():
    """Retorna o número do último concurso"""
    try:
        arquivo_csv = os.path.join(DADOS_DIR, 'resultados.csv')
        if os.path.exists(arquivo_csv):
            df = pd.read_csv(arquivo_csv)
            if not df.empty:
                return jsonify({
                    'success': True,
                    'ultimo_concurso': int(df['Concurso'].max())
                })
    except Exception as e:
        pass
    
    return jsonify({
        'success': False,
        'error': 'Não foi possível obter o último concurso'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
