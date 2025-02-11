from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import csv
from models import LotofacilPredictor
import json
import joblib

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configurações
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR = os.path.join(BASE_DIR, 'dados')

# Criar diretório de dados se não existir
if not os.path.exists(DADOS_DIR):
    os.makedirs(DADOS_DIR)

MODELO_DIR = os.path.join(BASE_DIR, 'modelo')
HISTORICO_DIR = os.path.join(DADOS_DIR, 'historico')

# Criar diretórios necessários
for dir_path in [MODELO_DIR, HISTORICO_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def carregar_modelo():
    """Carrega o modelo treinado"""
    modelo_path = os.path.join(MODELO_DIR, 'modelo_lotofacil.joblib')
    if os.path.exists(modelo_path):
        return joblib.load(modelo_path)
    return None

def treinar_modelo(df):
    """Treina o modelo com os dados atualizados"""
    try:
        predictor = LotofacilPredictor(os.path.join(DADOS_DIR, 'resultados.csv'))
        print("Treinando Random Forest...")
        predictor.train_random_forest()
        print("Treinando LSTM...")
        predictor.train_lstm()
        print("Treinando Prophet...")
        predictor.train_prophet()
        print("Treinando XGBoost...")
        predictor.train_xgboost()
        print("Salvando modelos...")
        predictor.save_models(MODELO_DIR)
        return True
    except Exception as e:
        print(f"Erro ao treinar modelo: {str(e)}")
        return False

def atualizar_resultados():
    try:
        print("Iniciando atualização dos resultados...")
        # Verifica se o arquivo existe
        if not os.path.exists(os.path.join(DADOS_DIR, 'resultados.csv')):
            print("Criando arquivo resultados.csv...")
            with open(os.path.join(DADOS_DIR, 'resultados.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Concurso', 'Data', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 
                               'Bola6', 'Bola7', 'Bola8', 'Bola9', 'Bola10', 'Bola11', 'Bola12', 
                               'Bola13', 'Bola14', 'Bola15'])

        print("Carregando dados existentes...")
        df = pd.read_csv(os.path.join(DADOS_DIR, 'resultados.csv'))
        
        # Obtém o último concurso
        ultimo_concurso = df['Concurso'].max() if not df.empty else 0
        print(f"Último concurso encontrado: {ultimo_concurso}")

        # URL da API da Caixa
        url = f'https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil/'
        print(f"Fazendo requisição para: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'Origin': 'https://loterias.caixa.gov.br',
            'Referer': 'https://loterias.caixa.gov.br/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Erro na requisição: Status {response.status_code}")
            return jsonify({'success': False, 'error': f'Erro ao obter dados da API: {response.status_code}'}), 500
        
        data = response.json()
        print(f"Dados recebidos da API: {data}")
        
        novo_concurso = int(data['numero'])
        print(f"Novo concurso: {novo_concurso}")
        
        if novo_concurso > ultimo_concurso:
            print("Atualizando resultados...")
            numeros = [int(n) for n in data['dezenasSorteadasOrdemSorteio']]
            data_sorteio = data['dataApuracao']
            
            nova_linha = [novo_concurso, data_sorteio] + numeros
            df_nova = pd.DataFrame([nova_linha], columns=df.columns)
            df = pd.concat([df, df_nova], ignore_index=True)
            
            print("Salvando arquivo atualizado...")
            df.to_csv(os.path.join(DADOS_DIR, 'resultados.csv'), index=False)
            treinar_modelo(df)
            return jsonify({'success': True, 'message': 'Resultados atualizados com sucesso!'})
        else:
            print("Nenhuma atualização necessária")
            return jsonify({'success': True, 'message': 'Resultados já estão atualizados!'})
            
    except Exception as e:
        print(f"Erro durante atualização: {str(e)}")
        return jsonify({'success': False, 'error': f'Erro ao atualizar resultados: {str(e)}'}), 500

def gerar_numeros(metodo='aleatorio', quantidade=15):
    """Gera números para jogar na Lotofácil
    
    Args:
        metodo (str): 'aleatorio' ou 'ia'
        quantidade (int): quantidade de números a gerar (15-18)
    """
    try:
        quantidade = min(max(15, quantidade), 18)  # Garante que está entre 15 e 18
        
        if metodo == 'aleatorio':
            numeros = sorted(np.random.choice(range(1, 26), size=quantidade, replace=False))
        else:
            df = pd.read_csv(os.path.join(DADOS_DIR, 'resultados.csv'))
            predictor = LotofacilPredictor(os.path.join(DADOS_DIR, 'resultados.csv'))
            predictor.load_models(MODELO_DIR)
            
            # Pegar os últimos 10 jogos
            last_10_games = df.tail(10)[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5',
                                       'Bola6', 'Bola7', 'Bola8', 'Bola9', 'Bola10',
                                       'Bola11', 'Bola12', 'Bola13', 'Bola14', 'Bola15']].values
            
            # Gerar previsão base com 15 números
            numeros_base = predictor.predict(last_10_games)
            numeros_base = sorted([int(n) for n in numeros_base])
            
            # Se precisar de mais números, adicionar os próximos mais prováveis
            if quantidade > 15:
                # Pegar todos os números não selecionados
                numeros_restantes = list(set(range(1, 26)) - set(numeros_base))
                
                # Calcular probabilidade para cada número restante
                probabilidades = []
                for num in numeros_restantes:
                    freq = df[df.apply(lambda row: num in row[2:17].values, axis=1)].shape[0]
                    probabilidades.append((num, freq))
                
                # Ordenar por frequência e pegar os números adicionais necessários
                numeros_adicionais = sorted([x[0] for x in sorted(probabilidades, key=lambda x: x[1], reverse=True)[:quantidade-15]])
                numeros = sorted(numeros_base + numeros_adicionais)
            else:
                numeros = numeros_base
        
        return numeros
    except Exception as e:
        print(f"Erro ao gerar números: {str(e)}")
        return sorted(np.random.choice(range(1, 26), size=quantidade, replace=False))

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
    return atualizar_resultados()

@app.route('/api/gerar', methods=['POST'])
def gerar():
    """Endpoint para gerar números"""
    try:
        data = request.get_json()
        metodo = data.get('metodo', 'aleatorio')
        quantidade = int(data.get('quantidade', 15))
        numeros = gerar_numeros(metodo, quantidade)
        return jsonify({
            'success': True,
            'numeros': numeros
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
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
    # Configuração para o Glitch
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
