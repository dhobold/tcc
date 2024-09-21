# camada de endpoints Flask - interface entre o front-end e codigo de analise
from flask import Flask, jsonify, render_template, request, send_from_directory
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import TCC_server as tcc


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Renderiza o arquivo HTML

@app.route('/ingestionAndTreatment', methods=['GET'])   
def ingestionAndTreatment():
    try:
        ret = tcc.start_server()
        return jsonify({"resultado": f"{ret}"})
    except Exception as e:
        return jsonify({"resultado": e.args[0]})
    
@app.route('/LoadList', methods=['GET'])
def LoadListNivel1_route():
    try:
        _resultado_nivel = tcc.LoadList()
        return jsonify(_resultado_nivel)  # Usa jsonify para converter o dicionário em JSON
    except Exception as e:
        return jsonify({})

@app.route('/getDataAnalysis', methods=['GET'])
def get_AnalyzingData():
    dados = tcc.dataAnalysis()
    dados['PV'] = dados['PV'].round(2)
    
    # Contar valores nulos na coluna 'Score'
    nulos_email = dados['Score'].isna().sum()
    print(f"Número de valores nulos na coluna 'Score': {nulos_email}")
    
    
    dados['Score'] = dados['Score'].round(3)
    df_clean = dados.replace({np.nan: None, np.inf: None, -np.inf: None})
    dados_json = df_clean.to_dict(orient='records')
    return jsonify(dados_json)

@app.route('/AnalyzingData', methods=['POST'])
def analyzing_data():

    try:
        # Recebe os dados do POST request como JSON
        data = request.get_json()
        nivel_2 = data.get('nivel_2')
        tipo = data.get('tipo')

        # Verifica se ao menos dois valores foram recebidos corretamente
        valid_values_count = sum(1 for value in [nivel_2, tipo] if value)

        if valid_values_count != 2:
            return jsonify({'erro': 'Os dois campos são obrigatórios!'}), 500

        tcc.AnalyzingData(data)   
        
        # Retorna a resposta JSON com o resultado da análise
        return jsonify({"resultado": "Anãlise realizada com sucesso!"}), 200

    except Exception as e:
        # Em caso de erro, retorna uma mensagem de erro
        return jsonify({'erro': str(e)}), 500
    
@app.route('/download-pdf')
def download_pdf():
    tcc.copiar_arquivo()
    
    # Substitua o caminho e o nome do arquivo de acordo com sua estrutura
    pdf_folder = 'static'  # Diretório onde o PDF está
    pdf_filename = 'relatorio.pdf'  # Nome do arquivo PDF

    return send_from_directory(pdf_folder, pdf_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)