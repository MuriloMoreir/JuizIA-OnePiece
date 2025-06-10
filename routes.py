from flask import Flask, render_template, request, jsonify
from modelo import responder_pergunta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.get_json(force=True)
    pergunta = data.get('pergunta', '').strip()
    
    if not pergunta:
        return jsonify({'erro': 'Pergunta não fornecida'}), 400

    try:
        # Chama a função do modelo
        resposta = responder_pergunta(pergunta)
        return jsonify({
            'pergunta': pergunta,
            'resposta': resposta
        })
    except Exception as e:
        # Em caso de erro interno, retorna 500
        return jsonify({'erro': f'Ocorreu um erro: {str(e)}'}), 500

if __name__ == '__main__':
    # Ajuste host/port se necessário
    app.run(debug=True)
