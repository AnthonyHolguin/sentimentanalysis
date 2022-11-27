from flask import Flask, jsonify, abort, request, make_response
import numpy as np
from word2vecPreprocess import preproccess_data, tokenizer,pad_sequences,model,maxlen

app = Flask(__name__)
PORT=5000
DEBUG=False

@app.errorhandler(404)
def not_found(error):
  return "Not Found"

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!a"
@app.route('/predict_comments', methods=['POST'])
def predictComments():
    print("holantto")
    json = request.get_json()
    print(json)
    if not json:
        abort(404)
    predictedTexto = []
    print("LLEGA AQUI")
    comentarios_proc = preproccess_data(json["comentarios"])
    print("ya preproces'o")
    sequences = tokenizer.texts_to_sequences(comentarios_proc)
    print("la hace sequence")
    print(sequences)
    data_proc = pad_sequences(sequences, maxlen=maxlen)
    print("totalmete procesada")
    print(data_proc)
    polaritys = model.predict(data_proc)

    predictedmax = np.argmax(polaritys, axis=1) #[0,1,0] = 1
    # cambios necesarios para luego reasignar valores
    predictedmax[predictedmax == 2] = 4
    predictedmax[predictedmax == 1] = 5
    predictedmax[predictedmax == 0] = 6
    # cambios de los valores reasignados a como deberían de tener
    predictedmax[predictedmax == 4] = 1
    predictedmax[predictedmax == 5] = 0
    predictedmax[predictedmax == 6] = 2
    for p in predictedmax:
      if p == 2:
        predictedTexto.append("-1")
      elif p == 0:
        predictedTexto.append("0")
      else:
        predictedTexto.append("1")
    print(predictedTexto)
    return make_response(jsonify({'polaridades' : predictedTexto}), 200)

if __name__ == '__main__':
  app.run(debug = DEBUG,host='0.0.0.0', port=os.getenv("PORT", default=5000))