from django.shortcuts import render
import pandas as pd

from joblib import load

random = load('./modelos/random_forest_model.pkl')
lineal = load('./modelos/linear_model.pkl')

# Create your views here.



def home(request):

    
    prediction = None

    if request.method == 'POST':
        if 'regresion' in request.POST:
            try:
                data = {
                    'Subsegmento': [int(request.POST['subsegmento'])],
                    'Antiguedad': [float(request.POST['antiguedad'])],
                    'Cuentas': [float(request.POST['cuentas'])],
                    'TC': [float(request.POST['tc'])],
                    'CUPO_MX': [float(request.POST['cupo_mx'])]
                }

                input_df = pd.DataFrame(data)

                pred = lineal.predict(input_df)

                print("ESTA ES LA PRED QUE LLEGA ", pred)
                prediction = pred[0]
                request.session['prediction'] = prediction  # Guardamos la predicción para mostrarla en el contexto

                prediction = request.session.get('prediction', None)

            except Exception as e:
                print("Error en la predicción: ", e)
                prediction = "Error en la entrada de datos."


    return render(request, 'index.html', { 'pred' : prediction})


'''
Subsegmento: 160 (int64)
Antigüedad: 2.553894704554291 (float64)
Cuentas: 0.0 (float64)
TC: -0.7399128501103254 (float64)
CUPO_MX: 0.9020878457232103

L1 REAL: 1.4446278462946316
'''
