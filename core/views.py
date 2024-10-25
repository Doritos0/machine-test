from django.shortcuts import render
import pandas as pd

from joblib import load

random = load('./modelos/random_forest_model.pkl')
lineal = load('./modelos/linear_model.pkl')

# Create your views here.




def home(request):

    regre = None
    clasi = None

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
                request.session['prediction'] = prediction 

                regre = request.session.get('prediction', None)

            except Exception as e:
                print("Error en la predicción: ", e)
                prediction = "Error en la entrada de datos."

        elif 'clasificacion' in request.POST:
            try:
                data = {
                    'Edad': [float(request.POST['edad'])],
                    'Internauta': [float(request.POST['internauta'])],
                    'Ctacte': [float(request.POST['cuenta_cte'])],
                    'Debito': [float(request.POST['debito'])],
                    'CambioPin': [float(request.POST['cambio_pin'])],
                    'Cuentas': [float(request.POST['cuentas'])],
                    'TC': [float(request.POST['tc'])]
                }
                input_df = pd.DataFrame(data)
                pred = random.predict(input_df)
                clasi = pred[0]

            except Exception as e:
                print("Error en la predicción de clasificación: ", e)
                clasi = "Error en la entrada de datos."



    return render(request, 'index.html', { 'regre' : regre, 'clasi' : clasi })


'''
Subsegmento: 160 (int64)
Antigüedad: 2.553894704554291 (float64)
Cuentas: 0.0 (float64)
TC: -0.7399128501103254 (float64)
CUPO_MX: 0.9020878457232103

L1 REAL: 1.4446278462946316
'''

'''
Index(['Edad', 'Internauta', 'Ctacte', 'Debito', 'CambioPin', 'Cuentas', 'TC'

Edad: 0.323
Internauta: 1
Cta Cte: 1
Débito: 1
Cambio PIN: 0.0
Cuentas: -0.74
TC: 1.445
'''