import pickle

import numpy as np
from flask import Flask, render_template, request
!pip3 install xgboost

def get_prediction(values: list) -> float:

    model = pickle.load(open('model/model_xgb.pkl', 'rb'))

    scaler_X = pickle.load(open('model/scaler_X.pkl', 'rb'))
    scaler_y = pickle.load(open('model/scaler_y.pkl', 'rb'))

    X_scl = scaler_X.transform(np.array(values).reshape(1, -1))

    y_pred_scl = model.predict(X_scl)

    y_pred = scaler_y.inverse_transform(y_pred_scl.reshape(-1, 1))
    y_pred = round(float(y_pred), 2)

    return y_pred

names = ['Количество отвердителя, м.%',
         'Содержание эпоксидных групп, %',
         'Температура вспышки, С',
         'Потребление смолы, г/м2',
         'Угол нашивки, град',
         'Шаг нашивки',
         'Плотность нашивки',
         'Плотность, кг/м3',
         'Поверхностная плотность, г/м2',
         'Модуль упругости, ГПа',
         'Соотношение матрица-наполнитель']


app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

@app.route('/', methods= ['POST', 'GET'])
def main():

    if request.method == 'GET':
        return render_template('main.html')

    if request.method == 'POST':
        input_values = []
        errors = []
        for field_name in names:
            value = float(request.form.get(field_name))
            if not (0 <= value <= 7000):
                errors.append(field_name)
            input_values.append(value)

        y = get_prediction(input_values)

        if len(errors) > 0:
            return render_template('main.html', errors=errors)
        return render_template('main.html', result=y)


if __name__ == '__main__':
    app.run()
