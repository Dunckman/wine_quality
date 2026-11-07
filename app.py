"""демо модели из ДЗ-3, подготовлено с помощью OpenAI Codex"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Винная подборка', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model(model_version):
    # загружаем модель один раз, без повторного обучения
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test(test_version):
    # возвращаем копию таблицы, чтобы не менять данные в кэше
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X, model_version):
    return _model.predict_proba(X)[:, 1]


def predict_csv(model, uploaded, feature_names, threshold):
    """проверяем CSV и получаем прогнозы для всех строк"""
    table = pd.read_csv(uploaded)
    missing = [col for col in feature_names if col not in table.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if table.empty:
        raise ValueError('CSV не содержит строк с данными.')
    features = table[feature_names].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('В признаках есть пропуски, текст или бесконечные значения. Нужны числа.')
    if not features['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен принимать значения 0 (белое) или 1 (красное).')
    result = table.copy()
    result['probability'] = model.predict_proba(features)[:, 1]
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    return result


# если модель или данные изменились, обновляем кэш
model_version = (APP_DIR / 'model.joblib').stat().st_mtime_ns
test_version = (APP_DIR / 'test.csv').stat().st_mtime_ns
model = load_model(model_version)
test = load_test(test_version)
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target']
probability = test_proba(model, X_test, model_version)
config = json.loads((APP_DIR / 'config.json').read_text())

st.title('Винная подборка')
st.caption('Лабораторные признаки → вероятность оценки не ниже 7 → решение для подборки')
threshold = st.sidebar.slider('Порог отбора', 0.0, 1.0, float(config['threshold']), 0.001,
                              key='threshold', format='%.3f')
st.sidebar.caption('Порог по умолчанию выбран на валидации по максимуму F1.')
single, quality, batch, weights = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with single:
    st.subheader('Характеристики вина')
    values = {}
    columns = st.columns(3)
    for i, name in enumerate(features):
        with columns[i % 3]:
            if name == 'is_red':
                values[name] = int(st.toggle('Красное вино', value=False))
            else:
                values[name] = st.number_input(
                    name, min_value=float(X_test[name].min()), max_value=float(X_test[name].max()),
                    value=float(X_test[name].median()), format='%.4f')
    one = pd.DataFrame([values], columns=features)
    p_good = float(model.predict_proba(one)[0, 1])
    st.metric('Вероятность хорошего вина', f'{p_good:.1%}')
    st.success('Добавить в подборку' if p_good >= threshold else 'Не добавлять в подборку')

with quality:
    labels = probability >= threshold
    st.subheader('Качество на отложенных винах')
    for column, name, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [
        metrics.precision_score(y_test, labels, zero_division=0),
        metrics.recall_score(y_test, labels), metrics.f1_score(y_test, labels)]):
        column.metric(name, f'{value:.3f}')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, labels, ax=axes[0], colorbar=False)
    metrics.RocCurveDisplay.from_predictions(y_test, probability, ax=axes[1])
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, labels, labels=[0, 1]).ravel()
    axes[1].scatter(fp / (fp + tn), tp / (tp + fn), color='red', label='Текущий порог')
    metrics.PrecisionRecallDisplay.from_predictions(y_test, probability, ax=axes[2])
    axes[2].axhline(y_test.mean(), linestyle='--', color='gray', label='Случайный отбор')
    axes[2].scatter(metrics.recall_score(y_test, labels),
                    metrics.precision_score(y_test, labels, zero_division=0),
                    color='red', label='Текущий порог')
    axes[1].legend(); axes[2].legend(); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)
    st.info('При увеличении порога подборка становится меньше, обычно точнее, но пропускает больше хороших вин. '
            'При уменьшении порога выбор шире и полнота выше, но посредственные вина попадают чаще. '
            'Recall — доля найденных хороших вин; precision — доля хороших среди отобранных.')

with batch:
    st.subheader('Загрузите лабораторные данные')
    st.caption('CSV с теми же 12 признаками. is_red: 0 — белое, 1 — красное. Порог общий для всех вкладок.')
    st.download_button('Скачать пример входного CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
                        'wine_example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с признаками', type=['csv'])
    if uploaded is not None:
        try:
            result = predict_csv(model, uploaded, features, threshold)
            st.dataframe(result, use_container_width=True)
            st.download_button('Скачать прогнозы', result.to_csv(index=False).encode('utf-8'),
                                'wine_predictions.csv', mime='text/csv')
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(str(error))

with weights:
    st.subheader('Как признаки связаны с решением')
    coefficients = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    coefficients.plot.barh(ax=ax, color=np.where(coefficients >= 0, '#3d8061', '#ac5262'))
    ax.set_xlabel('Коэффициент после StandardScaler'); ax.axvline(0, color='gray')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(coefficients.tail(3).index[::-1]) + '.')
    st.write('Сильнее всего против: ' + ', '.join(coefficients.head(3).index) + '.')
    st.caption('Знак и величина описывают связь при фиксированных остальных признаках, а не причинный эффект.')
