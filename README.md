# Text Clustering Research

Програма для дипломного дослідження алгоритмів кластеризації природномовних текстів.
Реалізовано:
- 5 векторизаторів (TF-IDF, Word2Vec, FastText, Doc2Vec, Sentence-BERT)
- 8 алгоритмів кластеризації (K-Means, MiniBatch, DBSCAN, Agglomerative, Birch, Spectral, Affinity Propagation, HDBSCAN)
- Вимірювання часу, пам'яті, CPU для кожної комбінації
- Внутрішні та зовнішні метрики якості
- Візуалізація кластерів (t-SNE/PCA)

## Встановлення
1. Клонуйте репозиторій
2. Встановіть залежності:
    pip install -r requirements.txt
3. Покладіть CSV-файли з колонками `text` та `label` (опціонально) в папку `datasets/`
4. Запустіть `python main.py`
5. Звіт з'явиться в `reports/full_report_*.csv`, графіки – там же.

## Налаштування
У файлі `config.py` можна змінити:
- список векторизаторів та алгоритмів
- кількість повторів для бенчмарку
- параметри візуалізації
- мову препроцесингу

## Тести
    python -m pytest tests/