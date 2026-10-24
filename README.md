# Sun_spots_and_activity_model

Проект направлен на отработку теории по ML.

Модель состоит из 2 частей:
  1) **SunspotModelRegressor** - использует **ResNet18** для регрессии количества пятен на Солнце. Функция потерь задана как MSE, используется оптимизатор Adam.
  2) **RF_model** использует алгоритм Random Forest для M-классовой классификации уровня активности Солнца. Классы: "quiet", "low", "medium", "high". Используется энкодер для 4 классов.

Обе части объединены в **result_model_prod**, выход от **SunspotModelRegressor** отправляется на вход **RF_model**, как результат получаем предсказание пятен и класса.

Метрики: 
  SunspotModelRegressor: **MSE, RMSE, R2**
  RF_model: **F1, Precision, Recall, ROC-AUC**
  
