import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from prophet import Prophet
import optuna
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

class LotofacilPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.rf_model = None
        self.lstm_model = None
        self.prophet_models = []
        self.xgb_model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        # Preparar features para cada número
        features = []
        labels = []
        
        for i in range(len(df) - 10):
            # Últimos 10 jogos como features
            last_10_games = df.iloc[i:i+10][['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5',
                                           'Bola6', 'Bola7', 'Bola8', 'Bola9', 'Bola10',
                                           'Bola11', 'Bola12', 'Bola13', 'Bola14', 'Bola15']].values
            next_game = df.iloc[i+10][['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5',
                                     'Bola6', 'Bola7', 'Bola8', 'Bola9', 'Bola10',
                                     'Bola11', 'Bola12', 'Bola13', 'Bola14', 'Bola15']].values
            
            features.append(last_10_games.flatten())
            labels.append(next_game)
            
        X = np.array(features)
        y = np.array(labels)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_random_forest(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            return model.score(X_test, y_test)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        self.rf_model = RandomForestClassifier(**study.best_params, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
    def train_lstm(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Reshape data for LSTM [samples, time steps, features]
        X_train_reshaped = X_train.reshape((X_train.shape[0], 10, 15))
        
        model = Sequential([
            LSTM(128, input_shape=(10, 15), return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(15)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)
        
        self.lstm_model = model
        
    def train_prophet(self):
        df = pd.read_csv(self.data_path)
        
        # Treinar um modelo Prophet para cada número possível (1-25)
        for num in range(1, 26):
            # Preparar dados para o Prophet
            occurrences = []
            for _, row in df.iterrows():
                if num in row[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5',
                              'Bola6', 'Bola7', 'Bola8', 'Bola9', 'Bola10',
                              'Bola11', 'Bola12', 'Bola13', 'Bola14', 'Bola15']].values:
                    occurrences.append(1)
                else:
                    occurrences.append(0)
            
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['Data']),
                'y': occurrences
            })
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.fit(prophet_df)
            self.prophet_models.append((num, model))
    
    def train_xgboost(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            return model.score(X_test, y_test)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        self.xgb_model = xgb.XGBRegressor(**study.best_params, random_state=42)
        self.xgb_model.fit(X_train, y_train)
    
    def predict(self, last_10_games):
        # Ensemble prediction
        predictions = []
        
        # Random Forest prediction
        if self.rf_model:
            rf_pred = self.rf_model.predict(last_10_games.reshape(1, -1))
            predictions.append(rf_pred[0])
        
        # LSTM prediction
        if self.lstm_model:
            lstm_pred = self.lstm_model.predict(last_10_games.reshape(1, 10, 15))
            predictions.append(lstm_pred[0])
        
        # XGBoost prediction
        if self.xgb_model:
            xgb_pred = self.xgb_model.predict(last_10_games.reshape(1, -1))
            predictions.append(xgb_pred[0])
        
        # Prophet predictions
        if self.prophet_models:
            prophet_probs = []
            for num, model in self.prophet_models:
                future = model.make_future_dataframe(periods=1)
                forecast = model.predict(future)
                prob = forecast.iloc[-1]['yhat']
                prophet_probs.append((num, prob))
            
            # Get top 15 numbers with highest probability
            prophet_nums = sorted(prophet_probs, key=lambda x: x[1], reverse=True)[:15]
            prophet_pred = [x[0] for x in prophet_nums]
            predictions.append(prophet_pred)
        
        # Ensemble the predictions
        final_prediction = np.mean(predictions, axis=0)
        
        # Ensure we have exactly 15 unique numbers
        return np.unique(final_prediction.round())[:15]
    
    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            
        if self.rf_model:
            joblib.dump(self.rf_model, os.path.join(path, 'rf_model.joblib'))
        if self.lstm_model:
            self.lstm_model.save(os.path.join(path, 'lstm_model'))
        if self.xgb_model:
            joblib.dump(self.xgb_model, os.path.join(path, 'xgb_model.joblib'))
        if self.prophet_models:
            joblib.dump(self.prophet_models, os.path.join(path, 'prophet_models.joblib'))
    
    def load_models(self, path):
        if os.path.exists(os.path.join(path, 'rf_model.joblib')):
            self.rf_model = joblib.load(os.path.join(path, 'rf_model.joblib'))
        if os.path.exists(os.path.join(path, 'lstm_model')):
            self.lstm_model = tf.keras.models.load_model(os.path.join(path, 'lstm_model'))
        if os.path.exists(os.path.join(path, 'xgb_model.joblib')):
            self.xgb_model = joblib.load(os.path.join(path, 'xgb_model.joblib'))
        if os.path.exists(os.path.join(path, 'prophet_models.joblib')):
            self.prophet_models = joblib.load(os.path.join(path, 'prophet_models.joblib'))
