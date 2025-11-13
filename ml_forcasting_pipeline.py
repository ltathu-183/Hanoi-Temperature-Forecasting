import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class WeatherForecastPipeline:
    """Pipeline dự đoán nhiệt độ 5 ngày với dữ liệu 10 năm"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_prepare_data(self):
        """Load và chuẩn bị dữ liệu 10 năm"""
        print("=== LOADING 10-YEAR WEATHER DATA ===")

        BASE_DIR = Path(__file__).parent

        data_path = BASE_DIR / "data" / "raw" / "daily_data.csv"
        df = pd.read_csv(data_path)
        
        print(f"Original data shape: {df.shape}")
        
        # Convert datetime và sort
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"Data period: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Total days: {len(df)} days ({len(df)/365.25:.1f} years)")
        
        # Kiểm tra missing values
        missing_info = df.isnull().sum()
        missing_pct = (missing_info / len(df)) * 100
        
        print(f"\nMissing data analysis:")
        for col in missing_info[missing_info > 0].index:
            print(f"  {col}: {missing_info[col]} ({missing_pct[col]:.1f}%)")
        
        # Loại bỏ features với quá nhiều missing values (>20%)
        high_missing = missing_pct[missing_pct > 20].index.tolist()
        if high_missing:
            print(f"\nRemoving features with >20% missing: {high_missing}")
            df = df.drop(columns=high_missing)
        
        # Loại bỏ các cột không cần thiết
        exclude_cols = ['name', 'resolvedAddress', 'tzoffset']
        df = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        
        # Fill missing values đơn giản
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
        for col in df.select_dtypes(include=[object]).columns:
            if col != 'datetime' and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        print(f"Final data shape: {df.shape}")
        
        return df
    
    def add_time_features(self, df):
        """Thêm time-based features"""
        print("Adding time-based features...")
        
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['dayofyear'] = df['datetime'].dt.dayofyear
        df['weekday'] = df['datetime'].dt.weekday
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # Season encoding
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                       3: 1, 4: 1, 5: 1,   # Spring
                                       6: 2, 7: 2, 8: 2,   # Summer
                                       9: 3, 10: 3, 11: 3}) # Fall
        
        print(f"Added time features. New shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
        if 'datetime' in categorical_cols:
            categorical_cols.remove('datetime')
            
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  Encoded: {col}")
            
        return df
    
    def prepare_features(self, df):
        """Chuẩn bị features cuối cùng"""
        print("Preparing final features...")
        
        # Loại bỏ datetime và các cột không cần thiết
        feature_cols = [col for col in df.columns if col not in ['datetime']]
        
        print(f"Selected features: {len(feature_cols)}")
        print(f"Feature types: {df[feature_cols].dtypes.value_counts().to_dict()}")
        
        return df, feature_cols
    
    def split_train_test(self, df, train_ratio=0.8):
        """Chia train/test theo thời gian"""
        print(f"\n=== TRAIN/TEST SPLIT ({int(train_ratio*100)}/{int((1-train_ratio)*100)}) ===")
        
        split_idx = int(len(df) * train_ratio)
        
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        
        print(f"Train period: {df_train['datetime'].min()} to {df_train['datetime'].max()}")
        print(f"Test period: {df_test['datetime'].min()} to {df_test['datetime'].max()}")
        print(f"Train samples: {len(df_train)} ({len(df_train)/365.25:.1f} years)")
        print(f"Test samples: {len(df_test)} ({len(df_test)/365.25:.1f} years)")
        
        return df_train, df_test
    
    def create_rolling_windows(self, df, features, window_size=30, forecast_days=5):
        """Tạo rolling windows cho training"""
        print(f"\n=== CREATING ROLLING WINDOWS ===")
        print(f"Window size: {window_size} days")
        print(f"Forecast horizon: {forecast_days} days")
        
        X, y, dates = [], [], []
        
        for i in range(window_size, len(df) - forecast_days + 1):
            # Input: window_size ngày trước
            X_window = df[features].iloc[i-window_size:i].values.flatten()
            
            # Output: nhiệt độ forecast_days ngày tiếp theo
            y_window = df['temp'].iloc[i:i+forecast_days].values
            
            # Date corresponding to forecast start
            date_window = df['datetime'].iloc[i]
            
            X.append(X_window)
            y.append(y_window)
            dates.append(date_window)
            
            if len(X) % 500 == 0:
                print(f"  Created {len(X)} windows...")
        
        X = np.array(X)
        y = np.array(y)
        dates = np.array(dates)
        
        print(f"Windows created:")
        print(f"  X shape: {X.shape} (samples, features)")
        print(f"  y shape: {y.shape} (samples, forecast_days)")
        print(f"  Date range: {dates[0]} to {dates[-1]}")
        
        return X, y, dates
    
    def initialize_models(self):
        """Khởi tạo các ML models"""
        models = {
            'Linear Regression': MultiOutputRegressor(LinearRegression()),
            'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0)),
            'Lasso Regression': MultiOutputRegressor(Lasso(alpha=1.0)),
            'Random Forest': MultiOutputRegressor(RandomForestRegressor(
                n_estimators=100, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1)),
            'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42)),
            'LightGBM': MultiOutputRegressor(lgb.LGBMRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
            'CatBoost': MultiOutputRegressor(cb.CatBoostRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                random_state=42, verbose=False)),
            'Decision Tree': MultiOutputRegressor(DecisionTreeRegressor(
                max_depth=15, min_samples_split=5, random_state=42)),
            'KNN': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5, weights='distance')),
            'SVR': MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
        }
        
        return models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, dates_test):
        """Train và evaluate tất cả models"""
        print(f"\n=== MODEL TRAINING & EVALUATION ===")
        
        models = self.initialize_models()
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                start_time = datetime.now()
                
                # Training
                model.fit(X_train, y_train)
                train_time = (datetime.now() - start_time).total_seconds()
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics cho từng ngày
                daily_metrics = {}
                for day in range(5):
                    # Test metrics
                    test_rmse = np.sqrt(mean_squared_error(y_test[:, day], y_pred_test[:, day]))
                    test_mae = mean_absolute_error(y_test[:, day], y_pred_test[:, day])
                    test_r2 = r2_score(y_test[:, day], y_pred_test[:, day])
                    
                    # Train metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train[:, day], y_pred_train[:, day]))
                    train_r2 = r2_score(y_train[:, day], y_pred_train[:, day])
                    
                    daily_metrics[f'day_{day+1}'] = {
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'train_r2': train_r2
                    }
                
                # Overall metrics
                overall_test_rmse = np.mean([daily_metrics[f'day_{i+1}']['test_rmse'] for i in range(5)])
                overall_test_mae = np.mean([daily_metrics[f'day_{i+1}']['test_mae'] for i in range(5)])
                overall_test_r2 = np.mean([daily_metrics[f'day_{i+1}']['test_r2'] for i in range(5)])
                
                results[name] = {
                    'model': model,
                    'overall_test_rmse': overall_test_rmse,
                    'overall_test_mae': overall_test_mae,
                    'overall_test_r2': overall_test_r2,
                    'daily_metrics': daily_metrics,
                    'train_time': train_time,
                    'predictions_test': y_pred_test,
                    'dates_test': dates_test
                }
                
                print(f"  ✓ {name}: RMSE={overall_test_rmse:.3f}, MAE={overall_test_mae:.3f}, R²={overall_test_r2:.3f}")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)}")
                continue
        
        return results
    
    def save_results(self, results):
        """Lưu kết quả chi tiết"""
        print(f"\n=== SAVING RESULTS ===")
        
        # Summary results
        summary_data = []
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Overall_RMSE': metrics['overall_test_rmse'],
                'Overall_MAE': metrics['overall_test_mae'],
                'Overall_R2': metrics['overall_test_r2'],
                'Train_Time_sec': metrics['train_time']
            }
            
            # Add daily metrics
            for day in range(1, 6):
                day_key = f'day_{day}'
                if day_key in metrics['daily_metrics']:
                    day_data = metrics['daily_metrics'][day_key]
                    row[f'Day{day}_RMSE'] = day_data['test_rmse']
                    row[f'Day{day}_MAE'] = day_data['test_mae']
                    row[f'Day{day}_R2'] = day_data['test_r2']
            
            summary_data.append(row)
        
        df_results = pd.DataFrame(summary_data)
        df_results = df_results.sort_values('Overall_R2', ascending=False)
        df_results.to_csv('weather_forecast_results.csv', index=False)
        
        print("✓ Results saved to weather_forecast_results.csv")
        return df_results
    
    def create_visualizations(self, results):
        """Tạo visualizations"""
        print(f"\n=== CREATING VISUALIZATIONS ===")
        
        # Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall Performance
        models = list(results.keys())
        rmse_scores = [results[m]['overall_test_rmse'] for m in models]
        r2_scores = [results[m]['overall_test_r2'] for m in models]
        
        axes[0,0].barh(models, rmse_scores, color='lightcoral')
        axes[0,0].set_title('Overall Test RMSE by Model')
        axes[0,0].set_xlabel('RMSE (°C)')
        
        axes[0,1].barh(models, r2_scores, color='lightblue')
        axes[0,1].set_title('Overall Test R² by Model')
        axes[0,1].set_xlabel('R² Score')
        
        # 2. Daily performance heatmap
        daily_rmse_data = []
        for model in models:
            row = []
            for day in range(1, 6):
                rmse = results[model]['daily_metrics'][f'day_{day}']['test_rmse']
                row.append(rmse)
            daily_rmse_data.append(row)
        
        daily_rmse_df = pd.DataFrame(daily_rmse_data, 
                                   index=models, 
                                   columns=[f'Day {i}' for i in range(1, 6)])
        
        sns.heatmap(daily_rmse_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1,0])
        axes[1,0].set_title('RMSE by Model and Forecast Day')
        
        # 3. R² heatmap
        daily_r2_data = []
        for model in models:
            row = []
            for day in range(1, 6):
                r2 = results[model]['daily_metrics'][f'day_{day}']['test_r2']
                row.append(r2)
            daily_r2_data.append(row)
        
        daily_r2_df = pd.DataFrame(daily_r2_data,
                                 index=models,
                                 columns=[f'Day {i}' for i in range(1, 6)])
        
        sns.heatmap(daily_r2_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1,1])
        axes[1,1].set_title('R² by Model and Forecast Day')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualization saved to model_comparison.png")
    
    def print_summary(self, results, df_results):
        """In summary chi tiết"""
        print(f"\n{'='*80}")
        print("WEATHER FORECAST PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        # Top performers
        print(f"\n📊 MODEL RANKING (by Overall R²):")
        print(f"{'Rank':<5} {'Model':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Time(s)':<8}")
        print("-" * 65)
        
        for idx, (_, row) in enumerate(df_results.iterrows(), 1):
            print(f"{idx:<5} {row['Model']:<20} {row['Overall_RMSE']:<8.3f} "
                  f"{row['Overall_MAE']:<8.3f} {row['Overall_R2']:<8.3f} "
                  f"{row['Train_Time_sec']:<8.1f}")
        
        # Best model details
        best_model_name = df_results.iloc[0]['Model']
        best_metrics = results[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Overall: RMSE={best_metrics['overall_test_rmse']:.3f}°C, "
              f"MAE={best_metrics['overall_test_mae']:.3f}°C, R²={best_metrics['overall_test_r2']:.3f}")
        
        print(f"\n📅 DAILY PERFORMANCE ({best_model_name}):")
        for day in range(1, 6):
            day_data = best_metrics['daily_metrics'][f'day_{day}']
            print(f"   Day {day}: RMSE={day_data['test_rmse']:.3f}°C, "
                  f"MAE={day_data['test_mae']:.3f}°C, R²={day_data['test_r2']:.3f}")
        
        # Performance degradation
        day1_rmse = best_metrics['daily_metrics']['day_1']['test_rmse']
        day5_rmse = best_metrics['daily_metrics']['day_5']['test_rmse']
        degradation = ((day5_rmse - day1_rmse) / day1_rmse) * 100
        
        print(f"\n📈 FORECAST QUALITY:")
        print(f"   Day 1 RMSE: {day1_rmse:.3f}°C")
        print(f"   Day 5 RMSE: {day5_rmse:.3f}°C")
        print(f"   Performance degradation: {degradation:.1f}%")
        
        print(f"\n💡 INSIGHTS:")
        print(f"   • Best performing model class: {'Ensemble' if 'Forest' in best_model_name or 'GB' in best_model_name else 'Linear'}")
        print(f"   • Forecast accuracy decreases with time horizon (expected)")
        print(f"   • 10-year dataset provides robust training foundation")
    
    def run_complete_pipeline(self, window_size=30, forecast_days=5):
        """Chạy pipeline hoàn chỉnh"""
        print("🌡️  HANOI TEMPERATURE FORECASTING PIPELINE")
        print("🗓️  10-Year Historical Data | 5-Day Forecast Horizon")
        print(f"🔄  Rolling Window: {window_size} days")
        print("🤖  Machine Learning Models Comparison\n")
        
        # 1. Load và prepare data
        df = self.load_and_prepare_data()
        
        # 2. Add time features
        df = self.add_time_features(df)
        
        # 3. Encode categorical
        df = self.encode_categorical_features(df)
        
        # 4. Prepare features
        df, features = self.prepare_features(df)
        
        # 5. Split train/test
        df_train, df_test = self.split_train_test(df)
        
        # 6. Create rolling windows
        X_train, y_train, dates_train = self.create_rolling_windows(
            df_train, features, window_size, forecast_days
        )
        X_test, y_test, dates_test = self.create_rolling_windows(
            df_test, features, window_size, forecast_days
        )
        
        # 7. Scale features
        print(f"\n=== FEATURE SCALING ===")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"✓ Features scaled using StandardScaler")
        print(f"  Train shape: {X_train_scaled.shape}")
        print(f"  Test shape: {X_test_scaled.shape}")
        
        # 8. Train và evaluate models
        results = self.train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, dates_test)
        
        # 9. Save results
        df_results = self.save_results(results)
        
        # 10. Create visualizations  
        self.create_visualizations(results)
        
        # 11. Print summary
        self.print_summary(results, df_results)
        
        return results, df_results
    # Add this method to WeatherForecastPipeline class
    def save_best_model(self, results, model_path="best_model.pkl"):
        """Save the best performing model"""
        import joblib
        
        # Find best model by R²
        best_model_name = max(results.keys(), key=lambda k: results[k]['overall_test_r2'])
        best_model = results[best_model_name]['model']
        
        # Save model + scaler + feature info
        model_package = {
            'model': best_model,
            'scaler': self.scaler,
            'window_size': 30,  # your window size
            'forecast_days': 5,
            'best_model_name': best_model_name
        }
        
        joblib.dump(model_package, model_path)
        print(f"✅ Best model ({best_model_name}) saved to {model_path}")
        return model_package
def main():
    pipeline = WeatherForecastPipeline()
    results, df_results = pipeline.run_complete_pipeline(window_size=30, forecast_days=5)
    
    # Save best model
    model_path = Path(__file__).parent.parent / "best_model.pkl"
    pipeline.save_best_model(results, model_path)
    
    return pipeline, results, df_results
    

if __name__ == "__main__":
    pipeline, results, df_results = main()