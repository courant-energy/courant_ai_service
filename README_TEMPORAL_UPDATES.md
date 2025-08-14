# Courant AI Service - Temporal Anomaly Detection

## Recent Improvements

This service has been updated to support temporal-aware anomaly detection with efficient daily retraining capabilities.

## Key Changes

### 1. Temporal Feature Engineering
- **Isolation Forest**: Now uses 33 temporal features instead of 5 static ones
- **LSTM**: Proper sequence-based training with timestamps
- **Time Context**: Includes hour of day, day of week, and weekend indicators

### 2. Efficient Daily Retraining
- **30-Day Window**: Only processes last 30 days of data instead of entire history
- **Incremental Updates**: Updates existing models rather than training from scratch
- **Model Persistence**: Preserves existing models if retraining fails

### 3. Improved Data Handling
- **Timestamps**: All queries now include `created_at` for proper temporal ordering
- **Missing Values**: Better handling with forward/backward fill
- **Temporal Splits**: No random shuffling for time series data

## File Structure

```
courant_ai_service/
├── app.py                           # Flask API (needs updating for temporal features)
├── train_models_by_profile.py       # Improved Isolation Forest training
├── train_lstm_models_by_profile.py  # Improved LSTM training
├── historical_storage.py            # New: Temporal data management
├── daily_training.bat              # New: Daily training automation
├── backups/                        # New: Model backup directory
├── trained_models.joblib           # Isolation Forest models
├── trained_lstm_models.joblib      # LSTM models
└── requirements.txt
```

## Training Scripts

### Isolation Forest (`train_models_by_profile.py`)
- **Temporal Features**: 33 features including trends, volatility, and time context
- **30-Day Window**: Only processes recent data for efficiency
- **Model Metadata**: Stores scaler, sensitivity, and training info

### LSTM (`train_lstm_models_by_profile.py`)
- **Sequence-Based**: Uses proper time sequences for training
- **Incremental Learning**: Updates existing models with fewer epochs
- **Minimum Requirements**: 50+ sequences needed for training

## Daily Training

### Automated Script
Run `daily_training.bat` to train both model types:
```cmd
daily_training.bat
```

### Manual Training
```cmd
# Train Isolation Forest
python train_models_by_profile.py

# Train LSTM models
python train_lstm_models_by_profile.py
```

### Scheduling (Windows Task Scheduler)
```cmd
# Create daily task for 2 AM
schtasks /create /tn "Courant Daily Training" /tr "c:\path\to\daily_training.bat" /sc daily /st 02:00
```

## Configuration

### Database Connection
Update these variables in both training scripts:
```python
DB_HOST = 'localhost'
DB_NAME = 'courant'
DB_USER = 'user'
DB_PASS = 'your_password'
```

### Training Parameters
```python
DAYS_TO_RETRAIN = 30              # Days of data to use
TEMPORAL_WINDOW = 5               # Window for temporal features
MIN_SEQUENCES_FOR_TRAINING = 50   # Minimum sequences needed
SEQUENCE_LENGTH = 10              # LSTM sequence length
```

## Model Outputs

### Isolation Forest Models
```python
{
    'model': IsolationForest(),
    'scaler': StandardScaler(),
    'sensitivity': 'medium',
    'contamination': 0.10,
    'temporal_window': 5,
    'feature_count': 33,
    'training_samples': 1500,
    'last_updated': datetime.now(),
    'feature_names': [...]
}
```

### LSTM Models
```python
{
    'model': Sequential(),
    'scaler': MinMaxScaler(),
    'sensitivity': 'medium',
    'last_updated': datetime.now(),
    'training_samples': 200,
    'sequence_length': 10,
    'feature_columns': [...]
}
```

## Next Steps

### 1. Update Flask API
The `app.py` file needs to be updated to support temporal predictions:
- Add temporal feature creation for Isolation Forest
- Implement sequence building for LSTM
- Add historical data storage

### 2. Add Monitoring
- Log training times and performance
- Alert on training failures
- Monitor model drift

### 3. Performance Optimization
- Consider model compression
- Implement caching for frequent predictions
- Add batch prediction endpoints

## Benefits

1. **Faster Training**: 30-day window reduces training time significantly
2. **Better Accuracy**: Temporal features capture time-based patterns
3. **Automated Operations**: Daily retraining with minimal manual intervention
4. **Model Persistence**: Graceful handling of training failures
5. **Scalable**: Efficient memory usage and processing

## Dependencies

Ensure these packages are installed:
```txt
pandas
numpy
scikit-learn
tensorflow
joblib
sqlalchemy
mysql-connector-python
```

## Troubleshooting

### Common Issues
1. **Insufficient Data**: Ensure profiles have enough historical readings
2. **Memory Issues**: Reduce `DAYS_TO_RETRAIN` if experiencing memory problems
3. **Training Failures**: Check database connectivity and data quality
4. **Model Loading**: Verify `.joblib` files are not corrupted
