# Courant AI Service

**AI/ML Microservice for Real-Time Anomaly Detection & Predictive Maintenance**

---

## Overview

The Courant AI Service is a Flask-based microservice that provides machine learning capabilities for the main Courant application. It implements LSTM (Long Short-Term Memory) neural networks for time-series anomaly detection and predictive maintenance analysis of power consumption data from IoT devices.

## Features

### 🧠 **Machine Learning Capabilities**
- **LSTM Neural Networks**: Time-series anomaly detection for each appliance profile
- **Isolation Forest**: Global outlier detection for batch processing
- **Real-Time Inference**: Sub-second prediction response times
- **Model Management**: Automated training and model persistence

### 📊 **Data Processing**
- **Time-Series Analysis**: Sequential pattern recognition in power data
- **Feature Engineering**: Automatic extraction of relevant power metrics
- **Data Normalization**: MinMax scaling for optimal model performance
- **Missing Data Handling**: Robust preprocessing for incomplete datasets

### 🔄 **Training & Updates**
- **Incremental Learning**: Models adapt to new usage patterns
- **Profile-Specific Models**: Separate models for each appliance type
- **Automated Retraining**: Scheduled model updates based on data drift
- **Model Versioning**: Historical model performance tracking

## Technical Stack

- **Framework**: Flask 3.1.0
- **ML Libraries**: TensorFlow 2.19, Keras 3.9, scikit-learn 1.6
- **Data Processing**: NumPy 2.1, Pandas 2.2, SQLAlchemy 2.0
- **Database**: MySQL Connector for direct database access
- **Model Persistence**: Joblib for model serialization

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Laravel App    │────│   AI Service    │────│   MySQL DB      │
│                 │    │                 │    │                 │
│ • HTTP Requests │    │ • Flask API     │    │ • Training Data │
│ • Job Dispatch  │    │ • LSTM Models   │    │ • Readings      │
│ • Alert Routing │    │ • Predictions   │    │ • Profiles      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
                       ┌─────────────────┐
                       │  Model Storage  │
                       │                 │
                       │ • LSTM Models   │
                       │ • Scalers       │
                       │ • Metadata      │
                       └─────────────────┘
```

## Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **MySQL**: Access to Courant database
- **Memory**: Minimum 2GB RAM for model training
- **Storage**: 1GB for model files and logs

### Installation Steps

1. **Navigate to AI Service Directory**:
```bash
cd courant_ai_service
```

2. **Create Virtual Environment**:
```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/MacOS:
source venv/bin/activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure Database Connection**:
Update the database configuration in `app.py`:
```python
# Database configuration
DB_HOST = 'localhost'
DB_NAME = 'courant'
DB_USER = 'your_db_user'
DB_PASS = 'your_db_password'
```

5. **Start the Service**:
```bash
# Development mode
python app.py

# Production mode (recommended)
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 wsgi:app
```

## API Endpoints

### **Health Check**
```http
GET /health
```
Returns service status and model information.

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "lstm_models_count": 5,
  "isolation_models_count": 3,
  "uptime": "2h 15m 30s"
}
```

### **Real-Time LSTM Anomaly Detection**
```http
POST /predict_lstm_anomaly
```
Analyzes current power reading for anomalies using LSTM model.

**Request Body**:
```json
{
  "device_uid": "ESP8266_001",
  "profile_id": 1,
  "current_data": {
    "voltage": 220.5,
    "current": 12.3,
    "power": 2710.15,
    "energy": 15.67,
    "power_factor": 0.92
  }
}
```

**Response**:
```json
{
  "status": "success",
  "anomaly_detected": true,
  "anomaly_score": 0.85,
  "prediction_accuracy": 0.92,
  "alert_metadata": {
    "severity": "high",
    "affected_fields": ["current", "power"],
    "business_impact": "Equipment may be overloading",
    "suggested_actions": ["Check motor bearings", "Inspect electrical connections"]
  }
}
```

### **Batch Anomaly Detection**
```http
POST /batch_anomaly_detection
```
Processes historical data for global anomaly patterns.

**Request Body**:
```json
{
  "profile_id": 1,
  "hours_back": 24
}
```

### **Model Training**
```http
POST /train_models/{profile_id}
```
Initiates training of LSTM model for specific appliance profile.

**Response**:
```json
{
  "status": "success",
  "training_started": true,
  "estimated_completion": "15 minutes",
  "data_points_used": 5040
}
```

### **Model Information**
```http
GET /models/info/{profile_id}
```
Returns model metadata and performance statistics.

## Model Training Process

### **Data Requirements**
- **Minimum Dataset**: 30 days of continuous readings (recommended)
- **Data Frequency**: 5-minute intervals for optimal performance
- **Quality Threshold**: 95% data completeness required
- **Feature Requirements**: All 5 power metrics (voltage, current, power, energy, power_factor)

### **Training Pipeline**

1. **Data Extraction**: Query historical readings from database
2. **Preprocessing**: Clean, normalize, and create sequences
3. **Feature Engineering**: Extract temporal patterns and trends
4. **Model Architecture**: LSTM with attention mechanism
5. **Training**: Supervised learning with anomaly labeling
6. **Validation**: Cross-validation with temporal splits
7. **Model Persistence**: Save trained models and scalers

### **Model Architecture**

```python
# LSTM Model Configuration
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(features)  # Output layer for reconstruction
])

# Optimization
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

### **Training Schedule**

Models are automatically retrained based on:
- **Data Drift Detection**: Significant changes in power patterns
- **Performance Degradation**: Accuracy drops below threshold
- **Scheduled Updates**: Weekly retraining for active profiles
- **Manual Triggers**: API-initiated training requests

## Performance Optimization

### **Memory Management**
- **Lazy Loading**: Models loaded on first request
- **Memory Pooling**: Efficient tensor allocation
- **Garbage Collection**: Automatic cleanup of unused models
- **Batch Processing**: Optimized for multiple predictions

### **Caching Strategy**
- **Model Caching**: Keep frequently used models in memory
- **Prediction Caching**: Cache recent predictions for identical inputs
- **Database Pooling**: Connection pool for database efficiency
- **Response Caching**: HTTP response caching for static data

### **Scaling Considerations**
- **Horizontal Scaling**: Multiple service instances supported
- **Load Balancing**: Stateless design for easy load distribution
- **Resource Limits**: Configurable memory and CPU constraints
- **Monitoring**: Built-in performance metrics and logging

## Monitoring & Logging

### **Application Logs**
- **Request Logging**: All API calls with timing information
- **Error Tracking**: Detailed error messages and stack traces
- **Model Performance**: Prediction accuracy and response times
- **Resource Usage**: Memory and CPU utilization tracking

### **Health Monitoring**
```bash
# Check service status
curl http://localhost:5000/health

# Monitor logs
tail -f logs/ai_service.log

# Performance metrics
curl http://localhost:5000/metrics
```

### **Log Levels**
- **DEBUG**: Detailed model training information
- **INFO**: API requests and normal operations
- **WARNING**: Performance issues and data quality problems
- **ERROR**: Failed predictions and system errors
- **CRITICAL**: Service failures and data corruption

## Troubleshooting

### **Common Issues**

#### **Model Loading Failures**
```bash
# Check model files exist
ls -la trained_*_models.joblib

# Verify permissions
chmod 644 *.joblib

# Regenerate models
python train_models_by_profile.py
```

#### **Database Connection Issues**
```python
# Test database connectivity
python -c "
from app import get_database_engine
engine = get_database_engine()
print('Database connection successful')
"
```

#### **Memory Issues**
```bash
# Monitor memory usage
ps aux | grep python

# Reduce model cache size in app.py
MAX_CACHED_MODELS = 3  # Reduce from default 10
```

#### **Performance Issues**
- **Slow Predictions**: Check model complexity and input size
- **High Memory Usage**: Reduce cached models and batch size
- **Database Timeouts**: Optimize queries and increase timeout values

### **Error Codes**

| Code | Description | Solution |
|------|-------------|----------|
| `MODEL_NOT_FOUND` | LSTM model not available for profile | Train model for profile |
| `INSUFFICIENT_DATA` | Not enough historical data | Collect more data (min 30 days) |
| `DATABASE_ERROR` | Database connection failed | Check DB credentials and connectivity |
| `INVALID_INPUT` | Malformed request data | Validate input format and required fields |
| `PREDICTION_FAILED` | Model inference error | Check model integrity and retrain if needed |

## Development & Testing

### **Running Tests**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html
```

### **Test Categories**
- **Unit Tests**: Individual function testing
- **Integration Tests**: API endpoint validation
- **Performance Tests**: Response time and throughput
- **Model Tests**: Prediction accuracy and consistency

### **Development Workflow**
```bash
# Start development server with auto-reload
export FLASK_ENV=development
python app.py

# Format code
black app.py *.py

# Run linting
flake8 app.py

# Check type hints
mypy app.py
```

## Production Deployment

### **WSGI Configuration**
```python
# wsgi.py
from app import app

if __name__ == "__main__":
    app.run()
```

### **Gunicorn Setup**
```bash
# Install Gunicorn
pip install gunicorn

# Production server
gunicorn --bind 0.0.0.0:5000 \
         --workers 4 \
         --timeout 120 \
         --worker-class sync \
         --max-requests 1000 \
         --preload \
         wsgi:app
```

### **Supervisor Configuration**
```ini
[program:courant-ai-service]
command=/path/to/venv/bin/gunicorn --bind 127.0.0.1:5000 --workers 2 wsgi:app
directory=/path/to/courant_ai_service
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/courant/ai-service.log
```

### **Docker Deployment**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "wsgi:app"]
```

## Security Considerations

- **Input Validation**: All API inputs validated and sanitized
- **Database Security**: Parameterized queries prevent SQL injection
- **Access Control**: Service should be behind authentication layer
- **Network Security**: Use HTTPS in production environments
- **Model Security**: Protect trained models from unauthorized access

## Performance Benchmarks

### **Response Times** (Average)
- **Health Check**: < 10ms
- **LSTM Prediction**: < 100ms
- **Batch Processing**: 2-5 seconds per 1000 records
- **Model Training**: 5-15 minutes per profile

### **Throughput**
- **Concurrent Requests**: Up to 50 simultaneous predictions
- **Daily Predictions**: > 100,000 predictions per day per instance
- **Model Updates**: Up to 10 concurrent training jobs

### **Resource Usage**
- **Memory**: 512MB base + 100MB per cached model
- **CPU**: 1-2 cores recommended for production
- **Storage**: 50MB per trained model file
- **Database**: 10-20 connections per instance

---

## Support & Maintenance

### **Regular Maintenance**
- **Model Retraining**: Weekly for active profiles
- **Log Rotation**: Daily cleanup of old log files
- **Performance Monitoring**: Continuous health checks
- **Database Optimization**: Monthly index and query optimization

### **Backup Procedures**
```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz *.joblib

# Backup configuration
cp app.py app.py.backup.$(date +%Y%m%d)
```

### **Updates & Patches**
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Apply security patches
pip audit --fix

# Version management
git tag -a v1.2.3 -m "AI Service v1.2.3"
```

For additional support, refer to the main Courant documentation or contact the development team.
