# Chichi Fraud Detection API Documentation

This API provides a dual-layer security system for Nigerian banking:
1. **NLP Alert Classifier**: Analyzes notification text (SMS/Email).
2. **Backend Fraud Engine**: Analyzes transaction metadata (NIBSS patterns).

**Base URL (Local)**: `http://localhost:8000`  
**Production URL**: `https://chimera-fraud-api.onrender.com`

---

## 1. Health Check
Checks if the system is live and models are loaded.

- **URL**: `/health`
- **Method**: `GET`
- **Response**:
    - `status`: "healthy" or "error"
    - `alert_classifier`: "active" or "inactive"
    - `fraud_engine`: "active" or "inactive"

**Example Response**:
```json
{
  "status": "healthy",
  "alert_classifier": "active",
  "fraud_engine": "active"
}
```

---

## 2. Alert Analysis
Analyzes the text of a banking alert to determine if it is legitimate, fake, or suspicious.

- **URL**: `/predict/alert`
- **Method**: `POST`
- **Request Body**:
    - `text` (string): The raw message content.

**Example Request**:
```json
{
  "text": "GTBank ALERT: Your acount has bin restricted due to BVN issues. Verify here: bit.ly/gtb-secure"
}
```

**Response Body**:
- `text` (string): The submitted text.
- `prediction` (string): "Legitimate", "Fake/Phishing", or "Suspicious".
- `label_id` (int): 0 (Legitimate), 1 (Fake), 2 (Suspicious).

**Example Response**:
```json
{
  "text": "...",
  "prediction": "Fake/Phishing",
  "label_id": 1
}
```

---

## 3. Transaction Analysis
Analyzes financial metadata to detect fraudulent transaction patterns. Uses an optimized XGBoost model with a 0.20 threshold for high recall (~82%).

- **URL**: `/predict/transaction`
- **Method**: `POST`
- **Request Body**:

| Field | Type | Description |
| :--- | :--- | :--- |
| `amount` | float | Transaction amount in NGN |
| `hour` | int | Hour of day (0-23) |
| `day_of_week` | int | Day of week (0-6) |
| `month` | int | Month (1-12) |
| `is_weekend` | bool | True if Saturday/Sunday |
| `is_peak_hour` | bool | True if between peak transaction hours |
| `tx_count_24h` | float | Count of transactions in last 24h |
| `amount_sum_24h` | float | Total amount spent in last 24h |
| `amount_mean_7d` | float | Average amount spent in last 7 days |
| `amount_std_7d` | float | Standard deviation of amount in last 7 days |
| `tx_count_total` | int | Lifetime transaction count for customer |
| `amount_mean_total` | float | Lifetime average amount |
| `amount_std_total` | float | Lifetime standard deviation |
| `channel_diversity` | int | Number of different channels used by user |
| `location_diversity` | int | Number of different locations used by user |
| `amount_vs_mean_ratio` | float | `amount / amount_mean_total` |
| `online_channel_ratio` | float | Ratio of web/mobile transactions to total |
| `channel` | string | "Web", "Mobile", "POS", "ATM", etc. |
| `merchant_category` | string | e.g., "Grocery", "Electronics", "Transfer" |
| `bank` | string | Name of the bank |

**Example Request**:
```json
{
  "amount": 250000.0,
  "hour": 3,
  "day_of_week": 1,
  "month": 3,
  "is_weekend": false,
  "is_peak_hour": false,
  "tx_count_24h": 15.0,
  "amount_sum_24h": 1000000.0,
  "amount_mean_7d": 50000.0,
  "amount_std_7d": 20000.0,
  "tx_count_total": 200,
  "amount_mean_total": 45000.0,
  "amount_std_total": 15000.0,
  "channel_diversity": 3,
  "location_diversity": 2,
  "amount_vs_mean_ratio": 5.5,
  "online_channel_ratio": 0.8,
  "channel": "Web",
  "merchant_category": "Entertainment",
  "bank": "Zenith"
}
```

**Response Body**:
- `is_fraud` (bool): `true` if the transaction is flagged as fraud.
- `fraud_probability` (float): Probability score (0.0 to 1.0).
- `risk_level` (string): "Low", "Medium", or "High".

**Example Response**:
```json
{
  "is_fraud": true,
  "fraud_probability": 0.82,
  "risk_level": "High"
}
```
