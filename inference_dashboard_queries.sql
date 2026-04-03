-- SQLite queries for monitoring and adoption-style analytics.
-- Run these against artifacts/prediction_logs.sqlite.

-- 1) Overall model usage + quality
SELECT
    COUNT(*) AS scans,
    AVG(CASE WHEN predicted_label = true_label THEN 1.0 ELSE 0.0 END) AS accuracy,
    AVG(confidence) AS avg_confidence,
    SUM(CASE WHEN predicted_label != true_label THEN 1 ELSE 0 END) AS errors
FROM prediction_logs;

-- 2) Daily trend of scan volume and accuracy
SELECT date(created_at) AS day,
       COUNT(*) AS scans,
       AVG(CASE WHEN predicted_label = true_label THEN 1.0 ELSE 0.0 END) AS accuracy,
       AVG(confidence) AS avg_confidence
FROM prediction_logs
GROUP BY date(created_at)
ORDER BY day;

-- 3) Split-level performance
SELECT split_name,
       COUNT(*) AS scans,
       AVG(CASE WHEN predicted_label = true_label THEN 1.0 ELSE 0.0 END) AS accuracy,
       AVG(confidence) AS avg_confidence
FROM prediction_logs
GROUP BY split_name
ORDER BY split_name;

-- 4) Error breakdown
SELECT true_label,
       predicted_label,
       COUNT(*) AS count
FROM prediction_logs
WHERE predicted_label != true_label
GROUP BY true_label, predicted_label
ORDER BY count DESC;

-- 5) Highest-confidence mistakes (for manual review)
SELECT scan_id, split_name, confidence, true_label, predicted_label, created_at
FROM prediction_logs
WHERE predicted_label != true_label
ORDER BY confidence DESC
LIMIT 10;
