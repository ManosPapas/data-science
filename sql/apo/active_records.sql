-- Example parameterized query for the "apo" workspace.
-- Bound params (:start, :end) are supplied via read_sql("apo/active_records", conn="apo", params=...).
SELECT *
FROM dbo.SomeTable
WHERE CreatedAt >= :start
  AND CreatedAt <= :end
ORDER BY CreatedAt DESC;
