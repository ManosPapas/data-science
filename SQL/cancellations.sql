SELECT
    *
FROM
    Cancellations c
WHERE
    c.CreatedAt >= '{start_at}'
    AND c.CreatedAt <= '{end_at}'
ORDER BY
    c.CreatedAt DESC