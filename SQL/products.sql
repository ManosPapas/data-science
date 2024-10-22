SELECT
    *
FROM
    Products p
WHERE
    p.CreatedAt >= '{start_at}'
    AND p.CreatedAt <= '{end_at}'
ORDER BY
    p.CreatedAt DESC