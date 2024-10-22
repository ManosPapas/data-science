SELECT
    *
FROM
    Bookings b
WHERE
    b.CreatedAt >= '{start_at}'
    AND p.CreatedAt <= '{end_at}'
ORDER BY
    b.CreatedAt DESC