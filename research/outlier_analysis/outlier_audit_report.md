# Outlier Timestamp + OHLCV Audit -- A-HMM4

DIAGNOSTIC ONLY. No production files, features, scaler, or HMM were modified. Reuses the existing, already-fit A-HMM4 model (results/model_A_HMM4.pkl), re-decoded (not refit) against the existing gap-aware feature matrix.

## Summary counts

- **total_rows_scanned**: 933924
- **total_valid_rows**: 923766
- **total_4sigma_obs**: 46883
- **total_5sigma_obs**: 26424
- **total_6sigma_obs**: 10441
- **total_outlier_events**: 18488
- **total_top100_ret5m_events**: 100
- **total_state3_obs**: 137590
- **total_state3_pct_of_dataset**: 14.894464615497865
- **total_state3_outlier_obs_over4s**: 33629
- **total_state3_outlier_pct**: 24.441456501199216
- **total_state3_over5s**: 19511
- **total_state3_over6s**: 10396
- **n_state3_segments**: 8509
- **n_state3_segments_near_outlier**: 2957
- **pct_state3_segments_near_outlier**: 34.75143965213304
- **outlier_transition_pct_at**: 11.404986882238765
- **normal_transition_pct_at**: 9.502864122123476
- **runtime_s**: 39.84806275367737

## Q1-3: Extreme observation counts

- >4sigma: 46,883
- >5sigma: 26,424
- >6sigma: 10,441

## Q4: Which features produce the most extreme observations

max_zscore_feature
vol_change    12842
skew_4h        6086
ret_5m         3594
vol_15m        3452
ret_4h         3270
ret_15m        3122
ret_2h         2858
ret_30m        2755
ret_1h         2715
vol_2h         2499
vol_1h         2130
vwap_dist      1542
skew_1h          18

## Q5-6: Top 20 |ret_5m| events (exact timestamps + OHLCV)

            timestamp_utc      open      high       low     close      volume  ret_5m_pct  hmm_state       regime_label
2020-03-13 02:35:00+00:00   4409.50   5252.49   4378.66   5222.12 4493.537984   16.914215          3 Ranging / High-Vol
2020-03-12 10:55:00+00:00   6697.94   6780.00   6012.50   6014.90 4275.473838  -10.786780          3 Ranging / High-Vol
2021-09-07 15:05:00+00:00  47818.21  47832.82  42900.00  43088.74 5375.452315  -10.414503          3 Ranging / High-Vol
2021-05-19 13:05:00+00:00  33358.60  33700.00  30100.00  30101.00 5348.920054  -10.200150          3 Ranging / High-Vol
2023-08-17 21:40:00+00:00  27578.74  27605.86  25188.00  25188.01 5543.128560   -9.067710          3 Ranging / High-Vol
2021-05-19 13:15:00+00:00  32437.76  35700.00  32200.00  35337.60 4205.029510    8.986608          3 Ranging / High-Vol
2019-06-26 20:40:00+00:00  13082.91  13082.91  11801.00  11958.77 3550.058723   -8.349292          3 Ranging / High-Vol
2017-12-10 23:00:00+00:00  14356.49  15700.00  14356.49  15620.11  147.150960    8.272083          3 Ranging / High-Vol
2020-05-10 00:15:00+00:00   9343.04   9349.00   8600.00   8628.15 8277.172300   -7.935858          3 Ranging / High-Vol
2021-05-19 12:50:00+00:00  35512.32  36914.56  32750.00  32904.67 6921.875616   -7.626509          3 Ranging / High-Vol
2025-10-10 21:15:00+00:00 112209.36 112510.92 103800.00 103975.26 5120.929630   -7.621343          3 Ranging / High-Vol
2018-01-16 22:30:00+00:00   9100.01   9800.00   9035.00   9800.00  747.770835    7.410687          3 Ranging / High-Vol
2020-03-13 02:25:00+00:00   3936.77   4240.75   3936.77   4230.94 2287.703678    7.206360          3 Ranging / High-Vol
2019-09-24 19:40:00+00:00   8435.93   8438.96   7800.00   7846.93 3758.058849   -7.113570          3 Ranging / High-Vol
2021-05-19 13:10:00+00:00  30101.00  32666.00  30000.00  32300.46 4736.700407    7.052308          3 Ranging / High-Vol
2017-12-22 14:50:00+00:00  11231.02  12099.00  11203.49  12050.81  278.912766    7.045228          3 Ranging / High-Vol
2017-11-29 19:40:00+00:00   8700.00   9450.00   8520.00   9330.80  216.640357    6.999773          3 Ranging / High-Vol
2017-09-04 08:40:00+00:00   4299.99   4372.20   4299.99   4372.20   15.670147    6.964738          3 Ranging / High-Vol
2017-11-12 07:05:00+00:00   5779.00   6189.00   5779.00   6189.00   99.179875    6.940844          3 Ranging / High-Vol
2021-05-19 12:55:00+00:00  32915.29  36600.00  32488.89  35234.62 5692.078771    6.841453          3 Ranging / High-Vol

## Q7: What % of extreme ret_5m observations belong to State 3

 threshold_pct  number_of_observations  state_3_count  state_3_percentage
           0.1                     924            924               100.0
           0.5                    4619           4619               100.0
           1.0                    9238           9238               100.0

## Q8: Isolated vs sustained volatility (major events)

case_classification
B: sustained high-vol regime    30

 event_id           start_timestamp  peak_max_abs_z  state_at_event  state_3_duration_after_event_bars          case_classification
       52 2017-09-04 07:10:00+00:00       29.635708               3                                163 B: sustained high-vol regime
      194 2017-09-15 06:05:00+00:00       24.670200               3                                427 B: sustained high-vol regime
     1109 2017-11-08 17:15:00+00:00       27.725761               3                                118 B: sustained high-vol regime
     1185 2017-11-12 03:30:00+00:00       35.849419               3                                574 B: sustained high-vol regime
     1447 2017-11-29 15:15:00+00:00       38.327559               3                                610 B: sustained high-vol regime
     1610 2017-12-10 21:15:00+00:00       35.198826               3                                124 B: sustained high-vol regime
     1698 2017-12-19 21:10:00+00:00       27.317692               3                                437 B: sustained high-vol regime
     1737 2017-12-22 00:55:00+00:00       30.087284               3                                546 B: sustained high-vol regime
     2019 2018-01-11 02:55:00+00:00       28.482704               3                                477 B: sustained high-vol regime
     2091 2018-01-16 17:15:00+00:00       31.533346               3                               1070 B: sustained high-vol regime
     2347 2018-02-02 12:35:00+00:00       23.256022               3                                383 B: sustained high-vol regime
     2440 2018-02-05 19:20:00+00:00       24.219309               3                                638 B: sustained high-vol regime
     4075 2018-10-15 05:40:00+00:00       28.355584               3                                 84 B: sustained high-vol regime
     5112 2019-06-26 20:30:00+00:00       41.302935               3                                546 B: sustained high-vol regime
     5846 2019-09-24 18:45:00+00:00       31.246134               3                                 83 B: sustained high-vol regime
     5988 2019-10-26 00:30:00+00:00       27.579630               3                                 72 B: sustained high-vol regime
     6188 2019-12-04 13:20:00+00:00       24.759898               3                                 44 B: sustained high-vol regime
     6581 2020-03-12 10:30:00+00:00       48.801749               3                                766 B: sustained high-vol regime
     6588 2020-03-12 20:15:00+00:00       71.973500               3                                649 B: sustained high-vol regime
     6996 2020-05-10 00:10:00+00:00       33.770476               3                                 66 B: sustained high-vol regime
     7131 2020-06-02 14:45:00+00:00       26.107480               3                                 48 B: sustained high-vol regime
     7363 2020-08-02 04:35:00+00:00       23.173670               3                                 48 B: sustained high-vol regime
     8414 2021-04-18 03:15:00+00:00       29.091410               3                                 55 B: sustained high-vol regime
     8489 2021-05-12 22:05:00+00:00       23.227422               3                                 90 B: sustained high-vol regime
     8581 2021-05-19 11:25:00+00:00       56.262585               3                                472 B: sustained high-vol regime
     8611 2021-05-21 14:10:00+00:00       23.030816               3                                351 B: sustained high-vol regime
     9066 2021-07-26 00:50:00+00:00       23.292568               3                                 50 B: sustained high-vol regime
     9243 2021-09-07 14:45:00+00:00       46.707362               3                                 93 B: sustained high-vol regime
    12131 2023-08-17 21:40:00+00:00       38.586822               3                                 41 B: sustained high-vol regime
    15997 2025-10-10 21:10:00+00:00       32.432127               3                                 56 B: sustained high-vol regime

## Q9: State 3 duration after major events (bars, 1 bar = 5 min)

count      30.000000
mean      306.033333
std       278.821842
min        41.000000
25%        67.500000
50%       143.500000
75%       528.750000
max      1070.000000

## Q10: Outlier vs normal timestamps -- HMM transition coincidence

Outlier bars: {'n': 46883, 'pct_changed_at': 11.404986882238765, 'pct_changed_before': 5.087131796173453, 'pct_changed_after': 3.5257982637629843}
Normal bars:  {'n': 876883, 'pct_changed_at': 9.502864122123476, 'pct_changed_before': 9.840651489423333, 'pct_changed_after': 9.924128988702027}

Outlier bars are 1.20x more likely to coincide with a state transition than normal bars.

## Q11: What does State 3 represent?

- Total State 3 observations: 137,590 (14.89% of dataset)
- State 3 observations containing a >4sigma feature: 33,629 (24.44% of State 3)
- State 3 segments beginning at/near (+/-1 bar) an outlier event: 2,957 / 8,509 (34.75%)

**Verdict (from measured data only): MIXTURE: most State 3 observations/segments are NOT directly outlier-triggered, suggesting State 3 is a genuine (if noisy) high-volatility regime that ALSO reliably absorbs extreme events when they occur -- not purely an outlier bucket.**

## Q12: Does fat-tail behavior plausibly contribute to rapid state switching?

This must be judged from the Q10 transition-coincidence evidence above, not assumed. The data does NOT show a strong disproportionate link between outlier bars and state transitions relative to normal bars -- so this analysis does not support claiming fat tails are a primary driver of the excessive switching seen elsewhere; the switching appears to happen at a broadly similar rate regardless of whether the bar is an outlier.
