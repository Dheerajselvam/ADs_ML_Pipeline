# Final Ads ML Report
Generated: 2025-12-11T11:27:14.726750Z

## 1) Offline Model Comparison

```
{
  "auc": 0.5895247591217817,
  "log_loss": 0.11984538533246777,
  "brier": 0.025283367936654734
}
```

## 2) A/B Test Summary

```
{
  "n_total": 20000,
  "treatment_pct": 0.5,
  "expected_counts": {
    "control": 10000,
    "treatment": 10000
  },
  "summary": {
    "control": {
      "impressions": 9423,
      "clicks": 255,
      "revenue": 393.85,
      "ctr": 0.027061445399554282,
      "rpm": 41.796667727899816
    },
    "treatment": {
      "impressions": 10577,
      "clicks": 266,
      "revenue": 406.63,
      "ctr": 0.02514890800794176,
      "rpm": 38.44473858371939
    }
  },
  "sample_ratio_mismatch": {
    "control_pct_diff": -0.0577,
    "treatment_pct_diff": 0.0577,
    "flag": true
  },
  "ctr_ztest": {
    "z": -0.8476149143937818,
    "p_value": 0.39665246554734845,
    "p1": 0.02514890800794176,
    "p2": 0.027061445399554282
  },
  "rpm_bootstrap": {
    "mean_diff": -0.0033286081909941955,
    "ci": [
      -0.011061385424881205,
      0.004385042914524896
    ]
  }
}
```

## 3) RTB Simulation Summary

```
{
  "total_impressions_served": 1274,
  "total_revenue": 25.0,
  "global_ctr": 0.019623233908948195,
  "global_rpm": 19.623233908948194,
  "per_advertiser": [
    {
      "advertiser": 0,
      "impressions": 96,
      "clicks": 3,
      "revenue": 3.0,
      "spend": 57.6038,
      "ctr": 0.03125,
      "rpm": 31.25
    },
    {
      "advertiser": 1,
      "impressions": 101,
      "clicks": 0,
      "revenue": 0.0,
      "spend": 65.8611,
      "ctr": 0.0,
      "rpm": 0.0
    },
    {
      "advertiser": 2,
      "impressions": 91,
      "clicks": 1,
      "revenue": 1.0,
      "spend": 59.3503,
      "ctr": 0.01098901098901099,
      "rpm": 10.989010989010989
    },
    {
      "advertiser": 3,
      "impressions": 82,
      "clicks": 2,
      "revenue": 2.0,
      "spend": 61.6655,
      "ctr": 0.024390243902439025,
      "rpm": 24.390243902439025
    },
    {
      "advertiser": 4,
      "impressions": 74,
      "clicks": 2,
      "revenue": 2.0,
      "spend": 57.782000000000004,
      "ctr": 0.02702702702702703,
      "rpm": 27.027027027027028
    },
    {
      "advertiser": 5,
      "impressions": 44,
      "clicks": 1,
      "revenue": 1.0,
      "spend": 35.4242,
      "ctr": 0.022727272727272728,
      "rpm": 22.727272727272727
    },
    {
      "advertiser": 6,
      "impressions": 38,
      "clicks": 2,
      "revenue": 2.0,
      "spend": 37.7094,
      "ctr": 0.05263157894736842,
      "rpm": 52.63157894736842
    },
    {
      "advertiser": 7,
      "impressions": 27,
      "clicks": 0,
      "revenue": 0.0,
      "spend": 20.773699999999998,
      "ctr": 0.0,
      "rpm": 0.0
    },
    {
      "advertiser": 8,
      "impressions": 31,
      "clicks": 0,
      "revenue": 0.0,
      "spend": 27.3536,
      "ctr": 0.0,
      "rpm": 0.0
    },
    {
      "advertiser": 9,
      "impressions": 21,
      "clicks": 1,
      "revenue": 1.0,
      "spend": 22.7138,
      "ctr": 0.047619047619047616,
      "rpm": 47.61904761904761
    },
    {
      "advertiser": 10,
      "impressions": 108,
      "clicks": 3,
      "revenue": 3.0,
      "spend": 72.6261,
      "ctr": 0.027777777777777776,
      "rpm": 27.777777777777775
    },
    {
      "advertiser": 11,
      "impressions": 110,
      "clicks": 1,
      "revenue": 1.0,
      "spend": 73.888,
      "ctr": 0.00909090909090909,
      "rpm": 9.09090909090909
    },
    {
      "advertiser": 12,
      "impressions": 121,
      "clicks": 4,
      "revenue": 4.0,
      "spend": 79.3728,
      "ctr": 0.03305785123966942,
      "rpm": 33.057851239669425
    },
    {
      "advertiser": 13,
      "impressions": 67,
      "clicks": 2,
      "revenue": 2.0,
      "spend": 46.983,
      "ctr": 0.029850746268656716,
      "rpm": 29.850746268656717
    },
    {
      "advertiser": 14,
      "impressions": 66,
      "clicks": 1,
      "revenue": 1.0,
      "spend": 53.8257,
      "ctr": 0.015151515151515152,
      "rpm": 15.151515151515152
    },
    {
      "advertiser": 15,
      "impressions": 62,
      "clicks": 2,
      "revenue": 2.0,
      "spend": 57.6345,
      "ctr": 0.03225806451612903,
      "rpm": 32.25806451612903
    },
    {
      "advertiser": 16,
      "impressions": 37,
      "clicks": 0,
      "revenue": 0.0,
      "spend": 27.884,
      "ctr": 0.0,
      "rpm": 0.0
    },
    {
      "advertiser": 17,
      "impressions": 39,
      "clicks": 0,
      "revenue": 0.0,
      "spend": 35.3605,
      "ctr": 0.0,
      "rpm": 0.0
    },
    {
      "advertiser": 18,
      "impressions": 33,
      "clicks": 0,
      "revenue": 0.0,
      "spend": 35.7296,
      "ctr": 0.0,
      "rpm": 0.0
    },
    {
      "advertiser": 19,
      "impressions": 26,
      "clicks": 0,
      "revenue": 0.0,
      "spend": 25.6064,
      "ctr": 0.0,
      "rpm": 0.0
    }
  ]
}
```

## 4) Recommendations

Do not deploy automatically; run longer experiment / check SRE issues
