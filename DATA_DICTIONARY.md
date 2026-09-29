# Data dictionary

The bundled data is synthetic. User-confirmed semantics: one machine-day per row; daily outcomes; an indicator of 1 is an actual maintenance event. Date format: YYYY-MM-DD. All local exports preserve this interpretation in project metadata.

| Field | Meaning / unit |
| --- | --- |
| Machine_ID | Machine identifier, retained for grouping; excluded as predictor |
| Machine_Category | Machine category; one consistent category per machine |
| Observation_Date | Calendar day whose outcomes are recorded |
| Maintenance_Date | Actual event date when Maintenance_Performed=1; never assumed an observation date |
| Maintenance_Performed | 0/1 event indicator, explicitly confirmed by user |
| Maintenance_Type | Recorded event type; absent values mean unknown/not recorded |
| Downtime_Hours | Recorded daily downtime hours; missing is unknown, not zero |
| Production_Volume | Daily output in unspecified production units; units may differ by category |
| Failure_Count | Nonnegative integer daily failure count |
| Planned_Production_Hours | Supplied scheduled hours; not automatically assumed equivalent to elapsed duration |
| Planned_Maintenance_Hours | Supplied planned maintenance hours; not a maintenance cost |
| Operating_Hours (optional) | Actual supplied operating exposure hours |
| Period_Start / Period_End (optional) | Explicit interval endpoints; ordering is validated; not invented |
| Maintenance_Event_ID (optional) | Distinguishes confirmed same-day events when complete |

Default analysis uses downtime per observed day. It does not infer operating hours or subtract planned maintenance from recorded downtime. Production comparisons across heterogeneous categories need engineering context and common units. Outcome support counts are supplied for each aggregate. Category estimates weight machines equally. Dates with multiple non-identical machine-day observations are disabled for temporal analysis until corrected.

Forecast prediction time is the end of the completed prior period. Historical event features include only events on or before that point. Unknown days since maintenance are median-imputed inside the training-fitted pipeline. Missing targets are excluded. The supplied model is a next-day forecast, not an estimate of the causal effect of preventive maintenance.

If Observation_Date is absent or semantics are unconfirmed, daily before/after comparison and forecasting are unavailable. A grouped explanatory model may remain available after row meaning is confirmed. Frequency rates require a known common calendar horizon. Explicit intervals currently require upstream daily disaggregation for paired analysis. Missing type prevents preventive-only claims. Costs, machine condition, manufacturer guidance and workload controls are not provided; no cost savings or exact maintenance interval recommendations are made.
