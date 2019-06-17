Input takes predictors and neuronal activity

Predictors:
+ Event: 3D boolean array, feature * trial * {time points}
    + feature: audio tone
    + feature: reward
+ Trial: 2D float array, feature * trial
    + feature: push success
    + feature: delay
    + feature: amplitude
    + feature: max speed
+_Temporal (continuous): 3D float array, feature * trial * {time point}
    + feature: trajectory
    + feature: speed
