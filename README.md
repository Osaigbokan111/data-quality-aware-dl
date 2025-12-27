Functional Dependency–Aware Learning under Data Quality Noise.
Investigating data quality issues in deep learning through functional dependency noise, feature pruning, and FD-aware regularization.
This repository studies how functional dependency (FD) violations affect machine learning models and how FD-aware training can explicitly control the tradeoff between predictive accuracy and data quality. 
The project moves from observing FD noise effects to enforcing FD constraints during training.
Motivation
More often than not, datasets violate structural constraints like functional dependencies.
Example FD:
Airport Country Code → Country Name
When this kind of constraints is violated:
Models can learn shortcut correlations
Accuracy may remain high while predictions become semantically inconsistent
Downstream systems relying on data integrity may fail
Hence, this project asks:
Can machine learning models be made aware of data quality constraints, and what accuracy–consistency tradeoffs emerge?
Dataset
Airline passenger dataset has information on passenger demographics, airport and country metadata and Flight information
Functional dependencies are discovered and manually validated before experimentation.
Experiments
Experiment 1: FD Noise Sensitivity
Goal:
Measure how FD violations affect model accuracy.
Method:
Inject random FD violations into the training data, train MLP and XGBoost classifier models and evaluate accuracy under increasing FD noise
Finding:
The results show high accuracy even with severe FD violations as moodels exploit shortcut correlations rather than learning robust structure.
Conclusion:
High accuracy does not imply high data quality.
Experiment 2: FD-Based Feature Pruning (Canonical Cover)
Goal:
Evaluate robustness when FD-redundant features are removed.
Method:
Compare full feature set vs FD-pruned feature set by injecting FD noise and measuring accuracy degradation
Finding:
Canonical features reduce leakage and improve stability
Full features achieve higher accuracy but are fragile under noise
Conclusion:
Feature pruning exposes an accuracy–robustness tradeoff.
Experiment 3: FD-Aware Training
Goal:
Explicitly enforce FD consistency during learning.
Method:
Add FD regularization to the loss function:
𝐿𝑡𝑜𝑡𝑎𝑙 = 𝐿𝑡𝑎𝑠𝑘 + 𝜆 * 𝐿𝐹𝐷
Where the FD loss penalizes inconsistent predictions within FD groups.
Finding:
FD-aware models produce consistent predictions and accuracy drops compared to unconstrained baselines
Conclusion:
By enforcing data quality constraints model behaviour can be reshaped.
Experiment 3 (Extended): Accuracy–Consistency Frontier
Goal:
Understand how FD regularization strength controls model behaviour with respect to accuracy and noise level.
Method:
Sweep λ over multiple values and measure accuracy and FD violation rate
Finding:
No single λ optimizes both accuracy and consistency
λ acts as a control knob for data quality
Conclusion:
Data quality becomes an explicit optimization dimension.
Key Insights
Machine learning models are not FD-aware by default
FD violations enable shortcut learning
Canonical features reduce leakage but limit expressiveness
FD-aware regularization enforces semantic consistency
Accuracy and data quality are fundamentally competing objectives
