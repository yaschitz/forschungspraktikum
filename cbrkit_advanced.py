import polars as pd
from cbrkit.sim.taxonomy import weights
from hypothesis.extra.pandas import columns
from sklearn.preprocessing import MinMaxScaler
import cbrkit
# Load the hospital readmissions dataset
data = pd.read_csv('hospital_readmissions.csv')

# Define numerical and categorical columns separately
numerical_cols = [
    "time_in_hospital", "n_procedures", "n_lab_procedures",
    "n_medications", "n_outpatient", "n_inpatient", "n_emergency"
]
categorical_cols = [
    "age", "medical_specialty", "diag_1", "diag_2", "diag_3",
    "glucose_test", "A1Ctest", "change", "diabetes_med"
]

casebase = cbrkit.loaders.polars(data)

query = {
    "age": "[70-80)",
    "time_in_hospital": 5,
    "n_procedures": 1,
    "n_lab_procedures": 50,
    "n_medications": 12,
    "n_outpatient": 0,
    "n_inpatient": 0,
    "n_emergency": 1,
    "medical_specialty": "InternalMedicine",
    "diag_1": "Circulatory",
    "diag_2": "Respiratory",
    "diag_3": "Digestive",
    "glucose_test": "normal",
    "A1Ctest": "normal",
    "change": "no",
    "diabetes_med": "yes"
}

attribute_weights = {
    "age": 0.9769,
    "time_in_hospital": 0.6637,
    "n_procedures":  0.8212,
    "n_lab_procedures": 1.5203,
    "n_medications": 1.3360,
    "n_outpatient": 1.5037,
    "n_inpatient": 1.3035,
    "n_emergency": 0.6157,
    "medical_specialty": 1.5068,
    "diag_1": 0.7647,
    "diag_2": 0.6014,
    "diag_3": 0.5871,
    "glucose_test": 1.2808,
    "A1Ctest": 1.2001,
    "change": 0.5,
    "diabetes_med": 0.5,
}

sim = cbrkit.sim.attribute_value(
    attributes={
        "age": cbrkit.sim.strings.levenshtein(),
        "diag_1": cbrkit.sim.strings.levenshtein(),
        "diag_2": cbrkit.sim.strings.levenshtein(),
        "diag_3": cbrkit.sim.strings.levenshtein(),
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean",pooling_weights=attribute_weights),
)


# Build retriever
retriever = cbrkit.retrieval.build(sim)
retriever1 = cbrkit.retrieval.dropout(retriever, min_similarity=0.9 ,limit=10)
#retriever1 = cbrkit.retrieval.dropout(sim, min_similarity=0.5, limit=20)
#retriever2 = cbrkit.retrieval.dropout(sim, limit=10)

sim_unweighted = cbrkit.sim.attribute_value(
    attributes={
        "age": cbrkit.sim.strings.levenshtein(),
        "diag_1": cbrkit.sim.strings.levenshtein(),
        "diag_2": cbrkit.sim.strings.levenshtein(),
        "diag_3": cbrkit.sim.strings.levenshtein(),
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean")  # No weights
)

retriever_unweighted = cbrkit.retrieval.build(sim_unweighted)
result_unweighted = cbrkit.retrieval.apply_query(casebase, query, retriever_unweighted)




# Retrieve cases similar to the query
result = cbrkit.retrieval.apply_query(casebase, query, retriever1)

# Extract readmission predictions
prediction_weighted = result.casebase["readmitted"].mode()[0]  # Majority vote
prediction_unweighted = result_unweighted.casebase["readmitted"].mode()[0]  # Majority vote

# Print results
print(result)
print(result.ranking)
print("//////")
print(result.similarities)
print("//////")
print(result.casebase)