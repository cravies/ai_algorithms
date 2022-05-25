from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the network structure
alarm_model = BayesianNetwork(
    [
        ("Pollution", "Cancer"),
        ("Smoker", "Cancer"),
        ("Cancer", "XRay"),
        ("Cancer", "Dyspnoea"),
    ]
)

# Define the probability tables by TabularCPD
cpd_burglary = TabularCPD(
    variable="Pollution", variable_card=2, values=[[0.10], [0.90]]
)

cpd_earthquake = TabularCPD(
    variable="Smoker", variable_card=2, values=[[0.70], [0.30]]
)

cpd_alarm = TabularCPD(
    variable="Cancer",
    variable_card=2,
    values=[[0.999, 0.97, 0.98, 0.95], [0.001, 0.03, 0.02, 0.05]],
    evidence=["Pollution", "Smoker"],
    evidence_card=[2, 2],
)

cpd_johncall = TabularCPD(
    variable="XRay",
    variable_card=2,
    values=[[0.80, 0.1], [0.20, 0.9]],
    evidence=["Cancer"],
    evidence_card=[2],
)

cpd_marycall = TabularCPD(
    variable="Dyspnoea",
    variable_card=2,
    values=[[0.70, 0.35], [0.30, 0.65]],
    evidence=["Cancer"],
    evidence_card=[2],
)

# Associating the probability tables with the model structure
alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncall, cpd_marycall
)

#calculate probability of pollution based on xray
alarm_infer = VariableElimination(alarm_model)
q = alarm_infer.query(variables=["Pollution"], evidence={"XRay": 1})
print(q)
