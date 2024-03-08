import json
import os
from multiprocessing import Pool

import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from tqdm import tqdm

output_folder = "../output"
log_folder = "../logs"

test_names = ["C", "D", "CD", "C-D"]
rounded_version = [None, 1, 0]  # Rounding applied to x decimals
k_values = [5]

grouping_columns = [
    "Sex",
    "Age",
    "Contract",
    "Statute",
    "Jurisdiction",
    "Region"
]

linking_columns_non_aggr = [
    "Duration"
]

linking_columns_aggr = [
    "count",
    "sum",
    "average",
    "std"
]


def constraints_C(model, m, non_aggregate_grouped, p):
    person_idx, person = p

    # Person has count absences
    model.addConstr(gp.quicksum(
        m[person_idx][idx] for idx, _ in enumerate(non_aggregate_grouped.items())) == person.count)

    # Person is a total of sum days absent
    model.addConstr(gp.quicksum(
        duration * m[person_idx][idx] for idx, (duration, count) in enumerate(non_aggregate_grouped.items())) ==
                    person.sum)


def constraints_D(model, m, non_aggregate_grouped, p, rounding, k):
    person_idx, person = p

    p_count = model.addVar(vtype=GRB.INTEGER, name=f'count absences person {person_idx}', lb=1,
                           ub=sum(non_aggregate_grouped)-k)
    p_sum = model.addVar(vtype=GRB.INTEGER, name=f'total days absent person {person_idx}', lb=1)

    # Person has count absences in current solution
    model.addConstr(gp.quicksum(
        m[person_idx][idx] for idx, _ in enumerate(non_aggregate_grouped.items())) == p_count)

    # Person is a total of sum days absent in current solution
    model.addConstr(gp.quicksum(
        duration * m[person_idx][idx] for idx, (duration, _) in enumerate(non_aggregate_grouped.items())) == p_sum)

    if rounding is None:
        model.addConstr(p_count * person.average == p_sum)

        # Person standard deviation must match absences
        model.addConstr(gp.quicksum(
            m[person_idx][idx] * ((duration - person.average) ** 2) for idx, (duration, _) in
            enumerate(non_aggregate_grouped.items())) == (person.std ** 2) * p_count)

    else:
        # Since rounding introduces error on the values, we can no longer work with the static values.
        # Instead, extra variables have to be introduced to properly insert the aggregation function into Gurobi
        # Note: We assumed 0 degrees of freedom for the standard deviation calculation
        tolerance = 0.5 * (10 ** -rounding)

        average = model.addVar(vtype=GRB.CONTINUOUS, name=f'average of person {person_idx}',
                               lb=person.average - tolerance, ub=person.average + tolerance)
        model.addConstr(p_count * average == p_sum)

        std = model.addVar(vtype=GRB.CONTINUOUS, name=f'std of person {person_idx}',
                           lb=person.std - tolerance, ub=person.std + tolerance)
        tmp_diff_duration_avg = [
            model.addVar(vtype=GRB.CONTINUOUS,
                         name=f"Difference between duration {duration} and curr average for person {person_idx}",
                         lb=-max(non_aggregate_grouped.index))
            for idx, (duration, count) in enumerate(non_aggregate_grouped.items())]
        tmp_diff_duration_avg_squared = [
            model.addVar(vtype=GRB.CONTINUOUS,
                         name=f"Squared difference between duration {duration} and curr average for person {person_idx}")
            for idx, (duration, count) in enumerate(non_aggregate_grouped.items())]
        tmp_weighted = [
            model.addVar(vtype=GRB.CONTINUOUS,
                         name=f"Weighted squared differences for person {person_idx}")
            for idx, (duration, count) in enumerate(non_aggregate_grouped.items())]
        helper_a = model.addVar(vtype=GRB.CONTINUOUS, name=f"1/count for person {person_idx}")
        helper_b = model.addVar(vtype=GRB.CONTINUOUS, name=f"intermediate value root for person {person_idx}")

        for idx, (duration, _) in enumerate(non_aggregate_grouped.items()):
            model.addConstr(duration - average == tmp_diff_duration_avg[idx])
            model.addConstr(tmp_diff_duration_avg[idx] * tmp_diff_duration_avg[idx] ==
                            tmp_diff_duration_avg_squared[idx])
            model.addConstr(m[person_idx][idx] * tmp_diff_duration_avg_squared[idx] == tmp_weighted[idx])

        model.addConstr(helper_a * p_count == 1)
        model.addConstr(helper_a * gp.quicksum(tmp_weighted) == helper_b)
        model.addGenConstrPow(helper_b, std, 0.5)


def constraints_CD(model, m, non_aggregate_grouped, p, rounding):
    person_idx, person = p

    # Person has count absences
    model.addConstr(gp.quicksum(
        m[person_idx][idx] for idx, _ in enumerate(non_aggregate_grouped.items())) == person.count)

    # Person is a total of sum days absent
    model.addConstr(gp.quicksum(
        duration * m[person_idx][idx] for idx, (duration, _) in enumerate(non_aggregate_grouped.items())) == person.sum)

    if rounding is None:
        # Person standard deviation must match absences
        model.addConstr(gp.quicksum(
            m[person_idx][idx] * ((duration - person.average) ** 2) for idx, (duration, _) in
            enumerate(non_aggregate_grouped.items())) == (person.std ** 2) * person.count)

    else:
        # Since rounding introduces error on the values, we can no longer work with the static values.
        # Instead, extra variables have to be introduced to properly insert the aggregation function into Gurobi
        # Note: We assumed 0 degrees of freedom for the standard deviation calculation
        tolerance = 0.5 * (10 ** -rounding)

        std = model.addVar(vtype=GRB.CONTINUOUS, name=f'std of person {person_idx}',
                           lb=person.std - tolerance, ub=person.std + tolerance)
        helper_a = model.addVar(vtype=GRB.CONTINUOUS, name=f"1/count for person {person_idx}")
        helper_b = model.addVar(vtype=GRB.CONTINUOUS, name=f"intermediate value root for person {person_idx}")

        model.addConstr(helper_a * person.count == 1)
        model.addConstr(helper_a * gp.quicksum(
            m[person_idx][idx] * ((duration - (person.sum / person.count)) ** 2) for idx, (duration, _) in
            enumerate(non_aggregate_grouped.items())) == helper_b)
        model.addGenConstrPow(helper_b, std, 0.5)


def calculate(d):
    k = d["K"]
    test_name = d["test_name"]
    rounding = d["rounding"]
    aggregate_records = d["aggregate_records"][linking_columns_aggr].copy()
    non_aggregate_records = d["non_aggregate_records"][linking_columns_non_aggr].sort_values(
        by=linking_columns_non_aggr)

    if rounding != None:
        aggregate_records[["average", "std"]] = aggregate_records[["average", "std"]].round(decimals=rounding)

    non_aggregate_grouped = non_aggregate_records.groupby(linking_columns_non_aggr).size()

    with gp.Env(empty=True) as env:
        # env.setParam('OutputFlag', 0)
        env.setParam('LogFile', f"{log_folder}/{k}_{test_name}_{rounding}.log")
        env.start()
        with gp.Model(env=env) as model:
            m = [[model.addVar(vtype=GRB.INTEGER, name=f'how many of {i} has person {person}', lb=0)
                  for i in non_aggregate_grouped.index] for person in range(len(aggregate_records))]

            # Default constraints
            for idx, (duration, count) in enumerate(non_aggregate_grouped.items()):
                model.addConstr(gp.quicksum(m[person_idx][idx] for person_idx in range(len(aggregate_records))) <= count)

            for p in enumerate(aggregate_records.itertuples(index=False)):
                match test_name:
                    case "C":
                        constraints_C(model, m, non_aggregate_grouped, p)
                    case "D":
                        constraints_D(model, m, non_aggregate_grouped, p, rounding, k)
                    case "CD":
                        constraints_CD(model, m, non_aggregate_grouped, p, rounding)
                    case "C-D":
                        person_idx, person = p
                        # Check if individual records can be linked between both aggregates:
                        # Since the average value can be calculated from the counting values, it is possible to check
                        # if a record can be linked to the derived dataset if the average of a person occurs only once
                        # in the aggregate
                        if len(aggregate_records[aggregate_records["average"] == person.average]) == 1:
                            constraints_CD(model, m, non_aggregate_grouped, p, rounding)
                        else:
                            # Note that you could tighten the imposed constraints by taking the possible matching
                            # counting values into consideration
                            constraints_D(model, m, non_aggregate_grouped, p, rounding, k)

            if rounding is not None:
                model.setParam("NonConvex", 2)

            model.setParam("PoolSearchMode", 2)
            model.setParam("PoolSolutions", 2000000000)  # Calculate at most 2B solutions
            model.setParam("LogToConsole", 0)
            model.setParam("SoftMemLimit", 48)  # Limit Gurobi to 48G of memory
            model.setParam("Threads", 8)
            # model.setParam("FeasibilityTol", 1e-6)
            model.setParam("TimeLimit", 60)
            model.optimize()

            n_solutions = model.SolCount
            runtime = model.Runtime

            # Get M^p_* per person
            person_solutions = [{} for _ in range(len(aggregate_records))]
            for solN in range(model.SolCount):
                model.Params.SolutionNumber = solN
                for person in range(len(aggregate_records)):
                    tmp = str([int(round(value.xn)) for value in m[person]])
                    if tmp not in person_solutions[person]:
                        person_solutions[person][tmp] = 0
                    person_solutions[person][tmp] += 1

            d["n_solutions"] = n_solutions
            d["runtime"] = runtime
            d["person_solutions"] = person_solutions
            model.dispose()
    return d


if __name__ == '__main__':
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    eq_to_process = []
    for k in k_values:
        # The code assumes that the same equivalence classes match between both datasets (removes the blocking constraints)
        non_aggregate_path = f"../Examples/A_{k}_non_aggregate.csv"
        aggregate_path = f"../Examples/A_{k}_aggregate.csv"
        df_non_aggregate = pd.read_csv(non_aggregate_path, sep=";").sort_values(by=grouping_columns)
        df_aggregate = pd.read_csv(aggregate_path, sep=";").sort_values(by=grouping_columns)

        for index, ((QID_non_aggr, group_non_aggr), (QID_aggr, group_aggr)), in enumerate(zip(
                df_non_aggregate.groupby(by=grouping_columns, dropna=True),
                df_aggregate.groupby(by=grouping_columns, dropna=True))):
            if QID_non_aggr != QID_aggr:
                print(f"Mismatch between datasets occurred, manual linking step required for eq {QID_non_aggr}")
            else:
                for test_name in test_names:
                    if test_name != "C":
                        for rounding in rounded_version:
                            eq_to_process.append({
                                "K": k,
                                "QID": QID_aggr,
                                "test_name": test_name,
                                "rounding": rounding,
                                "aggregate_records": group_aggr,
                                "non_aggregate_records": group_non_aggr,
                            })
                    else:
                        eq_to_process.append({
                            "K": k,
                            "QID": QID_aggr,
                            "test_name": test_name,
                            "rounding": None,
                            "aggregate_records": group_aggr,
                            "non_aggregate_records": group_non_aggr,
                        })

    with Pool(processes=4, maxtasksperchild=1) as pool:
        for d in tqdm(pool.imap_unordered(calculate, eq_to_process), total=len(eq_to_process),
                      desc="Calculating"):
            with open(f"{output_folder}/population_{d['K']}_{d['test_name']}_{d['rounding']}.txt", 'a') as f:
                f.write(
                    f"""{d['QID']}\n{d["n_solutions"]}\n{d["runtime"]}s\n{json.dumps(d['person_solutions'])}\n""")
