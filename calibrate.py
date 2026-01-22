import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from neo4j import GraphDatabase
from run_policy_simulation import run_simulation_with_policies
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from scipy.stats import wasserstein_distance
# from some_simulator import run_simulation  # your simulator function
from skopt import gp_minimize  # Bayesian optimization
from skopt.space import Categorical, Integer, Real

# --- Neo4j connection setup ---
URI = "bolt://localhost:7690"  # adjust if using neo4j+s:// for Aura
USER = "neo4j"
PASSWORD = "openreview"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def truncate_right_tail(data, quantile=0.99, max_value=None):
    """
    Truncate values above a quantile or a fixed max_value.
    Returns the truncated array.
    """
    if max_value is None:
        cutoff = np.quantile(data, quantile)
    else:
        cutoff = max_value
    return data[data <= cutoff]


def get_collaborators_per_author_or():
    # --- Query for coauthors ---
    # This query collects coauthors for a given author
    COAUTHOR_QUERY = """
    MATCH (a:Author)<-[:_HAS_AUTHOR]-(p:Paper)-[:_HAS_AUTHOR]->(coauthor:Author),
        (p)-[:_IS_SUBMITTED_TO]->(c:Conference)
    WHERE coauthor <> a
    RETURN a.name AS author, collect(DISTINCT coauthor.name) AS coauthors, COLLECT(DISTINCT c.year) AS years, COUNT(p) as p_count
    """
    coauthor_dict = {}
    with driver.session(database="open-review-data") as session:
        results = session.run(COAUTHOR_QUERY)
        for record in results:
            author = record["author"]
            coauthor_dict[author] = record
    return coauthor_dict


def mean_yoy_growth_rate_or():
    QUERY = """
        MATCH (p:Paper)-[:_HAS_AUTHOR]->(a:Author),
            (p)-[:_IS_SUBMITTED_TO]->(c:Conference {name: "ICLR"})
        RETURN c.year AS year, count(DISTINCT a) AS numAuthors
        ORDER BY year ASC
    """
    with driver.session(database="open-review-data") as session:
        results = session.run(QUERY)
        data = [(int(record["year"]), record["numAuthors"]) for record in results]

    # Ensure sorted by year
    data.sort(key=lambda x: x[0])

    # Compute YoY growth rates
    growth_rates = []
    for i in range(1, len(data)):
        prev_year, prev_val = data[i - 1]
        curr_year, curr_val = data[i]
        if prev_val > 0:
            growth = (curr_val - prev_val) / prev_val
            growth_rates.append(growth)

    if growth_rates:
        return np.mean(growth_rates)
    else:
        return None


def get_review_scores_or(
    confs=[
        "ICLR.cc_2018",
        "ICLR.cc_2020",
        "ICLR.cc_2021",
        "ICLR.cc_2022",
        "ICLR.cc_2023",
    ]
):
    score_query = """
        MATCH (p:Paper)-[:_IS_SUBMITTED_TO]->(c:Conference {{id: "{0}"}})
        MATCH (p)-[:_HAS_REVIEW]->(r:Review)
        WITH p, avg(toFloat(r.score)) AS score
        RETURN score
    """
    scores = []
    with driver.session(database="open-review-data") as session:
        for conf in confs:
            results = session.run(score_query.format(conf))
            for record in results:
                if record["score"] is not None:
                    scores.append(record["score"])
    return scores


def get_acceptance_rates_or(
    confs=[
        "ICLR.cc_2018",
        "ICLR.cc_2020",
        "ICLR.cc_2021",
        "ICLR.cc_2022",
        "ICLR.cc_2023",
    ]
):
    score_query = """
        MATCH (p:Paper)-[:_IS_SUBMITTED_TO]->(c:Conference {{id: "{0}"}})
        WITH p.accepted AS accepted
        RETURN accepted
    """
    scores = []
    with driver.session(database="open-review-data") as session:
        for conf in confs:
            results = session.run(score_query.format(conf))
            for record in results:
                if record["accepted"] is not None:
                    scores.append(record["accepted"])
    return scores


def get_authors_per_paper_openalex(papers):
    counts = []
    for paper in papers:
        authorships = paper.get("authorships", [])
        counts.append(len(authorships))
    return counts


def get_papers_per_author_openalex(authors):
    totals = []
    for author in authors:
        total_works = sum(
            entry.get("works_count", 0) for entry in author.get("counts_by_year", [])
        )
        totals.append(total_works)
    return totals


def get_author_lifespans_openalex(authors):
    lifespans = []
    for author in authors:
        years = [int(entry.get("year")) for entry in author.get("counts_by_year", [])]
        if years:  # make sure author has data
            lifespan = max(years) - min(years) + 1
        else:
            lifespan = 0
        lifespans.append(lifespan)
    return lifespans


def get_author_lifespans():
    pass


def get_papers_per_author():
    pass


def get_authors_per_paper():
    pass


def build_stats(projects):
    contributor_times = defaultdict(list)
    contributor_papers = defaultdict(int)

    authors_per_paper = []
    quality_scores = []
    acceptances = []

    # collect info
    for proj in projects:
        start_time = proj.get("start_time")
        if start_time < 100:
            continue
        contributors = proj.get("contributors", [])
        authors_per_paper.append(len(contributors))
        quality_scores.append(proj.get("quality_score"))
        acceptances.append((1 if proj.get("final_reward") > 0 else 0))
        for c in contributors:
            contributor_times[c].append(start_time)
            contributor_papers[c] += 1

    # compute ages
    author_lifespan = []
    papers_per_author = []
    for c, times in contributor_times.items():
        age = max(times) - min(times) if len(times) > 1 else 0
        author_lifespan.append(age)
        papers_per_author.append(contributor_papers[c])

    return {
        "papers_per_author": np.array(papers_per_author),
        "authors_per_paper": np.array(authors_per_paper),
        "lifespan": np.array(author_lifespan) / 52,
        "quality": np.array(quality_scores) * 10,
        "acceptance": np.array(acceptances),
    }


def save_real_world_data():
    with open("../data/target_corpus_meta_info.json", "r") as f:
        papers = json.load(f)
    papers = np.random.choice(list(papers.values()), 10_000, replace=False)
    with open("../data/openalex_authors_sample.json", "r") as f:
        authors = json.load(f)

    print(f"Sampled {len(papers)} papers")
    print(f"Loaded {len(authors)} authors")
    np.save("author_lifespan.npy", np.array(get_author_lifespans_openalex(authors)))
    np.save("papers_per_author.npy", np.array(get_papers_per_author_openalex(authors)))
    np.save("authors_per_paper.npy", np.array(get_authors_per_paper_openalex(papers)))
    np.save("acceptance.npy", np.array(get_review_scores_or()))
    np.save("quality.npy", np.array(get_acceptance_rates_or()))


def generate_proportions(step=0.1):
    proportions = []
    steps = int(1 / step) + 1
    for i in range(steps):
        for j in range(steps - i):
            k = steps - i - j - 1
            p1, p2, p3 = i * step, j * step, k * step
            proportions.append((round(p1, 5), round(p2, 5), round(p3, 5)))
    return proportions


def sensitivity_analysis(problem):
    # --- Step 2: Sample parameter combinations ---
    param_values = sobol_sample.sample(problem, 64, calc_second_order=False)

    # --- Step 3: Run simulation and collect outputs ---
    def run_model(params):
        acceptance, novelty, prestige, effort, rewardless, group_align = params
        try:
            sim_run = run_simulation_with_policies(
                n_agents=400,
                start_agents=100,
                max_steps=400,
                n_groups=10,
                max_peer_group_size=100,
                max_rewardless_steps=rewardless,
                policy_distribution={
                    "careerist": 1 / 3,
                    "orthodox_scientist": 1 / 3,
                    "mass_producer": 1 / 3,
                },
                output_file_prefix="sensitivity",
                group_policy_homogenous=group_align,
                acceptance_threshold=acceptance,
                novelty_threshold=novelty,
                prestige_threshold=prestige,
                effort_threshold=effort,
            )
        except Exception as e:
            print(e)
            return [np.nan] * 5
        with open("log/sensitivity_projects.json", "r") as f:
            run_projects = json.load(f)
        sim_data = build_stats(run_projects)

        # Outputs for sensitivity
        return [
            float(np.nanmean(sim_data["papers_per_author"])),
            float(np.nanmean(sim_data["authors_per_paper"])),
            float(np.nanmean(sim_data["lifespan"])),
            float(np.nanmean(sim_data["quality"])),
            float(np.nanmean(sim_data["acceptance"])),
        ]

    Y = []
    for i, p in enumerate(param_values):
        print(f"Sensitivity Analysis run {i+1}/{len(param_values)}")
        outputs = run_model(p)
        Y.append(outputs)

    Y = np.array(Y)
    for i in range(Y.shape[0]):
        if np.isnan(Y[i]).any():
            Y[i] = np.nanmean(Y, axis=0)
    # --- Step 4: Sobol sensitivity analysis + Save results ---
    output_names = [
        "papers_per_author",
        "authors_per_paper",
        "lifespan",
        "quality",
        "acceptance",
    ]

    for i, output_name in enumerate(output_names):
        Si = sobol_analyze.analyze(problem, Y[:, i], calc_second_order=False)
        results = {
            "S1": dict(zip(problem["names"], Si["S1"].tolist())),
            "ST": dict(zip(problem["names"], Si["ST"].tolist())),
        }
        out_file = f"sensitivity_{output_name}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved sensitivity results for {output_name} → {out_file}")


def calibrate(problem, real_data):
    # Bounds: adjust for your parameter ranges
    # np.random.seed(42)
    # grid_props = generate_proportions(step=0.2)
    # rand_props = np.random.dirichlet([1, 1, 1], size=20).tolist()
    # candidates = [tuple(prop) for prop in rand_props + grid_props]
    # np.random.shuffle(candidates)
    # print(candidates)
    names = problem["names"]
    bounds = problem["bounds"]
    if len(names) == 6:
        param_space = [
            Real(*bounds[0], name=names[0]),
            Real(*bounds[1], name=names[1]),
            Real(*bounds[2], name=names[2]),
            Integer(*bounds[3], name=names[3]),
            Integer(*bounds[4], name=names[4]),
            Categorical(bounds[5], name=names[5]),  # Boolean
            # Categorical(candidates, name="policy_population_proportions"),
        ]
    else:
        param_space = [
            Real(*bounds[0], name=names[0]),
            Integer(*bounds[1], name=names[1]),
            Integer(*bounds[2], name=names[2]),
        ]

    # ---- Step 2–3: Define loss function ----
    def loss(theta):
        print(list(zip(names, theta)))
        try:
            if len(names) == 6:
                sim_run = run_simulation_with_policies(
                    n_agents=2_000,
                    # n_agents=600,
                    start_agents=200,
                    # start_agents=60,
                    max_steps=600,
                    # max_steps=120,
                    n_groups=20,
                    # n_groups=6,
                    max_peer_group_size=300,
                    # max_rewardless_steps=theta[2],
                    max_rewardless_steps=theta[names.index("max_rewardless_steps")],
                    policy_distribution={
                        "careerist": 1 / 3,  # theta[4][0],
                        "orthodox_scientist": 1 / 3,  # theta[4][1],
                        "mass_producer": 1 / 3,  # theta[4][2],
                    },
                    output_file_prefix="calibration",
                    # group_policy_homogenous = 0,
                    group_policy_homogenous=bool(
                        theta[names.index("policy_aligned_in_group")]
                    ),
                    # acceptance_threshold=theta[0],
                    acceptance_threshold=theta[names.index("acceptance_threshold")],
                    # novelty_threshold = 0.4,
                    novelty_threshold=theta[names.index("orthodox_novelty_threshold")],
                    # prestige_threshold = 0.29,
                    prestige_threshold=theta[
                        names.index("careerist_prestige_threshold")
                    ],
                    # effort_threshold=theta[1],
                    effort_threshold=theta[
                        names.index("mass_producer_effort_threshold")
                    ],
                )
            else:
                sim_run = run_simulation_with_policies(
                    n_agents=2_000,
                    # n_agents=600,
                    start_agents=200,
                    # start_agents=60,
                    max_steps=600,
                    # max_steps=120,
                    n_groups=20,
                    # n_groups=6,
                    max_peer_group_size=300,
                    max_rewardless_steps=theta[2],
                    # max_rewardless_steps=theta[names.index("max_rewardless_steps")],
                    policy_distribution={
                        "careerist": 1 / 3,  # theta[4][0],
                        "orthodox_scientist": 1 / 3,  # theta[4][1],
                        "mass_producer": 1 / 3,  # theta[4][2],
                    },
                    output_file_prefix="calibration",
                    group_policy_homogenous=0,
                    # group_policy_homogenous = bool(
                    #     theta[names.index("policy_aligned_in_group")]
                    # ),
                    acceptance_threshold=theta[0],
                    # acceptance_threshold=theta[names.index("acceptance_threshold")],
                    novelty_threshold=0.4,
                    # novelty_threshold = theta[names.index("orthodox_novelty_threshold")],
                    prestige_threshold=0.4,
                    # prestige_threshold = theta[names.index("careerist_prestige_threshold")],
                    effort_threshold=theta[1],
                    # effort_threshold=theta[names.index("mass_producer_effort_threshold")],
                )
        except Exception as e:
            print(e)
            return 1e6

        with open("log/calibration_projects.json", "r") as f:
            run_projects = json.load(f)
        sim_data = build_stats(run_projects)
        n_bins_ppa = min(
            max(sim_data["papers_per_author"]), max(real_data["papers_per_author"])
        )  # 200
        n_bins_ppa = 200 if n_bins_ppa < 200 else n_bins_ppa
        n_bins_app = min(
            max(sim_data["authors_per_paper"]), max(real_data["authors_per_paper"])
        )  #
        n_bins_app = 5 if n_bins_app < 5 else n_bins_app
        n_bins_ls = min(int(max(sim_data["lifespan"])), max(real_data["lifespan"]))  #
        n_bins_ls = 5 if n_bins_ls < 5 else n_bins_ls
        n_bins_q = min(int(max(sim_data["quality"])), max(real_data["quality"]))  #
        n_bins_q = 10 if n_bins_q < 10 else n_bins_q
        # Extract histograms (same bins as real)
        H_sim1 = np.histogram(sim_data["papers_per_author"], bins=n_bins_ppa)[0]
        H_sim2 = np.histogram(sim_data["authors_per_paper"], bins=n_bins_app)[0]
        H_sim3 = np.histogram(sim_data["lifespan"], bins=n_bins_ls)[0]
        H_sim4 = np.histogram(sim_data["quality"], bins=n_bins_q)[0]
        # Normalize
        H_sim1 = H_sim1 / H_sim1.sum()
        H_sim2 = H_sim2 / H_sim2.sum()
        H_sim3 = H_sim3 / H_sim3.sum()
        H_sim4 = H_sim4 / H_sim4.sum()
        sim_acceptance_rate = np.array(sim_data["acceptance"]).mean()
        # Normalize real data histograms
        H_real_papers_per_author = np.histogram(
            truncate_right_tail(real_data["papers_per_author"], max_value=n_bins_ppa),
            bins=n_bins_ppa,
        )[0]
        H_real_authors_per_paper = np.histogram(
            truncate_right_tail(
                real_data["authors_per_paper"][real_data["authors_per_paper"] > 0],
                max_value=n_bins_app,
            ),
            bins=n_bins_app,
        )[0]
        H_real_lifespan = np.histogram(real_data["lifespan"], bins=n_bins_ls)[0]
        H_real_quality = np.histogram(real_data["quality"], bins=n_bins_q)[0]
        real_acceptance_rate = real_data["acceptance"].mean()
        H_real_papers_per_author = (
            H_real_papers_per_author / H_real_papers_per_author.sum()
        )
        H_real_authors_per_paper = (
            H_real_authors_per_paper / H_real_authors_per_paper.sum()
        )
        H_real_lifespan = H_real_lifespan / H_real_lifespan.sum()
        H_real_quality = H_real_quality / H_real_quality.sum()
        # Distances
        d1 = wasserstein_distance(H_real_papers_per_author, H_sim1)
        d2 = wasserstein_distance(H_real_authors_per_paper, H_sim2)
        d3 = wasserstein_distance(H_real_lifespan, H_sim3)
        d4 = wasserstein_distance(H_real_quality, H_sim4)
        d5 = np.abs(real_acceptance_rate - sim_acceptance_rate)
        print(
            (
                f"PPA dist: {round(d1, 5)}, ",
                f"APP dist: {round(d2, 5)}, ",
                f"LS dist: {round(d3, 5)}, ",
                f"PQ dist: {round(d4, 5)}, ",
                f"AR dist {round(d5, 5)}",
            )
        )
        return d1 + d2 + d3 + d4 + (d5 * 0.1)  # weighted sum possible

    res = gp_minimize(loss, param_space, n_calls=50, random_state=42)
    # res = gp_minimize(loss, param_space, n_calls=10, random_state=42)
    print("Best parameters:", list(zip(names, res.x)))


# --- Build normalized histograms with edges ---
def build_normalized_hist(data, bins):
    counts, edges = np.histogram(data, bins=bins)
    counts = counts / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])  # bin centers
    return counts, centers


def plot_calibration_overlays(
    real_data, sim_data, real_centers, sim_centers, outfile=None
):
    metrics = [
        ("papers_per_author", "Distribution of Papers per Author", "Papers per author"),
        ("authors_per_paper", "Authors per Paper", "Authors per paper"),
        ("lifespan", "Career Length", "Years active"),
        ("quality", "Paper Quality", "Quality score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (key, title, xlabel) in zip(axes.flatten(), metrics):
        ax.plot(real_centers[key], real_data[key], label="Real", lw=2)
        ax.plot(
            sim_centers[key], sim_data[key], label="Simulation", lw=2, linestyle="--"
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.legend()

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    plt.show()


def main():
    sweep_1_problem = {
        "num_vars": 6,
        "names": [
            "acceptance_threshold",
            "orthodox_novelty_threshold",
            "careerist_prestige_threshold",
            "mass_producer_effort_threshold",
            "max_rewardless_steps",
            "policy_aligned_in_group",
        ],
        "bounds": [
            [0.2, 0.8],  # Real
            [0.2, 0.8],  # Real
            [0.2, 0.8],  # Real
            [10, 50],  # Integer (approx. continuous for SA)
            [50, 500],  # Integer
            [0, 1],  # Boolean → treat as 0/1
        ],
    }
    sweep_2_problem = {
        "num_vars": 3,
        "names": [
            "acceptance_threshold",
            "mass_producer_effort_threshold",
            "max_rewardless_steps",
        ],
        "bounds": [
            [0.2, 0.8],  # Real
            [10, 50],  # Integer (approx. continuous for SA)
            [50, 500],  # Integer
        ],
    }
    # sensitivity_analysis(problem)
    real_data = {
        "papers_per_author": np.load("papers_per_author.npy"),
        "authors_per_paper": np.load("authors_per_paper.npy"),
        "lifespan": np.load("author_lifespan.npy"),
        "quality": np.load("quality_histogram.npy"),
        "acceptance": np.load("acceptance_histogram.npy"),
    }

    calibrate(sweep_2_problem, real_data)


if __name__ == "__main__":
    main()
