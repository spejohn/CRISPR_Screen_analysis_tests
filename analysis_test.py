import argparse
from multiprocessing import Value
import os
import re
import sys
import subprocess
from pathlib import Path, PurePosixPath
from typing import Optional, Union
import logging

import docker
from gooey import Gooey, GooeyParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from plotly import graph_objects as go
from scipy.stats import pearsonr
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from profilehooks import profile

DOCKER_CLIENT = docker.from_env()

# Configure logging
logging.basicConfig(filename= r'C:\Users\spejo\Documents\2_CRISPR_analysis_test_output\validation_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def generate_sample_sheet(reads_dir):
    # Make a mapping of dirname:fastq for each directory in the reads directory
    reads_dir = Path(reads_dir)
    samples = list()

    for file in reads_dir.iterdir():
        if file.suffix == ("_cnttbl"):
            continue
        elif file.endswith('.fastq'):
            samples.append(
                [
                    file.name,
                    str(reads_dir / file)
                ]
            )

    sample_sheet = pd.DataFrame(samples, columns=["sample_name", "fastq_path"])
    sample_sheet = sample_sheet.sort_values("sample_name").reset_index(drop=True)
    return sample_sheet


def mageck_count(
    sample_sheet: pd.DataFrame,
    library_file: str,
    output_dir: str,
):
    """
    Run MAGeCK count to generate read counts from FASTQ files.

    Args:
        rc_table (str): Path to sgRNA library CSV. Expected columns are [sgRNA, sequence, gene].
        sample_sheet (pd.DataFrame): DataFrame with columns [sample, fastq1, fastq2],
            where fastq1 and fastq2 are absolute paths to the relevant R1/R2 read files for each sample.
        output_dir (str): Path to directory for storing output files.

    Returns:
        None

    Raises:
        ChildProcessError: If the Docker container running MAGeCK count exits with a non-zero status code.
    """
    image = "samburger/mageck"
    volumes = [
        # TODO: edit library volume to point towards data folder with ess_genes - should deposit sgrna library files there
        f"{str(Path(os.getcwd()).absolute())}:/work",
        f"{str(Path(library_file).parent.absolute())}:/library",
        f"{str(Path(output_dir).absolute())}:/output",
    ]
    #TODO: make conditional statements for library names in fastq file dirs
    #if 'tkov3' in folder_name:
    count_command = [
        "mageck",
        "count",
        "-l",
        "/library/" + Path(library_file).name,
        "-n",
        Path(sample_sheet).name,
        "--trim-5",
        "0",
        "--sample-label",
        ]

    # Isn't it redundant to have the sample sheet spit out as a list, convert to df, then convert back to list?
    count_command.append(",".join(sample_sheet["sample_name"].tolist()))
    count_command.append("--fastq")
    count_command.append(" /work/".join(sample_sheet["fastq_path"].tolist()))
    count_command_line = " ".join(count_command)
    container_workdir = PurePosixPath("/output")
    #TODO: decide where I want to copy count file to output
    copy_command_line = (
        f"mkdir -p {container_workdir} && cp {Path(sample_sheet).name}* {container_workdir}"
    )
    # Run the command in the container
    shell_cmd = f"docker run -it"
    for vol in volumes:
        shell_cmd += f' -v "{vol}"'
    shell_cmd += (
        f' samburger/mageck "{count_command_line + " && " + copy_command_line}"'
    )
    print(shell_cmd)
    container = DOCKER_CLIENT.containers.run(
        image,
        f'"{count_command_line + " && " + copy_command_line}"',
        volumes=volumes,
        remove=False,
        detach=True,  # Run in the background
        stdout=True,
        stderr=True,
    )
    # Stream the logs
    for line in container.logs(stream=True, stdout=True, stderr=True, follow=True):
        print(line.decode("utf-8").strip())
    container_status = container.wait()
    if not container_status["StatusCode"] == 0:
        print(
            f"Error: Container exited with status code {container_status['StatusCode']}"
        )
    count_df = pd.read_csv(Path(sample_sheet).glob("*.count.txt"), sep="\t+")
    return count_df

def mageck_test(
    contrasts: str, 
    input_file: str,
    output_dir: str, 
):
    contrasts_df = pd.read_csv(Path(contrasts), sep="\t+")
    image = "samburger/mageck"
    volumes = [
        f"{str(Path(os.getcwd()).absolute())}:/work",
        f"{str(Path(input_file).absolute())}:/input",
        f"{str(Path(output_dir).absolute())}:/output"
    ]

    container_indir = PurePosixPath("/input")
    container_outdir = PurePosixPath("/output")
    test_command = [
        "mageck",
        "test",
        "--count-table",
        container_indir,
        "--norm-method",
        "median",
        "--adjust-method",
        "fdr",
    ]
    for line in contrasts_df.itertuples():
        # Make an extension of test_command for each contrast
        # Using the treatment and control columns e.g.:
        # -t SampleA,SampleB -c SampleD,SampleE
        contrast_extension = f"-t {line.treatment} -c {line.control}"
        full_command = test_command + [
            "--output-prefix",
            f"{line.contrast}",
            contrast_extension,
        ]
        # Add a copying of output
        copy_command = f"&& cp {line.contrast}* {container_outdir}"
        full_command.append(copy_command)
        # Run the full command for the current contrast
        shell_cmd = f"docker run -it"
        for vol in volumes:
            shell_cmd += f' -v "{vol}"'
        shell_cmd += f' samburger/mageck "{" ".join(map(str, full_command))}"'
        print(shell_cmd)
        container = DOCKER_CLIENT.containers.run(
            image,
            f'"{" ".join(map(str, full_command))}"',
            volumes=volumes,
            remove=False,
            detach=True,  # Run in the background
        )

        # Create new .csv for readability/downstream QA_QC from .txt output
        output_path = Path(output_dir)
        for txt_file in output_path.glob("*.gene_summary.txt"):

            csv_file = str(txt_file).replace(".gene_summary.txt", "_gMGK")
    
            df = pd.read_csv(txt_file, sep="\t")
            df.to_csv(Path(csv_file).with_suffix(".csv"), index=False)

            print(f"Converted {txt_file} to {csv_file}")

        # Stream the logs
        for line in container.logs(stream=True, follow=True):
            print(line.decode("utf-8").strip())
        container_status = container.wait()
        if not container_status["StatusCode"] == 0:
            raise ChildProcessError(
                f"Error: Container exited with status code {container_status['StatusCode']}"
            )
    return


def run_drugz(contrasts: str, 
              input_file: str,
              output_dir: Union[str, Path], 
              nontarget_gene_lib: Optional[str] = None, 
              remove_nontarget_genes: Optional[bool] = None):
    
    # Ensure drugz.py is available
    drugz_path = Path(os.getcwd()) / "drugz" / "drugz.py"
    if not drugz_path.exists():
        raise ModuleNotFoundError(
            "Drugz not found.  Ensure you have run `git submodule update --init --recursive`."
        )
    contrasts_df = pd.read_csv(Path(contrasts), sep="\t+")

    read_file = Path(input_file)
    if not read_file.exists():
        raise FileNotFoundError(f"Count file {read_file.absolute()} not found.")

    # Generate a drugz command for each contrast in the contrasts file
    commands = []

    for c in contrasts_df.itertuples():
        drugz_command = [
            "python",
            str(drugz_path.absolute()),
            "-i",
            str(read_file.absolute()),
            "-o",
            str(
                (
                    Path(output_dir) / f"{c.contrast}_DZ_output.tsv"
                ).absolute()
            ),
            "-c",
            c.control,
            "-x",
            c.treatment,
            "-unpaired"
        ]
        commands.append(drugz_command)
    
    # Run each drugz command
    for c in commands:
        subprocess.run(c, check=True)

        # Create new .csv for readability/downstream QA_QC from .txt output
        output_path = Path(output_dir)
        for txt_file in output_path.glob(f"*_DZ_output.tsv"):

            csv_file = str(txt_file).replace("_DZ_output.tsv", "_gDZ")
    
            df = pd.read_csv(txt_file, sep="\t")
            df.to_csv(Path(csv_file).with_suffix(".csv"), index=False)

            print(f"Converted {txt_file} to {csv_file}")
    return


def plot_QA(
    screen_title: str,
    ess_genes: dict[str, list[str]],
    noness_genes: Optional[dict[str, list[str]]] = None,
):
    # TODO: add guide- and gene-level count correlation between replicates
    # TODO: make figures dir if does not exist
    # Load mageck sgRNA counts
    sgrna_counts = pd.read_csv(
        Path("data") / "mageck" / screen_title / f"{screen_title}.count.txt", sep="\t"
    )
    plot_QA_sgRNA_corr(screen_title, sgrna_counts)

    # Load mageck gene LFCs
    dfs = []
    for filename in Path(f"data/mageck/{screen_title}").iterdir():
        if "gene_summary" in filename.name:
            contrast = filename.name[
                filename.name.find("_") + 1 : filename.name.find(".gene_summary")
            ]
            df = pd.read_csv(filename, sep="\t")
            df["contrast"] = contrast
            dfs.append(df)
    mageck_results = pd.concat(dfs, ignore_index=True)
    # drop nontargeting guides
    mageck_results = mageck_results[mageck_results["id"] != "Non_Targeting_Human_CRko"]
    for genelist in ess_genes:
        if genelist not in noness_genes:
            # Label all genes not in essential list as non-essential.
            mageck_results["essential"] = mageck_results["id"].apply(
                lambda x: 1 if x in ess_genes[genelist] else 0
            )
            plot_QA_ROC(screen_title, mageck_results, genelist)
            plot_QA_PRC(screen_title, mageck_results, genelist)
        else:
            # Limit to only genes in essential/nonessential lists
            mageck_results = mageck_results[
                (mageck_results["id"].isin(ess_genes[genelist]))
                | (mageck_results["id"].isin(noness_genes[genelist]))
            ]
            mageck_results["essential"] = mageck_results["id"].apply(
                lambda x: 1 if x in ess_genes[genelist] else 0
            )
            plot_QA_ROC(screen_title, mageck_results, genelist)
            plot_QA_PRC(screen_title, mageck_results, genelist)


def plot_QA_sgRNA_corr(screen_title: str, counts: pd.DataFrame):
    print("Generating sgRNA replicate correlation plots...")
    # Create a dictionary to hold the replicates
    replicate_dict = {}
    # Iterate over the columns and populate the dictionary with replicates
    for column in counts.columns:
        if "Rep" in column:
            # Extract the sample name and replicate number
            match = re.findall("Sample[_-](.*)[_-]Rep", column)
            if match:
                sample_name = match[0]
                replicate_number = re.findall(r"Rep[_-]?(\d+)", column)[0]
                # Add the replicate to the dictionary
                replicate_dict.setdefault(sample_name, []).append(
                    (replicate_number, column)
                )
    # Sort the replicates and pair them up
    replicates = []
    for sample, reps in replicate_dict.items():
        # Sort by replicate number
        sorted_reps = sorted(reps, key=lambda x: int(x[0]))
        # Pair up the replicates, assuming they are in order
        for i in range(0, len(sorted_reps) - 1, 2):
            replicates.append((sorted_reps[i][1], sorted_reps[i + 1][1]))
    for rep1, rep2 in replicates:
        plt.figure()
        g = sns.jointplot(
            x=rep1,
            y=rep2,
            data=counts,
            kind="reg",
            joint_kws={"line_kws": {"color": "black", "linewidth": 1}},
        )

        rep_guide_pearsonr = pearsonr(counts[rep1], counts[rep2]).statistic
        sample_title = re.findall("Sample[_-](.*)[_-]Rep", rep1)[0]
        plt.suptitle(
            f"{sample_title}: " + r"$\rho=$" + f"{rep_guide_pearsonr:.3f}", y=1.05
        )
        plt.xlabel("Rep 1 guide counts")
        plt.ylabel("Rep 2 guide counts")
        plt.savefig(
            Path("figures")
            / screen_title
            / f"QA_sgRNA_count_corr_{screen_title}_{sample_title}.png"
        )
        print(f'"{screen_title}","{sample_title}",{rep_guide_pearsonr},"pearson_r"')


def plot_QA_ROC(
    screen_title: str,
    mageck_results: pd.DataFrame,
    genelist: str,
):
    print("Generating ROC plots...")
    plt.figure()
    for contrast in mageck_results["contrast"].unique():
        subset = mageck_results[mageck_results["contrast"] == contrast].sort_values(
            "neg|lfc", ascending=False
        )
        fpr, tpr, _ = roc_curve(subset["essential"], -subset["neg|lfc"])
        roc_auc = auc(fpr, tpr)
        print(f'"{screen_title}","{contrast}",{roc_auc},"ROC_AUC"')
        plt.plot(fpr, tpr, lw=2, label=f"{contrast} (AUC = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {screen_title}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(
        Path("figures") / screen_title / f"QA_ROC_{screen_title}_{genelist}.png",
        bbox_inches="tight",
    )
    # Make plotly chart for debug with separate lines for each contrast
    fig = go.Figure()
    for contrast in mageck_results["contrast"].unique():
        subset = mageck_results[mageck_results["contrast"] == contrast].sort_values(
            "neg|lfc", ascending=False
        )
        fpr, tpr, threshold = roc_curve(subset["essential"], -subset["neg|lfc"])
        roc_auc = auc(fpr, tpr)
        roc_data = pd.DataFrame(
            {
                "threshold": threshold,
                "fpr": fpr,
                "tpr": tpr,
            }
        )
        # count ess genes at each threshold
        roc_data["n_genes_threshold"] = roc_data["threshold"].apply(
            lambda x: (-subset["neg|lfc"] > x).sum()
        )
        fig.add_trace(
            go.Scatter(
                x=roc_data.fpr,
                y=roc_data.tpr,
                mode="lines",
                name=f"{contrast} (AUC = {roc_auc:0.2f})",
                customdata=roc_data[["threshold", "n_genes_threshold"]],
                hovertemplate="<br>".join(
                    [
                        "LFC Threshold: %{customdata[0]}",
                        "TPR: %{y}",
                        "FPR: %{x}",
                        "# Genes > Threshold: %{customdata[1]}",
                    ]
                ),
            )
        )

    fig.update_layout(
        title=f"ROC Curve - {screen_title}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Contrasts",
    )
    fig.write_html(
        Path("figures") / screen_title / f"QA_ROC_{screen_title}_{genelist}.html"
    )


def plot_QA_PRC(screen_title: str, mageck_results: pd.DataFrame, genelist: str):
    print("Generating Precision-Recall plots...")
    plt.figure()
    for contrast in mageck_results["contrast"].unique():
        subset = mageck_results[mageck_results["contrast"] == contrast].sort_values(
            "neg|lfc", ascending=False
        )
        precision, recall, _ = precision_recall_curve(
            subset["essential"], -subset["neg|lfc"]
        )
        pr_auc = auc(recall, precision)
        print(f'"{screen_title}","{contrast}",{pr_auc},"PR_AUC"')
        plt.plot(recall, precision, lw=2, label=f"{contrast} (AUC = {pr_auc:0.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"P/R - {screen_title}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(
        Path("figures") / screen_title / f"QA_PRC_{screen_title}_{genelist}.png",
        bbox_inches="tight",
    )
    # Make plotly chart for debug with separate lines for each contrast
    fig = go.Figure()
    for contrast in mageck_results["contrast"].unique():
        subset = mageck_results[mageck_results["contrast"] == contrast].sort_values(
            "neg|lfc", ascending=False
        )
        precision, recall, threshold = precision_recall_curve(
            subset["essential"], -subset["neg|lfc"]
        )
        pr_auc = auc(recall, precision)
        pr_data = pd.DataFrame(
            {
                "threshold": threshold,
                "precision": precision[:-1],
                "recall": recall[:-1],
            }
        )
        # count ess genes at each threshold
        pr_data["n_genes_threshold"] = pr_data["threshold"].apply(
            lambda x: (-subset["neg|lfc"] > x).sum()
        )
        fig.add_trace(
            go.Scatter(
                x=pr_data.recall,
                y=pr_data.precision,
                mode="lines",
                name=f"{contrast} (AUC = {pr_auc:0.2f})",
                customdata=pr_data[["threshold", "n_genes_threshold"]],
                hovertemplate="<br>".join(
                    [
                        "LFC Threshold: %{customdata[0]}",
                        "Precision: %{y}",
                        "Recall: %{x}",
                        "# Genes > Threshold: %{customdata[1]}",
                    ]
                ),
            )
        )

    fig.update_layout(
        title=f"Precision-Recall Curve - {screen_title}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend_title="Contrasts",
    )
    fig.write_html(
        Path("figures") / screen_title / f"QA_PRC_{screen_title}_{genelist}.html"
    )


def plot_hits_mageck():
    # TODO: volcano plots and essentiality rank plots for mageck outputs
    pass


def plot_hits_drugz(screen_title: str, contrasts: pd.DataFrame, output_dir: str):
    # Load each drugz output file and combine to a single dataframe
    dfs = []
    for c in contrasts.itertuples():
        df = pd.read_csv(
            Path(output_dir)
            / "drugz"
            / screen_title
            / f"{screen_title}_{c.contrast}.tsv",
            sep="\t",
        )
        df["Contrast"] = c.contrast
        dfs.append(df)
    drugz_results = pd.concat(dfs, ignore_index=True)
    # drop nontargeting
    drugz_results = drugz_results[drugz_results["GENE"] != "Non_Targeting_Human_CRko"]
    drugz_results = drugz_results[
        ["Contrast", "GENE"] + list(drugz_results.columns[1:-1])
    ]
    drugz_results = drugz_results.sort_values(
        by=["Contrast", "normZ"], ascending=[True, True]
    )
    drugz_results = drugz_results.reset_index(drop=True)
    drugz_results.to_csv(
        Path(output_dir) / "drugz" / screen_title / f"drugz_results_{screen_title}.csv",
        index=False,
    )
    # Generate figures
    for contrast in drugz_results["Contrast"].unique():
        df = drugz_results.loc[drugz_results["Contrast"] == contrast]
        fig = px.scatter(
            df,
            x="rank_synth",
            y="normZ",
            hover_name="GENE",
            labels={"rank_synth": "Gene Rank"},
            title=contrast,
        )
        fig.write_html(
            Path(output_dir)
            / "drugz"
            / screen_title
            / f"{screen_title}_{contrast}_drugz_plot.html"
        )
    return

def validate_input_tables(
    contrasts: str,
    sample_sheet: Optional[pd.DataFrame] = None,
    counts: Optional[str] = None,
) -> bool:
    
    contrasts_df = pd.read_csv(Path(contrasts), sep='\t')
    try:
        assert set(contrasts_df.columns) == {"contrast", "treatment", "control"}
    except AssertionError:
        error_message = f"Error in {contrasts} column names - verify 'contrast', 'treatment', 'control' in table"
        print(error_message)
        logging.error(error_message)
        return False
    if counts is not None:
        try:
            counts_df = pd.read_csv(Path(counts), sep='\t')
            for col in ["sgrna", "gene"]:
                assert col in map(str.lower, set(counts_df.columns))
        except AssertionError:
            error_message = f"{counts} is expected to contain columns 'sgRNA' and 'Gene'."
            print(error_message)
            logging.error(error_message)
            return False

    if sample_sheet is not None:
        try:
            assert set(sample_sheet.columns) == {"sample_name", "fastq_path"}
        except AssertionError:
            error_message = "Sample sheet columns are expected to be exactly: 'sample_name' and 'fastq_path'"
            print(error_message)
            logging.error(error_message)
            return False
    return True

def longest_common_substring(s1:str, s2:str):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    length = 0
    end_pos = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > length:
                    length = dp[i][j]
                    end_pos = i

    return s1[end_pos - length:end_pos]

def reverse_dir(input_dir:str, root:str, output_dir:str):
    # Finds and removes the common substring between input_dir and root
    # This allows us to prepend output_dir and search for files if the input and output directory structures are mirrored
    if input_dir in root:
        commmon_substring = longest_common_substring(input_dir, root)
        #print(f"This is the longest common string: {commmon_substring}")
        dir_str = root.replace(commmon_substring, "", 1)
    else:
        print(f"Input directory string not found for {root} file.")

    # Combine the paths to mirror the input directory
    out_path = Path(output_dir) / Path(dir_str.lstrip('\\'))

    # Create dir is doesn't exist
    if not out_path.exists:
        os.makedirs(out_path, parents=True)

    return out_path

def check_existing_files(input_dir: str, output_dir: str):
    input_path = Path(input_dir)

    unique_existing_dir = set()

    # Iterate through input_dir for files ending in "_RC" or ".fastq"
    for root, dirs, files in os.walk(input_path):
        if dirs:
            continue
        for file in files:
            if "_rc" in file.lower() or file.endswith(('.fq', '.fastq')):

                existing_output = reverse_dir(input_dir=input_dir, root=root, output_dir=output_dir)
                
                MGK_output_exists = any(existing_output.glob("*_gMGK.csv"))
                DZ_output_exists = any(existing_output.glob("*_gDZ.csv"))

                # Check if the parent directory exists in the output directory and add to existing_dir list
                if MGK_output_exists or DZ_output_exists:
                    print(f"Analysis for {file} exists in {existing_output}")
                    unique_existing_dir.add(str(existing_output))
                else:
                    # Could create directory and continue with loop to avoid doing this in run_pipeline()
                    #print(f"{existing_output} does not exist in the input directory.")
                    pass
            elif '_cnttbl' in file.lower():
                continue
            else:
                print(f"No files found in {input_dir} with _RC or .fastq in name.")
    
    # Return existing_dir list to skip directory analysis in later functions
    return unique_existing_dir

def assign_file_path(root:str, file_str:str, fastq_dirs: set = None):
    if fastq_dirs is None:
        fastq_dirs = set()

    file_name = str(Path(root) / Path(file_str))

    # Check for correct file names and endings for each type, create file path, and return "key" to be used for column name in path df later with "value" being file path
    if "_rc" in file_name.lower() and Path(file_str).suffix == '.csv':
        key = 'count_file_path'

        rc_table_path = (Path(root) / file_str)

        rc_data = pd.read_csv(rc_table_path)
        print(f"Found RC file: {file_str}")

        # Make count table tab-delimited ending with ".count.txt" for mageck_test() recognition
        # TODO: move this to mageck_count() as it is currently making count tables for read count tables that have no cnttbl?
        count_file = file_str.replace(".csv", ".count.txt")
        rc_data.to_csv(
            (Path(root) / count_file),
            sep="\t",
            index=False,
        )
        value = str(Path(root) / count_file)
        print(f"Made count table for: {file_str}")

        return key, value
    
    elif "_cnttbl" in file_name.lower() and Path(file_str).suffix in ['.txt', '.tsv']:
        key = 'cnttbl_path'
        value = str(Path(root) / file_str)
        print(f"Found cnttbl file: {file_str}")
        return key, value

    elif "fastq" in file_name.lower() and Path(file_str).suffix in ['.fastq', '.fq']:
        key = 'fastq_dir_path'
        if root not in fastq_dirs:
            fastq_dirs.add(root)
            value = str(Path(root))
            print(f"Found fastq directory: {root}")
            return key, value

    elif Path(file_str).suffix == '.xlsx':
        print(f"Convert all .xlsx files to .csv before running")
        print(f"Skipping .xlsx file: {file_str}")
    elif '.count' in file_name.lower():
        pass
    else:
        raise FileNotFoundError("Make sure fastq file ends in .fastq or .fq, cnttbl is tab-delimited and ends in .txt or .txv, and read count table is comma-delimited ending with .csv not .xlsx.")
@profile(stdout=False, filename = r'C:\Users\spejo\Documents\2_CRISPR_analysis_test_output\crispr_analysis_pipeline_baseline.prof')
def run_pipeline(input_dir:str, output_dir:str, overwrite:bool = False):
    library = r"C:\Users\spejo\Documents\1_CRISPR_analysis_test_input\FASTQ\TKOv3_guide_sequences_key.csv"
    
    # Load essential gene lists
    # TODO: make these configurable rather than hard-coded
    '''depmap_genes = pd.read_csv(Path("data") / "DepMap 22Q4 Common Essentials.csv")
    depmap_genes.columns = ["gene"]
    depmap_genes["gene"] = depmap_genes["gene"].apply(lambda x: x.split(" ")[0])
    depmap_genes = list(depmap_genes["gene"].unique())
    cellecta_ess = pd.read_csv(Path("data") / "cellecta_essential_genes.csv")
    cellecta_ess = list(cellecta_ess["gene"].unique())
    cellecta_noness = pd.read_csv(Path("data") / "cellecta_nonessential_genes.csv")
    cellecta_noness = list(cellecta_noness["gene"].unique())
    ess_genes = {
        "depmap": depmap_genes,
        "cellecta": cellecta_ess,
    }
    noness_genes = {"cellecta": cellecta_noness}'''

    rc_paths_list = {'count_file_path': [], 
                     'cnttbl_path': [],
                     'rc_out_dir': []}
    
    fastq_paths_list = {'fastq_dir_path': [],
                        'cnttbl_path': [],
                        'fastq_out_dir': []}
    
    rc_outpath_list = []
    fastq_outpath_list = []
    # Create list of screens that have already been analyzed
    analyzed_screen_list = check_existing_files(input_dir, output_dir)
    
    # Recursively walk through input directory - should have FASTQ and Read_count subdirectories
    for root, dirs, files in os.walk(input_dir):
        dir_df = pd.DataFrame(os.walk(input_dir))
        dir_df.to_csv(Path(output_dir) / Path("dir_df_test.csv"))

        # Skip directories for screens that have already been analyzed
        if root in analyzed_screen_list and not overwrite:
            print(f"{root} already analyzed, residing in {output_dir}")
            continue
        # Check if dirs is empty. Dirs is empty at the bottom of a directory tree. This should put you in place to find files.
        elif dirs:
            continue
        elif "read_count" in root.lower():
            # Check for cnttbl and read count table in directory. If both aren't present, it will ruin our paths df later.
            if any('_cnttbl' in f.lower() for f in files) and any('_rc' in f.lower() and '.csv' in f.lower() for f in files):
                for file in files:
                    rc_paths_df = None

                    # Define output path to mirror the input directory by removing input dir from root then prepending output dir to root, create dir if doesn't exist
                    rc_outpath = reverse_dir(input_dir=input_dir, root=root, output_dir=output_dir)

                    

                    # Check if test output files exist
                    rc_output_files_exist = any(list(rc_outpath.glob(pattern)) for pattern in ["*gene_summary.txt", "*DZ_output.tsv"])

                    # TODO: rc_output_files_exist always seems to evalate to true even when suffixes aren't in file name
                    # Does this matter? analyzed_screen_list should eliminate the pre-analyzed files AND this overwrite is redundant then
                    
                    if rc_output_files_exist and not overwrite:
                        print(f"Output files for {file} already exist, run with --overwrite to overwrite.")
                        continue

                    else:
                        rc_outpath_str = str(rc_outpath)
                        if rc_outpath_str not in rc_outpath_list:
                            rc_outpath_list.append(rc_outpath_str)
                        
                        # If files don't exist and/or overwrite=True, create path list for paired files to analyze
                        result = assign_file_path(root=root, file_str=file)
                        if result is not None:
                            key, value = result
                            new_rows = {key: [value]}
                            rc_paths_list[key].extend(new_rows[key])
                            
                        else:
                            pass
            else:
                continue
                    
        elif "fastq" in root.lower():
            # Check for cnttbl and fastq files in directory. If both aren't present, it will ruin our paths df later.
            if any('_cnttbl' in f.lower() for f in files) and any('.fq' in f.lower() or '.fastq' in f.lower() for f in files):      
                for file in files:
                    fastq_paths_df = None

                    # Define output path to mirror the input directory by removing input dir from root then prepending output dir to root, create dir if doesn't exist
                    fastq_outpath = reverse_dir(input_dir=input_dir, root=root, output_dir=output_dir)

                    # Check if test output files exist and return
                    fastq_output_files_exist = any(list(fastq_outpath.glob(pattern)) for pattern in ["*gene_summary.txt", "*DZ_output.tsv"])
                    
                    # TODO: fastq_output_files_exist always seems to evalate to true even when suffixes aren't in file name
                    # Does this matter? analyzed_screen_list should eliminate the pre-analyzed files AND this overwrite is redundant then
                    if fastq_output_files_exist and not overwrite:
                        print(f"Output files for {file} already exist, run with --overwrite to overwrite.")
                        continue
                    else:
                        fastq_outpath_str = str(fastq_outpath)
                        if fastq_outpath_str not in fastq_outpath_list:
                            fastq_outpath_list.append(fastq_outpath_str)
                            
                        # If files don't exist and/or overwrite=True, create path list for paired files to analyze
                        result = assign_file_path(root=root, file_str=file)
                        if result is not None:
                            key, value = result
                            new_rows = {key: [value]}
                            fastq_paths_list[key].extend(new_rows[key])
                            
                        else:
                            pass
        
        else:
            raise IsADirectoryError("Confirm Read_count or FASTQ are names of subdirectories within input directory.")
        
    rc_paths_list["rc_out_dir"] = rc_outpath_list
    fastq_paths_list["fastq_out_dir"] = fastq_outpath_list

    if not rc_paths_list:
        raise ValueError("Read count path list empty.")
    else:
        rc_paths_df = pd.DataFrame.from_dict(rc_paths_list)

        # Validate and analyze read count data
        for row in rc_paths_df.itertuples():
            
            if not validate_input_tables(contrasts=row.cnttbl_path, counts=row.count_file_path):
                print(f"Skipping analysis for files in {row.cnttbl_path} due to validation error.")
                continue

            mageck_test(input_file=row.count_file_path, 
                    contrasts=row.cnttbl_path, 
                    output_dir=row.rc_out_dir)
            # plot_QA(args.title, ess_genes, noness_genes)
            run_drugz(input_file=row.count_file_path, 
                  contrasts=row.cnttbl_path, 
                  output_dir=row.rc_out_dir)
            #plot_hits_drugz(title, cnttbl_rc, output_dir)
            print(f"Finished analysis on files in {root}")

    if not fastq_paths_list:
        raise ValueError("Fastq files directory list empty.")
    else:
        fastq_paths_df = pd.DataFrame.from_dict(fastq_paths_list)

        # Create count table from fastq, validate, analyze
        for row in fastq_paths_df.itertuples():
            sample_sheet = generate_sample_sheet(row.fastq_dir_path)

            if not validate_input_tables(contrasts=row.cnttbl_path, sample_sheet=sample_sheet):
                print(f"Skipping analysis for files in {row.cnttbl_path} due to validation error.")
                continue

            count_df = mageck_count(sample_sheet=sample_sheet, library_file=library, output_dir=row.fastq_dir_path)
            mageck_test(input_file=count_df, 
                        contrasts=row.cnttbl_path, 
                        output_dir=row.fastq_out_dir)
            run_drugz(input_file=count_df, 
                        contrasts=row.cnttbl_path, 
                        output_dir=row.fastq_out_dir)
            print(f"Finished analysis on files in {root}")
        

# TODO: could change mageck_count() and mageck_test() so count table output is easier to find/read

if __name__ == "__main__":
    # Example usage
    input_folder = r"C:\Users\spejo\Documents\1_CRISPR_analysis_test_input"
    output_folder = r"C:\Users\spejo\Documents\2_CRISPR_analysis_test_output"

    run_pipeline(input_folder, output_folder, overwrite=False)