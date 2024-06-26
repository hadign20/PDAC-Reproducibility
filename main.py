import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.data_loader import *
from src.preprocessing.data_cleaning import clean_clinical_data, normalize_radiomics_features
from src.feature_extraction.radiomics_features import extract_radiomics_features
#from src.feature_extraction.deep_features import extract_deep_features
from src.feature_selection.correlation import calculate_correlation_matrix, select_highly_correlated_features
from src.model.train_test_split import split_data
from src.model.train import train_model, evaluate_model
from src.visualization.plotting import plot_auc_with_ci
from src.feature_selection import inter_reader_agreement
from src.feature_selection.inter_reader_agreement import *
from src.visualization.plotting import *


data_path = r'D:\projects\pdac_reproducibility\PDACreproducibility'
result_path = r'D:\projects\pdac_reproducibility\pdac_reproducibility\results'



def main():
    # Paths to folders
    folders = {
        "Natally": "NatallySegmentations",
        "Maria": "MariaSegmentations",
        "Burcin": "Burcin_Segmentations",
        "Onur": "OnurSegmentations"
    }

    # Expert and Novice groups
    experts = ["Natally", "Maria"]
    novices = ["Burcin", "Onur"]

    # =========================================================
    # Radiomics Feature Extraction
    # =========================================================


    # =========================================================
    # Calculate dice scores
    # =========================================================
    dice_results = []

    # Loop through each case and calculate Dice scores
    for case in range(1, 39):  # Assuming there are 10 cases; adjust as necessary
        if case in [2, 8, 11, 14, 16, 17, 22, 37]: continue
        case_str = f"Case_{case:02d}"
        for seg_type in ["pre", "post"]:
            for organ in ["pancreas", "tumor"]:
                files = [f"{case_str}_{seg_type}_{organ}.mha"]
                print(f"Processing {case_str}_{seg_type}_{organ}.mha")
                segmentations = {}
                reference_images = {}
                for reader, folder in folders.items():
                    file_path = os.path.join(data_path, folder, files[0])
                    image = sitk.ReadImage(file_path)
                    segmentations[reader] = image
                    reference_images[reader] = image

                # Find the smallest shape among all images
                min_shape = min((image.GetSize() for image in reference_images.values()), key=lambda x: x[2])

                # Resample images to the smallest shape
                for reader in folders.keys():
                    reference_image = sitk.Image(min_shape, reference_images[reader].GetPixelID())
                    reference_image.CopyInformation(reference_images[reader])
                    segmentations[reader] = resample_image(reference_images[reader], reference_image)
                    segmentations[reader] = sitk.GetArrayFromImage(segmentations[reader])

                # Compare experts vs novices
                for exp in experts:
                    for nov in novices:
                        dice_score = dice_coefficient(segmentations[exp], segmentations[nov])
                        dice_results.append({
                            "Case": case_str,
                            "Comparison": "Expert vs Novice",
                            "Reader1": exp,
                            "Reader2": nov,
                            "Pre/Post": seg_type,
                            "Organ": organ,
                            "Dice Score": dice_score
                        })

                # Compare experts with each other
                for i in range(len(experts)):
                    for j in range(i + 1, len(experts)):
                        dice_score = dice_coefficient(segmentations[experts[i]], segmentations[experts[j]])
                        dice_results.append({
                            "Case": case_str,
                            "Comparison": "Experts",
                            "Reader1": experts[i],
                            "Reader2": experts[j],
                            "Pre/Post": seg_type,
                            "Organ": organ,
                            "Dice Score": dice_score
                        })

                # Compare novices with each other
                for i in range(len(novices)):
                    for j in range(i + 1, len(novices)):
                        dice_score = dice_coefficient(segmentations[novices[i]], segmentations[novices[j]])
                        dice_results.append({
                            "Case": case_str,
                            "Comparison": "Novices",
                            "Reader1": novices[i],
                            "Reader2": novices[j],
                            "Pre/Post": seg_type,
                            "Organ": organ,
                            "Dice Score": dice_score
                        })

    # Convert results to DataFrame
    dice_df = pd.DataFrame(dice_results)

    # Calculate statistics
    stats = dice_df.groupby(["Comparison", "Pre/Post", "Organ"])["Dice Score"].agg(["mean", "median", "std"]).reset_index()

    # Save statistics table
    stats.to_csv(os.path.join(result_path, "dice_score_statistics.csv"), index=False)
    dice_df.to_csv(os.path.join(result_path, "detailed_dice_scores.csv"), index=False)

    # Plotting
    sns.set(style="whitegrid")

    # Boxplot for Expert vs Novice (Overall)
    plot_boxplot(
        data=dice_df[dice_df["Comparison"] == "Expert vs Novice"],
        x="Organ",
        y="Dice Score",
        hue="Pre/Post",
        title="Dice Scores for Expert vs Novice (Overall)",
        save_path=os.path.join(result_path, "dice_scores_expert_vs_novice.png")
    )

    # Boxplot for Expert vs Novice (Per Case) in a single figure
    plot_facet_grid_boxplot(
        data=dice_df[dice_df["Comparison"] == "Expert vs Novice"],
        x="Organ",
        y="Dice Score",
        hue="Pre/Post",
        col="Case",
        col_wrap=5,
        order=["pancreas", "tumor"],
        hue_order=["pre", "post"],
        title="Dice Scores for Expert vs Novice (Per Case)",
        save_path=os.path.join(result_path, "dice_scores_expert_vs_novice_per_case.png")
    )

    # Boxplot for Experts (Overall)
    plot_boxplot(
        data=dice_df[dice_df["Comparison"] == "Experts"],
        x="Organ",
        y="Dice Score",
        hue="Pre/Post",
        title="Dice Scores for Experts (Overall)",
        save_path=os.path.join(result_path, "dice_scores_experts.png")
    )

    # Boxplot for Novices (Overall)
    plot_boxplot(
        data=dice_df[dice_df["Comparison"] == "Novices"],
        x="Organ",
        y="Dice Score",
        hue="Pre/Post",
        title="Dice Scores for Novices (Overall)",
        save_path=os.path.join(result_path, "dice_scores_novices.png")
    )




if __name__ == "__main__":
    main()


