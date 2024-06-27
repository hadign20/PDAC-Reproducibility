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


EXTRACT_RADIOMICS = False
CALCULATE_DICE_SCORE = False
CALCULATE_ICC_VALUES = True


data_path = r'D:\projects\pdac_reproducibility\PDACreproducibility'
result_path = r'D:\projects\pdac_reproducibility\pdac_reproducibility\results'
params_path = r'D:\projects\pdac_reproducibility\pdac_reproducibility\src\feature_extraction\CT.yaml'


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
    if EXTRACT_RADIOMICS:
        for seg_type in ["pre", "post"]:
            for organ in ["pancreas", "tumor"]:
                reader_data = {reader: [] for reader in folders}

                for case in range(1, 39):
                    if case in [2, 8, 11, 14, 16, 17, 22, 37]: continue
                    case_str = f"Case_{case:02d}"

                    seg_file = f"{case_str}_{seg_type}_{organ}.mha"
                    print(f"Processing {case_str}_{seg_type}_{organ}.mha")

                    for reader, folder in folders.items():
                        seg_full_path = os.path.join(data_path, folder, seg_file)
                        if not os.path.exists(seg_full_path):
                            print(f"file {seg_full_path} don't exist..!")
                            continue
                        else:
                            image_file = f"{case_str.split('_')[1]}_neo_pdac_{seg_type}_volume.mha"
                            image_full_path = os.path.join(data_path, "WholeCT_MHA", image_file)
                            if not os.path.exists(image_full_path):
                                print(f"file {image_full_path} don't exist..!")
                                continue
                            else:
                                features = extract_radiomics_features(image_full_path, seg_full_path, params_path)
                                features_row = {'Case_ID': case_str}
                                features_row.update(features)
                                reader_data[reader].append(features_row)

                output_excel_path = os.path.join(result_path, "PyRadiomics_" + seg_type + "_" + organ + ".xlsx")
                with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
                    for reader, data in reader_data.items():
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=reader, index=False)
                print(f"Radiomics features saved to {output_excel_path}")



    # =========================================================
    # Calculate dice scores
    # =========================================================
    if CALCULATE_DICE_SCORE:
        dice_results = []

        # Loop through each case and calculate Dice scores
        for case in range(1, 39):
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

    # =========================================================
    # Calculate ICC values
    # =========================================================
    if CALCULATE_ICC_VALUES:
        excel_files = {
            'post_pancreas': os.path.join(result_path, 'PyRadiomics_post_pancreas.xlsx'),
            'post_tumor': os.path.join(result_path, 'PyRadiomics_post_tumor.xlsx'),
            'pre_pancreas': os.path.join(result_path, 'PyRadiomics_pre_pancreas.xlsx'),
            'pre_tumor': os.path.join(result_path, 'PyRadiomics_pre_tumor.xlsx')
        }

        comparisons = [
            ('experts', experts),
            ('novices', novices),
            ('experts_vs_novices', experts + novices)
        ]

        for key, file_path in excel_files.items():
            sheets = ["Natally", "Maria", "Burcin", "Onur"]
            data = load_excel_sheets(file_path, sheets)

            # Combine all data into one DataFrame for each feature
            combined_data = {}
            features = data[sheets[0]].columns[39:]  # Assuming the first two columns are Case_ID and Rater
            for feature in features:
                combined_data[feature] = []
                for sheet in sheets:
                    df = data[sheet]
                    df['Rater'] = sheet
                    combined_data[feature].append(df[['Case_ID', feature]].assign(Rater=sheet))
                combined_data[feature] = pd.concat(combined_data[feature])

            # Calculate ICC3 for each comparison
            for comp_name, comp_raters in comparisons:
                comp_data = {feature: combined_data[feature][combined_data[feature]['Rater'].isin(comp_raters)] for
                             feature in features}
                comp_prepared_data = {feature: prepare_data_for_icc(comp_data[feature], feature) for feature in
                                      features}
                icc3_df = calculate_icc3_for_features(comp_prepared_data)

                # Save ICC3 values to Excel
                output_path = os.path.join(result_path, f'ICC3_{comp_name}_{key}.xlsx')
                icc3_df.to_excel(output_path, index=False)
                print(f"ICC3 values saved to {output_path}")






if __name__ == "__main__":
    main()


