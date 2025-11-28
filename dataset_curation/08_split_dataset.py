import json
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path


def build_stain_patient_counts(filepaths_IHC, patient_mapping):
    """ Groups images by IHC stain and patient via idnr → patient.
    
    Parameters:
        filepaths_IHC (list): List of file paths for IHC images.
        patient_mapping (dict): Dictionary mapping idnr to patient.
    
    Returns:
        stain_patient_counts (dict): Dictionary with stains as keys and dictionaries of patients and their image counts as values.
        patient_image_counts (dict): Dictionary with patients as keys and their total image counts as values. """
    
    # Group images by stain and patient via idnr → patient
    stain_patient_counts = defaultdict(lambda: defaultdict(int)) # dict. containing stain: {patient: nr_images, ...}
    patient_image_counts = defaultdict(int) # dict. containing patient: nr_images

    for file in filepaths_IHC:
        parts = file.split("_")
        stain = parts[2]
        idnr = parts[3] + '_' + parts[4]
        patient = patient_mapping.get(idnr)
        if patient:
            stain_patient_counts[stain][patient] += 1
            patient_image_counts[patient] += 1
        else:
            print(f"Warning: for ID {idnr}, no patient is found in patient_mapping. Skipping this file.")

    return stain_patient_counts, patient_image_counts  


def patient_overlap_across_stains(stain_patient_counts, print_results=False):
    """ Check for patients that are in multiple stains. 
    
    Parameters:
        stain_patient_counts (dict): Dictionary with stains as keys and dictionaries of patients and their image counts as values.
        print_results (bool): If True, prints which patients are in multiple stains and which stains they are associated with.
    
    Returns:
        patient_to_stains (dict): Dictionary with patients as keys and sets of stains they are associated with as values. """
    
    # Check for patients that are in multiple stains
    patient_to_stains = defaultdict(set) # dict. containing patient: {stain1, stain2, ...}
    for stain, patient_counts in stain_patient_counts.items():
        for patient in patient_counts:
            patient_to_stains[patient].add(stain)

    # Filter for patients that are in multiple stains
    overlapping_patients = {patient: stains for patient, stains in patient_to_stains.items() if len(stains) > 1} # dict. containing patient: {stain1, stain2, ...} for patients in multiple stains

    # Print the results
    if print_results:
        if overlapping_patients:
            print(f"Found {len(overlapping_patients)} patients in multiple stains.")
            for patient, stains in overlapping_patients.items():
                print(f"  {patient}: {sorted(stains)}")
        else:
            print("No patients shared across stains.")

    return patient_to_stains 


def split_dataset(stain_patient_counts_IHC, patient_to_stains_IHC, patient_image_counts_HE, train_ratio=0.7, val_ratio=0.1, target_per_stain_test=91):
    """ Splits the dataset into train, validation, and test sets based on the specified ratios and target number of images per stain in the test set.

    Parameters:
        stain_patient_counts_IHC (dict): Dictionary with stains as keys and dictionaries of patients and their image counts as values.
        patient_to_stains_IHC (dict): Dictionary with patients as keys and sets of stains they are associated with as values.
        patient_image_counts_HE (dict): Dictionary with patients as keys and their total image counts for HE images as values.
        train_ratio (float): Ratio of patients to assign to the training set. Default is 0.7.
        val_ratio (float): Ratio of patients to assign to the validation set. Default is 0.1.
        target_per_stain_test (int): Target number of images per stain in the test set. Default is 91.

    Returns:
        assigned_patients (dict): Dictionary with patients as keys and their assigned set (train/val/test) as values. """

    # Step 1: Shuffle patients 
    all_patients = list(patient_to_stains_IHC.keys())
    random.shuffle(all_patients)

    # Keep track of splitting
    assigned_patients = {} # dict. containing patient: train/val/test
    remaining_patients = set(all_patients) # patients that are not assigned yet

    # Step 2: test set filling
    stain_test_counts = Counter() # dict. containing stain: nr_image_in_test_set
    for stain, patient_counts in stain_patient_counts_IHC.items():
        while stain_test_counts[stain] < target_per_stain_test:
            candidates = [(patient, patient_counts[patient]) for patient in patient_counts if patient in remaining_patients]
            if not candidates:
                break  # No unassigned candidates left for this stain
            
            images_needed = target_per_stain_test - stain_test_counts[stain]

            # if more than 2/3 full, use closest-fit strategy
            if stain_test_counts[stain] >= (5/6) * target_per_stain_test:
                # Closest-fit: minimize overshoot
                best_patient, _ = min(candidates, key=lambda x: abs(images_needed - x[1])) 
            else:
                # Encourage more patients: choose randomly from the candidates
                best_patient, _ = random.choice(candidates) # or minimal to encourage more patients? --> min(candidates, key=lambda x: x[1])  
                        
            # Assign to test set
            assigned_patients[best_patient] = "test"
            remaining_patients.remove(best_patient)

            # Update test counts for all stains this patient touches because they are all in the test set
            for s in patient_to_stains_IHC[best_patient]:
                stain_test_counts[s] += stain_patient_counts_IHC[s].get(best_patient, 0)

    # Step 3: Assign remaining patients to train and validation sets
    val_cutoff = int(len(remaining_patients) * val_ratio/(val_ratio+train_ratio))
    val_patients = list(remaining_patients)[:val_cutoff]
    train_patients = list(remaining_patients)[val_cutoff:]

    for p in val_patients:
        assigned_patients[p] = "val"
    for p in train_patients:
        assigned_patients[p] = "train"

    # Step 4: Assign remainging patients to train set (patients that are not in IHC images but in HE images)
    nr_images_extra_in_train = 0
    for patient in set(patient_image_counts_HE.keys()):
        if patient not in assigned_patients.keys():
            nr_images_extra_in_train += 1
            assigned_patients[patient] = "train"
    print(f"{nr_images_extra_in_train} patients in train set that are not in IHC images but in HE images\n")

    return assigned_patients


def analyze_split(assigned_patients, patient_to_stains_IHC, patient_image_counts_IHC, patient_image_counts_HE, stain_patient_counts_IHC):
    """ Analyzes the split of the dataset into train, validation, and test sets.
    
    Parameters:
        assigned_patients (dict): Dictionary with patients as keys and their assigned set (train/val/test) as values.
        patient_to_stains_IHC (dict): Dictionary with patients as keys and sets of stains they are associated with as values.
        patient_image_counts_IHC (dict): Dictionary with patients as keys and their total image counts for IHC images as values.
        patient_image_counts_HE (dict): Dictionary with patients as keys and their total image counts for HE images as values.
        stain_patient_counts_IHC (dict): Dictionary with stains as keys and dictionaries of patients and their image counts as values.
        
    Returns:
        None. The function prints the analysis results. """
    
    # Look at how many images there are in each set and how many patients are in the test set per stain
    set_image_counts_IHC = {"train": 0, "val": 0, "test": 0}
    set_image_counts_HE = {"train": 0, "val": 0, "test": 0}
    test_stain_distribution = Counter()
    stain_to_test_patients = defaultdict(set)

    for patient, subset in assigned_patients.items():
        set_image_counts_IHC[subset] += patient_image_counts_IHC[patient]
        set_image_counts_HE[subset] += patient_image_counts_HE[patient]
        if subset == "test":
            for stain in patient_to_stains_IHC[patient]:
                test_stain_distribution[stain] += stain_patient_counts_IHC[stain].get(patient, 0)
                stain_to_test_patients[stain].add(patient)

    print("Patient-to-set assignment:")
    total_patients = len(assigned_patients)
    for subset in ["train", "val", "test"]:
        patients_in_set = [p for p, s in assigned_patients.items() if s == subset]
        print(f"  {subset}: {len(patients_in_set)} patients, {len(patients_in_set)/total_patients*100:.2f}% of total patients") 

    print("\nImage-to-set assignment for IHC images:")
    total_images = sum(set_image_counts_IHC.values())
    for subset in ["train", "val", "test"]:
        print(f"  {subset}: {set_image_counts_IHC[subset]} images, {set_image_counts_IHC[subset]/total_images*100:.2f}% of total patients")
    
    print("\nImage-to-set assignment for HE images:")
    total_images = sum(set_image_counts_HE.values())
    for subset in ["train", "val", "test"]:
        print(f"  {subset}: {set_image_counts_HE[subset]} images, {set_image_counts_HE[subset]/total_images*100:.2f}% of total patients")

    print("\nImage-to-stain assignment in test set:")
    for stain, count in test_stain_distribution.items():
        print(f"  {stain}: {count} images")

    print("\nPatient-to-stain assignment in test set:")
    for stain, patient_set in stain_to_test_patients.items():
        print(f"  {stain}: {len(patient_set)} patients")


def organize_images_by_split(data_folder, assigned_split, patient_mapping):
    """ Organizes images into train, validation, and test folders based on the assigned split.
    
    Parameters:
        data_folder (Path): Path to the folder containing the images.
        assigned_split (dict): Dictionary with patients as keys and their assigned set (train/val/test) as values.
        patient_mapping (dict): Dictionary mapping idnr to patient.
        
    Returns:
        None. The function moves the images into the corresponding folders. """

    output_dirs = {"train": data_folder / "train", "val": data_folder / "val", "test": data_folder / "test"}

    # Create output subfolders
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    # Loop over all images in the folder
    for img_path in data_folder.iterdir():
        if not img_path.is_file() or img_path.name in output_dirs:
            continue

        # Extract ID from filename 
        parts = img_path.stem.split("_")
        idnr = parts[3] + '_' + parts[4]
        patient = patient_mapping.get(idnr)

        if not patient:
            print(f"No patient found for ID: {idnr}")
            continue

        split = assigned_split.get(patient)
        if not split:
            print(f"No split found for patient: {patient}")
            continue

        # Copy to corresponding folder
        dest_path = output_dirs[split] / img_path.name
        shutil.move(img_path, dest_path)

    print(f"Image organization complete for {data_folder}.")

# set seed for reproducibility
random.seed(42)

# define paths
patient_mapping_path = r""
images_HE = r""
images_IHC = r""
masks_HE = r""
masks_IHC = r""
assigned_split_path = r""

if __name__ == "__main__":

    # Load the patient to ID mapping
    with open(patient_mapping_path, "r") as f:
        patient_mapping = json.load(f)

    # Get all images
    images_HE = Path(images_HE)
    images_IHC = Path(images_IHC)
    filepaths_HE = [f.name for f in images_HE.iterdir() if f.is_file()]
    filepaths_IHC = [f.name for f in images_IHC.iterdir() if f.is_file()]

    masks_HE = Path(masks_HE)
    masks_IHC = Path(masks_IHC)
    
    # Prepare mappings so so we can split per patient 
    stain_patient_counts_IHC, patient_image_counts_IHC  = build_stain_patient_counts(filepaths_IHC, patient_mapping)
    _, patient_image_counts_HE  = build_stain_patient_counts(filepaths_HE, patient_mapping)
    patient_to_stains_IHC = patient_overlap_across_stains(stain_patient_counts_IHC)
    
    # Split the dataset
    assigned_split = split_dataset(stain_patient_counts_IHC, patient_to_stains_IHC, patient_image_counts_HE, train_ratio=0.7, val_ratio=0.1, target_per_stain_test=91)

    # Analyze the split
    analyze_split(assigned_split, patient_to_stains_IHC, patient_image_counts_IHC, patient_image_counts_HE, stain_patient_counts_IHC)

    # Save the assigned split to a JSON file
    with open(assigned_split_path, "w") as f:
        json.dump(assigned_split, f, indent=4)

    # Organize images into train, validation, and test folders
    organize_images_by_split(images_HE, assigned_split, patient_mapping)
    organize_images_by_split(images_IHC, assigned_split, patient_mapping)
    organize_images_by_split(masks_HE, assigned_split, patient_mapping)
    organize_images_by_split(masks_IHC, assigned_split, patient_mapping)
