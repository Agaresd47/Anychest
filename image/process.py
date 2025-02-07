import math
import random
from collections import Counter





def count_transformation_methods(transform_combinations):
    """
    Counts the occurrences of each transformation method in a set of transformation combinations.

    Args:
        transform_combinations (list): A list of dictionaries where each dictionary defines a transformation combination.

    Returns:
        list: A list of tuples where each tuple contains a transformation method and its total count.
    """
    

    method_counter = Counter()

    for combination in transform_combinations:
        for method in combination.keys():
            method_counter[method] += 1

    # Convert to a list of tuples (method, count)
    method_counts = list(method_counter.items())

    return method_counts


def calculate_intervals(method_counts, total_samples, parameter_ranges, special_list=None):
    """
    Calculate interval sizes for each transformation method based on their ranges and total usage.

    Args:
        method_counts (list): A list of tuples where each tuple contains a transformation method and its count.
        total_samples (int): Total number of samples to process.
        parameter_ranges (dict): A dictionary of parameter ranges for each transformation method.

    Returns:
        dict: A dictionary containing interval sizes for each method.
    """
    
    if special_list is None:
        special_list = []
        
    overlap = set(parameter_ranges.keys()).intersection(set(special_list))
    if overlap:
        raise ValueError(f"The following methods appear in both parameter_ranges and special_list: {overlap}")

    # Ensure all methods in method_counts are covered in parameter_ranges or special_list
    all_methods = set([method for method, _ in method_counts])
    covered_methods = set(parameter_ranges.keys()).union(set(special_list))

    if not all_methods.issubset(covered_methods):
        missing_methods = all_methods - covered_methods
        raise ValueError(f"The following methods are missing from parameter_ranges or special_list: {missing_methods}")
    
    
    intervals = {}

    for method, count in method_counts:
        total_appearance_count = count * total_samples * 2  # Each method applied twice per sample

        if method in parameter_ranges:
            # Handle evenly sampled ranges
            if isinstance(parameter_ranges[method], tuple):
                range_start, range_end = parameter_ranges[method]
                interval_size = (range_end - range_start) / total_appearance_count
                intervals[method] = interval_size

            # Handle special cases (like random_occlusions)
            elif isinstance(parameter_ranges[method], dict):
                special_intervals = {}
                for sub_param, (start, end) in parameter_ranges[method].items():
                    interval_size = (end - start) / total_appearance_count
                    special_intervals[sub_param] = interval_size
                intervals[method] = special_intervals

    return intervals

def special_list_function(method, params, order, parameter_ranges, intervals, method_counts):
    """
    Handles special cases for transformation methods.
    Args:
        method (str): The method name.
        params (any): The parameters for the method.
        order (int): The current order number.
        parameter_ranges (dict): Parameter ranges for methods.
        intervals (dict): Calculated intervals for each transformation method.
        method_counts (dict): Count of occurrences for each method.
    Returns:
        any: Updated parameters for the method.
    """
    if method == "random_occlusions":
        # Special handling for random_occlusions using intervals and lower parameter ranges
        num_occlusions = float(parameter_ranges[method]["num_occlusions"][0]) + float(((order - 1) * intervals[method]["num_occlusions"] / method_counts[method]))
        size_variation = random.uniform(1, 5)  # Add random variation factor between 1 and 5
        size_variation_sec = random.uniform(1, 3)  # Add random variation factor between 1 and 5
        size = (
            parameter_ranges[method]["size"][0] + ((order - 1) * intervals[method]["size"] / method_counts[method]) * size_variation,
            parameter_ranges[method]["size"][0] + ((order - 1) * intervals[method]["size"] / method_counts[method]) * size_variation_sec
        )
        return [math.ceil(num_occlusions), size]
    elif method == "add_gaussian_noise":
        # Handle add_gaussian_noise: mean is fixed at 0, std_dev varies
        std_dev = parameter_ranges[method][0] + ((order - 1) * intervals[method] / method_counts[method])
        return [0, std_dev]  # Always pass mean as 0
    elif method == "axis_aligned_flip":
        # Alternate between "vertical" and "horizontal" based on the order
        return ["vertical"] if order % 2 == 1 else ["horizontal"]
    elif method == "histogram_equalization":
        # Always return True for histogram_equalization
        return True
    return params

def process_transformations(order, transform_combinations, intervals, special_list, parameter_ranges, method_counts):
    """
    Process a single image-mask pair with specified transformation combinations and intervals.

    Args:
        order (int): The order number of the transformation (e.g., 1 to total_samples*2).
        transform_combinations (list): List of transformation combination dictionaries.
        intervals (dict): Calculated intervals for each transformation method.
        special_list (list): List of special methods that require custom handling.
        parameter_ranges (dict): Parameter ranges for methods.
        method_counts (dict): Count of occurrences for each method.

    Returns:
        list: Modified transform_combinations with updated parameters.
    """
    # Ensure method_counts is always a dictionary
    if isinstance(method_counts, list):
        method_counts = dict(method_counts)
        
        
    updated_combinations = []
    current_method_orders = {method: 0 for method in method_counts}
    
    

    for combination in transform_combinations:
        updated_combination = {}
        for method, params in combination.items():
            current_order = (order - 1) * method_counts[method] + current_method_orders[method] + 1
            if method in special_list:
                # Call special_list_function for special handling
                updated_combination[method] = special_list_function(method, params, current_order, parameter_ranges, intervals, method_counts)
            elif method in intervals:
                # Regular calculation process with base as lower parameter range
                if isinstance(intervals[method], dict):
                    updated_combination[method] = {}
                    for sub_param in intervals[method]:
                        try:
                            # Check if params[sub_param] exists and is a list
                            if sub_param not in params or not isinstance(params[sub_param], list):
                                raise ValueError(f"Parameter '{sub_param}' for method '{method}' must be a list.")
                            updated_combination[method][sub_param] = parameter_ranges[method][sub_param][0] + ((current_order - 1) * intervals[method][sub_param] / method_counts[method])
                        except Exception as e:
                            raise ValueError(f"Error with method '{method}' and parameter '{sub_param}': {e}")
                else:
                    updated_combination[method] = parameter_ranges[method][0] + ((current_order - 1) * intervals[method] / method_counts[method])
            else:
                raise ValueError(f"Method {method} is not recognized in intervals or special_list.")

            current_method_orders[method] = current_method_orders[method] + 1

        updated_combinations.append(updated_combination)

    return updated_combinations


def print_updated_combinations(updated_combinations):
    """
    Helper function to print updated combinations in a readable format.

    Args:
        updated_combinations (list): List of updated transformation combinations.
    """
    print("Updated Combinations:")
    for idx, combination in enumerate(updated_combinations, start=1):
        print(f"Combination {idx}:")
        for method, params in combination.items():
            print(f"  {method}: {params}")

