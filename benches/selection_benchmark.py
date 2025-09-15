# You'll need to install psutil first:
# pip install psutil

import timeit
import random
import os
import psutil
from digit_bin_index import DigitBinIndex

def get_memory_usage_mb():
    """Returns the memory usage of the current process in megabytes."""
    p = psutil.Process(os.getpid())
    # rss (Resident Set Size) is a good measure of the actual physical memory a process is using
    return p.memory_info().rss / (1024 * 1024)

def run_benchmark(num_items_to_test):
    """
    Runs a heavily scaled benchmark and reports memory usage.
    """
    print(f"--- Benchmarking with {num_items_to_test:,} items ---")
    print("This may take some time...")

    # --- Memory and Time Benchmark for Populating the Index ---
    mem_before_add = get_memory_usage_mb()

    setup_code_add = f"""
from digit_bin_index import DigitBinIndex
import random
# Prepare the data beforehand so we don't time the random number generation
data = [(i, random.random()) for i in range({num_items_to_test})]
index = DigitBinIndex.with_precision_and_capacity(3, {num_items_to_test})
"""
    add_stmt = "for item_id, weight in data: index.add(id=item_id, weight=weight)"
    
    add_time = timeit.timeit(stmt=add_stmt, setup=setup_code_add, number=1)
    
    # To measure memory, we need to actually create the index in this process
    # The setup for timeit runs in a separate context.
    data_for_mem_test = [(i, random.random()) for i in range(num_items_to_test)]
    index = DigitBinIndex.with_precision_and_capacity(3, num_items_to_test)
    for item_id, weight in data_for_mem_test:
        index.add(id=item_id, weight=weight)
    
    mem_after_add = get_memory_usage_mb()
    memory_used_by_index = mem_after_add - mem_before_add
    
    print(f"Time to add {num_items_to_test:,} items: {add_time:.6f} seconds")
    print(f"Estimated memory for index: {memory_used_by_index:.2f} MB")

    # --- Setup for selection benchmarks ---
    # We reuse the created index for the selection benchmarks to save time
    # Note: timeit will still run its own setup, but this makes the script flow logically.
    setup_code_select = f"""
from digit_bin_index import DigitBinIndex
import random
# Re-create the index within timeit's context
index = DigitBinIndex.with_precision_and_capacity(3, {num_items_to_test})
for i in range({num_items_to_test}):
    index.add(id=i, weight=random.random())
"""

    # --- Benchmark: select_and_remove ---
    number_of_runs_single = 100_000
    select_stmt = "index.select_and_remove()"
    select_time = timeit.timeit(stmt=select_stmt, setup=setup_code_select, number=number_of_runs_single)
    print(f"{number_of_runs_single:,} single selections: {select_time:.6f} seconds")


    # --- Benchmark: select_many_and_remove ---
    number_of_runs_many = 1_000
    select_many_stmt = "index.select_many_and_remove(100)"
    select_many_time = timeit.timeit(stmt=select_many_stmt, setup=setup_code_select, number=number_of_runs_many)
    print(f"{number_of_runs_many:,} multi-selections of 100: {select_many_time:.6f} seconds")
    print("-" * (35 + len(f"{num_items_to_test:,}")))
    
    # Return memory used to help decide if we should continue
    return memory_used_by_index


if __name__ == "__main__":
    # Test with significantly larger index sizes
    benchmark_sizes = [100_000, 1_000_000, 10_000_000, 100_000_000]
    
    # Get available system memory
    available_mem_mb = psutil.virtual_memory().available / (1024 * 1024)
    print(f"Total available system memory: {available_mem_mb:,.2f} MB")
    
    # Set a safety margin of 10% of available RAM, since next size is 10 times larger
    safety_margin = 0.10
    max_safe_mem = available_mem_mb * safety_margin

    for num in benchmark_sizes:
        print(f"\nChecking benchmark for {num:,} items...")
        
        # We'll do a rough pre-calculation. Let's assume memory scales linearly.
        # This is a bit of a chicken-and-egg problem. We need to run the test
        # to know the memory, but we want to know the memory before we run it.
        # For now, we will run and check afterwards. A more advanced version could
        # extrapolate from the first run.
        
        mem_used = run_benchmark(num)
        
        if mem_used > max_safe_mem:
            print(f"\nWARNING: Benchmark used {mem_used:.2f} MB, which exceeds the safety limit of {max_safe_mem:.2f} MB.")
            print("Stopping further benchmarks to prevent out-of-memory errors.")
            break