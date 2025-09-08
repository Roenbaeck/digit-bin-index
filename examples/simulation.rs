// examples/simulation.rs

use digit_bin_index::DigitBinIndex;
use rust_decimal_macros::dec;

fn main() {
    println!("--- Simulation 1: Default Precision (3) ---");
    let mut dbi1 = DigitBinIndex::new();

    dbi1.add(101, dec!(0.543));    // 0.543
    dbi1.add(102, dec!(0.120));    // 0.120 (explicitly 3 decimal places)
    dbi1.add(103, dec!(0.12345));  // 0.12345 (will be binned as 0.123)
    
    println!("Initial state: {} individuals, total weight = {}", dbi1.count(), dbi1.total_weight());
    if let Some((id, _)) = dbi1.select_and_remove() {
        println!("Selected ID: {}", id);
    }
    println!("Final state: {} individuals, total weight = {}\n", dbi1.count(), dbi1.total_weight());


    println!("--- Simulation 2: High Precision (5) ---");
    let mut dbi2 = DigitBinIndex::with_precision(5);

    dbi2.add(201, dec!(0.543,));    // 0.54300
    dbi2.add(202, dec!(0.12));      // 0.12000
    dbi2.add(203, dec!(0.12345));   // 0.12345

    println!("Initial state: {} individuals, total weight = {}", dbi2.count(), dbi2.total_weight());
     if let Some((id, _)) = dbi2.select_and_remove() {
        println!("Selected ID: {}", id);
    }
    println!("Final state: {} individuals, total weight = {}", dbi2.count(), dbi2.total_weight());
}