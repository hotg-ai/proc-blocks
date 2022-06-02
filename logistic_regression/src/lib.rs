use linfa::traits::{Fit, Predict};
use linfa_logistic::LogisticRegression;
use linfa_datasets::winequality;

// Example on using binary labels different from 0 and 1
let dataset = winequality().map_targets(|x| if *x > 6 { "good" } else { "bad" });
let model = LogisticRegression::default().fit(&dataset).unwrap();
let prediction = model.predict(&dataset);