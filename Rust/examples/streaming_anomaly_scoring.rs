//! Streaming anomaly scores command line application.
//!
//! This example shows how to read data from an input CSV file and output
//! streaming anomaly scores. By "streaming", we mean that each observation is
//! first scored and then the model is updated with the observation.
//!
//! In this example, we use the `clap` package for a basic CLI. We use the `csv`
//! package to parse the input CSV data to be fed into an RCF model.
//!
extern crate clap;
use clap::{AppSettings, Clap};

extern crate csv;

use random_cut_forest::{RandomCutForest, RandomCutForestBuilder};

use std::error::Error;
use std::io;
use std::process;

/// Streaming random cut forest anomaly scoring.
///
/// Comma-delimited data is accepted via stdin. Anomaly score are output to
/// stdout. To read from file use the standard redirects.
///
/// Headers are automatically ignored. Many data contains a timestamp column
/// in the first column. The --ignore-first-column flag is useful in this
/// situation.
///
#[derive(Clap)]
#[clap(setting=AppSettings::ColoredHelp)]
struct Opts {
    /// Dimensionality of the input
    #[clap(short, long)]
    dimension: usize,

    /// Number of trees used in the model
    #[clap(short, long, default_value="50")]
    num_trees: usize,

    /// Number of samples per tree
    #[clap(short, long, default_value="256")]
    sample_size: usize,

    /// Parameter for time-decay reservoir sampling
    #[clap(short, long, default_value="0.000390625")]
    time_decay: f32,

    /// Ignore the first column of input. (e.g. timestamps)
    #[clap(long)]
    ignore_first_column: bool,
}

fn run(rcf: &mut RandomCutForest<f32>, ignore_first_column: bool) -> Result<(), Box<dyn Error>> {
    let dimension = rcf.dimension();
    let start_index: usize = match ignore_first_column {
        true => 1,
        false => 0,
    };

    let mut rdr = csv::Reader::from_reader(io::stdin());
    for result in rdr.records() {
        let record = result?;

        let mut point: Vec<f32> = Vec::with_capacity(dimension);
        for i in start_index..(dimension + start_index) {
            let value: f32 = record.get(i)?.parse::<f32>()?;
            point.push(value);
        }

        let score = rcf.anomaly_score(&point);
        rcf.update(point);
        println!("{}", score);
    }
    Ok(())
}

fn main() {
    let opts = Opts::parse();
    let mut rcf: RandomCutForest<f32> = RandomCutForestBuilder::new(opts.dimension)
        .num_trees(opts.num_trees)
        .sample_size(opts.sample_size)
        .time_decay(opts.time_decay)
        .build();

    if let Err(err) = run(&mut rcf, opts.ignore_first_column) {
        println!("error running example: {}", err);
        process::exit(1);
    }
}