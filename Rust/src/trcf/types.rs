use std::fmt;

#[derive(Clone, Copy)]
#[derive(PartialEq)]
pub enum TransformMethod {
    NONE,
    WEIGHTED,
    DIFFERENCE,
    SUBTRACT_MA,
    NORMALIZE,
    NORMALIZE_DIFFERENCE
}

impl fmt::Display for TransformMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let printable = match *self {
            TransformMethod::NONE => "NONE",
            TransformMethod::NORMALIZE => "NORMALIZE" ,
            TransformMethod::NORMALIZE_DIFFERENCE=> "NORMALIZE_DIFFERENCE",
            TransformMethod::SUBTRACT_MA=> "SUBTRACT_MA",
            TransformMethod::DIFFERENCE => "DIFFERENCE",
            TransformMethod::WEIGHTED=> "WEIGHTED",
        };
        write!(f, "{}", printable)
    }
}

#[derive(Clone, Copy)]
#[derive(PartialEq)]
pub enum ImputationMethod {

    //use a fixed set of specified values (same as input dimension)
    FIXED,
    // last known value in each input dimension
    PREVIOUS,
    //next seen value in each input dimension
    NEXT,
    // linear interpolation
    LINEAR,
    // use the RCF imputation; but would often require a minimum number of
    // observations and would use defaults (often LINEAR) till that point
    USE_RCF
}

#[derive(Clone, Copy)]
#[derive(PartialEq)]
pub enum ForestMode {
    /**
     * a standard mode that uses shingling and most known applications; it uses the
     * last K data points where K=1 would correspond to non time series (population)
     * analysis
     */
    STANDARD,
    /**
     * time stamp is added automatically to data to correlate within RCF itself;
     * this is useful for event streaams and for modeling sparse events. Option is
     * provided to normalize the time gaps.
     */
    TIME_AUGMENTED,
    /**
     * uses various Fill-In strageies for data with gaps but not really sparse. Must
     * have shingleSize greater than 1, typically larger shingle size is better, and
     * so is fewer input dimensions
     */
    STREAMING_IMPUTE
}

// alternate scoring that can be thresholded differently

#[derive(Clone, Copy)]
#[derive(PartialEq)]
pub enum ScoringStrategy{
    EXPECTED_INVERSE_HEIGHT,
    /**
     * This is the same as STANDARD mode where the scoring function is switched to
     * distances between the vectors. Since RCFs build a multiresolution tree, and
     * in the aggregate, preserves distances to some approximation, this provides an
     * alternate anomaly detection mechanism which can be useful for shingleSize = 1
     * and (dynamic) population analysis via RCFs. Specifially it switches the
     * scoring to be based on the distance computation in the Density Estimation
     * (interpolation). This allows for a direct comparison of clustering based
     * outlier detection and RCFs over numeric vectors. All transformations
     * available to the STANDARD mode in the ThresholdedRCF are available for this
     * mode as well; this does not affect RandomCutForest core in any way. For
     * timeseries analysis the STANDARD mode is recommended, but this does provide
     * another option in combination with the TransformMethods.
     */
    DISTANCE
}

#[derive(Clone, Copy)]
#[derive(PartialEq)]
pub enum CorrectionMode {

    /**
     * default behavior, no correction
     */
    NONE,

    /**
     * due to transforms, or due to input noise
     */
    NOISE,

    /**
     * elimination due to multi mode operation, not in use currently
     */

    MULTI_MODE,

    /**
     * effect of an anomaly in shingle
     */

    ANOMALY_IN_SHINGLE,

    /**
     * conditional forecast, using conditional fields
     */

    CONDITIONAL_FORECAST,

    /**
     * forecasted value was not very different
     */

    FORECAST,

    /**
     * data drifts and level shifts, will not be corrected unless level shifts are
     * turned on
     */

    DATA_DRIFT

}

#[derive(Clone, Copy)]
#[derive(PartialEq)]
pub enum Calibration {

    NONE,

    /**
     * a basic staring point where the intervals are adjusted to be the minimal
     * necessary based on past error the intervals are smaller -- but the interval
     * precision will likely be close to 1 - 2 * percentile
     */
    MINIMAL,

    /**
     * a Markov inequality based interval, where the past error and model errors are
     * additive. The interval precision is likely higher than MINIMAL but so are the
     * intervals.
     */
    SIMPLE,

}